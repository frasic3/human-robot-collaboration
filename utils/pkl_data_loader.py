"""CHICO Dataset Loader

Loads and prepares the CHICO dataset for pose forecasting.
Each sample contains a temporal window of past poses (input)
and the sequence of future poses to predict (target).

Dataset structure:
- 20 subjects (S00-S19)
- 7 collaborative human-robot actions
- Each frame: 15 human joints + 9 robot joints = 24 total joints
- 3D coordinates (x, y, z) in millimeters
"""
import pickle
import numpy as np
import os
import random
from typing import List, Tuple, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class BalancedSequenceBatchSampler:
    """Batch sampler for sequence mode with ~equal class composition per batch.

    Uses per-sequence label = max(output_risk) (safety-first).
    Produces batches with approximately 1/3 Safe, 1/3 Near, 1/3 Collision.

    Notes:
    - This enforces balance per batch (not just per epoch).
    - Sampling is with replacement when a class has insufficient samples.
    """

    def __init__(self, dataset: 'CHICODataset', batch_size: int, drop_last: bool = True, seed: int | None = None):
        if dataset.mode != 'sequence':
            raise ValueError('BalancedSequenceBatchSampler requires dataset.mode=="sequence"')
        if batch_size <= 0:
            raise ValueError('batch_size must be > 0')

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.rng = np.random.default_rng(seed)

        labels = _get_sampling_labels(dataset)
        self.class_indices: dict[int, np.ndarray] = {
            c: np.where(labels == c)[0] for c in range(3)
        }

        # Within collision class, prefer sequences with more collision frames.
        # This increases the density of positive frames without changing class balance.
        self._collision_sampling_p: np.ndarray | None = None
        coll_idx = self.class_indices.get(2, None)
        if coll_idx is not None and len(coll_idx) > 0:
            collision_frames = np.zeros(len(coll_idx), dtype=np.float64)
            for j, seq_i in enumerate(coll_idx.tolist()):
                try:
                    y = dataset.sequences[int(seq_i)].get('output_risk', None)
                except Exception:
                    y = None
                if y is None:
                    continue
                y = np.asarray(y)
                collision_frames[j] = float(np.sum(y == 2))

            # Fixed heuristic weights: (1 + collision_frames)
            w = 1.0 + collision_frames
            s = float(w.sum())
            if s > 0:
                self._collision_sampling_p = w / s

        # Precompute per-batch counts (handle non-multiple of 3)
        base = self.batch_size // 3
        rem = self.batch_size % 3
        # Distribute remainder as [Safe, Near, Collision] with rotation per batch
        self.base_counts = [base, base, base]
        self.rem = rem

        # Define epoch length: we mimic standard DataLoader length
        n = len(dataset)
        if self.drop_last:
            self.num_batches = n // self.batch_size
        else:
            self.num_batches = int(np.ceil(n / self.batch_size))

        if self.num_batches <= 0:
            self.num_batches = 1

    def __len__(self) -> int:
        return self.num_batches

    def _sample_from_class(self, c: int, k: int) -> list[int]:
        idx = self.class_indices.get(c, None)
        if idx is None or len(idx) == 0 or k <= 0:
            return []

        # Collision class: weighted by collision-frame density
        if c == 2 and self._collision_sampling_p is not None and len(self._collision_sampling_p) == len(idx):
            if len(idx) >= k:
                return self.rng.choice(idx, size=k, replace=False, p=self._collision_sampling_p).tolist()
            return self.rng.choice(idx, size=k, replace=True, p=self._collision_sampling_p).tolist()

        if len(idx) >= k:
            return self.rng.choice(idx, size=k, replace=False).tolist()
        # Not enough samples: sample with replacement
        return self.rng.choice(idx, size=k, replace=True).tolist()

    def __iter__(self):
        # Rotate which class receives the remainder to avoid always biasing class 0
        for b in range(self.num_batches):
            counts = self.base_counts.copy()
            for r in range(self.rem):
                counts[(b + r) % 3] += 1

            batch = []
            for c in range(3):
                batch.extend(self._sample_from_class(c, counts[c]))

            self.rng.shuffle(batch)

            if self.drop_last and len(batch) != self.batch_size:
                continue
            yield batch

class CHICODataset(Dataset):
    """
    Dataset for CHICO 3D Skeleton
    Each sample contains:
    - Input: sequence of T past frames (human + robot poses)
    - Target: sequence of P future frames (human + robot poses)
    """
    
    def __init__(self, 
                 dataset_path: str,
                 subjects: List[str] = None,
                 actions: List[str] = None,
                 input_frames: int = 10,  # T frames di input
                 output_frames: int = 25,  # P frames da predire (1 sec @ 25fps)
                 stride: int = 1,
                 use_crash: bool = True,
                 normalize: bool = True,
                 stats: Tuple[np.ndarray, np.ndarray] = None,
                 mode: str = 'sequence',  # 'sequence' or 'single_frame'
                 augment_collision: bool = False,  # Data augmentation per classe Collision
                 augment_factor: int = 200):  # Numero di copie augmentate per ogni sample Collision
        """
        Args:
            dataset_path: path to the dataset folder
            subjects: list of subjects to use (e.g. ['S00', 'S01'])
            actions: list of actions to use (e.g. ['hammer', 'lift'])
            input_frames: number of input frames (past history)
            output_frames: number of frames to predict (future)
            stride: stride to create sequences
            use_crash: if True, include _CRASH versions
            normalize: if True, normalize coordinates (root-relative)
            stats: tuple (mean, std) for Z-score normalization. If None, no Z-score.
            mode: 'sequence' for LSTM (T+P frames), 'single_frame' for MLP (1 frame)
            augment_collision: if True, apply data augmentation to Collision class
            augment_factor: number of augmented copies for each Collision sample
        """
        self.dataset_path = dataset_path
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.stride = stride
        self.normalize = normalize
        self.stats = stats
        self.mode = mode
        self.augment_collision = augment_collision
        self.augment_factor = augment_factor
        
        # Default: all subjects and all actions
        if subjects is None:
            subjects = [f'S{i:02d}' for i in range(20)]
        
        if actions is None:
            actions = ['hammer', 'lift', 'place-hp', 'place-lp', 
                      'polish', 'span_heavy', 'span_light']
        
        self.subjects = subjects
        self.actions = actions
        self.use_crash = use_crash
        
        # Load all data
        self.sequences = []
        self.load_data()
        
        if self.normalize:
            self._process_dataset()
        
        # Apply data augmentation to Collision class (after normalization)
        if self.augment_collision and self.mode == 'single_frame':
            self._augment_collision_samples()
        
        print(f"Dataset loaded: {len(self.sequences)} samples (Mode: {self.mode})")
        
    def load_data(self):
        """Load all .pkl files and create sequences"""
        for subject in self.subjects:
            subject_path = os.path.join(self.dataset_path, subject)
            
            if not os.path.exists(subject_path):
                print(f"Warning: {subject} not found")
                continue
            
            for action in self.actions:
                # Normal version
                file_path = os.path.join(subject_path, f'{action}.pkl')
                if os.path.exists(file_path):
                    self._process_file(file_path)
                
                # CRASH version
                if self.use_crash:
                    crash_path = os.path.join(subject_path, f'{action}_CRASH.pkl')
                    if os.path.exists(crash_path):
                        self._process_file(crash_path)
    
    def _process_file(self, file_path: str):
        """Process a single file and create sequences"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        total_frames = len(data)
        
        if self.mode == 'single_frame': # For MLP treat each frame as a sample

            for i in range(0, total_frames, self.stride):
                frame = data[i]
                human_joints = np.array(frame[0], dtype=np.float32) # (15, 3)
                robot_joints = np.array(frame[1], dtype=np.float32) # (9, 3)
                full_pose = np.vstack([human_joints, robot_joints]) # (24, 3)
                risk = frame[2]

                self.sequences.append({
                    'input': full_pose, # (24, 3)
                    'risk': risk
                })
                
        else: # Sequence mode for LSTM
            sequence_length = self.input_frames + self.output_frames
            
            # Create sliding sequences, be careful with bounds
            for start_idx in range(0, total_frames - sequence_length + 1, self.stride):
                
                input_seq_list = []
                output_risk_list = []
                
                for i in range(sequence_length):
                    frame_idx = start_idx + i
                    frame = data[frame_idx]
                    
                    if i < self.input_frames:
                        # Input part (First T frames): Extract Poses
                        human_joints = np.array(frame[0], dtype=np.float32) # (15, 3)
                        robot_joints = np.array(frame[1], dtype=np.float32) # (9, 3)
                        full_pose = np.vstack([human_joints, robot_joints]) # (24, 3)
                        input_seq_list.append(full_pose)
                    else:
                        # Output part (Next P frames): Extract Risk Labels
                        risk = frame[2]
                        output_risk_list.append(risk)
                
                # Convert to numpy arrays with explicit shapes
                input_seq = np.array(input_seq_list)                     # Shape: (T, 24, 3)
                output_risk = np.array(output_risk_list, dtype=np.int64) # Shape: (P,)

                self.sequences.append({
                    'input': input_seq,
                    'output_risk': output_risk
                })
    
    # Get dataset length, useful for DataLoader
    def __len__(self) -> int:
        return len(self.sequences)
    
    def _augment_collision_samples(self):
        """
        Data augmentation per la classe Collision (risk=2).
        Applica trasformazioni per aumentare i campioni della classe minoritaria.
        """
        print("Applying data augmentation to Collision class...")
        
        # Trova tutti i campioni Collision
        collision_samples = [s for s in self.sequences if s['risk'] == 2]
        original_count = len(collision_samples)
        
        if original_count == 0:
            print("Warning: No Collision samples found for augmentation")
            return
        
        augmented_samples = []
        
        for sample in collision_samples:
            pose = sample['input'].copy()  # (24, 3) normalizzato
            
            for _ in range(self.augment_factor):
                aug_pose = self._apply_augmentation(pose.copy())
                augmented_samples.append({
                    'input': aug_pose,
                    'risk': 2  # Collision
                })
        
        # Aggiungi i campioni augmentati al dataset
        self.sequences.extend(augmented_samples)
        print(f"Collision augmentation: {original_count} -> {original_count + len(augmented_samples)} samples (+{len(augmented_samples)})")
    
    def _apply_augmentation(self, pose: np.ndarray) -> np.ndarray:
        """
        Applica trasformazioni random a una posa 3D.
        Args:
            pose: (24, 3) array di joint positions
        Returns:
            Posa augmentata (24, 3)
        """
        # Scegli casualmente quali augmentation applicare
        aug_type = np.random.randint(0, 4)
        
        if aug_type == 0:
            # 1. Gaussian Noise
            noise = np.random.normal(0, 0.02, pose.shape).astype(np.float32)
            pose = pose + noise
            
        elif aug_type == 1:
            # 2. Random Scaling (95% - 105%)
            scale = np.random.uniform(0.95, 1.05)
            pose = pose * scale
        
        elif aug_type == 2:
            # 3. Rotation around Y-axis (vertical)
            angle = np.random.uniform(-15, 15) * np.pi / 180  # +/- 15 gradi
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ], dtype=np.float32)
            pose = pose @ rotation_matrix.T
        
        elif aug_type == 3:
            # Joint Dropout (Simulazione errore sensore)
            num_dropout = np.random.randint(1, 3) 
            idxs = np.random.choice(pose.shape[0], num_dropout, replace=False)
            pose[idxs] = 0.0

        else:
            # 4. Combination: Noise + Scaling + Small Rotation
            noise = np.random.normal(0, 0.01, pose.shape).astype(np.float32)
            scale = np.random.uniform(0.98, 1.02)
            angle = np.random.uniform(-5, 5) * np.pi / 180
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ], dtype=np.float32)
            pose = (pose + noise) * scale @ rotation_matrix.T
        
        return pose
    
    def _process_dataset(self):
        """
        Applies normalization to the entire dataset in memory.
        1. Root-Relative Normalization
        2. Statistics Calculation (if not provided)
        3. Z-Score Normalization
        """
        print("Processing dataset normalization...")
        need_stats = self.stats is None
        all_poses_for_stats = [] if need_stats else None
        
        if self.mode == 'single_frame':
            # 1. Root-relative normalization + collect stats
            for i in range(len(self.sequences)):
                pose = self.sequences[i]['input'] # (24, 3)
                pose = pose - pose[0, :] # Subtract root (first joint)
                self.sequences[i]['input'] = pose
                if need_stats:
                    all_poses_for_stats.append(pose)
            
            # 2. Compute stats if needed
            if need_stats:
                print("Computing dataset statistics (Single Frame)...")
                all_poses = np.stack(all_poses_for_stats, axis=0) # (N, 24, 3)
                mean = all_poses.mean(axis=(0, 1))
                std = all_poses.std(axis=(0, 1))
                self.stats = (mean, std)
                print(f"Computed stats - Mean: {mean}, Std: {std}")
            
            # 3. Z-score normalization
            mean, std = self.stats
            for i in range(len(self.sequences)):
                self.sequences[i]['input'] = (self.sequences[i]['input'] - mean) / (std + 1e-8)
                
        else:
            # 1. Root-relative normalization + collect stats
            for i in range(len(self.sequences)):
                input_seq = self.sequences[i]['input']
                input_seq = input_seq - input_seq[0, 0, :] # Subtract root (first joint, first frame)
                self.sequences[i]['input'] = input_seq
                if need_stats:
                    all_poses_for_stats.append(input_seq)

            # 2. Compute stats if needed
            if need_stats:
                print("Computing dataset statistics (Sequence)...")
                all_poses = np.concatenate(all_poses_for_stats, axis=0)
                mean = all_poses.mean(axis=(0, 1))
                std = all_poses.std(axis=(0, 1))
                self.stats = (mean, std)
                print(f"Computed stats - Mean: {mean}, Std: {std}")
                
            # 3. Z-score normalization
            mean, std = self.stats
            for i in range(len(self.sequences)):
                self.sequences[i]['input'] = (self.sequences[i]['input'] - mean) / (std + 1e-8)

    # Get a sample, useful for DataLoader
    def __getitem__(self, idx: int):
        sample = self.sequences[idx]
        
        # Convert input to tensor
        x = torch.from_numpy(sample['input']).float()
        
        if self.mode == 'single_frame':
            # (24, 3) -> (72,)
            x = x.view(-1)
            y = torch.tensor(sample['risk'], dtype=torch.long)
        else:
            # (T, 24, 3) -> (T, 72)
            x = x.view(x.size(0), -1)
            y = torch.from_numpy(sample['output_risk']).long()
            
        return x, y

# Helper functions for WeightedRandomSampler
def _get_sampling_labels(dataset: CHICODataset) -> np.ndarray:
    """
    Estrae un array di label (una per sample) usabile per il WeightedRandomSampler.
    
    - single_frame: usa direttamente 'risk'
    - sequence: usa max(output_risk) — se almeno un frame futuro è Collision,
      l'intero sample è considerato Collision (safety-first)
    """
    if dataset.mode == 'single_frame':
        return np.array([s['risk'] for s in dataset.sequences], dtype=np.int64)
    return np.array([int(np.max(s['output_risk'])) for s in dataset.sequences], dtype=np.int64)


def _augment_sequence_sample(sample: dict, augment_type: int = None) -> dict:
    """Augmentation avanzata per sequenze (LSTM) preservando la geometria.

    Implementa 4 trasformazioni (scelte a caso con probabilità pesate):
      0) Time warping (variazione velocità) via interpolazione temporale
      1) Rotazione globale della scena attorno all'asse Y
      2) Traslazione globale piccola (shift costante)
      3) Scaling leggero

    Nota: il rischio futuro ('output_risk') non cambia.
    """

    input_seq = sample['input'].copy()
    original_shape = input_seq.shape

    # Support both (T, 24, 3) and flattened (T, 72)
    if input_seq.ndim == 2:
        T, F = input_seq.shape
        input_seq_3d = input_seq.reshape(T, -1, 3)
    else:
        input_seq_3d = input_seq
        T = input_seq_3d.shape[0]

    if augment_type is None:
        augment_type = int(np.random.choice([0, 1, 2, 3], p=[0.4, 0.2, 0.2, 0.2]))

    def _resample_to_length(data_TF: np.ndarray, new_len: int) -> np.ndarray:
        """Resample (T, F) -> (new_len, F) with linear interpolation."""
        old_len = data_TF.shape[0]
        if new_len <= 1 or old_len <= 1 or new_len == old_len:
            return data_TF
        old_x = np.linspace(0.0, 1.0, old_len, dtype=np.float64)
        new_x = np.linspace(0.0, 1.0, new_len, dtype=np.float64)
        out = np.empty((new_len, data_TF.shape[1]), dtype=np.float32)
        for f in range(data_TF.shape[1]):
            out[:, f] = np.interp(new_x, old_x, data_TF[:, f].astype(np.float64)).astype(np.float32)
        return out

    # 0) TIME WARPING (cambio di velocità)
    if augment_type == 0:
        speed_factor = float(np.random.uniform(0.8, 1.2))
        flat = input_seq_3d.reshape(T, -1).astype(np.float32)
        mid_len = int(max(2, round(T * speed_factor)))
        warped = _resample_to_length(flat, mid_len)
        flat = _resample_to_length(warped, T)
        input_seq_3d = flat.reshape(input_seq_3d.shape)

    # 1) GLOBAL SCENE ROTATION (sicura: ruota tutto insieme)
    elif augment_type == 1:
        angle_deg = float(np.random.uniform(-45.0, 45.0))
        angle_rad = angle_deg * np.pi / 180.0
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array(
            [
                [cos_a, 0.0, sin_a],
                [0.0, 1.0, 0.0],
                [-sin_a, 0.0, cos_a],
            ],
            dtype=np.float32,
        )
        input_seq_3d = input_seq_3d @ rotation_matrix.T

    # 2) GLOBAL TRANSLATION (shift piccolo)
    elif augment_type == 2:
        shift_x = float(np.random.uniform(-0.05, 0.05))
        shift_z = float(np.random.uniform(-0.05, 0.05))
        shift_vec = np.array([shift_x, 0.0, shift_z], dtype=np.float32)
        input_seq_3d = input_seq_3d + shift_vec

    # 3) RANDOM SCALING (leggero)
    else:
        scale = float(np.random.uniform(0.95, 1.05))
        input_seq_3d = input_seq_3d * scale

    # Restore original shape
    if len(original_shape) == 2:
        input_seq = input_seq_3d.reshape(original_shape)
    else:
        input_seq = input_seq_3d

    input_seq = input_seq.astype(np.float32)
    return {
        'input': input_seq,
        'output_risk': sample['output_risk'].copy(),
    }


def augment_collision_sequences(dataset: CHICODataset, augment_factor: int = 10) -> None:
    """
    Data augmentation per sequenze contenenti Collision (per LSTM).
    Una sequenza è considerata Collision se max(output_risk) == 2.
    
    Args:
        dataset: CHICODataset in mode='sequence'
        augment_factor: numero di copie augmentate per ogni sample Collision
    """
    if dataset.mode != 'sequence':
        print("Warning: augment_collision_sequences only works with sequence mode")
        return
    
    # quiet augmentation (no verbose prints)
    
    # Trova tutte le sequenze con almeno una Collision
    collision_samples = [s for s in dataset.sequences if np.max(s['output_risk']) == 2]
    original_count = len(collision_samples)
    
    if original_count == 0:
        return
        return
    
    augmented_samples = []
    
    for sample in collision_samples:
        for _ in range(augment_factor):
            aug_sample = _augment_sequence_sample(sample)
            augmented_samples.append(aug_sample)
    
    # Aggiungi i campioni augmentati al dataset
    dataset.sequences.extend(augmented_samples)


def _build_weighted_sampler(labels: np.ndarray, num_classes: int = 3) -> WeightedRandomSampler:
    """
    Crea un WeightedRandomSampler a partire da label intere.
    
    Peso di ogni classe = 1 / (numero di sample in quella classe)
    → classi rare vengono pescate più spesso
    """
    # {0:2000, 1:500, 2:100} 
    class_counts = np.bincount(labels, minlength=num_classes)
    
    # Evita divisione per zero se una classe è assente
    class_weights = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        if class_counts[c] > 0:
            class_weights[c] = 1.0 / class_counts[c]
    
    # Assegna a ogni sample il peso della sua classe
    sample_weights = class_weights[labels]
    sample_weights_tensor = torch.from_numpy(sample_weights).double()
    
    return WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True
    )


def _build_weighted_sampler_from_sample_weights(sample_weights: np.ndarray) -> WeightedRandomSampler:
    """Crea un WeightedRandomSampler direttamente da pesi per-sample."""
    sample_weights = np.asarray(sample_weights, dtype=np.float64)
    # clamp minimo per evitare pesi 0
    sample_weights = np.clip(sample_weights, 1e-12, None)
    sample_weights_tensor = torch.from_numpy(sample_weights).double()
    return WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True
    )


def _build_equal_class_mass_sequence_sampler_with_collision_frames(
    train_dataset: CHICODataset,
    sampler_collision_scale: float = 1.0,
    sampler_collision_power: float = 1.0,
    sampler_min_weight: float = 1.0,
) -> WeightedRandomSampler:
    """Sampler per sequenze con massa uguale per classe (Safe/Near/Collision).

    - La classe per-sequenza è definita come max(output_risk) (safety-first).
    - La probabilità totale di pescare ciascuna classe presente è uguale.
    - Dentro la classe Collision, le sequenze con più collision-frame pesano di più.

    Nota: garantisce bilanciamento *in media sull'epoca* (WeightedRandomSampler).
    """
    seq_labels = _get_sampling_labels(train_dataset)  # (N,) in {0,1,2}
    n = len(seq_labels)
    class_counts = np.bincount(seq_labels, minlength=3)

    present_classes = [c for c in range(3) if class_counts[c] > 0]
    if not present_classes:
        return _build_weighted_sampler_from_sample_weights(np.ones(n, dtype=np.float64))

    # raw per-sample weights within each class
    raw = np.ones(n, dtype=np.float64)

    collision_frames = np.zeros(n, dtype=np.float64)
    for i, s in enumerate(train_dataset.sequences):
        y = s.get('output_risk', None)
        if y is None:
            continue
        y = np.asarray(y)
        collision_frames[i] = float(np.sum(y == 2))

    min_w = float(sampler_min_weight)
    scale = float(sampler_collision_scale)
    power = float(sampler_collision_power)

    is_collision_seq = (seq_labels == 2)
    if np.any(is_collision_seq):
        raw[is_collision_seq] = min_w + scale * (collision_frames[is_collision_seq] ** power)

    # equalize class mass
    class_mass = 1.0 / float(len(present_classes))
    sample_weights = np.zeros(n, dtype=np.float64)
    for c in present_classes:
        idx = np.where(seq_labels == c)[0]
        w = raw[idx]
        s = float(w.sum())
        if s <= 0:
            w = np.ones_like(w)
            s = float(w.sum())
        sample_weights[idx] = (w / s) * class_mass

    return _build_weighted_sampler_from_sample_weights(sample_weights)

# main function to create DataLoaders
def create_pkl_dataloaders(dataset_path: str,
                       train_subjects: List[str],
                       val_subjects: List[str],
                       test_subjects: List[str],
                       input_frames: int = 10,
                       output_frames: int = 25,
                       batch_size: int = 32,
                       num_workers: int = 0,
                       mode: str = 'sequence',
                       use_weighted_sampler: bool = False,
                       augment_collision: bool = False,
                       augment_factor: int = 5,
                       sampler_strategy: str = 'max',
                       sampler_collision_scale: float = 1.0,
                       sampler_collision_power: float = 1.0,
                       sampler_min_weight: float = 1.0,
                       seed: int | None = None,
                       train_stride: int | None = None,
                       val_stride: int | None = None,
                       test_stride: int | None = None) -> Tuple[DataLoader, DataLoader, DataLoader, Tuple[np.ndarray, np.ndarray]]:
    """
    Create dataloaders for train, validation and test
    
    Args:
        use_weighted_sampler: if True, use WeightedRandomSampler for balanced training
        augment_collision: if True, apply data augmentation to Collision class (training only)
        augment_factor: number of augmented copies for each Collision sample
    """
    
    # Resolve default strides
    if train_stride is None:
        train_stride = 1 if mode == 'sequence' else 10
    if val_stride is None:
        val_stride = 5 if mode == 'sequence' else 20
    if test_stride is None:
        test_stride = 10 if mode == 'sequence' else 30

    # 1. Create training dataset first (stats will be computed internally)
    train_dataset = CHICODataset(
        dataset_path=dataset_path,
        subjects=train_subjects,
        input_frames=input_frames,
        output_frames=output_frames,
        stride=int(train_stride),
        use_crash=True,
        normalize=True,
        stats=None,
        mode=mode,
        augment_collision=augment_collision,
        augment_factor=augment_factor
    )
    
    # 2. Get calculated stats
    stats = train_dataset.stats
    
    val_dataset = CHICODataset(
        dataset_path=dataset_path,
        subjects=val_subjects,
        input_frames=input_frames,
        output_frames=output_frames,
        stride=int(val_stride),
        use_crash=True,
        normalize=True,
        stats=stats,
        mode=mode
    )
    
    test_dataset = CHICODataset(
        dataset_path=dataset_path,
        subjects=test_subjects,
        input_frames=input_frames,
        output_frames=output_frames,
        stride=int(test_stride),
        use_crash=True,
        normalize=True,
        stats=stats,
        mode=mode
    )
    
    # Check if CUDA is available for pin_memory
    pin_memory = torch.cuda.is_available()

    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

        def _seed_worker(worker_id: int):
            s = (int(seed) + int(worker_id)) % (2**32)
            np.random.seed(s)
            random.seed(s)
            torch.manual_seed(s)

        worker_init_fn = _seed_worker
    
    # Apply sequence augmentation for LSTM if requested (must happen before sampling setup)
    if augment_collision and mode == 'sequence':
        augment_collision_sequences(train_dataset, augment_factor)

    # Sampling setup
    sampler = None
    batch_sampler = None
    shuffle_train = True

    if use_weighted_sampler:
        shuffle_train = False

        if mode == 'single_frame':
            labels = _get_sampling_labels(train_dataset)
            sampler = _build_weighted_sampler(labels)
        else:
            # Standardized behavior: balanced batches across classes (no collision-length preference)
            batch_sampler = BalancedSequenceBatchSampler(train_dataset, batch_size=batch_size, drop_last=True, seed=seed)
    
    if batch_sampler is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
            generator=generator
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
            generator=generator
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        generator=generator
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        generator=generator
    )
    
    return train_loader, val_loader, test_loader, stats

