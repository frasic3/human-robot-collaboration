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
from typing import List, Tuple, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

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
        aug_type = np.random.randint(0, 5)
        
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
        
        elif aug_type == 4:
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
                       augment_factor: int = 5) -> Tuple[DataLoader, DataLoader, DataLoader, Tuple[np.ndarray, np.ndarray]]:
    """
    Create dataloaders for train, validation and test
    
    Args:
        use_weighted_sampler: if True, use WeightedRandomSampler for balanced training
        augment_collision: if True, apply data augmentation to Collision class (training only)
        augment_factor: number of augmented copies for each Collision sample
    """
    
    # 1. Create training dataset first (stats will be computed internally)
    print("Initializing Training Dataset...")
    train_dataset = CHICODataset(
        dataset_path=dataset_path,
        subjects=train_subjects,
        input_frames=input_frames,
        output_frames=output_frames,
        stride=1 if mode == 'sequence' else 10, # Higher stride for single frame to avoid too much data
        use_crash=True,
        normalize=True,
        stats=None,
        mode=mode,
        augment_collision=augment_collision,  # Data augmentation solo per training
        augment_factor=augment_factor
    )
    
    # 2. Get calculated stats
    stats = train_dataset.stats
    
    print("Initializing Validation Dataset...")
    val_dataset = CHICODataset(
        dataset_path=dataset_path,
        subjects=val_subjects,
        input_frames=input_frames,
        output_frames=output_frames,
        stride=5 if mode == 'sequence' else 20,
        use_crash=True,
        normalize=True,
        stats=stats,
        mode=mode
    )
    
    print("Initializing Test Dataset...")
    test_dataset = CHICODataset(
        dataset_path=dataset_path,
        subjects=test_subjects,
        input_frames=input_frames,
        output_frames=output_frames,
        stride=10 if mode == 'sequence' else 30,
        use_crash=True,
        normalize=True,
        stats=stats,
        mode=mode
    )
    
    # Check if CUDA is available for pin_memory
    pin_memory = torch.cuda.is_available()
    
    # Create WeightedRandomSampler for balanced training if requested
    sampler = None
    shuffle_train = True
    
    if use_weighted_sampler and mode == 'single_frame':
        print("Creating WeightedRandomSampler for balanced training...")
        # Get all labels from training set
        labels = np.array([seq['risk'] for seq in train_dataset.sequences])
        
        # Count samples per class
        class_counts = np.bincount(labels)
        print(f"Class distribution: {dict(enumerate(class_counts))}")
        
        # Compute weight for each sample (inverse of class frequency)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]
        sample_weights = torch.from_numpy(sample_weights).double()
        
        # Create sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle_train = False  # Cannot use shuffle with sampler
        print(f"WeightedRandomSampler created with {len(sample_weights)} samples")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader, stats

