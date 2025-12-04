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
from torch.utils.data import Dataset, DataLoader

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
                 stats: Tuple[np.ndarray, np.ndarray] = None):
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
        """
        self.dataset_path = dataset_path
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.stride = stride
        self.normalize = normalize
        self.stats = stats
        
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
        
        print(f"Dataset loaded: {len(self.sequences)} sequences")
        
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
                    self._process_file(file_path, subject, action, crash=False)
                
                # CRASH version
                if self.use_crash:
                    crash_path = os.path.join(subject_path, f'{action}_CRASH.pkl')
                    if os.path.exists(crash_path):
                        self._process_file(crash_path, subject, action, crash=True)
    
    def _process_file(self, file_path: str, subject: str, action: str, crash: bool):
        """Process a single file and create sequences"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # data is a list of frames
        # Old format: [human_joints, robot_joints]
        # New format: [human_joints, robot_joints, risk_label]
        
        total_frames = len(data)
        sequence_length = self.input_frames + self.output_frames
        
        # Create sliding sequences
        for start_idx in range(0, total_frames - sequence_length + 1, self.stride):
            end_idx = start_idx + sequence_length
            
            # Extract the sequence
            sequence = []
            risk_sequence = []
            
            for frame_idx in range(start_idx, end_idx):
                frame = data[frame_idx]
                human_joints = np.array(frame[0], dtype=np.float32)  # (15, 3)
                robot_joints = np.array(frame[1], dtype=np.float32)  # (9, 3)
                
                # Read risk from file if available, else compute/default
                if len(frame) >= 3:
                    risk = frame[2]
                else:
                    # Fallback: compute on the fly if missing (e.g. old dataset)
                    diff = human_joints[:, None, :] - robot_joints[None, :, :]
                    min_dist = np.linalg.norm(diff, axis=-1).min()
                    if min_dist < 200: risk = 2
                    elif min_dist < 300: risk = 1
                    else: risk = 0
                
                # Concatenate human and robot: (24, 3)
                full_pose = np.vstack([human_joints, robot_joints])
                sequence.append(full_pose)
                risk_sequence.append(risk)
            
            sequence = np.array(sequence)  # (T+P, 24, 3)
            risk_sequence = np.array(risk_sequence, dtype=np.int64)
            
            # Split into input and output
            input_seq = sequence[:self.input_frames]  # (T, 24, 3)
            output_seq = sequence[self.input_frames:]  # (P, 24, 3)
            
            input_risk = risk_sequence[:self.input_frames]
            output_risk = risk_sequence[self.input_frames:]

            self.sequences.append({
                'input': input_seq,
                'output': output_seq,
                'input_risk': input_risk,
                'output_risk': output_risk,
                'subject': subject,
                'action': action,
                'crash': crash,
                'start_frame': start_idx
            })
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def _process_dataset(self):
        """
        Applies normalization to the entire dataset in memory.
        1. Root-Relative Normalization
        2. Statistics Calculation (if not provided)
        3. Z-Score Normalization
        """
        print("Processing dataset normalization...")
        all_poses_for_stats = []
        
        # 1. Apply Root-Relative Normalization to ALL sequences
        for i in range(len(self.sequences)):
            input_seq = self.sequences[i]['input']
            output_seq = self.sequences[i]['output']
            
            # Root is the first joint of the first frame of input
            root = input_seq[0, 0, :].copy()
            
            self.sequences[i]['input'] = input_seq - root
            self.sequences[i]['output'] = output_seq - root
            
            # Collect data for stats if we need to compute them
            if self.stats is None:
                # Concatenate input and output for stats calculation
                seq = np.concatenate([self.sequences[i]['input'], self.sequences[i]['output']], axis=0)
                all_poses_for_stats.append(seq)

        # 2. Calculate Statistics if not provided
        if self.stats is None:
            print("Computing dataset statistics...")
            all_poses = np.concatenate(all_poses_for_stats, axis=0)
            mean = all_poses.mean(axis=(0, 1))
            std = all_poses.std(axis=(0, 1))
            self.stats = (mean, std)
            print(f"Computed stats - Mean: {mean}, Std: {std}")
            
        # 3. Apply Z-Score Normalization
        mean, std = self.stats
        for i in range(len(self.sequences)):
            self.sequences[i]['input'] = (self.sequences[i]['input'] - mean) / (std + 1e-8)
            self.sequences[i]['output'] = (self.sequences[i]['output'] - mean) / (std + 1e-8)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, bool]:
        sample = self.sequences[idx]
        # Data is already normalized in memory
        return (torch.from_numpy(sample['input']).float(), 
                torch.from_numpy(sample['output']).float(), 
                torch.from_numpy(sample['input_risk']).long(),
                torch.from_numpy(sample['output_risk']).long(),
                sample['action'], 
                sample['crash'])


def create_pkl_dataloaders(dataset_path: str,
                       train_subjects: List[str],
                       val_subjects: List[str],
                       test_subjects: List[str],
                       input_frames: int = 10,
                       output_frames: int = 25,
                       batch_size: int = 32,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader, Tuple[np.ndarray, np.ndarray]]:
    """
    Create dataloaders for train, validation and test
    
    Example split:
    - Train: S00-S15 (16 subjects)
    - Val: S16-S17 (2 subjects)
    - Test: S18-S19 (2 subjects)
    """
    
    # 1. Create training dataset first (stats will be computed internally)
    print("Initializing Training Dataset...")
    train_dataset = CHICODataset(
        dataset_path=dataset_path,
        subjects=train_subjects,
        input_frames=input_frames,
        output_frames=output_frames,
        stride=1,
        use_crash=True,
        normalize=True,
        stats=None
    )
    
    # 2. Get calculated stats
    stats = train_dataset.stats
    
    print("Initializing Validation Dataset...")
    val_dataset = CHICODataset(
        dataset_path=dataset_path,
        subjects=val_subjects,
        input_frames=input_frames,
        output_frames=output_frames,
        stride=5,  # Larger stride for validation
        use_crash=True,
        normalize=True,
        stats=stats
    )
    
    print("Initializing Test Dataset...")
    test_dataset = CHICODataset(
        dataset_path=dataset_path,
        subjects=test_subjects,
        input_frames=input_frames,
        output_frames=output_frames,
        stride=10,  # Larger stride for test
        use_crash=True,
        normalize=True,
        stats=stats
    )
    
    # Check if CUDA is available for pin_memory
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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

