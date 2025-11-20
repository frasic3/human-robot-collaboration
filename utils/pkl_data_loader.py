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
        
        # data is a list of frames, each frame contains [human_joints, robot_joints]
        # human_joints: (15, 3), robot_joints: (9, 3)
        
        total_frames = len(data)
        sequence_length = self.input_frames + self.output_frames
        
        # Create sliding sequences
        for start_idx in range(0, total_frames - sequence_length + 1, self.stride):
            end_idx = start_idx + sequence_length
            
            # Extract the sequence
            sequence = []
            for frame_idx in range(start_idx, end_idx):
                frame = data[frame_idx]
                human_joints = np.array(frame[0], dtype=np.float32)  # (15, 3)
                robot_joints = np.array(frame[1], dtype=np.float32)  # (9, 3)
                
                # Concatenate human and robot: (24, 3)
                full_pose = np.vstack([human_joints, robot_joints])
                sequence.append(full_pose)
            
            sequence = np.array(sequence)  # (T+P, 24, 3)
            
            # Split into input and output
            input_seq = sequence[:self.input_frames]  # (T, 24, 3)
            output_seq = sequence[self.input_frames:]  # (P, 24, 3)
            
            self.sequences.append({
                'input': input_seq,
                'output': output_seq,
                'subject': subject,
                'action': action,
                'crash': crash,
                'start_frame': start_idx
            })
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, bool]:
        sample = self.sequences[idx]
        
        input_seq = sample['input'].copy()  # (T, 24, 3)
        output_seq = sample['output'].copy()  # (P, 24, 3)
        
        if self.normalize:
            # Normalize relative to the first frame (root-relative)
            root = input_seq[0, 0, :].copy()  # first joint of first frame
            input_seq = input_seq - root
            output_seq = output_seq - root
        
        # Apply Z-score normalization if stats are provided
        if self.stats is not None:
            mean, std = self.stats
            input_seq = (input_seq - mean) / (std + 1e-8)
            output_seq = (output_seq - mean) / (std + 1e-8)
        
        # Convert to torch tensors
        input_tensor = torch.from_numpy(input_seq).float()
        output_tensor = torch.from_numpy(output_seq).float()
        
        return input_tensor, output_tensor, sample['action'], sample['crash']
    
    def get_statistics(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate dataset statistics for normalization (on root-relative data)"""
        all_poses = []
        print("Computing dataset statistics...")
        
        # We need to iterate manually to apply root-relative normalization
        # but NOT Z-score (since we are computing it)
        for sample in self.sequences:
            input_seq = sample['input']
            output_seq = sample['output']
            
            # Concatenate to process full sequence
            seq = np.concatenate([input_seq, output_seq], axis=0)
            
            if self.normalize:
                root = seq[0, 0, :].copy()
                seq = seq - root
            
            all_poses.append(seq)
        
        all_poses = np.concatenate(all_poses, axis=0)  # (Total_Frames, 24, 3)
        
        # Compute mean and std across frames and joints (global for x, y, z)
        mean = all_poses.mean(axis=(0, 1))  # (3,)
        std = all_poses.std(axis=(0, 1))    # (3,)
        
        print(f"Computed stats - Mean: {mean}, Std: {std}")
        return mean, std


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
    
    # 1. Create training dataset first (without stats)
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
    
    # 2. Calculate statistics from training data
    stats = train_dataset.get_statistics()
    
    # 3. Update training dataset with stats
    train_dataset.stats = stats
    
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
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, stats

