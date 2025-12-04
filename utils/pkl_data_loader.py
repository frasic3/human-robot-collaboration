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
                 stats: Tuple[np.ndarray, np.ndarray] = None,
                 mode: str = 'sequence'): # 'sequence' or 'single_frame'
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
        """
        self.dataset_path = dataset_path
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.stride = stride
        self.normalize = normalize
        self.stats = stats
        self.mode = mode
        
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
                       num_workers: int = 4,
                       mode: str = 'sequence') -> Tuple[DataLoader, DataLoader, DataLoader, Tuple[np.ndarray, np.ndarray]]:
    """
    Create dataloaders for train, validation and test
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
        mode=mode
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

