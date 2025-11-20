import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import pickle
from typing import List, Tuple

class CHICOFrameDataset(Dataset):
    """
    Dataset for Single Frame Pose Estimation
    Input: Single Video Frame
    Target: Corresponding 3D Skeleton (24 joints)
    """
    
    def __init__(self, 
                 base_path: str,
                 subjects: List[str],
                 actions: List[str] = None,
                 stride: int = 10, # High stride to reduce dataset size (video is 30fps)
                 resize: Tuple[int, int] = (224, 224)):
        
        self.base_path = base_path
        self.episodes_path = os.path.join(base_path, 'datasets', 'episodes')
        self.skeletons_path = os.path.join(base_path, 'datasets', '3d_skeletons')
        
        self.subjects = subjects
        self.stride = stride
        self.resize = resize
        
        if actions is None:
            actions = ['hammer', 'lift', 'place-hp', 'place-lp', 
                      'polish', 'span_heavy', 'span_light']
        self.actions = actions
        
        self.samples = []
        self._load_dataset_index()
        
    def _load_dataset_index(self):
        print(f"Indexing frame dataset for subjects: {self.subjects}")
        
        for subject in self.subjects:
            for action in self.actions:
                # Load Skeleton
                skel_file = os.path.join(self.skeletons_path, subject, f"{action}.pkl")
                if not os.path.exists(skel_file):
                    continue
                with open(skel_file, 'rb') as f:
                    skeleton_data = pickle.load(f)
                
                # Load Video Info
                video_folder = os.path.join(self.episodes_path, f"{subject}_{action}")
                if not os.path.exists(video_folder):
                    continue
                
                video_files = sorted([f for f in os.listdir(video_folder) if f.endswith('.mp4')])
                
                # CHICO dataset structure:
                # The videos (e.g. 00_03.mp4, 00_06.mp4, 00_12.mp4) are DIFFERENT VIEWS of the SAME action.
                # They are synchronized. Frame N in any video corresponds to index N in the PKL.
                #
                # LOGIC:
                # We iterate through ALL 3 video files.
                # For each video, we pair Frame(i) with PKL_Row(i).
                # This means the model learns that:
                #   Image_View1(i) -> Pose(i)
                #   Image_View2(i) -> Pose(i)
                #   Image_View3(i) -> Pose(i)
                # This makes the model robust to different camera angles (Top, Side, Front).
                
                for v_file in video_files:
                    v_path = os.path.join(video_folder, v_file)
                    cap = cv2.VideoCapture(v_path)
                    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    
                    # The PKL and Video should have same length, but take min to be safe
                    limit = min(num_frames, len(skeleton_data))
                    
                    # Add samples with stride
                    for i in range(0, limit, self.stride):
                        self.samples.append({
                            'video_path': v_path,
                            'frame_idx': i,          # Frame index in video
                            'skeleton_data': skeleton_data, 
                            'skeleton_idx': i        # Corresponding index in PKL
                        })
                            
        print(f"Indexed {len(self.samples)} frames.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load Frame
        cap = cv2.VideoCapture(sample['video_path'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample['frame_idx'])
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            frame = np.zeros((self.resize[1], self.resize[0], 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, self.resize)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        # Normalize
        frame = np.array(frame, dtype=np.float32) / 255.0
        frame = torch.from_numpy(frame).permute(2, 0, 1) # (C, H, W)
        
        # Load Skeleton
        # [human(15,3), robot(9,3)]
        # Use the specific index for this frame
        frame_data = sample['skeleton_data'][sample['skeleton_idx']]
        human = np.array(frame_data[0], dtype=np.float32)
        robot = np.array(frame_data[1], dtype=np.float32)
        pose = np.vstack([human, robot]) # (24, 3)
        
        # Normalize Pose? 
        # CHICO is in mm. Neural networks like normalized data (e.g. -1 to 1 or 0 to 1).
        # But for now let's keep it raw and maybe use a higher learning rate or normalize in the model.
        # Better: Normalize by dividing by a constant (e.g. 2000mm)
        pose = pose / 1000.0 # Convert to meters
        
        pose = torch.from_numpy(pose)
        
        return frame, pose

def create_pose_estimation_dataloaders(base_path: str,
                                     train_subjects: List[str],
                                     val_subjects: List[str],
                                     batch_size: int = 32,
                                     stride: int = 10,
                                     num_workers: int = 4):
    
    train_ds = CHICOFrameDataset(base_path, train_subjects, stride=stride)
    val_ds = CHICOFrameDataset(base_path, val_subjects, stride=stride)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader
