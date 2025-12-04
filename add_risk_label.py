import pickle
import numpy as np
import os
from tqdm import tqdm

def compute_frame_risk(human_joints, robot_joints):
    """
    Compute risk level for a single frame.
    Risk classes: 0 (Safe), 1 (Near-collision), 2 (Collision)
    Thresholds: Safe > 630mm, Near-collision 130-630mm, Collision <= 130mm
    """
    # human_joints: (15, 3)
    # robot_joints: (9, 3)
    
    # Expand dims for broadcasting: (15, 1, 3) - (1, 9, 3)
    diff = human_joints[:, None, :] - robot_joints[None, :, :]
    # Calculate Euclidean distances between all pairs of joints
    dists = np.linalg.norm(diff, axis=-1) # (15, 9)
    
    min_dist = dists.min()
    
    if min_dist <= 130:
        return 2 # Collision
    elif min_dist <= 630:
        return 1 # Near-collision
    else:
        return 0 # Safe

def process_dataset(src_root, dst_root):
    print(f"Processing dataset from {src_root} to {dst_root}")
    
    # Count files for tqdm
    pkl_files = []
    for root, dirs, files in os.walk(src_root):
        for file in files:
            if file.endswith(".pkl"):
                pkl_files.append(os.path.join(root, file))
    
    print(f"Found {len(pkl_files)} .pkl files.")
    
    for src_path in tqdm(pkl_files, desc="Processing files"):
        # Determine relative path to maintain structure
        rel_path = os.path.relpath(src_path, src_root)
        dst_path = os.path.join(dst_root, rel_path)
        
        # Create destination directory
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        
        process_file(src_path, dst_path)

def process_file(src_path, dst_path):
    with open(src_path, 'rb') as f:
        data = pickle.load(f)
    
    new_data = []
    for frame in data:
        # frame is expected to be [human_joints, robot_joints]
        # human_joints might be list or numpy array
        human = np.array(frame[0])
        robot = np.array(frame[1])
        
        risk = compute_frame_risk(human, robot)
        
        # Create new frame structure: [human, robot, risk]
        new_frame = [frame[0], frame[1], risk]
        new_data.append(new_frame)
        
    with open(dst_path, 'wb') as f:
        pickle.dump(new_data, f)

if __name__ == "__main__":
    base_path = r"c:\Users\Proprietario\Desktop\human-robot-collaboration"
    src_dataset = os.path.join(base_path, "datasets", "3d_skeletons")
    # Creating a new folder with suffix _risk
    dst_dataset = os.path.join(base_path, "datasets", "3d_skeletons_risk")
    
    if not os.path.exists(src_dataset):
        print(f"Error: Source dataset not found at {src_dataset}")
        exit(1)
        
    process_dataset(src_dataset, dst_dataset)
    print("Dataset processing complete.")
