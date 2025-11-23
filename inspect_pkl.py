"""
Visualizer for CHICO 3D Skeleton dataset
Human-robot interaction dataset with 2 entities:
- Entity 0: Human skeleton (15 points)
- Entity 1: Robot arm/Tower (9 points)
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os

# Human skeleton connections definition (15 points)
HUMAN_SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),  # Spine
    (1, 4), (4, 5), (5, 6),  # Right arm
    (1, 7), (7, 8), (8, 9),  # Left arm
    (0, 10), (10, 11), (11, 12),  # Right leg
    (0, 13), (13, 14),  # Left leg/others
]

# Robot connections definition (9 points)
ROBOT_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (4, 5), (5, 6), (6, 7), (7, 8)
]

def load_skeleton_data(file_path):
    """Load skeleton data from a pickle file"""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def visualize_frame(data, frame_idx=0, title="Frame", save_path=None):
    """
    Visualize a single frame with human and robot
    data: list of frames, each frame contains [human_joints, robot_joints]
    """
    if frame_idx >= len(data):
        frame_idx = 0
    
    frame = data[frame_idx]
    human_joints = np.array(frame[0])  # 15x3
    robot_joints = np.array(frame[1])  # 9x3
    
    fig = plt.figure(figsize=(18, 6))
    
    # Plot 1: Combined 3D view
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Draw human skeleton
    ax1.scatter(human_joints[:, 0], human_joints[:, 1], human_joints[:, 2], 
                c='red', marker='o', s=100, label='Human', alpha=0.7)
    for connection in HUMAN_SKELETON_CONNECTIONS:
        if connection[0] < len(human_joints) and connection[1] < len(human_joints):
            points = human_joints[[connection[0], connection[1]]]
            ax1.plot(points[:, 0], points[:, 1], points[:, 2], 'r-', linewidth=2, alpha=0.6)
    
    # Draw robot
    ax1.scatter(robot_joints[:, 0], robot_joints[:, 1], robot_joints[:, 2], 
                c='blue', marker='s', s=100, label='Robot', alpha=0.7)
    for connection in ROBOT_CONNECTIONS:
        if connection[0] < len(robot_joints) and connection[1] < len(robot_joints):
            points = robot_joints[[connection[0], connection[1]]]
            ax1.plot(points[:, 0], points[:, 1], points[:, 2], 'b-', linewidth=2, alpha=0.6)
    
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title(f'{title}\nFrame {frame_idx}/{len(data)}')
    ax1.legend()
    
    # Plot 2: Top view (XY)
    ax2 = fig.add_subplot(132)
    ax2.scatter(human_joints[:, 0], human_joints[:, 1], c='red', marker='o', s=100, label='Human', alpha=0.7)
    for connection in HUMAN_SKELETON_CONNECTIONS:
        if connection[0] < len(human_joints) and connection[1] < len(human_joints):
            points = human_joints[[connection[0], connection[1]]]
            ax2.plot(points[:, 0], points[:, 1], 'r-', linewidth=2, alpha=0.6)
    
    ax2.scatter(robot_joints[:, 0], robot_joints[:, 1], c='blue', marker='s', s=100, label='Robot', alpha=0.7)
    for connection in ROBOT_CONNECTIONS:
        if connection[0] < len(robot_joints) and connection[1] < len(robot_joints):
            points = robot_joints[[connection[0], connection[1]]]
            ax2.plot(points[:, 0], points[:, 1], 'b-', linewidth=2, alpha=0.6)
    
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title('Top view (XY)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axis('equal')
    
    # Plot 3: Lateral view (XZ)
    ax3 = fig.add_subplot(133)
    ax3.scatter(human_joints[:, 0], human_joints[:, 2], c='red', marker='o', s=100, label='Human', alpha=0.7)
    for connection in HUMAN_SKELETON_CONNECTIONS:
        if connection[0] < len(human_joints) and connection[1] < len(human_joints):
            points = human_joints[[connection[0], connection[1]]]
            ax3.plot(points[:, 0], points[:, 2], 'r-', linewidth=2, alpha=0.6)
    
    ax3.scatter(robot_joints[:, 0], robot_joints[:, 2], c='blue', marker='s', s=100, label='Robot', alpha=0.7)
    for connection in ROBOT_CONNECTIONS:
        if connection[0] < len(robot_joints) and connection[1] < len(robot_joints):
            points = robot_joints[[connection[0], connection[1]]]
            ax3.plot(points[:, 0], points[:, 2], 'b-', linewidth=2, alpha=0.6)
    
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Z (mm)')
    ax3.set_title('Lateral view (XZ)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.axis('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')    

def save_animation(data, action_name, save_path, max_frames=100, fps=25):
    """
    Save animation of an action as a GIF/Video
    """    
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    
    num_frames = min(len(data), max_frames)
    
    def update(frame_idx):
        ax1.clear()
        ax2.clear()
        ax3.clear()
        
        frame = data[frame_idx]
        human_joints = np.array(frame[0])
        robot_joints = np.array(frame[1])
        
        # Plot 1: Combined 3D view
        ax1.scatter(human_joints[:, 0], human_joints[:, 1], human_joints[:, 2], 
                    c='red', marker='o', s=50, label='Human', alpha=0.7)
        for connection in HUMAN_SKELETON_CONNECTIONS:
            if connection[0] < len(human_joints) and connection[1] < len(human_joints):
                points = human_joints[[connection[0], connection[1]]]
                ax1.plot(points[:, 0], points[:, 1], points[:, 2], 'r-', linewidth=2, alpha=0.6)
        
        ax1.scatter(robot_joints[:, 0], robot_joints[:, 1], robot_joints[:, 2], 
                    c='blue', marker='s', s=50, label='Robot', alpha=0.7)
        for connection in ROBOT_CONNECTIONS:
            if connection[0] < len(robot_joints) and connection[1] < len(robot_joints):
                points = robot_joints[[connection[0], connection[1]]]
                ax1.plot(points[:, 0], points[:, 1], points[:, 2], 'b-', linewidth=2, alpha=0.6)
        
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_zlabel('Z (mm)')
        ax1.set_title(f'{action_name}\nFrame {frame_idx}/{num_frames}')
        
        # Plot 2: Top view (XY)
        ax2.scatter(human_joints[:, 0], human_joints[:, 1], c='red', marker='o', s=50, alpha=0.7)
        for connection in HUMAN_SKELETON_CONNECTIONS:
            if connection[0] < len(human_joints) and connection[1] < len(human_joints):
                points = human_joints[[connection[0], connection[1]]]
                ax2.plot(points[:, 0], points[:, 1], 'r-', linewidth=2, alpha=0.6)
        
        ax2.scatter(robot_joints[:, 0], robot_joints[:, 1], c='blue', marker='s', s=50, alpha=0.7)
        for connection in ROBOT_CONNECTIONS:
            if connection[0] < len(robot_joints) and connection[1] < len(robot_joints):
                points = robot_joints[[connection[0], connection[1]]]
                ax2.plot(points[:, 0], points[:, 1], 'b-', linewidth=2, alpha=0.6)
        
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.set_title('Top view (XY)')
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        # Plot 3: Side view (XZ)
        ax3.scatter(human_joints[:, 0], human_joints[:, 2], c='red', marker='o', s=50, alpha=0.7)
        for connection in HUMAN_SKELETON_CONNECTIONS:
            if connection[0] < len(human_joints) and connection[1] < len(human_joints):
                points = human_joints[[connection[0], connection[1]]]
                ax3.plot(points[:, 0], points[:, 2], 'r-', linewidth=2, alpha=0.6)
        
        ax3.scatter(robot_joints[:, 0], robot_joints[:, 2], c='blue', marker='s', s=50, alpha=0.7)
        for connection in ROBOT_CONNECTIONS:
            if connection[0] < len(robot_joints) and connection[1] < len(robot_joints):
                points = robot_joints[[connection[0], connection[1]]]
                ax3.plot(points[:, 0], points[:, 2], 'b-', linewidth=2, alpha=0.6)
        
        ax3.set_xlabel('X (mm)')
        ax3.set_ylabel('Z (mm)')
        ax3.set_title('Side view (XZ)')
        ax3.grid(True, alpha=0.3)
        ax3.axis('equal')

    anim = FuncAnimation(fig, update, frames=num_frames, interval=1000/fps)
    
    # Try saving as mp4 if ffmpeg is available, else gif
    try:
        anim.save(save_path, writer='pillow', fps=fps)
    except Exception as e:
        print(f"Error saving animation: {e}")
    
    plt.close(fig)

if __name__ == "__main__":
    dataset_path = r"c:\Users\Proprietario\Desktop\human-robot-collaboration\datasets\3d_skeletons"
    output_dir = r"c:\Users\Proprietario\Desktop\human-robot-collaboration\inspection_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define actions to process (Normal vs Crash)
    # Using place-hp as it has both versions
    actions_to_process = [
        {'name': 'place-hp', 'file': 'place-hp.pkl', 'title': 'PLACE-HP (Normal)'},
        {'name': 'place-hp_CRASH', 'file': 'place-hp_CRASH.pkl', 'title': 'PLACE-HP (Crash)'}
    ]
    
    for action in actions_to_process:
        action_file = os.path.join(dataset_path, 'S00', action['file'])
        
        if os.path.exists(action_file):
            data = load_skeleton_data(action_file)
            
            # 1. Save first frame image
            image_path = os.path.join(output_dir, f"S00_{action['name']}_frame0.png")
            visualize_frame(data, 0, title=f"{action['title']} Frame 0", save_path=image_path)
            
            # 2. Save video/gif (1/4 of total frames)
            video_path = os.path.join(output_dir, f"S00_{action['name']}_video.gif")
            num_frames_to_save = len(data) // 4
            save_animation(data, action['title'], video_path, max_frames=num_frames_to_save)
            
        else:
            print(f"File not found: {action_file}")
            
    print(f"Results in '{output_dir}'")
