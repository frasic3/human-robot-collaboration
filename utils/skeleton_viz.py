"""
Visualizzazione 3D degli scheletri per analisi errori
Dataset CHICO: 15 joint umani + 9 joint robot
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# HUMAN: 15 joints (OpenPose body model)
# 0:Nose, 1:Neck, 2:RShoulder, 3:RElbow, 4:RWrist, 
# 5:LShoulder, 6:LElbow, 7:LWrist, 8:RHip, 9:RKnee, 
# 10:RAnkle, 11:LHip, 12:LKnee, 13:LAnkle, 14:Head
HUMAN_CONNECTIONS = [
    (0, 1),  # Nose to Neck
    (1, 2), (2, 3), (3, 4),  # Right arm
    (1, 5), (5, 6), (6, 7),  # Left arm  
    (1, 8), (8, 9), (9, 10),  # Right leg
    (1, 11), (11, 12), (12, 13),  # Left leg
    (0, 14)  # Nose to Head
]

# ROBOT: 9 joints (UR10 robot arm)
# Sequential connections for robot arm
ROBOT_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)
]


def plot_skeleton_3d(skeleton_data, title="3D Skeleton", save_path=None):
    """
    Visualizza scheletro umano (15 joint) + robot (9 joint)
    
    Args:
        skeleton_data: array (72,) o (24, 3) - primi 15 joint umano, ultimi 9 robot
        title: titolo del grafico
        save_path: percorso dove salvare l'immagine
    """
    # Reshape se necessario
    if skeleton_data.shape[0] == 72:
        skeleton_data = skeleton_data.reshape(24, 3)
    
    human_joints = skeleton_data[:15]  # Primi 15 joint
    robot_joints = skeleton_data[15:]  # Ultimi 9 joint
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot HUMAN joints (blue)
    ax.scatter(human_joints[:, 0], human_joints[:, 1], human_joints[:, 2], 
               c='blue', s=80, marker='o', label='Human Joints', alpha=0.8)
    
    # Plot HUMAN connections
    for connection in HUMAN_CONNECTIONS:
        joint1, joint2 = connection
        x = [human_joints[joint1, 0], human_joints[joint2, 0]]
        y = [human_joints[joint1, 1], human_joints[joint2, 1]]
        z = [human_joints[joint1, 2], human_joints[joint2, 2]]
        ax.plot(x, y, z, 'b-', linewidth=2, alpha=0.6)
    
    # Plot ROBOT joints (red)
    ax.scatter(robot_joints[:, 0], robot_joints[:, 1], robot_joints[:, 2], 
               c='red', s=80, marker='^', label='Robot Joints', alpha=0.8)
    
    # Plot ROBOT connections
    for connection in ROBOT_CONNECTIONS:
        joint1, joint2 = connection
        x = [robot_joints[joint1, 0], robot_joints[joint2, 0]]
        y = [robot_joints[joint1, 1], robot_joints[joint2, 1]]
        z = [robot_joints[joint1, 2], robot_joints[joint2, 2]]
        ax.plot(x, y, z, 'r-', linewidth=2, alpha=0.6)
    
    # Labels per joint chiave umano
    human_labels = {
        0: 'Nose', 1: 'Neck', 2: 'R-Shoulder', 5: 'L-Shoulder',
        4: 'R-Wrist', 7: 'L-Wrist', 8: 'R-Hip', 11: 'L-Hip'
    }
    for joint_id, label in human_labels.items():
        ax.text(human_joints[joint_id, 0], human_joints[joint_id, 1], 
                human_joints[joint_id, 2], label, fontsize=7, color='darkblue')
    
    # Label per base e end-effector robot
    ax.text(robot_joints[0, 0], robot_joints[0, 1], robot_joints[0, 2],
            'Robot Base', fontsize=7, color='darkred')
    ax.text(robot_joints[8, 0], robot_joints[8, 1], robot_joints[8, 2],
            'Robot EE', fontsize=7, color='darkred')
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    
    # Set equal aspect ratio per tutti gli assi
    all_joints = skeleton_data
    max_range = np.array([all_joints[:, 0].max() - all_joints[:, 0].min(),
                          all_joints[:, 1].max() - all_joints[:, 1].min(),
                          all_joints[:, 2].max() - all_joints[:, 2].min()]).max() / 2.0
    
    mid_x = (all_joints[:, 0].max() + all_joints[:, 0].min()) * 0.5
    mid_y = (all_joints[:, 1].max() + all_joints[:, 1].min()) * 0.5
    mid_z = (all_joints[:, 2].max() + all_joints[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Skeleton visualization saved to: {save_path}")
    
    plt.close()


def visualize_missed_collision(skeleton_data, probs, true_label, pred_label, 
                               class_names, sample_idx, threshold, save_path):
    """
    Visualizza scheletro umano + robot con informazioni sulla predizione errata
    
    Args:
        skeleton_data: array (72,) o (24, 3) - 15 joint umano + 9 joint robot
        probs: array (3,) - probabilità delle 3 classi
        true_label: int - label vera
        pred_label: int - label predetta
        class_names: list - nomi delle classi
        sample_idx: int - indice del sample
        threshold: float - soglia usata
        save_path: str - percorso dove salvare
    """
    # Reshape se necessario
    if skeleton_data.shape[0] == 72:
        skeleton_data = skeleton_data.reshape(24, 3)
    
    human_joints = skeleton_data[:15]
    robot_joints = skeleton_data[15:]
    
    # Crea figura con 2 subplot
    fig = plt.figure(figsize=(18, 8))
    
    # Subplot 1: Skeleton 3D
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Plot HUMAN joints (blue)
    ax1.scatter(human_joints[:, 0], human_joints[:, 1], human_joints[:, 2], 
                c='blue', s=100, marker='o', label='Human', alpha=0.8, edgecolors='darkblue', linewidths=2)
    
    # Plot HUMAN connections
    for connection in HUMAN_CONNECTIONS:
        joint1, joint2 = connection
        x = [human_joints[joint1, 0], human_joints[joint2, 0]]
        y = [human_joints[joint1, 1], human_joints[joint2, 1]]
        z = [human_joints[joint1, 2], human_joints[joint2, 2]]
        ax1.plot(x, y, z, 'b-', linewidth=3, alpha=0.7)
    
    # Plot ROBOT joints (red)
    ax1.scatter(robot_joints[:, 0], robot_joints[:, 1], robot_joints[:, 2], 
                c='red', s=100, marker='^', label='Robot', alpha=0.8, edgecolors='darkred', linewidths=2)
    
    # Plot ROBOT connections
    for connection in ROBOT_CONNECTIONS:
        joint1, joint2 = connection
        x = [robot_joints[joint1, 0], robot_joints[joint2, 0]]
        y = [robot_joints[joint1, 1], robot_joints[joint2, 1]]
        z = [robot_joints[joint1, 2], robot_joints[joint2, 2]]
        ax1.plot(x, y, z, 'r-', linewidth=3, alpha=0.7)
    
    # Calcola distanza minima
    diff = human_joints[:, None, :] - robot_joints[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    min_dist = dists.min()
    min_idx = np.unravel_index(dists.argmin(), dists.shape)
    
    # Disegna linea di distanza minima
    closest_human = human_joints[min_idx[0]]
    closest_robot = robot_joints[min_idx[1]]
    ax1.plot([closest_human[0], closest_robot[0]], 
             [closest_human[1], closest_robot[1]], 
             [closest_human[2], closest_robot[2]], 
             'yellow', linewidth=3, linestyle='--', label=f'Min Dist: {min_dist:.1f}mm')
    
    ax1.set_xlabel('X (mm)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Y (mm)', fontsize=11, fontweight='bold')
    ax1.set_zlabel('Z (mm)', fontsize=11, fontweight='bold')
    ax1.set_title(f'MISSED COLLISION - Sample #{sample_idx}\nMin Distance: {min_dist:.1f} mm', 
                  fontsize=14, fontweight='bold', color='red')
    ax1.legend(fontsize=10)
    
    # Set equal aspect ratio
    all_joints = skeleton_data
    max_range = np.array([all_joints[:, 0].max() - all_joints[:, 0].min(),
                          all_joints[:, 1].max() - all_joints[:, 1].min(),
                          all_joints[:, 2].max() - all_joints[:, 2].min()]).max() / 2.0
    
    mid_x = (all_joints[:, 0].max() + all_joints[:, 0].min()) * 0.5
    mid_y = (all_joints[:, 1].max() + all_joints[:, 1].min()) * 0.5
    mid_z = (all_joints[:, 2].max() + all_joints[:, 2].min()) * 0.5
    
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Subplot 2: Info e probabilità
    ax2 = fig.add_subplot(122)
    ax2.axis('off')
    
    # Testo informativo
    info_text = f"""
PREDICTION ANALYSIS
{'='*50}

Sample Index: {sample_idx}

True Label:      {class_names[true_label]} ({true_label})
Predicted Label: {class_names[pred_label]} ({pred_label})

{'='*50}
DISTANCE ANALYSIS:
{'='*50}

Minimum Distance: {min_dist:.2f} mm

Risk Thresholds:
  - Collision:      <= 130 mm
  - Near-Collision: 130-630 mm  
  - Safe:           > 630 mm

Expected Class: {
    'Collision' if min_dist <= 130 else 
    'Near-Collision' if min_dist <= 630 else 
    'Safe'
}

{'='*50}
PROBABILITIES:
{'='*50}

Safe:           {probs[0]:.4f} ({probs[0]*100:.2f}%)
Near-Collision: {probs[1]:.4f} ({probs[1]*100:.2f}%)
Collision:      {probs[2]:.4f} ({probs[2]*100:.2f}%)

{'='*50}
THRESHOLD INFO:
{'='*50}

Collision Threshold: {threshold:.4f}
Collision Prob:      {probs[2]:.4f}

Status: {'BELOW' if probs[2] < threshold else 'ABOVE'} threshold
-> Predicted as: {class_names[pred_label]}

{'='*50}
DIAGNOSIS:
{'='*50}

The model assigned only {probs[2]*100:.1f}% probability 
to Collision, below the {threshold*100:.0f}% threshold.

Instead, it preferred:
- {class_names[np.argmax(probs)]}: {probs[np.argmax(probs)]*100:.1f}%

This is a FALSE NEGATIVE (missed collision).
    """
    
    ax2.text(0.05, 0.5, info_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.5))
    
    # Bar chart delle probabilità
    ax3 = fig.add_axes([0.6, 0.1, 0.3, 0.25])
    colors = ['green' if i == true_label else 'red' if i == pred_label else 'gray' 
              for i in range(3)]
    bars = ax3.bar(class_names, probs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax3.axhline(y=threshold, color='orange', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    ax3.set_ylabel('Probability', fontsize=11, fontweight='bold')
    ax3.set_title('Class Probabilities', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.set_ylim(0, 1.0)
    ax3.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
