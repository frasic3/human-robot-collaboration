import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_min_distance(human_joints, robot_joints):
    diff = human_joints[:, None, :] - robot_joints[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    return dists.min()

def find_and_plot_examples(base_path):
    # We'll look into a CRASH file to ensure we find collisions
    file_path = os.path.join(base_path, "datasets", "3d_skeletons_risk", "S01", "place-hp_CRASH.pkl")
    
    # Create output directory
    output_dir = os.path.join(base_path, "risk_examples")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots to {output_dir}")
    
    print(f"Scanning {file_path} for examples...")
    
    if not os.path.exists(file_path):
        print("File not found.")
        return

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    found_class_0 = False
    found_class_1 = False
    found_class_2 = False
    
    for i, frame in enumerate(data):
        human_joints = np.array(frame[0])
        robot_joints = np.array(frame[1])
        risk_label = frame[2]
        
        min_dist = calculate_min_distance(human_joints, robot_joints)
        
        if risk_label == 0 and not found_class_0:
            print(f"\n--- Found Class 0 (Safe) at Frame {i} ---")
            print(f"Risk Label: {risk_label}")
            print(f"Calculated Min Distance: {min_dist:.2f} mm")
            plot_frame(human_joints, robot_joints, risk_label, i, os.path.join(output_dir, "risk_safe.png"))
            found_class_0 = True

        if risk_label == 1 and not found_class_1:
            print(f"\n--- Found Class 1 (Near-collision) at Frame {i} ---")
            print(f"Risk Label: {risk_label}")
            print(f"Calculated Min Distance: {min_dist:.2f} mm")
            plot_frame(human_joints, robot_joints, risk_label, i, os.path.join(output_dir, "risk_near_collision.png"))
            found_class_1 = True
            
        if risk_label == 2 and not found_class_2:
            print(f"\n--- Found Class 2 (Collision) at Frame {i} ---")
            print(f"Risk Label: {risk_label}")
            print(f"Calculated Min Distance: {min_dist:.2f} mm")
            plot_frame(human_joints, robot_joints, risk_label, i, os.path.join(output_dir, "risk_collision.png"))
            found_class_2 = True
            
        if found_class_0 and found_class_1 and found_class_2:
            break
            
    if not found_class_0:
        print("Could not find Class 0 example.")
    if not found_class_1:
        print("Could not find Class 1 example.")
    if not found_class_2:
        print("Could not find Class 2 example.")

def plot_frame(human_joints, robot_joints, label, frame_idx, filename):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    label_map = {0: "Safe", 1: "Near-collision", 2: "Collision"}
    label_str = label_map.get(label, str(label))

    # Plot Human Joints: Blue circles + Numbers
    ax.scatter(human_joints[:, 0], human_joints[:, 1], human_joints[:, 2], c='blue', marker='o', label='Human')
    for i, (x, y, z) in enumerate(human_joints):
        # Add a small offset to text so it doesn't overlap exactly with the point
        ax.text(x, y, z, f"H{i}", color='black', fontsize=8)

    # Plot Robot Joints: Red triangles + Numbers
    ax.scatter(robot_joints[:, 0], robot_joints[:, 1], robot_joints[:, 2], c='red', marker='^', label='Robot')
    for i, (x, y, z) in enumerate(robot_joints):
        ax.text(x, y, z, f"R{i}", color='black', fontsize=8)

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'Frame {frame_idx} - Risk Class: {label_str}')
    
    ax.legend()
    
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close(fig)

if __name__ == "__main__":
    base_path = r"c:\Users\Proprietario\Desktop\human-robot-collaboration"
    find_and_plot_examples(base_path)
