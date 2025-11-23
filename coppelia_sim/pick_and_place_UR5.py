import time
import math
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

print('=== PICK AND PLACE - OPTIMIZED SEQUENCE ===\n')

# --- CONFIGURATION ---
UR5_BASE_POS = [0.575, -0.700, 0.710]
CUBE_POS = [-0.250, -0.325, 0.710]
BOWL_POS = [-0.275, -0.8, 0.710]

# Connection
client = RemoteAPIClient()
sim = client.getObject('sim')

def get_all_joints(sim):
    """Gets all 6 joints of the UR5."""
    joints = []
    for i in range(1, 7):
        try:
            joint = sim.getObjectHandle(f'UR5_joint{i}')
            joints.append(joint)
        except:
            try:
                joint = sim.getObject(f'/UR5/joint{i}')
                joints.append(joint)
            except:
                pass
    return joints

def set_joint(sim, joint, angle_deg):
    """Moves a single joint."""
    angle_rad = math.radians(angle_deg)
    sim.setJointTargetPosition(joint, angle_rad)

try:
    print('Connected to CoppeliaSim.\n')
    
    # Get joints
    joints = get_all_joints(sim)
    print(f'✓ Found {len(joints)} joints\n')
    
    # Get objects
    try:
        cuboid = sim.getObject('/Cuboid')
    except:
        cuboid = sim.getObjectHandle('Cuboid')
    
    try:
        tip = sim.getObject('/UR5/connection')
    except:
        tip = sim.getObjectHandle('UR5_connection')

    # Start simulation
    sim.startSimulation()
    print('\n✓ Simulation started.\n')
    time.sleep(3)
    
    # === SEQUENCE ===
    
    time.sleep(3)
    # 1. Read current position of Joint 1
    print('='*50)
    print('STEP 1: Reading current position')
    print('='*50)
    initial_joint1 = sim.getJointPosition(joints[0])
    initial_joint2 = sim.getJointPosition(joints[1])
    initial_joint3 = sim.getJointPosition(joints[2])
    
    initial_joint1_deg = math.degrees(initial_joint1)
    initial_joint2_deg = math.degrees(initial_joint2)
    initial_joint3_deg = math.degrees(initial_joint3)
    
    print(f'  Current Joint 1: {initial_joint1_deg:.1f}°')
    time.sleep(1)
    
    # 2. Rotate Joint 1: -30° from current position
    new_joint1_angle = initial_joint1_deg - 30
    set_joint(sim, joints[0], new_joint1_angle)
    time.sleep(1)
    
    # Read current tip position
    tip_pos = sim.getObjectPosition(tip, -1)
    print(f'  Tip now at Z={tip_pos[2]:.3f}m, Cube at Z={CUBE_POS[2]:.3f}m')
    print(f'  Z Difference: {tip_pos[2] - CUBE_POS[2]:.3f}m')
    
    # 3. Lower Joint 2 
    # Read current position Joint 2
    current_joint2 = sim.getJointPosition(joints[1])
    current_joint2_deg = math.degrees(current_joint2)    

    # Use angle +87° as definitive
    final_angle = current_joint2_deg + 85
    print(f'\n  Using Joint 2 = {final_angle:.1f}° (increment of 87°)')
    set_joint(sim, joints[1], final_angle)
    time.sleep(1)
    
    # 4. PICK the cube
    print('\n' + '='*50)
    print('STEP 4: PICK the cube')
    print('='*50)
    print('  → PICK!')
    sim.setObjectParent(cuboid, tip, True)
    time.sleep(1)
    
    # 5. Lift joint 2 (go back up)
    print('\n' + '='*50)
    print('STEP 5: Lift Joint 2 with the cube')
    print('='*50)
    current_joint2 = sim.getJointPosition(joints[1])
    current_joint2_deg = math.degrees(current_joint2)    
    final_angle = current_joint2_deg - 20
    set_joint(sim, joints[1], final_angle)
    time.sleep(1)
    
    # 7. Rotate joint 1 towards the bowl
    print('\n' + '='*50)
    print('STEP 7: Rotate Joint 1 towards bowl')
    print('='*50)
    current_joint1_deg = sim.getJointPosition(joints[0])
    current_joint1_deg = math.degrees(current_joint1_deg)    
    final_angle = current_joint1_deg + 30
    set_joint(sim, joints[0], final_angle)
    time.sleep(1)

    # 8. RELEASE the cube (10cm above the bowl)
    print('\n' + '='*50)
    print('STEP 8: RELEASE into the bowl')
    print('='*50)
    print('  → PLACE!')
    sim.setObjectParent(cuboid, -1, True)
    time.sleep(1)
    
    print('\n' + '='*60)
    print('✓✓✓ PICK AND PLACE COMPLETED SUCCESSFULLY! ✓✓✓')
    print('='*60 + '\n')
    
    # Return joints to initial position
    print('Returning joints to initial position...')
    initial_joint1_deg = math.degrees(initial_joint1)
    initial_joint2_deg = math.degrees(initial_joint2)
    initial_joint3_deg = math.degrees(initial_joint3)
    set_joint(sim, joints[0], initial_joint1_deg)
    set_joint(sim, joints[1], initial_joint2_deg)
    set_joint(sim, joints[2], initial_joint3_deg)
    time.sleep(2)
    
    print('✓ Joints returned to initial position.\n')
    sim.stopSimulation()

except Exception as e:
    print(f'\n✗ Error: {e}')
    import traceback
    traceback.print_exc()
    try:
        sim.stopSimulation()
    except:
        pass
