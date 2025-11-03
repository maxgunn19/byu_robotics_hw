import numpy as np
import time
from kinematics_project import SerialArm
from visualization_project import VizScene
from rrt_planner import RRT

# Define some colors for visualization
VIZ_GREY = (0.5, 0.5, 0.5, 0.4)
VIZ_RED = (0.8, 0.0, 0.0, 0.8)
VIZ_GREEN = (0.0, 0.8, 0.0, 0.9)
VIZ_BLUE = (0.0, 0.0, 0.8, 0.9)
VIZ_GOAL = (0.1, 0.9, 0.1, 0.5)

def define_6dof_arm():
    """
    Defines a generic 6-DOF (PUMA-style) arm.
    DH parameters: [theta, d, a, alpha]
    """
    pi = np.pi
    dh = [[0, 0.5, 0.1, pi/2],
          [0, 0.0, 0.6, 0.0],
          [0, 0.0, 0.1, pi/2],
          [0, 0.6, 0.0, -pi/2],
          [0, 0.0, 0.0, pi/2],
          [0, 0.1, 0.0, 0.0]]
    
    # Joint limits [min_row; max_row]
    qlim = np.array([
        [-pi,   -pi/2, -pi/2, -2*pi, -pi/2, -2*pi],  # Min angles
        [ pi,    pi/2,  pi,    2*pi,  pi/2,  2*pi]   # Max angles
    ])
    
    # Define a tip transform (e.g., a tool)
    T_tip = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0.2], # 20cm tool
        [0, 0, 0, 1]
    ])

    arm = SerialArm(dh, joint_limits=qlim.T, tip=T_tip)
    print(arm)
    print(f"Arm reach: {arm.reach:.3f} m")
    return arm

def run_visualization(viz: VizScene, arm: SerialArm, q_start: np.ndarray,
                      all_nodes: list,
                      raw_path_nodes: list,
                      pruned_path_nodes: list,
                      motion_qs: np.ndarray):
    """
    Runs the multi-stage presentation animation.
    """
    
    print("--- STAGE 1: Show Initial Setup ---")
    viz.update(qs=q_start) # Show arm at start config
    viz.hold(2) # Pause for 2 seconds

    print("--- STAGE 2: Animate RRT Generation ---")
    tree_lines = []
    # Add nodes one-by-one for animation effect
    for node in all_nodes[1:]: # Skip root
        if node.parent:
            p_parent = arm.fk(node.parent.q, base=True, tip=True)[:3, 3]
            p_node = arm.fk(node.q, base=True, tip=True)[:3, 3]
            line = viz.add_line(p_parent, p_node, color=VIZ_GREY, width=1)
            tree_lines.append(line)
        
        if len(tree_lines) % 20 == 0:
            viz.app.processEvents() # Update the display every 20 lines
            time.sleep(0.001)

    print("RRT generation complete.")
    viz.hold(2)

    print("--- STAGE 3: Highlight Raw Path ---")
    raw_path_lines = []
    for i in range(len(raw_path_nodes) - 1):
        p1 = arm.fk(raw_path_nodes[i].q, base=True, tip=True)[:3, 3]
        p2 = arm.fk(raw_path_nodes[i+1].q, base=True, tip=True)[:3, 3]
        line = viz.add_line(p1, p2, color=VIZ_RED, width=5)
        raw_path_lines.append(line)
        viz.app.processEvents()
        time.sleep(0.05)
        
    print("Raw path highlighted.")
    viz.hold(2)

    print("--- STAGE 4: Show Pruned Path ---")
    # Remove all grey tree lines and red path lines
    for line in tree_lines:
        viz.remove_item(line)
    for line in raw_path_lines:
        viz.remove_item(line)
    
    viz.app.processEvents()
    time.sleep(0.5)

    # Add the bold, pruned path
    pruned_path_lines = []
    for i in range(len(pruned_path_nodes) - 1):
        p1 = arm.fk(pruned_path_nodes[i].q, base=True, tip=True)[:3, 3]
        p2 = arm.fk(pruned_path_nodes[i+1].q, base=True, tip=True)[:3, 3]
        line = viz.add_line(p1, p2, color=VIZ_GREEN, width=8)
        pruned_path_lines.append(line)
    
    print("Path pruned and highlighted.")
    viz.hold(3)

    print("--- STAGE 5: Animate Robot Motion ---")
    for q in motion_qs:
        viz.update(qs=q)
        time.sleep(0.02) # 50 FPS
        
    print("Motion complete.")
    viz.hold(2)
    
    # Remove pruned path lines and just show final pose
    for line in pruned_path_lines:
        viz.remove_item(line)
        
    print("--- Visualization Finished. Close window to exit. ---")
    viz.hold()


def main():
    np.set_printoptions(precision=4, suppress=True)

    # 1. Define Arm
    arm = define_6dof_arm()

    # 2. Define Environment
    # Obstacles: list of (center, radius)
    obstacles = [
        (np.array([0.5, -1, 0.6]), 0.15),  # Sphere 1
        (np.array([0.5, 0, 0.7]), 0.15), # Sphere 2
        (np.array([0.4, 0.2, 0.5]), 0.15) # Sphere 3
    ]
    
    # Start and Goal
    q_start = np.array([0.0, 0.0, 0, 0.0, 0.0, 0.0])
    p_goal_target = np.array([0.75, 0.5, 1.0])

    # 3. Solve for Goal Configuration (q_goal)
    print("Solving IK for goal position...")
    K_gain = np.eye(3) * 0.8
    q_goal, err, iters, converged = arm.ik_position(
        p_goal_target,
        q0=q_start,
        method='pinv',
        K=K_gain,
        tol=1e-4,
        max_iter=500
    )
    
    if not converged:
        print(f"IK failed to converge! Error: {np.linalg.norm(err):.5f}")
        # We can still try, but the goal might be unreachable
    else:
        print(f"IK converged in {iters} iterations.")
        
    # Check if IK goal is in collision
    if RRT(arm, obstacles, q_start, q_goal)._is_collision_q(q_goal):
        print("ERROR: Target goal configuration is in collision!")
        return

    # 4. --- PRE-CALCULATION ---
    print("\n--- Starting RRT Pre-calculation ---")
    
    # Initialize RRT
    rrt = RRT(arm, obstacles, q_start, q_goal,
              max_iter=15000,    # Lower for faster demo, raise for harder problems
              step_size=0.4,    # C-space step size
              goal_bias=0.25,   # 25% chance to sample goal
              collision_step_size=0.15)
              
    # Build the tree
    start_time = time.time()
    all_nodes, goal_node = rrt.build()
    print(f"RRT build time: {time.time() - start_time:.2f}s")

    if goal_node is None:
        print("RRT failed to find a path. Visualizing tree anyway.")
        raw_path_nodes = []
        pruned_path_nodes = []
        motion_qs = []
    else:
        # Get paths and interpolate
        raw_path_nodes = rrt.final_path_nodes
        pruned_path_nodes = rrt.prune_path()
        motion_qs = rrt.interpolate_path(steps_per_segment=20)
        
    print(f"All nodes: {len(all_nodes)}")
    print(f"Raw path nodes: {len(raw_path_nodes)}")
    print(f"Pruned path nodes: {len(pruned_path_nodes)}")
    print(f"Interpolated motion steps: {len(motion_qs)}")
    
    print("\n--- Pre-calculation Complete. Starting Visualization ---")

    # 5. --- VISUALIZATION ---
    viz = VizScene()
    viz.add_arm(arm)
    
    # Add obstacles
    for center, radius in obstacles:
        viz.add_obstacle(center, rad=radius)
        
    # Add goal marker
    viz.add_marker(p_goal_target, color=VIZ_GOAL, radius=0.05)
    
    # Run the animated presentation
    try:
        run_visualization(viz, arm, q_start, all_nodes, raw_path_nodes, pruned_path_nodes, motion_qs)
    except Exception as e:
        print(f"An error occurred during visualization: {e}")
    finally:
        viz.close_viz()


if __name__ == "__main__":
    main()