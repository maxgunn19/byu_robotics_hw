import numpy as np
import matplotlib.pyplot as plt
import re
import os
from kinematics_2 import SerialArm # Import from your file

"""
This script performs the following:
1. Parses all 'output_3_*.txt' files in the folder.
2. Uses the SerialArm kinematics to calculate end-effector linear and angular velocity.
3. Computes the average and variance of both velocity magnitudes.
4. Plots the results.
"""

def parse_output_file(filename):
    """
    Parses one of the text output files to extract t, q, and q_dot.
    
    This function is brittle and assumes the exact format from
    mat_to_txt_converter.py, and that the files are complete 
    (i.e., no '...' truncation).
    
    It also assumes the variables are in the order q, q_dot, t.
    """
    try:
        with open(filename, 'r') as f:
            content = f.read()

        # A more robust parsing method
        # 1. Split the file content by "Variable:"
        # This creates a list where the first item is the header
        # and subsequent items are the variable blocks.
        parts = content.split('Variable:')
        
        # We expect 4 parts: [header, q_block, q_dot_block, t_block]
        if len(parts) < 4:
            raise ValueError(f"File {filename} does not contain all three 'Variable:' sections. Found {len(parts)} parts.")
            
        # Helper to find the value string *within* a part
        def get_value_str(part):
            # Find "Value:" and get everything after it
            try:
                # We also need to trim off the '----' separator at the end
                val_and_end = part.split('Value:\n', 1)[1]
                return val_and_end
            except IndexError:
                raise ValueError("Could not find 'Value:\n' in variable block.")

        # 2. Per user, the order is q, q_dot, t
        # parts[0] is the header
        # parts[1] should start with " q\nType:..."
        # parts[2] should start with " q_dot\nType:..."
        # parts[3] should start with " t\nType:..."
        
        # Let's add a check for that
        if not parts[1].strip().startswith('q\n'):
            raise ValueError("Expected second block to be 'q'.")
        if not parts[2].strip().startswith('q_dot\n'):
            raise ValueError("Expected third block to be 'q_dot'.")
        if not parts[3].strip().startswith('t\n'):
            raise ValueError("Expected fourth block to be 't'.")

        # Extract the value string from each block
        q_str = get_value_str(parts[1])
        q_dot_str = get_value_str(parts[2])
        t_str = get_value_str(parts[3])

        # Helper to convert the numpy string to a flat array
        def clean_and_parse(s):
            """
            Robustly parses a string block containing a numpy array.
            Finds all floating-point numbers (incl. scientific notation)
            and returns them as a flat numpy array.
            """
            # Find all floating point numbers
            numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
            # Convert them to a numpy array of floats
            return np.array(numbers, dtype=float)

        # Parse 't' first to get the number of timesteps
        t = clean_and_parse(t_str)
        num_timesteps = len(t)
        
        if num_timesteps == 0:
            raise ValueError("Parsed 't' array is empty.")

        # Use the number of timesteps to reshape q and q_dot
        q = clean_and_parse(q_str).reshape((num_timesteps, 7))
        q_dot = clean_and_parse(q_dot_str).reshape((num_timesteps, 7))

        return t, q, q_dot

    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
        return None, None, None
    except Exception as e:
        print(f"Error parsing file {filename}. Make sure it is not truncated ('...').")
        print(f"Details: {e}")
        return None, None, None

def main():
    # --- 1. Define Robot Kinematics ---
    # DH Parameters from your image, in [theta, d, a, alpha] format
    # The 'theta' values are 0 here, as they are the joint variables.
    
    dh_table = [
        [0, 0.27035, 0.069, -np.pi/2],  # Link 1
        [np.pi/2, 0,      0,  np.pi/2],  # Link 2
        [0, 0.36435,  0.069, -np.pi/2],  # Link 3
        [0, 0,      0,  np.pi/2],  # Link 4
        [0, 0.37429, 0.010, -np.pi/2],  # Link 5
        [0, 0,      0,  np.pi/2],  # Link 6
        [0, 0.229525, 0,  0]         # Link 7
    ]
    
    # Create the arm instance
    arm = SerialArm(dh_table)

    # --- 2. Load and Process Data ---
    num_trials = 10
    file_template = "output_3_{}.txt"
    
    # Lists to store the 1D magnitude arrays from each trial
    all_ee_vel_magnitudes = []
    all_ee_vel_angular_magnitudes = [] # ADDED
    time_array = None

    print(f"Processing {num_trials} trial files...")

    for i in range(1, num_trials + 1):
        filename = file_template.format(i)
        
        if not os.path.exists(filename):
            print(f"Warning: File not found, skipping: {filename}")
            continue
            
        print(f"  - Parsing {filename}...")
        t, q, q_dot = parse_output_file(filename)

        if t is None:
            continue # Skip file if parsing failed

        if time_array is None:
            time_array = t
        
        num_steps = len(t)
        # Array to store the 3D velocities [vx, vy, vz] for each step
        ee_vel_linear = np.zeros((num_steps, 3))
        ee_vel_angular = np.zeros((num_steps, 3)) # ADDED

        # Calculate end-effector velocity for each timestep
        for j in range(num_steps):
            q_j = q[j, :]       # Joint positions at step j
            q_dot_j = q_dot[j,:]  # Joint velocities at step j
            
            # Calculate 6x7 Jacobian
            J = arm.jacob(q_j)
            
            # Calculate 6x1 end-effector velocity (linear + angular)
            # v_ee = [vx, vy, vz, wx, wy, wz]
            v_ee = J @ q_dot_j
            
            # Store the linear velocity part (first 3 components)
            ee_vel_linear[j, :] = v_ee[0:3]
            # Store the angular velocity part (last 3 components)
            ee_vel_angular[j, :] = v_ee[3:6] # ADDED

        # Calculate the magnitude (Euclidean norm) of the linear velocity at each step
        # This results in a 1D array of size (num_steps,)
        ee_vel_magnitude = np.linalg.norm(ee_vel_linear, axis=1)
        all_ee_vel_magnitudes.append(ee_vel_magnitude)

        # ADDED: Calculate and store angular velocity magnitude
        ee_vel_angular_magnitude = np.linalg.norm(ee_vel_angular, axis=1)
        all_ee_vel_angular_magnitudes.append(ee_vel_angular_magnitude)


    if not all_ee_vel_magnitudes:
        print("Error: No data was successfully processed. Exiting.")
        return

    print("All files processed.")

    # --- 3. Compute Averages and Variance ---
    
    # === Linear Velocity ===
    # Stack the list of 1D arrays into a 2D (num_trials x num_steps) array
    # e.g., (10, 4998)
    vel_data = np.stack(all_ee_vel_magnitudes, axis=0)

    # Calculate mean and variance across the trials (axis=0)
    avg_velocity = np.mean(vel_data, axis=0)
    var_velocity = np.var(vel_data, axis=0)
    std_velocity = np.sqrt(var_velocity)

    # === ADDED: Angular Velocity ===
    vel_angular_data = np.stack(all_ee_vel_angular_magnitudes, axis=0)
    avg_velocity_angular = np.mean(vel_angular_data, axis=0)
    var_velocity_angular = np.var(vel_angular_data, axis=0)
    std_velocity_angular = np.sqrt(var_velocity_angular)

    print("Statistics computed.")

    # --- 4. Plot Results ---
    # Create 4 separate figures

    # Plot 1: Average Linear Velocity
    plt.figure(figsize=(10, 7))
    plt.plot(time_array, avg_velocity, 'b-', label='Average Linear Velocity')
    # Shade the region +/- 3 standard deviation
    plt.fill_between(time_array, 
                     avg_velocity - std_velocity * 3, 
                     avg_velocity + std_velocity * 3, 
                     color='b', alpha=0.2, label='± 3 Std. Dev.')
    
    plt.title("Average End Effector Linear Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity Magnitude (m/s)")
    plt.legend()
    plt.grid(True)

    # Plot 2: Average Angular Velocity (ADDED)
    plt.figure(figsize=(10, 7))
    plt.plot(time_array, avg_velocity_angular, 'g-', label='Average Angular Velocity')
    plt.fill_between(time_array,
                     avg_velocity_angular - std_velocity_angular * 3,
                     avg_velocity_angular + std_velocity_angular * 3,
                     color='g', alpha=0.2, label='± 3 Std. Dev.')

    plt.title("Average End Effector Angular Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Velocity Mag. (rad/s)")
    plt.legend()
    plt.grid(True)

    # Plot 3: Linear Velocity Variance
    plt.figure(figsize=(10, 7))
    plt.plot(time_array, var_velocity, 'r-', label='Linear Variance')
    
    plt.title("Variance of End Effector Linear Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Variance (m/s)²")
    plt.legend()
    plt.grid(True)

    # Plot 4: Angular Velocity Variance (ADDED)
    plt.figure(figsize=(10, 7))
    plt.plot(time_array, var_velocity_angular, 'm-', label='Angular Variance')

    plt.title("Variance of End Effector Angular Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Variance (rad/s)²")
    plt.legend()
    plt.grid(True)

    print("Plots created. Showing results...")
    plt.show()


if __name__ == "__main__":
    main()

