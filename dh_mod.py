# DH Parameter Visualizer (Interactive - Rebuild + Update)
#
# This version combines both of the user's suggestions:
# 1. It uses the "rebuild" strategy (viz.clear() + viz.add_arm())
#    from the "maybe try deleting" suggestion.
# 2. It *also* calls viz.update() *inside* the button callback
#    to force an immediate redraw, as per the
#    "actually use viz.update()" suggestion.
#
# This uses the tkinter single-loop model (no viz.hold()).

import kinematics as kin
from visualization import VizScene
import numpy as np
import tkinter as tk
from tkinter import ttk, font

# --- DEFINE YOUR ROBOT'S INITIAL STATE HERE ---

initial_dh_parameters_deg = [
    [0, 4, 0, -90],
    [60, 0, 2, 0],
    [0, 0, 2, 0],
    [-60, 0, 2, 90]
]
initial_joint_types = ['r', 'r', 'r', 'r']

# --- END OF ROBOT DEFINITION ---


# --- GLOBAL VARIABLES ---
arm = None       # The *current* arm object
viz = None       # The one and only visualization scene
dh_entries = []  # GUI widgets
jt_entries = []  # GUI widgets
root = None      # GUI root window
q_current = []   # Current joint configuration

def update_robot_visualization():
    """
    Callback function triggered by the "Update" button.
    This function DESTROYS the old arm, builds a NEW one,
    and forces an immediate update.
    """
    global arm, viz, q_current
    
    if not viz:
        print("Error: Visualization not initialized.")
        return
        
    try:
        # 1. Read all new parameters from the GUI
        new_dh_deg = []
        for i, row_entries in enumerate(dh_entries):
            theta = float(row_entries[0].get())
            d = float(row_entries[1].get())
            a = float(row_entries[2].get())
            alpha = float(row_entries[3].get())
            new_dh_deg.append([theta, d, a, alpha])
        
        new_jt = [entry.get() for entry in jt_entries]
        new_dh_rad = [[np.deg2rad(th), d, a, np.deg2rad(al)] for th, d, a, al in new_dh_deg]

        # 2. Create a NEW arm object and its zero config
        arm = kin.SerialArm(new_dh_rad, jt=new_jt)
        q_current = [0.0] * arm.n
        
        # 3. Clear the visualization scene
        # (Assuming 'viz.clear()' is the correct method)
        viz.clear()
        
        # 4. Add the NEW arm back to the scene
        viz.add_arm(arm, joint_colors=[np.array([0.1, 0.8, 0.1, 1.0])]*arm.n)
        
        # 5. --- THIS IS THE NEW, CRITICAL STEP ---
        # Force a single, immediate update of the visualization
        # to draw the new arm in its zero state.
        viz.update(qs=[q_current])
        
        print("\n--- Robot Rebuilt and Redrawn ---")
        print("New DH Parameters (deg):", new_dh_deg)
        print("New Joint Types:", new_jt)
        
    except AttributeError as ae:
        print("\n--- ERROR ---")
        print(f"Failed to clear the scene: {ae}")
        print("This is an expected error if the API is different.")
        print("Please check `visualization.py` for the correct method")
        print("to remove all objects from the scene (e.g., 'viz.clear()',")
        print("'viz.remove_all()', or 'viz.scene.clear()') and")
        print("replace 'viz.clear()' in the code.")
    except ValueError as ve:
        print(f"\nError: Invalid input. Please ensure all values are numbers.")
        print(ve)
    except Exception as e:
        print(f"\nAn error occurred during update:")
        print(e)


# --- (The setup_gui function is identical to the previous answer) ---
def setup_gui(initial_dh, initial_jt):
    """
    Creates the tkinter control panel window. (Does not run mainloop).
    """
    global root, dh_entries, jt_entries
    
    root = tk.Tk()
    root.title("Robot Control Panel")
    
    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    # --- DH Parameters Grid ---
    dh_frame = ttk.LabelFrame(main_frame, text="DH Parameters", padding="10")
    dh_frame.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    
    headers = ["Joint", "θ (deg)", "d", "a", "α (deg)"]
    for j, header in enumerate(headers):
        lbl = ttk.Label(dh_frame, text=header, font=font.Font(weight="bold"))
        lbl.grid(row=0, column=j, padx=5, pady=5)

    dh_entries = []
    for i in range(len(initial_dh)):
        row_entries = []
        lbl = ttk.Label(dh_frame, text=f" {i+1} ")
        lbl.grid(row=i+1, column=0, padx=5)
        
        for j in range(4):
            entry = ttk.Entry(dh_frame, width=8)
            entry.insert(0, str(initial_dh[i][j]))
            entry.grid(row=i+1, column=j+1, padx=5, pady=2)
            row_entries.append(entry)
        dh_entries.append(row_entries)

    # --- Joint Types ---
    jt_frame = ttk.LabelFrame(main_frame, text="Joint Types ('r' or 'p')", padding="10")
    jt_frame.grid(row=1, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
    
    jt_entries = []
    for i in range(len(initial_jt)):
        lbl = ttk.Label(jt_frame, text=f"J{i+1}:")
        lbl.grid(row=0, column=i*2, padx=(10, 0))
        
        entry = ttk.Entry(jt_frame, width=3)
        entry.insert(0, initial_jt[i])
        entry.grid(row=0, column=i*2 + 1, padx=(2, 10))
        jt_entries.append(entry)

    # --- Update Button ---
    update_button = ttk.Button(
        main_frame, 
        text="Update Visualization", 
        command=update_robot_visualization
    )
    update_button.grid(row=2, column=0, padx=5, pady=10, sticky=(tk.W, tk.E))
# --- (End of identical setup_gui function) ---


def main_loop():
    """
    This is the single, unified loop.
    It updates the visualization and the GUI, then reschedules itself.
    """
    global viz, root, q_current
    
    try:
        # 1. Render the visualization
        if viz:
            # This call just keeps the window responsive and
            # would handle animations if q_current were changing.
            viz.update(qs=[q_current])
        
        # 2. Process GUI events
        if root:
            root.update()
            
        # 3. Reschedule this function to run again
        root.after(16, main_loop) # ~60 FPS
        
    except tk.TclError:
        # This error happens when the GUI window is closed.
        print("\nGUI window closed. Exiting.")
    except Exception as e:
        # This catches errors if the viz window is closed first.
        print(f"\nMain loop error (or viz window closed): {e}")
        if root:
            # Force the GUI to close if the viz window died
            root.destroy()

def main():
    global arm, viz, q_current
    
    print("--- Interactive DH Parameter Visualizer (Rebuild + Update) ---")
    
    if len(initial_dh_parameters_deg) != len(initial_joint_types):
        print("Error: DH parameters and joint types must have the same length.")
        return

    try:
        # 1. Create the *initial* robot arm object
        dh_parameters_rad = [[np.deg2rad(th), d, a, np.deg2rad(al)] for th, d, a, al in initial_dh_parameters_deg]
        arm = kin.SerialArm(dh_parameters_rad, jt=initial_joint_types)
        num_joints = arm.n
        q_current = [0.0] * num_joints
        
        print(f"Visualizing initial {num_joints}-DOF robot:")
        print(arm)

        # 2. Set up the visualization (DO NOT CALL viz.hold())
        viz = VizScene()
        viz.add_arm(arm, joint_colors=[np.array([0.1, 0.8, 0.1, 1.0])]*num_joints)
        
        # 3. Set up the GUI (DO NOT CALL root.mainloop() yet)
        setup_gui(initial_dh_parameters_deg, initial_joint_types)
        
        # 4. Start the unified loop
        print("Starting main loop. Close the GUI window to exit.")
        main_loop()
        
        # 5. This call is necessary to start the tkinter loop
        #    that main_loop() hooks into.
        root.mainloop()

    except Exception as e:
        print("\nAn error occurred during initialization:")
        print(e)
        print("\nPlease ensure that your `kinematics.py` and `visualization.py` files are in the same directory.")


if __name__ == "__main__":
    main()