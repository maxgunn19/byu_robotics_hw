import kinematics as kin  
from visualization import VizScene 
import time
import numpy as np

'''
DH Parameters: 
Apparently I am horrible at finding DH parameters. I cannot for the life of me figure out how
to get the parameters that give you a rotating link 2 (the one at 30 degrees in the picture),
so I am going to add 2 new joints and pretend like the joint (joint 3) on link 2 does not exist. 

I am adding an intermediate joint in the middle of link 1 that rotates about the a parallel axis to joint 2,
and then I am adding a joint in the middle of link 2 (the same place as the forgotten joint 3) that 
rotates about an axis parallel to joint 2's axis.
'''

# I am doing dh in degrees first because as I am finding out... I am bad at finding DH parameters
dh_degrees = [
            [0, 2, 0, 90],    # Joint 1
            [90, 0, 2, 0],   # Joint 2 (new joint added)
            [210, 0, 2, 90],  # Joint 3 (original joint 2)
            [0, 0, 2, 90],    # Joint 4 (new joint added)
            [-60, 0, 2, 90]    # Joint 5 (original joint 3)
]

dh = dh_degrees.copy()
for i in range(len(dh)):
    dh[i][0] = np.deg2rad(dh[i][0])  # convert to radians
    dh[i][3] = np.deg2rad(dh[i][3])  # convert to radians
    
joint_types = ['r'] * len(dh)

arm = kin.SerialArm(dh, jt=joint_types)

'''
Okay here is the fun and new stuff. I will be doing a potential field method to avoid the obstacle.
The idea is that the goal will attract the robot, while the obstacle repels it.
Hopefully it will create a nice safe and smooth path to the goal.
I have taken both Control Systems and Autonomous Aircraft Controls so I have a general idea of how it works.
'''

def potential_field(q_start, goal, obst_location, obst_radius, params):
      q = q_start.copy()
      q_s = [q.copy()]    
      iters = 0
      success = False
    
      # Params that it will use
      k_att = params['k_att']                     # Attractive gain
      k_rep = params['k_rep']                     # Repulsive gain
      eta_0 = params['eta_0']                     # Influence distance of the obstacle
      alpha = params['alpha']                     # Step size
      tolerance = params['tolerance']             # Tolerance for reaching the goal
      max_iters = params['max_iters']             # Maximum iterations 
      delta_q_max = params['delta_q_max']         # Maximum change in joint angles 
      control_points = params['control_points']   # Control points on the robot
      k_self_rep = params['k_self_rep']           # Self-repulsion gain (i am hoping that this will help with links hitting eachother)
      d_safe = params['d_safe']                   # Safety distance between links
          
      while iters < max_iters:
            pos_tip = arm.fk(q)[:3, 3]
            error = goal - pos_tip
          
            final_error_norm = np.linalg.norm(error)
          
            if final_error_norm < tolerance:
                  success = True
                  break
            
            # Attractive force stuff
            F_att = k_att * error
            J_tip = arm.jacob(q, index=arm.n)[:3, :]
            delta_q_att = J_tip.T @ F_att
            
            # Repulsive force stuff
            delta_q_rep = np.zeros(arm.n)
            
            for index in control_points:
                  pos_index = arm.fk(q, index=index)[:3, 3]
                  vec_to_obst = pos_index - obst_location
                  dist_to_obst = np.linalg.norm(vec_to_obst)
                  eta = dist_to_obst - obst_radius
                  
                  
                  if eta < eta_0:
                        if eta <= 1e-4:
                              F_rep_mag = k_rep * (1/1e-4 - 1/eta_0) / (1e-4**2)
                              F_rep_dir = vec_to_obst / dist_to_obst if dist_to_obst > 1e-9 else (np.random.rand(3) - 0.5)/np.linalg.norm(np.random.rand(3) - 0.5)
                        else:
                              F_rep_mag = k_rep * (1/eta - 1/eta_0) / (eta**2)
                              F_rep_dir = vec_to_obst / dist_to_obst
                        
                        F_rep_index = F_rep_mag * F_rep_dir
                        J_index = arm.jacob(q, index=index)[:3, :]
                        delta_q_rep += J_index.T @ F_rep_index
            
            # self repulsion stuff bc the links keep hitting eachother
            delta_q_self_rep = np.zeros(arm.n)
           
            self_rep_pairs = [(1, 3), (1, 4), (1, 5), (2, 4), (2, 5), (3, 5)] 

            control_point_positions = {}
            for index in control_points:
                  control_point_positions[index] = arm.fk(q, index=index)[:3, 3]

            for i, j in self_rep_pairs:
                  pos_i = control_point_positions[i]
                  pos_j = control_point_positions[j]
                  
                  vec_ij = pos_i - pos_j
                  dist_ij = np.linalg.norm(vec_ij)
                  
                  if dist_ij < d_safe:
                  # (same formula as before)
                        if dist_ij <= 1e-4:
                              F_self_mag = k_self_rep * (1/1e-4 - 1/d_safe) / (1e-4**2)
                        else:
                              F_self_mag = k_self_rep * (1/dist_ij - 1/d_safe) / (dist_ij**2)
                        
                        F_self_dir = vec_ij / dist_ij
                        F_self = F_self_mag * F_self_dir
                        
                        # (pushes i away from j)
                        J_i = arm.jacob(q, index=i)[:3, :]
                        delta_q_self_rep += J_i.T @ F_self
                        
                        # (pushes j away from i)
                        J_j = arm.jacob(q, index=j)[:3, :]
                        delta_q_self_rep -= J_j.T @ F_self
                              
            # combine it all
            delta_q_total = delta_q_att + delta_q_rep + delta_q_self_rep
            delta_q = alpha * delta_q_total
            delta_q_norm = np.linalg.norm(delta_q)
            
            if delta_q_norm > delta_q_max:
                  delta_q = delta_q * (delta_q_max / delta_q_norm)
                  
            q += delta_q
            
            q_s.append(q.copy())
            
            iters += 1
            
      return q_s, final_error_norm, iters, success

'''
Finds the path with the potential field method
'''
def compute_robot_path(q_init, goal, obst_location, obst_radius):

      q_s = []
      
      q_init_np = np.array(q_init, dtype=float)
      goal_np = np.array(goal, dtype=float)
      obst_location_np = np.array(obst_location, dtype=float)
      obst_radius_np = float(obst_radius)
      
      # some checks... they should all pass (unless there is some weird issue with future inputs or i have misunderstood something)
      for idx in [1, 2, 3, 4, 5]:
            p_index = arm.fk(q_init_np, index=idx)[:3, 3]
            dist_to_obst = np.linalg.norm(p_index - obst_location_np)
            if dist_to_obst <= obst_radius_np:
                  print(f"Initial collision: Frame {idx} is inside the obstacle.")
                  return q_s
      dist_goal_obst = np.linalg.norm(goal_np - obst_location_np)
      if dist_goal_obst <= obst_radius_np:
            print("Goal is inside the obstacle.")
            return q_s
      
      # Parameters for potential field
      params = {
            'k_att': 20.0,
            'k_rep': 7.5,
            'k_self_rep': 5.0,
            'd_safe': 3.0,
            'eta_0': obst_radius_np + 0.1,  # added a bit of wiggle room
            'alpha': 1.1,
            'tolerance': 0.1,
            'max_iters': 15000,
            'delta_q_max': 0.005,
            'control_points': [1, 2, 3, 4, 5] 
      }
      
      # running the potential field algorithm
      q_s, final_error, iters, success = potential_field(q_init_np, goal_np, obst_location_np, obst_radius_np, params)
      
      if success:
            print(f"Path found in {iters} iterations with final error {final_error:.4f}.")
      else:
            print(f"Failed to find path within max iterations. Final error: {final_error:.4f}.")
      
      return q_s

if __name__ == "__main__":
      q_0 = [0, 0, 0, 0, 0]
      goal = [0, 4, 4]
      obst_position = [0, 2.5, 2.5]
      obst_rad = 1.0

      q_ik_slns = compute_robot_path(q_0, goal, obst_position, obst_rad)

      viz = VizScene()
      viz.add_arm(arm, joint_colors=[np.array([0.95, 0.13, 0.13, 1])]*arm.n)
      viz.add_marker(goal, radius=0.1)
      viz.add_obstacle(obst_position, rad=obst_rad)
      for q in q_ik_slns:
            viz.update(qs=[q])

            #add pause for first iteration to see starting position
            if q is q_ik_slns[0]:
                  time.sleep(1.5)
            else:
                  time.sleep(0.01)
                  
      viz.hold()  # keep the visualization window open at the end