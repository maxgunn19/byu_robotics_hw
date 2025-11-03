import numpy as np
from kinematics_project import SerialArm # Import your arm class

class Node:
    """
    RRT Node class. Stores a configuration 'q' (joint angles)
    and a reference to its parent node.
    """
    def __init__(self, q, parent=None):
        self.q = np.array(q)
        self.parent = parent

class RRT:
    """
    RRT Path Planner class.
    """
    def __init__(self, arm: SerialArm, obstacles: list, 
                 q_start: np.ndarray, q_goal: np.ndarray,
                 max_iter=10000, step_size=0.15, goal_bias=0.1,
                 prune_max_iter=100, collision_step_size=0.1):
        """
        :param SerialArm arm: The robot arm object from kinematics.py
        :param list obstacles: List of obstacles. Each obstacle is a tuple
                                (center_position, radius). e.g., [(np.array([x,y,z]), r), ...]
        :param np.ndarray q_start: The starting joint configuration (N,)
        :param np.ndarray q_goal: The goal joint configuration (N,)
        :param int max_iter: Max iterations to build the tree
        :param float step_size: C-space distance to "steer" from a node
        :param float goal_bias: Probability (0-1) of sampling the goal
        :param int prune_max_iter: Max iterations for path pruning
        :param float collision_step_size: Step size for checking C-space paths
        """
        self.arm = arm
        self.obstacles = obstacles
        self.q_start = q_start
        self.q_goal = q_goal
        
        self.root = Node(q_start)
        self.goal_node = Node(q_goal) # We'll try to connect to this
        self.nodes = [self.root]
        
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.prune_max_iter = prune_max_iter
        self.collision_step_size = collision_step_size # for path checking
        
        self.q_min = arm.qlim[0, :]
        self.q_max = arm.qlim[1, :]
        
        self.all_nodes_generated = [] # For visualization
        self.final_path_nodes = []
        self.pruned_path_nodes = []

    def _get_link_positions(self, q: np.ndarray) -> list:
        """Helper to get 3D positions of each joint for a given q."""
        positions = []
        # Add base position
        positions.append(self.arm.base[:3, 3])
        
        # Add positions of each joint frame
        for i in range(self.arm.n):
            T = self.arm.fk(q, index=i+1, base=True)
            positions.append(T[:3, 3])
            
        # Add tip position
        T_tip = self.arm.fk(q, base=True, tip=True)
        positions.append(T_tip[:3, 3])
        
        return positions

    def _check_line_sphere_collision(self, p1, p2, sphere_center, sphere_radius):
        """
        Checks for collision between a line segment (p1-p2) and a sphere.
        """
        # Vector from line start to sphere center
        v_sc_p1 = sphere_center - p1
        # Line segment vector
        v_p1_p2 = p2 - p1
        
        # Project sphere center onto the line (not the line segment)
        # t = (v_sc_p1 . v_p1_p2) / |v_p1_p2|^2
        line_len_sq = np.dot(v_p1_p2, v_p1_p2)
        
        if line_len_sq < 1e-12:
            # p1 and p2 are the same point
            return np.linalg.norm(v_sc_p1) <= sphere_radius

        t = np.dot(v_sc_p1, v_p1_p2) / line_len_sq
        
        # Clamp t to [0, 1] to find the closest point *on the segment*
        t_clamped = np.clip(t, 0.0, 1.0)
        
        # Closest point on the segment to the sphere center
        closest_point = p1 + t_clamped * v_p1_p2
        
        # Check distance from closest point to sphere center
        dist_sq = np.sum((closest_point - sphere_center)**2)
        
        return dist_sq <= sphere_radius**2

    def _is_collision_q(self, q: np.ndarray) -> bool:
        """Checks if a single configuration 'q' is in collision."""
        # 1. Check joint limits
        if np.any(q < self.q_min) or np.any(q > self.q_max):
            return True
            
        # 2. Check for self-collision (simplified: not implemented)
        # ...
        
        # 3. Check for obstacle collision
        link_positions = self._get_link_positions(q)
        
        # Check each link segment (p_i to p_i+1)
        for i in range(len(link_positions) - 1):
            p1 = link_positions[i]
            p2 = link_positions[i+1]
            
            # Skip zero-length links (e.g., base to joint 1 if d=0, a=0)
            if np.linalg.norm(p1 - p2) < 1e-6:
                continue

            for obs_center, obs_radius in self.obstacles:
                if self._check_line_sphere_collision(p1, p2, obs_center, obs_radius):
                    return True # Collision found
                    
        return False # No collision

    def is_collision_path(self, q1: np.ndarray, q2: np.ndarray) -> bool:
        """
        Checks for collision on the straight-line C-space path from q1 to q2.
        """
        v = q2 - q1
        dist = np.linalg.norm(v)
        if dist < 1e-6:
            return self._is_collision_q(q1)
        
        num_steps = int(np.ceil(dist / self.collision_step_size))
        
        for i in range(num_steps + 1):
            t = i / num_steps
            q_interp = q1 + t * v
            if self._is_collision_q(q_interp):
                return True
        return False

    def sample_config(self) -> np.ndarray:
        """Sample a random configuration."""
        # Goal-biasing
        if np.random.rand() < self.goal_bias:
            return self.q_goal
        
        # Sample uniformly from joint limits
        return np.random.uniform(self.q_min, self.q_max)

    def find_nearest_node(self, q_sample: np.ndarray) -> Node:
        """Find the node in the tree closest to the sample."""
        min_dist = float('inf')
        nearest_node = None
        for node in self.nodes:
            dist = np.linalg.norm(node.q - q_sample)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        return nearest_node

    def steer(self, q_near: np.ndarray, q_sample: np.ndarray) -> np.ndarray:
        """Steer from q_near towards q_sample by self.step_size."""
        v = q_sample - q_near
        dist = np.linalg.norm(v)
        
        if dist < self.step_size:
            return q_sample
        
        q_new = q_near + (v / dist) * self.step_size
        return q_new

    def build(self) -> (list, Node | None):
        """
        Build the RRT tree.
        Returns (all_nodes_generated, goal_node_if_found)
        """
        self.all_nodes_generated = [self.root]
        
        for i in range(self.max_iter):
            if i % 500 == 0:
                print(f"RRT Iteration: {i}/{self.max_iter}, Nodes: {len(self.nodes)}")
                
            q_sample = self.sample_config()
            
            nearest_node = self.find_nearest_node(q_sample)
            
            q_new = self.steer(nearest_node.q, q_sample)
            
            if not self.is_collision_path(nearest_node.q, q_new):
                new_node = Node(q_new, parent=nearest_node)
                self.nodes.append(new_node)
                self.all_nodes_generated.append(new_node) # For viz
                
                # Check if we can connect to the goal
                dist_to_goal = np.linalg.norm(q_new - self.q_goal)
                if dist_to_goal < self.step_size:
                    if not self.is_collision_path(q_new, self.q_goal):
                        self.goal_node.parent = new_node
                        self.nodes.append(self.goal_node)
                        self.all_nodes_generated.append(self.goal_node)
                        print(f"Goal Reached! Total nodes: {len(self.nodes)}")
                        self.final_path_nodes = self.get_path(self.goal_node)
                        return self.all_nodes_generated, self.goal_node
                        
        print("RRT failed to find a path.")
        return self.all_nodes_generated, None

    def get_path(self, end_node: Node) -> list:
        """Extract the path from the end_node back to the root."""
        path = []
        curr = end_node
        while curr is not None:
            path.append(curr)
            curr = curr.parent
        return path[::-1] # Return from start to end

    def prune_path(self) -> list:
        """
        Prune the self.final_path_nodes by connecting non-adjacent nodes
        if the path between them is collision-free.
        """
        if not self.final_path_nodes:
            print("No path to prune.")
            return []
            
        pruned_path = [self.final_path_nodes[0]]
        curr_node_idx = 0
        
        while curr_node_idx < len(self.final_path_nodes) - 1:
            # Find the furthest node we can connect to
            best_next_node_idx = curr_node_idx + 1
            for j in range(len(self.final_path_nodes) - 1, curr_node_idx, -1):
                if not self.is_collision_path(pruned_path[-1].q, self.final_path_nodes[j].q):
                    best_next_node_idx = j
                    break
            
            pruned_path.append(self.final_path_nodes[best_next_node_idx])
            curr_node_idx = best_next_node_idx
            
        self.pruned_path_nodes = pruned_path
        print(f"Path pruned from {len(self.final_path_nodes)} to {len(pruned_path)} nodes.")
        return pruned_path

    def interpolate_path(self, steps_per_segment=15) -> np.ndarray:
        """
        Interpolate the pruned path for smooth animation.
        """
        if not self.pruned_path_nodes:
            print("No pruned path to interpolate.")
            return np.array([])
            
        path_qs = []
        for i in range(len(self.pruned_path_nodes) - 1):
            q_start = self.pruned_path_nodes[i].q
            q_end = self.pruned_path_nodes[i+1].q
            for t in np.linspace(0, 1, steps_per_segment):
                path_qs.append(q_start + t * (q_end - q_start))
        
        # Add the very last node
        path_qs.append(self.pruned_path_nodes[-1].q)
        return np.array(path_qs)