import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os

build_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../build'))
if build_path not in sys.path:
    sys.path.append(build_path)

try:
    import mig_core
except ImportError as e:
    print(f"\n[Error] Cannot import mig_core module! Make sure you compiled the C++ project.")
    sys.exit(1)

class MigOptEnv(gym.Env):
    def __init__(self, aig_files_list, target_mode='depth'):
        super(MigOptEnv, self).__init__()
        
        self.target_mode = target_mode.lower()
        valid_modes = ['depth', 'area', 'balanced']
        if self.target_mode not in valid_modes:
            raise ValueError(f"Invalid target_mode: {self.target_mode}. Must be one of {valid_modes}")
        
        print(f"[Env] Initialized with Optimization Mode: {self.target_mode.upper()}")

        # load aig file list
        if isinstance(aig_files_list, str):
            self.aig_files = [aig_files_list]
        else:
            self.aig_files = aig_files_list

        # initialize C++ manager
        self.current_aig_path = self.aig_files[0]
        try:
            self.mig_manager = mig_core.MigManager(self.current_aig_path)
        except Exception as e:
            print(f"C++ Init Failed: {e}")
            sys.exit(1)

        self.update_initial_stats()
        
        # 0:Rewrite, 1:Balance, 2:Resub, 3:Refactor
        self.action_space = spaces.Discrete(4)

        # 11-dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )
        
        self.last_action = -1 
        self.steps = 0
        self.max_steps = 40 
        self.repeat_count = 0 

    def update_initial_stats(self):
        self.initial_area = float(self.mig_manager.get_node_count())
        self.initial_depth = float(self.mig_manager.get_depth())
        
        if self.initial_area == 0: self.initial_area = 1.0
        if self.initial_depth == 0: self.initial_depth = 1.0
        
        self.initial_adp = self.initial_area * self.initial_depth

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # randomly select a circuit
        for _ in range(10):
            try:
                self.current_aig_path = np.random.choice(self.aig_files)
                self.mig_manager.reset(self.current_aig_path)
                if self.mig_manager.get_node_count() > 0:
                    break
            except:
                continue
        
        self.update_initial_stats()
        self.last_action = -1
        self.steps = 0
        self.repeat_count = 0

        return self._get_obs()

    def _get_obs(self):
        cur_area = float(self.mig_manager.get_node_count())
        cur_depth = float(self.mig_manager.get_depth())
        
        return self._compute_state_vector(cur_area, cur_depth), {
            "raw_area": cur_area, 
            "raw_depth": cur_depth,
            "filename": os.path.basename(self.current_aig_path)
        }

    def _compute_state_vector(self, cur_area, cur_depth):
        # normalize
        norm_area = cur_area / self.initial_area
        norm_depth = cur_depth / self.initial_depth
        
        # 2. 相对密度
        init_density = self.initial_area / (self.initial_depth + 1e-5)
        cur_density = cur_area / (cur_depth + 1e-5)
        rel_density = cur_density / (init_density + 1e-5)

        progress = self.steps / float(self.max_steps)

        repeat_penalty_feature = min(self.repeat_count / 5.0, 1.0)
        
        is_bloated = 1.0 if cur_area > self.initial_area else 0.0

        # 6. 动作历史 (One-Hot)
        action_one_hot = np.zeros(5, dtype=np.float32)
        if self.last_action == -1:
            action_one_hot[0] = 1.0
        else:
            action_one_hot[self.last_action + 1] = 1.0

        """
        Observation Space Description (Shape: 11,)
        -------------------------------------------------------
        [0] norm_area      : Float, Cur_Area / Init_Area. (<1.0 is better)
        [1] norm_depth     : Float, Cur_Depth / Init_Depth. (<1.0 is better)
        [2] rel_density    : Float, Relative circuit density (Area/Depth ratio).
        [3] progress       : Float, Step / Max_Steps (0.0 -> 1.0).
        [4] repeat_penalty : Float, Penalty intensity for repeating actions.
        [5] is_bloated     : Float, 1.0 if Cur_Area > Init_Area, else 0.0.
        
        [6-10] Action History (One-Hot Encoding):
            [6] : Start / None
            [7] : Rewrite
            [8] : Balance
            [9] : Resub
            [10]: Refactor
        -------------------------------------------------------
        """
        state = np.array([
            norm_area, norm_depth, rel_density, progress, 
            repeat_penalty_feature, is_bloated
        ], dtype=np.float32)
        
        return np.concatenate((state, action_one_hot))

    def step(self, action):
        self.steps += 1
        
        prev_area = float(self.mig_manager.get_node_count())
        prev_depth = float(self.mig_manager.get_depth())
        prev_adp = prev_area * prev_depth

        if action == 0: self.mig_manager.rewrite()
        elif action == 1: self.mig_manager.balance()
        elif action == 2: self.mig_manager.resub()
        elif action == 3: self.mig_manager.refactor()

        cur_area = float(self.mig_manager.get_node_count())
        cur_depth = float(self.mig_manager.get_depth())
        cur_adp = cur_area * cur_depth
        
        # compute reward
        reward = 0.0
        
        if prev_area != 0:
            area_imp = (prev_area - cur_area) / prev_area
        if prev_depth != 0:
            depth_imp = (prev_depth - cur_depth) / prev_depth
        if prev_adp != 0:
            adp_imp = (prev_adp - cur_adp) / prev_adp

        if self.target_mode == 'depth':
            score = (0.3 * area_imp) + (0.7 * depth_imp)

            if score > 0: reward += score * 60.0
            else: reward += score * 30.0
            
            if cur_area > prev_area and prev_area != 0: reward -= 20.0 * (cur_area / prev_area)
            if cur_depth > prev_depth and prev_depth != 0: reward -= 30.0 * (cur_depth / prev_depth)

        elif self.target_mode == 'area':
            score = (0.7 * area_imp) + (0.3 * depth_imp)

            if score > 0: reward += score * 60.0
            else: reward += score * 30.0
            
            if cur_area > prev_area and prev_area != 0: reward -= 30.0 * (cur_area / prev_area)
            if cur_depth > prev_depth and prev_depth != 0: reward -= 20.0 * (cur_depth / prev_depth)

        elif self.target_mode == 'balanced':
            score = (0.5 * area_imp) + (0.5 * depth_imp)
            
            if score > 0: reward += score * 60.0
            else: reward += score * 30.0
            
            if cur_area > prev_area and prev_area != 0: reward -= 20.0 * (cur_area / prev_area)
            if cur_depth > prev_depth and prev_depth != 0: reward -= 20.0 * (cur_depth / prev_depth)

        # restriction and penalty
        
        # no operation
        is_no_op = (prev_area == cur_area and prev_depth == cur_depth)
        if is_no_op:
            reward -= 2.0

        # repeat
        if action == self.last_action:
            self.repeat_count += 1
            if is_no_op:
                reward -= 1.0 * (self.repeat_count ** 2)
            elif self.repeat_count > 5:
                reward -= 0.5 * self.repeat_count
        else:
            self.repeat_count = 0

        terminated = False
        truncated = False

        success = False
        if self.target_mode == 'depth':
            if cur_area < self.initial_area * 1.1 and cur_depth < self.initial_depth * 0.75:
                success = True
        elif self.target_mode == 'area':
            if cur_area < self.initial_area * 0.8 and cur_depth <= self.initial_depth * 1.1:
                success = True
        elif self.target_mode == 'balanced':
            if cur_area < self.initial_area * 0.9 and cur_depth < self.initial_depth * 0.85:
                success = True

        if success:
            reward += 100.0
            terminated = True

        # truncated if too large
        bloat_limit = 3.0 if self.target_mode == 'depth' else 1.5
        if cur_area > self.initial_area * bloat_limit:
            reward -= 100.0
            truncated = True

        # truncated if take the same action while no improvment
        if self.repeat_count > 8 and is_no_op:
            truncated = True
            reward -= 20.0

        self.last_action = action
        state = self._compute_state_vector(cur_area, cur_depth)
        
        info = {
            "raw_area": cur_area,
            "raw_depth": cur_depth,
            "action_name": ["Rewrite", "Balance", "Resub", "Refactor"][action],
            "is_success": terminated,
            "mode": self.target_mode
        }
        
        return state, reward, terminated, truncated, info