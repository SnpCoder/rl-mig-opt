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
    print(f"\n[Error] Cannot import mig_core module!")
    sys.exit(1)

class MigOptEnv(gym.Env):
    def __init__(self, aig_file):
        super(MigOptEnv, self).__init__()
        self.aig_path = aig_file
        
        try:
            self.mig_manager = mig_core.MigManager(self.aig_path)
        except Exception as e:
            print(f"C++ Init Failed: {e}")
            sys.exit(1)

        self.initial_area = float(self.mig_manager.get_node_count())
        self.initial_depth = float(self.mig_manager.get_depth())
        
        if self.initial_area == 0: self.initial_area = 1.0
        if self.initial_depth == 0: self.initial_depth = 1.0

        # Action 0: Rewrite (Algebraic) - 小修小补
        # Action 1: Balance (SOP)       - 深度核武器 (副作用：面积爆炸)
        # Action 2: Resub               - 面积吸尘器 (专门修补 Balance 后的烂摊子)
        # Action 3: RewriteAgg          - 深度重构
        self.action_space = spaces.Discrete(4)

        # 观察空间增加到 5 维
        # [Norm_Area, Norm_Depth, Last_Action, Progress, Area_State_Flag]
        self.observation_space = spaces.Box(
            low=0.0, high=10.0, shape=(5,), dtype=np.float32
        )
        
        self.last_action = -1.0
        self.steps = 0
        self.max_steps = 40 # 增加步数，给它足够的时间“破坏再修复”
        self.consecutive_no_ops = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.mig_manager.reset(self.aig_path)
        
        self.last_action = -1.0
        self.steps = 0
        self.consecutive_no_ops = 0

        raw_area = self.mig_manager.get_node_count()
        raw_depth = self.mig_manager.get_depth()
        
        state = self._get_state(raw_area, raw_depth)
        info = {"raw_area": raw_area, "raw_depth": raw_depth}
        return state, info

    def _get_state(self, area, depth):
        # Area_State_Flag: 1.0 表示面积膨胀了(红灯), 0.0 表示安全(绿灯)
        is_bloated = 1.0 if area > self.initial_area else 0.0
        
        return np.array([
            area / self.initial_area,
            depth / self.initial_depth,
            (self.last_action + 1) / 4.0,
            self.steps / self.max_steps,
            is_bloated 
        ], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        
        prev_area = self.mig_manager.get_node_count()
        prev_depth = self.mig_manager.get_depth()

        # 执行动作
        if action == 0: self.mig_manager.rewrite()
        elif action == 1: self.mig_manager.balance()
        elif action == 2: self.mig_manager.resub()
        elif action == 3: self.mig_manager.rewrite_aggressive()

        cur_area = self.mig_manager.get_node_count()
        cur_depth = self.mig_manager.get_depth()

        # 计算归一化差异 (正数=变好，负数=变差)
        area_diff_norm = (prev_area - cur_area) / self.initial_area
        depth_diff_norm = (prev_depth - cur_depth) / self.initial_depth

        # ==========================================
        # 核心逻辑：动态权重 (Dynamic Weighting)
        # ==========================================
        
        # 场景 A: 面积膨胀了 (红灯模式)
        # 任务：全力修复面积，深度只要不恶化就行
        if cur_area > self.initial_area:
            w_area = 10.0  # 疯狂奖励面积减少
            w_depth = 1.0  # 深度暂时不重要
            
            # 这种状态下，如果能减少面积，给巨额奖励
            base_reward = (w_area * area_diff_norm) + (w_depth * depth_diff_norm)
            
            # 额外惩罚：依然处于膨胀状态，每一步都扣分（迫使它尽快修好）
            base_reward -= 0.5 

        # 场景 B: 面积安全 (绿灯模式)
        # 任务：全力冲击深度，允许牺牲一点面积
        else:
            w_area = 2.0
            w_depth = 8.0  # 疯狂奖励深度减少
            
            base_reward = (w_area * area_diff_norm) + (w_depth * depth_diff_norm)
            
            # 特殊激励：如果深度真的降了，即使面积稍微涨了一点，也不要罚太重
            if depth_diff_norm > 0 and area_diff_norm < 0:
                base_reward += 1.0 # 鼓励这种“有价值的牺牲”

        reward = base_reward

        # ==========================================
        # 辅助逻辑
        # ==========================================

        # 1. 遏制无脑刷屏 (No-Op Killer)
        # 如果动作无效，给予随次数指数增长的惩罚
        if prev_area == cur_area and prev_depth == cur_depth:
            self.consecutive_no_ops += 1
            reward -= (1.0 * self.consecutive_no_ops) # -1, -2, -3...
        else:
            self.consecutive_no_ops = 0 # 重置

        # 2. 终止条件
        terminated = False
        truncated = False

        # 如果连续 5 次操作都无效，直接重开，别浪费时间了
        if self.consecutive_no_ops >= 5:
            truncated = True
            reward -= 5.0

        # 如果面积实在太大 (2倍)，判定为不可挽回，强制结束
        if cur_area > self.initial_area * 2.0:
            truncated = True
            reward -= 10.0

        # 3. 完美结局奖励 (Jackpot)
        # 如果两者都显著优化 (>20%)，给一个巨额奖金
        if cur_area < self.initial_area * 0.8 and cur_depth < self.initial_depth * 0.8:
            reward += 10.0

        # 更新状态
        self.last_action = action
        state = self._get_state(cur_area, cur_depth)

        info = {
            "raw_area": cur_area,
            "raw_depth": cur_depth,
            "action_name": ["Rewrite", "Balance", "Resub", "RewriteAgg"][action]
        }

        return state, reward, terminated, truncated, info