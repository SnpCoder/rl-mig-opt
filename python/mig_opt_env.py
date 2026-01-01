import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os

# --- 关键：导入 C++ 模块 ---
# 我们需要把 build 目录加入系统路径，这样 Python 才能找到 mig_core.so
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))

# add_path = os.path.join(os.path.dirname(__file__), '../build')
# sys.path.append(add_path)

# # 打印关键信息，排查路径问题
# print("当前脚本__file__路径：", os.path.abspath(__file__))
# print("拼接后的build路径：", os.path.abspath(add_path))  # 转成绝对路径更易排查
# print("sys.path中是否包含该路径：", add_path in sys.path)
# print("build目录下是否存在.so文件：", os.path.exists(os.path.join(os.path.abspath(add_path), "mig_core.cpython-312-x86_64-linux-gnu.so")))

try:
    import mig_core
except ImportError as e:
    print(f"\nCRITICAL ERROR: 导入 mig_core 失败！")
    print(f"详细报错信息: {e}") 
    # 这里可能会显示: 
    # 1. "dynamic module does not define module export function" (版本不对)
    # 2. "undefined symbol: _ZN..." (编译缺库)
    sys.exit(1)

class MigOptEnv(gym.Env):
    """
    自定义的强化学习环境，用于优化 MIG 电路
    """
    def __init__(self, verilog_file):
        super(MigOptEnv, self).__init__()
        
        self.verilog_path = verilog_file
        
        # 初始化 C++ 后端
        try:
            self.mig_manager = mig_core.MigManager(self.verilog_path)
        except Exception as e:
            print(f"C++ 初始化失败: {e}")
            sys.exit(1)

        # 定义动作空间 (Action Space)
        # 目前 C++ 里只实现了 rewrite，我们可以定义：
        # 0: Rewrite (代数重写)
        # 1: No-op (什么都不做，或者以后你在 C++ 加了 balance 可以在这里映射)
        self.action_space = spaces.Discrete(2)

        # 定义观测空间 (Observation Space)
        # 我们用一个简单的向量表示状态：[当前门数量, 当前深度]
        # Box 用于表示连续或数值范围
        self.observation_space = spaces.Box(
            low=0, 
            high=float('inf'), 
            shape=(2,), 
            dtype=np.float32
        )

        # 记录初始状态用于计算 Reward
        self.initial_area = self.mig_manager.get_node_count()
        self.initial_depth = self.mig_manager.get_depth()

    def reset(self, seed=None, options=None):
        """
        重置环境：重新读取 Verilog 文件
        """
        super().reset(seed=seed)
        
        # 调用 C++ 的 reset 方法
        self.mig_manager.reset(self.verilog_path)
        
        # 获取初始状态
        current_area = self.mig_manager.get_node_count()
        current_depth = self.mig_manager.get_depth()
        
        observation = np.array([current_area, current_depth], dtype=np.float32)
        info = {}
        
        return observation, info

    def step(self, action):
        """
        执行一步动作
        """
        # 记录动作前的状态
        prev_area = self.mig_manager.get_node_count()
        prev_depth = self.mig_manager.get_depth()

        # --- 执行动作 ---
        if action == 0:
            # Action 0: 执行重写
            self.mig_manager.rewrite()
        elif action == 1:
            # Action 1: 这里目前是占位符，你可以去 C++ 实现 balance() 然后在这里调用
            pass 

        # 获取动作后的状态
        cur_area = self.mig_manager.get_node_count()
        cur_depth = self.mig_manager.get_depth()

        # --- 计算奖励 (Reward Design) ---
        # 这是一个关键的研究点。简单的逻辑是：面积减小给正分，深度减小给正分。
        # 如果变得更差，给负分。
        area_impr = prev_area - cur_area
        depth_impr = prev_depth - cur_depth
        
        # 奖励函数公式： 面积优化权重 * 面积减少量 + 深度优化权重 * 深度减少量
        reward = (1.0 * area_impr) + (2.0 * depth_impr)
        
        # 稍微给一点惩罚防止它无限在这个状态磨蹭（Time penalty）
        reward -= 0.1 

        # --- 终止条件 ---
        # 比如：如果深度降到了 0 (不可能)，或者 gate 很少，或者步数太长(由外部控制)
        terminated = False
        truncated = False
        
        observation = np.array([cur_area, cur_depth], dtype=np.float32)
        info = {
            "area": cur_area,
            "depth": cur_depth
        }

        return observation, reward, terminated, truncated, info
