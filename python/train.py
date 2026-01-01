import os
from stable_baselines3 import PPO
from mig_opt_env import MigOptEnv

# 1. 路径设置
# 假设你在 rl-mig-opt/python 目录下运行此脚本
# 所以 benchmarks 路径应该是 ../benchmarks/adder.v
verilog_file = os.path.join(os.path.dirname(__file__), '../benchmarks/test.v')

if not os.path.exists(verilog_file):
    print(f"Error: 找不到电路文件 {verilog_file}")
    exit()

# 2. 实例化环境
env = MigOptEnv(verilog_file)

# 3. 定义 RL 模型
# 使用 PPO (Proximal Policy Optimization)，目前最常用的算法
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

print("----------- 开始训练 -----------")
# 4. 开始训练
# timesteps 设置大一点可以看到 loss 变化，这里设为 2000 仅作演示
model.learn(total_timesteps=2000)
print("----------- 训练结束 -----------")

# 5. 测试训练好的模型
print("\n----------- 开始测试 -----------")
obs, _ = env.reset()
done = False
total_reward = 0

print(f"初始状态: Area={obs[0]}, Depth={obs[1]}")

for i in range(10): # 测试 10 步
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    action_str = "Rewrite" if action == 0 else "No-op"
    print(f"Step {i+1}: Action={action_str}, Reward={reward:.2f}, New State=[Area:{info['area']}, Depth:{info['depth']}]")

    if terminated or truncated:
        break
