import os
import time
import numpy as np
import subprocess  # <--- 新增：用于调用系统命令
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from mig_opt_env import MigOptEnv

# --- 1. 全局配置 ---
VERILOG_FILE = os.path.join(os.path.dirname(__file__), '../benchmarks/arithmetic/adder.aig')
MODEL_PATH = "mig_ppo_model"

# [重要] 请修改为你实际的 abc 可执行文件路径
# 如果你的 abc 在项目的 lib/abc/abc，路径可能是这样的：
ABC_BINARY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../lib/abc/abc')) 

if not os.path.exists(VERILOG_FILE):
    print(f"Error: 找不到电路文件 {VERILOG_FILE}")
    exit()

def print_performance_report(initial_metrics, final_metrics):
    """
    打印美观的成绩单
    """
    init_area, init_depth = initial_metrics
    final_area, final_depth = final_metrics
    
    area_imp = (init_area - final_area) / init_area * 100
    depth_imp = (init_depth - final_depth) / init_depth * 100
    
    print("\n" + "="*60)
    print(f"{'PERFORMANCE REPORT (优化成绩单)':^60}")
    print("="*60)
    print(f"{'Metric (指标)':<15} | {'Initial':<12} | {'Final':<12} | {'Improvement'}")
    print("-" * 60)
    
    area_sign = "+" if area_imp > 0 else ""
    print(f"{'Area (Gates)':<15} | {int(init_area):<12} | {int(final_area):<12} | {area_sign}{area_imp:.2f}%")
    
    depth_sign = "+" if depth_imp > 0 else ""
    print(f"{'Depth (Level)':<15} | {int(init_depth):<12} | {int(final_depth):<12} | {depth_sign}{depth_imp:.2f}%")
    print("-" * 60)
    
    if depth_imp > 0 and area_imp < -50:
        conclusion = "深度优先 (High Performance): 牺牲了面积换取速度。"
    elif depth_imp > 0 and area_imp > 0:
        conclusion = "双赢 (Pareto Optimal): 面积和深度都减少了！"
    elif depth_imp <= 0 and area_imp > 0:
        conclusion = "面积优先 (Area Saving): 牺牲了速度换取面积。"
    else:
        conclusion = "需要改进 (Need Improvement): 尚未达到理想效果。"
        
    print(f"Evaluation: {conclusion}")
    print("="*60 + "\n")

def verify_circuit(original_path, optimized_path):
    """
    调用 ABC 工具进行等价性验证 (CEC)
    """
    print(f"[*] 正在进行逻辑等价性验证 (CEC)...")
    
    if not os.path.exists(ABC_BINARY_PATH):
        print(f"[Warning] 找不到 ABC 工具: {ABC_BINARY_PATH}")
        print("跳过验证。请确保已编译 ABC 并正确设置路径。")
        return

    # 构造 ABC 命令
    # cec 命令会比较两个网络是否逻辑等价
    cmd = f'"{ABC_BINARY_PATH}" -c "&r {original_path}; &cec {optimized_path}"'
    
    try:
        # 执行命令
        result = subprocess.run(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        output = result.stdout
        
        # 检查输出结果
        if "Networks are equivalent" in output:
            print("\n [SUCCESS] 验证通过！优化后的电路与原电路逻辑完全等价。")
            
        else:
            print("\n [FAILURE] 验证失败！电路逻辑已被改变。")
            print("ABC Output snippet:\n", output[:300]) # 打印前300字符供调试
            
    except Exception as e:
        print(f"执行验证时出错: {e}")

def train():
    print(f"正在使用电路文件: {VERILOG_FILE}")
    
    num_cpu = 8 
    
    env = make_vec_env(
        lambda: MigOptEnv(VERILOG_FILE), 
        n_envs=num_cpu, 
        vec_env_cls=DummyVecEnv
    )

    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0003, 
        device="auto", 
        n_steps=512,       
        batch_size=64,
        n_epochs=10,
        ent_coef=0.05      
    )

    print(f"----------- 开始训练 (并行进程数: {num_cpu}) -----------")
    start_time = time.time()
    
    model.learn(total_timesteps=50000)
    
    end_time = time.time()
    print(f"----------- 训练结束 (耗时: {end_time - start_time:.2f}s) -----------")

    model.save(MODEL_PATH)
    return model

def test(model):
    print("\n----------- 开始测试 (展示优化序列) -----------")
    
    test_env = MigOptEnv(VERILOG_FILE)
    obs, info = test_env.reset()
    
    init_area = test_env.initial_area
    init_depth = test_env.initial_depth
    
    total_reward = 0
    print(f"初始状态: Gates={init_area}, Depth={init_depth}")

    final_info = info 

    for i in range(20): 
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        total_reward += reward
        final_info = info
        
        action_name = info['action_name']
        
        print(f"Step {i+1:02d}: {action_name:<12} | Reward: {reward:+.4f} | "
              f"Gates: {info['raw_area']} (Diff: {info['raw_area'] - init_area:+g}) | "
              f"Depth: {info['raw_depth']} (Diff: {info['raw_depth'] - init_depth:+g})")

        if terminated or truncated:
            break
            
    # 1. 打印成绩单
    final_metrics = (final_info['raw_area'], final_info['raw_depth'])
    initial_metrics = (init_area, init_depth)
    print_performance_report(initial_metrics, final_metrics)
    
    # 2. 保存并验证
    # 只要你在 C++ 里绑定了 save 方法，这段代码就能工作
    save_path = VERILOG_FILE.replace(".aig", "_opt.aig")
    try:
        print(f"正在保存优化结果到: {save_path}")
        test_env.mig_manager.save(save_path) # 调用 C++ save
        
        # 调用验证函数
        verify_circuit(VERILOG_FILE, save_path)
        
    except AttributeError:
        print("[Warning] 未在 C++ 环境中找到 save 方法，跳过保存和验证。")
    except Exception as e:
        print(f"[Error] 保存/验证过程中出错: {e}")

if __name__ == '__main__':
    # 训练
    trained_model = train()
    #trained_model = PPO.load(MODEL_PATH) # 如果不想训练可取消注释
    
    # 测试
    test(trained_model)