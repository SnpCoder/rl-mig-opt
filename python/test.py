import os
import time
import glob
import pandas as pd
import subprocess
from stable_baselines3 import PPO
from mig_opt_env import MigOptEnv

# 【核心】导入配置文件
import config as cfg

# MAX_STEPS 依然可以在这里微调
MAX_STEPS = 40 

def verify_equivalence(original_path, optimized_path):
    """ 调用 ABC 进行逻辑等价性检查 (CEC) """
    if not os.path.exists(cfg.ABC_BINARY_PATH):
        return "ABC_Not_Found"

    cmd = f'"{cfg.ABC_BINARY_PATH}" -c "&r {original_path}; &cec {optimized_path}"'
    try:
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if "Networks are equivalent" in result.stdout:
            return "PASS"
        else:
            return "FAIL"
    except Exception as e:
        return f"Error: {str(e)}"

def save_log_file(filename, initial_info, step_records, final_info, cec_status, duration):
    """ 将详细的优化过程（含功耗）写入 .log 文件 """
    log_filename = filename.replace(".aig", f"_opt_{cfg.CURRENT_MODE}.log")
    log_path = os.path.join(cfg.RESULTS_DIR, log_filename)
    
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Optimization Log for: {filename}\n")
        f.write(f"Mode: {cfg.CURRENT_MODE.upper()}\n")
        f.write("="*80 + "\n") # 加宽分割线以容纳更多列
        f.write(f"Initial State: Gates={initial_info['gates']}, Depth={initial_info['depth']}, WSA={initial_info['wsa']:.1f}\n")
        f.write("-" * 80 + "\n")
        # 增加 Power 列
        f.write(f"{'Step':<5} | {'Action':<10} | {'Reward':<10} | {'Gates':<15} | {'Depth':<15} | {'Power(WSA)':<15}\n")
        f.write("-" * 80 + "\n")
        
        for record in step_records:
            step_str = f"{record['step']:02d}"
            action_str = record['action']
            reward_str = f"{record['reward']:+.4f}"
            gate_change = f"{record['gates']} ({record['gate_diff']:+})"
            depth_change = f"{record['depth']} ({record['depth_diff']:+})"
            # Power 变化
            wsa_change = f"{record['wsa']:.1f}"
            
            f.write(f"{step_str:<5} | {action_str:<10} | {reward_str:<10} | {gate_change:<15} | {depth_change:<15} | {wsa_change:<15}\n")
            
        f.write("-" * 80 + "\n")
        f.write(f"Final State:   Gates={final_info['gates']}, Depth={final_info['depth']}, WSA={final_info['wsa']:.1f}\n")
        f.write(f"Improvement:   Gates {final_info['gate_imp']:.2f}% | Depth {final_info['depth_imp']:.2f}% | Power {final_info['wsa_imp']:.2f}%\n")
        f.write(f"Verification:  {cec_status}\n")
        f.write(f"Total Time:    {duration:.2f}s\n")
        f.write("="*80 + "\n")
    
    return log_path

def evaluate_single_circuit(model, aig_file):
    """ 测试单个电路并生成日志 """
    filename = os.path.basename(aig_file)
    
    # 初始化环境
    env = MigOptEnv(aig_file, target_mode=cfg.CURRENT_MODE)
    obs, info = env.reset()
    
    # 获取初始指标 (包含 WSA)
    init_gates = int(info['raw_area'])
    init_depth = int(info['raw_depth'])
    init_wsa = env.mig_manager.get_switching_activity()
    
    print(f"Processing: {filename:<30} | Init: G={init_gates}, D={init_depth}, WSA={init_wsa:.1f}")
    
    start_time = time.time()
    
    initial_stats = {'gates': init_gates, 'depth': init_depth, 'wsa': init_wsa}
    step_records = []
    
    current_gates = init_gates
    current_depth = init_depth

    for i in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        new_gates = int(info['raw_area'])
        new_depth = int(info['raw_depth'])
        new_wsa = env.mig_manager.get_switching_activity()
        
        step_records.append({
            'step': i + 1,
            'action': info['action_name'],
            'reward': reward,
            'gates': new_gates,
            'depth': new_depth,
            'wsa': new_wsa,
            'gate_diff': int(new_gates - current_gates),
            'depth_diff': int(new_depth - current_depth)
        })
        
        current_gates = new_gates
        current_depth = new_depth
        
        if terminated or truncated:
            break
            
    elapsed_time = time.time() - start_time
    
    # 1. 保存优化后的电路
    new_filename = filename.replace(".aig", f"_opt_{cfg.CURRENT_MODE}.aig")
    save_path = os.path.join(cfg.RESULTS_DIR, new_filename)
    env.mig_manager.save(save_path)
    
    # 2. 验证电路
    cec_status = verify_equivalence(aig_file, save_path)
    
    # 3. 计算最终指标
    final_wsa = env.mig_manager.get_switching_activity()
    
    final_stats = {
        'gates': current_gates,
        'depth': current_depth,
        'wsa': final_wsa,
        'gate_imp': (initial_stats['gates'] - current_gates) / initial_stats['gates'] * 100,
        'depth_imp': (initial_stats['depth'] - current_depth) / initial_stats['depth'] * 100,
        'wsa_imp': (initial_stats['wsa'] - final_wsa) / initial_stats['wsa'] * 100
    }
    
    # 4. 生成日志
    log_path = save_log_file(filename, initial_stats, step_records, final_stats, cec_status, elapsed_time)
    
    print(f"   -> Log: {os.path.basename(log_path)}")
    print(f"   -> Res: Gates {final_stats['gate_imp']:.2f}% | Depth {final_stats['depth_imp']:.2f}% | Power {final_stats['wsa_imp']:.2f}% | CEC: {cec_status}\n")
    
    # 返回给 CSV 的数据
    return {
        "Circuit": filename,
        "Mode": cfg.CURRENT_MODE,
        
        "Init_Gates": initial_stats['gates'],
        "Final_Gates": final_stats['gates'],
        "Gate_Imp(%)": round(final_stats['gate_imp'], 2),
        
        "Init_Depth": initial_stats['depth'],
        "Final_Depth": final_stats['depth'],
        "Depth_Imp(%)": round(final_stats['depth_imp'], 2),
        
        "Init_WSA": round(initial_stats['wsa'], 1),
        "Final_WSA": round(final_stats['wsa'], 1),
        "WSA_Imp(%)": round(final_stats['wsa_imp'], 2),
        
        "CEC_Check": cec_status,
        "Time(s)": round(elapsed_time, 2),
        "Steps": len(step_records)
    }

def main():
    # 0. 检查目录
    if not os.path.exists(cfg.RESULTS_DIR):
        os.makedirs(cfg.RESULTS_DIR)

    # 1. 加载模型
    if not os.path.exists(cfg.MODEL_PATH + ".zip"):
        print(f"[Error] Model file not found: {cfg.MODEL_PATH}.zip")
        print(f"Please run 'python python/train.py' to train the {cfg.CURRENT_MODE} model first.")
        return

    print(f"Loading model: {cfg.MODEL_PATH} ...")
    model = PPO.load(cfg.MODEL_PATH, device="cpu")

    # 2. 获取测试文件
    files = glob.glob(cfg.TEST_DATA_DIR)
    if not files:
        print(f"[Error] No .aig files found in {cfg.TEST_DATA_DIR}")
        return

    print(f"Found {len(files)} circuits. Testing Mode: {cfg.CURRENT_MODE.upper()}")
    print(f"Results will be saved to: {cfg.RESULTS_DIR}\n")
    
    # 3. 批量测试
    results_list = []
    for i, aig_file in enumerate(files):
        print(f"[{i+1}/{len(files)}] ", end="")
        try:
            res = evaluate_single_circuit(model, aig_file)
            results_list.append(res)
        except Exception as e:
            print(f"[Critical Error] Failed on {aig_file}: {e}")

    # 4. 生成汇总 CSV
    if results_list:
        df = pd.DataFrame(results_list)
        csv_name = f"benchmark_summary_{cfg.CURRENT_MODE}.csv"
        csv_path = os.path.join(cfg.RESULTS_DIR, csv_name)
        
        # 调整列顺序，把 Power 放在 Depth 后面
        cols = ["Circuit", "Mode", 
                "Init_Gates", "Final_Gates", "Gate_Imp(%)", 
                "Init_Depth", "Final_Depth", "Depth_Imp(%)",
                "Init_WSA", "Final_WSA", "WSA_Imp(%)",
                "CEC_Check", "Time(s)", "Steps"]
        # 确保 DataFrame 包含所有列再排序，防止报错
        df = df.reindex(columns=cols)
        
        df.to_csv(csv_path, index=False)
        
        print("="*60)
        print(f"FINAL REPORT: {csv_path}")
        print(f"Pass Rate: {len(df[df['CEC_Check']=='PASS'])}/{len(files)}")
        print("="*60)

if __name__ == "__main__":
    main()