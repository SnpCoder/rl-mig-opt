import os
import time
import glob
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from mig_opt_env import MigOptEnv

# 【核心】导入配置文件，所有路径和模式都在这里管理
import config as cfg

def print_performance_report(initial_metrics, final_metrics):
    """ 打印包含功耗(WSA)的演示报告 """
    # 解包数据：现在包含三个指标 (Area, Depth, Power)
    init_area, init_depth, init_wsa = initial_metrics
    final_area, final_depth, final_wsa = final_metrics
    
    # 计算改进率
    area_imp = (init_area - final_area) / init_area * 100
    depth_imp = (init_depth - final_depth) / init_depth * 100
    wsa_imp = (init_wsa - final_wsa) / init_wsa * 100
    
    print("\n" + "="*60)
    print(f"{'DEMO REPORT':^60}")
    print(f"{f'Mode: {cfg.CURRENT_MODE.upper()}':^60}")
    print("-" * 60)
    print(f"{'Metric':<15} | {'Initial':<12} | {'Final':<12} | {'Improvement'}")
    print("-" * 60)
    
    area_sign = "+" if area_imp > 0 else ""
    print(f"{'Area':<15} | {int(init_area):<12} | {int(final_area):<12} | {area_sign}{area_imp:.2f}%")
    
    depth_sign = "+" if depth_imp > 0 else ""
    print(f"{'Depth':<15} | {int(init_depth):<12} | {int(final_depth):<12} | {depth_sign}{depth_imp:.2f}%")
    
    # 新增：打印功耗 (WSA)
    wsa_sign = "+" if wsa_imp > 0 else ""
    print(f"{'Power (WSA)':<15} | {init_wsa:<12.1f} | {final_wsa:<12.1f} | {wsa_sign}{wsa_imp:.2f}%")
    
    print("="*60 + "\n")

def train():
    # 1. 获取数据集
    all_circuits = glob.glob(cfg.DATASET_PATH)
    train_circuits = [f for f in all_circuits if "new" not in f and "_opt" not in f]
    
    if not train_circuits:
        print(f"[Error] No training circuits found in {cfg.DATASET_PATH}")
        return None

    # 2. 打印训练信息看板
    print(f"\n{'='*60}")
    print(f"STARTING TRAINING SESSION")
    print(f"Mode:         {cfg.CURRENT_MODE.upper()}")
    print(f"Description:  {cfg.CURRENT_CONFIG['desc']}")
    print(f"Save Path:    {cfg.MODEL_PATH}.zip")
    print(f"Dataset Size: {len(train_circuits)} circuits")
    print(f"Device:       {cfg.DEVICE}")
    print(f"{'='*60}\n")
    
    # 3. 创建环境
    vec_env_cls = DummyVecEnv 

    env = make_vec_env(
        lambda: MigOptEnv(train_circuits, target_mode=cfg.CURRENT_MODE), 
        n_envs=cfg.NUM_CPU, 
        vec_env_cls=vec_env_cls
    )

    # 4. 定义模型
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4, 
        device=cfg.DEVICE, 
        n_steps=2048,      
        batch_size=512,
        n_epochs=10,
        ent_coef=0.05,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
        tensorboard_log=cfg.LOG_DIR 
    )

    print(f"----------- Training Started ({cfg.CURRENT_MODE}) -----------")
    start_time = time.time()
    
    total_timesteps = 100000 
    model.learn(total_timesteps=total_timesteps)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"----------- Training Finished ({duration:.2f}s) -----------")

    # 5. 保存模型
    model.save(cfg.MODEL_PATH)
    print(f"Model saved successfully to: {cfg.MODEL_PATH}.zip")
    
    return model

def demo(model):
    """ 训练结束后跑一个简单的 Demo 看看效果 """
    print(f"\n----------- Running Demo ({cfg.CURRENT_MODE}) -----------")
    
    if not os.path.exists(cfg.VERILOG_FILE):
        print(f"Demo file not found: {cfg.VERILOG_FILE}")
        return

    # 使用同样的配置初始化环境
    test_env = MigOptEnv(cfg.VERILOG_FILE, target_mode=cfg.CURRENT_MODE)
    obs, info = test_env.reset()
    
    # 【新增】获取初始功耗指标
    init_wsa = test_env.mig_manager.get_switching_activity()
    init_area = test_env.initial_area
    init_depth = test_env.initial_depth
    
    print(f"Target: {os.path.basename(cfg.VERILOG_FILE)}")
    
    for i in range(40):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = test_env.step(action)
        
        if i < 5: 
            print(f"Step {i+1:02d}: {info['action_name']:<10} | Reward: {reward:+.2f}")

        if term or trunc:
            break
            
    # 【新增】获取最终功耗指标
    final_wsa = test_env.mig_manager.get_switching_activity()
    final_area = test_env.mig_manager.get_node_count()
    final_depth = test_env.mig_manager.get_depth()
    
    # 打印最终报告 (传入三元组)
    print_performance_report(
        (init_area, init_depth, init_wsa), 
        (final_area, final_depth, final_wsa)
    )

if __name__ == '__main__':
    trained_model = train()
    # trained_model = PPO.load(cfg.MODEL_PATH, device="cpu")
    
    if trained_model:
        demo(trained_model)