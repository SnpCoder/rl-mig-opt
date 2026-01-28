import os

# ==================== 1. 全局控制中心 (只改这里！) ====================

# [当前模式] 可选: 'depth', 'area', 'balanced'
CURRENT_MODE = "area" 

# ===================================================================

# --- 2. 模式定义 (不需要改) ---
MODE_CONFIGS = {
    "depth": {
        "model_name": "mig_opt_model_depth",
        "desc": "DEPTH FOCUS: 追求极致速度，允许面积适当膨胀",
    },
    "area": {
        "model_name": "mig_opt_model_area",
        "desc": "AREA FOCUS: 追求最小面积，严禁深度恶化",
    },
    "balanced": {
        "model_name": "mig_opt_model_balanced",
        "desc": "BALANCED: 面积与深度的双重权衡",
    }
}

if CURRENT_MODE not in MODE_CONFIGS:
    raise ValueError(f"Unknown mode: {CURRENT_MODE}")

CURRENT_CONFIG = MODE_CONFIGS[CURRENT_MODE]
MODEL_NAME = CURRENT_CONFIG["model_name"]

# --- 3. 路径管理 (自动处理) ---
# 项目根目录 (python 文件夹的上一级)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 模型保存路径
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# 日志路径
LOG_DIR = os.path.join(PROJECT_ROOT, 'mig_opt_logs', CURRENT_MODE)

# 结果路径
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# 数据集路径
VERILOG_FILE = os.path.join(PROJECT_ROOT, 'benchmarks/big/mccarthy91.phx.aig')
DATASET_PATH = os.path.join(PROJECT_ROOT, 'benchmarks/small/*.aig')
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, 'benchmarks/big/*.aig')

# ABC 工具路径
ABC_BINARY_PATH = os.path.join(PROJECT_ROOT, 'lib/abc/abc')

# 训练参数
NUM_CPU = 8
DEVICE = "cpu" # 保持 CPU 以避免冲突