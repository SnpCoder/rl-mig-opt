import os
import sys
import glob
import subprocess
import shutil
import time

# ================= 配置区域 =================
# 1. 设置你的 AIG 文件所在目录
# DATASET_DIR = "../benchmarks/small"  # 请根据实际情况修改
DATASET_DIR = "../benchmarks/big" 

# 2. 设置 build 目录路径 (为了找到 mig_core)
# 假设脚本在 python/ 目录下，build 在 ../build
BUILD_DIR = "../build"

# 3. 设置隔离区 (有问题的电路会被移到这里)
QUARANTINE_DIR = "../benchmarks/quarantine"
# ===========================================

def get_abs_path(rel_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), rel_path))

def check_file_safe(file_path, build_path):
    """
    启动一个子进程来测试文件。
    返回: True (安全), False (崩溃/失败)
    """
    # 构造一段微型的 Python 代码，在子进程中运行
    # 这段代码只做一件事：导入库 -> 加载文件 -> 退出
    worker_code = f"""
import sys
import os
sys.path.append('{build_path}')
try:
    import mig_core
except ImportError:
    print("[Child] Error: Could not import mig_core")
    sys.exit(1)

try:
    # 尝试加载
    mgr = mig_core.MigManager('{file_path}')
    # 尝试简单操作确保没坏
    n = mgr.get_node_count()
    if n == 0: sys.exit(2) # 空电路也不行
except Exception as e:
    print(f"[Child] Exception: {{e}}")
    sys.exit(1)

sys.exit(0) # 成功存活
"""

    try:
        # 启动子进程
        result = subprocess.run(
            [sys.executable, "-c", worker_code],
            capture_output=True, # 捕获输出，防止刷屏
            text=True,
            timeout=5 # 设置超时，防止死锁（例如读取超大文件卡死）
        )
        
        # 检查返回值
        if result.returncode == 0:
            return True, None
        else:
            # 返回码 -11 通常是 Segmentation Fault
            error_msg = f"Process died with code {result.returncode}. Stderr: {result.stderr.strip()}"
            return False, error_msg

    except subprocess.TimeoutExpired:
        return False, "Timeout (Loading took too long)"
    except Exception as e:
        return False, str(e)

def main():
    abs_dataset_dir = get_abs_path(DATASET_DIR)
    abs_build_dir = get_abs_path(BUILD_DIR)
    abs_quarantine_dir = get_abs_path(QUARANTINE_DIR)

    print(f"[*] 正在扫描目录: {abs_dataset_dir}")
    print(f"[*] 库路径: {abs_build_dir}")
    
    # 获取所有 aig 文件
    aig_files = glob.glob(os.path.join(abs_dataset_dir, "*.aig"))
    # 排除掉 _opt 文件
    aig_files = [f for f in aig_files if "_opt" not in f]
    
    if not aig_files:
        print("[!] 未找到任何 .aig 文件！请检查路径配置。")
        return

    print(f"[*] 发现 {len(aig_files)} 个文件。开始体检...\n")

    if not os.path.exists(abs_quarantine_dir):
        os.makedirs(abs_quarantine_dir)

    good_count = 0
    bad_count = 0
    bad_file_list = []

    # 进度条效果
    total = len(aig_files)
    for i, file_path in enumerate(aig_files):
        filename = os.path.basename(file_path)
        print(f"\r[{i+1}/{total}] Testing: {filename:<40}", end="", flush=True)
        
        is_safe, error_msg = check_file_safe(file_path, abs_build_dir)
        
        if is_safe:
            good_count += 1
        else:
            bad_count += 1
            print(f"\n    ❌ DETECTED BAD FILE: {filename}")
            print(f"       Reason: {error_msg}")
            
            # 移动到隔离区
            try:
                dst = os.path.join(abs_quarantine_dir, filename)
                shutil.move(file_path, dst)
                print(f"       -> 已移动到隔离区: {QUARANTINE_DIR}")
                bad_file_list.append(filename)
            except Exception as e:
                print(f"       -> 移动失败: {e}")

    print("\n" + "="*50)
    print("扫描完成！")
    print(f"✅ 正常文件: {good_count}")
    print(f"❌ 损坏文件: {bad_count}")
    
    if bad_count > 0:
        print("\n以下文件已导致 C++ 崩溃并被移除:")
        for f in bad_file_list:
            print(f" - {f}")
    print("="*50)

if __name__ == "__main__":
    main()