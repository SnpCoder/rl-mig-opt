import sys
import os
import subprocess

# ==========================================
# 1. 关键修复：添加 build 路径
# ==========================================
# 获取当前脚本所在目录 (python/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取 build 目录 (../build)
build_path = os.path.abspath(os.path.join(current_dir, '../build'))

# 将 build 加入 Python 搜索路径
if build_path not in sys.path:
    sys.path.append(build_path)

# ==========================================
# 2. 导入模块 (注意是 mig_core 单数)
# ==========================================
try:
    import mig_core 
except ImportError as e:
    print(f"\n[Error] 无法导入 mig_core 模块!")
    print(f"Python 正在搜索的路径: {sys.path}")
    print(f"请检查 build 目录下是否有 .so 文件，且名字包含 'mig_core'")
    sys.exit(1)

# ==========================================
# 3. 测试逻辑
# ==========================================
# 配置路径
AIG_PATH = os.path.join(current_dir, '../benchmarks/arithmetic/adder.aig')
# ABC 路径 (根据你的实际安装位置调整)
ABC_PATH = os.path.join(current_dir, '../lib/abc/abc') 
SAVE_PATH = "sanity_test.aig"

def verify(original, optimized):
    print(f"[*] Verifying: {original} vs {optimized}")
    
    if not os.path.exists(ABC_PATH):
        print(f"[Warning] 找不到 ABC 工具: {ABC_PATH}")
        return False

    # 使用 &cec (基于结构/顺序的验证)
    cmd = f'"{ABC_PATH}" -c "&r {original}; &cec {optimized}"'
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if "Networks are equivalent" in res.stdout:
            print("✅ 验证通过！(Equivalent)")
            return True
        else:
            print("❌ 验证失败！(Not Equivalent)")
            print("--- ABC Output ---")
            print(res.stdout)
            print("------------------")
            return False
    except Exception as e:
        print(f"ABC 执行出错: {e}")
        return False

def main():
    print("=== 开始白盒测试 (Read -> Save -> Verify) ===")
    
    if not os.path.exists(AIG_PATH):
        print(f"找不到输入文件: {AIG_PATH}")
        return

    # 1. 初始化
    print("1. 读取电路...")
    try:
        mgr = mig_core.MigManager(AIG_PATH)
        print(f"   初始状态: Gates={mgr.get_node_count()}, Depth={mgr.get_depth()}")
    except Exception as e:
        print(f"读取 AIGER 失败: {e}")
        return

    # 2. 什么都不做，直接保存
    print(f"2. 直接保存到 {SAVE_PATH} (不做任何优化)...")
    mgr.save(SAVE_PATH)

    # 3. 验证
    print("3. 进行 CEC 验证...")
    if verify(AIG_PATH, SAVE_PATH):
        print("\n[结论]：工具链正常。MIG 读取/保存流程是安全的。")
        print("       说明之前验证失败是由于【优化算法】改坏了逻辑。")
    else:
        print("\n[结论]：工具链异常！仅仅是读取再保存，逻辑就变了。")
        print("       说明 mockturtle 在转换 adder.aig 时丢失了端口信息。")
        print("       建议更换测试用例 (如 adder.aig)。")

if __name__ == "__main__":
    main()