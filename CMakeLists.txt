cmake_minimum_required(VERSION 3.10)
project(mig_rl_project)

set(CMAKE_CXX_STANDARD 17)

# 1. 包含 pybind11
add_subdirectory(lib/pybind11)

# 2. 设置 Mockturtle 及其依赖的头文件路径
# 注意：根据你实际 clone 的位置调整这些路径
include_directories(
    lib/mockturtle/include
    lib/kitty/include
    lib/lorina/include
    lib/fmt/include
    lib/parallel_hashmap
)

# 显式包含 fmt 的原文件
set(FMT_SOURCES lib/fmt/src/format.cc)

# 3. 创建 Python 模块
# "mig_core" 必须和 C++ 代码中 PYBIND11_MODULE(mig_core, m) 里的名字一样
pybind11_add_module(mig_core src/mig_env.cpp ${FMT_SOURCES})

# 如果 mockturtle 或其依赖库需要链接特定的库（通常 header-only 不需要，但 fmt 可能需要）
# target_link_libraries(mig_core PRIVATE fmt) 
