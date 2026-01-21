cmake_minimum_required(VERSION 3.10)
project(mig_rl_project LANGUAGES CXX C) 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ==========================================
# 1. 强制全局宏定义
# ==========================================
add_compile_definitions(LINUX=1)
add_compile_definitions(LIN64=1)
add_compile_definitions(SIZEOF_VOID_P=8)
add_compile_definitions(ABC_NAMESPACE=pabc)

# 屏蔽警告
add_compile_options(-Wno-macro-redefined)
add_compile_options(-Wno-format)
add_compile_options(-Wno-return-type)
add_compile_options(-Wno-sign-compare)
add_compile_options(-Wno-implicit-function-declaration) # 针对部分老代码

# ==========================================
# 2. 精确查找源文件 (白名单模式)
# ==========================================

# --- Nauty 核心库文件白名单 ---
# 我们只编译这些库文件，忽略 poptest.c, geng.c 等工具文件
set(NAUTY_CORE_FILES
    nauty.c
    nautil.c
    naugraph.c
    naugroup.c
    naurng.c
    nausparse.c
    nautaux.c
    nautinv.c
    naututil.c
    schreier.c
    traces.c
    gtools.c
    gutil1.c
    gutil2.c
)

set(NAUTY_SRCS "")
# 在 mockturtle 或 percy 路径下查找这些核心文件
foreach(SRC ${NAUTY_CORE_FILES})
    # 尝试路径 1: mockturtle/lib/nauty
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lib/mockturtle/lib/nauty/${SRC}")
        list(APPEND NAUTY_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/lib/mockturtle/lib/nauty/${SRC}")
    # 尝试路径 2: percy/nauty
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lib/percy/nauty/${SRC}")
        list(APPEND NAUTY_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/lib/percy/nauty/${SRC}")
    endif()
endforeach()

# --- ABC/ESOP 源码 ---
# 这里通常比较干净，可以用 GLOB，但为了保险，也可以排除 demo
file(GLOB ABC_SRCS 
    "${CMAKE_CURRENT_SOURCE_DIR}/lib/mockturtle/lib/abcesop/*.c"
)

# ==========================================
# 3. Pybind11 & FMT
# ==========================================
add_subdirectory(lib/pybind11)

set(FMT_SOURCES "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lib/fmt/src/format.cc")
    set(FMT_SOURCES lib/fmt/src/format.cc)
endif()

# ==========================================
# 4. 定义模块
# ==========================================
pybind11_add_module(
    mig_core
    src/mig_env.cpp
    ${FMT_SOURCES}
    ${NAUTY_SRCS}
    ${ABC_SRCS}
)

# ==========================================
# 5. 头文件路径
# ==========================================
target_include_directories(mig_core PRIVATE
    src
    lib/mockturtle/include
    lib/kitty/include
    lib/lorina/include
    lib/fmt/include
    lib/parallel_hashmap
    lib/bill/include
    lib/percy/include
    lib/mockturtle/lib/abcesop
    lib/mockturtle/lib/abcesop/eabc 
    lib/percy/nauty
    lib/mockturtle/lib/nauty 
    lib/bill/include/bill/sat/solver
)

# ==========================================
# 6. 链接
# ==========================================
find_package(Threads REQUIRED)
target_link_libraries(mig_core PRIVATE Threads::Threads)