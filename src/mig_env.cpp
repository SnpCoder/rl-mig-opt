#include <iostream>
#include <pybind11/pybind11.h>
#include <string>

// Mockturtle 和相关库的头文件
#include <lorina/verilog.hpp>
#include <mockturtle/algorithms/mig_algebraic_rewriting.hpp>
#include <mockturtle/algorithms/node_resynthesis/akers.hpp>
#include <mockturtle/io/aiger_reader.hpp>
#include <mockturtle/networks/mig.hpp>
#include <mockturtle/views/depth_view.hpp>

namespace py = pybind11;
using namespace mockturtle;

class MigManager {
public:
  mig_network mig;

  // 构造函数：读取 Verilog 文件构建 MIG
  MigManager(std::string filename) {
    // 仅保留纯读取逻辑，无诊断引擎、无多余参数
    lorina::return_code result =
        lorina::read_aiger(filename, aiger_reader(mig));

    if (result != lorina::return_code::success) {
      throw std::runtime_error("Failed to parse Verilog file: " + filename);
    }

    // 调试打印保留，不影响编译和功能
    std::cout << "[C++ Debug] Loaded file: " << filename << std::endl;
    std::cout << "[C++ Debug] Gates: " << mig.num_gates() << std::endl;
    std::cout << "[C++ Debug] PIs (Inputs): " << mig.num_pis() << std::endl;
    std::cout << "[C++ Debug] POs (Outputs): " << mig.num_pos() << std::endl;

    if (mig.num_gates() == 0) {
      std::cout << "[C++ Warning] !!! The circuit is empty (0 gates). Check "
                   "Verilog format! !!!"
                << std::endl;
    }
  }

  // 获取当前节点数 (Area)
  int get_node_count() { return mig.num_gates(); }

  // 获取当前深度 (Depth)
  int get_depth() {
    // 为了获取深度，需要临时创建一个 depth_view
    depth_view<mig_network> depth_mig(mig);
    return depth_mig.depth();
  }

  // 动作 1: 代数深度重写 (Rewriting)
  void rewrite() {
    mig_algebraic_depth_rewriting_params ps;
    ps.allow_area_increase = false;

    // 【关键修改点】
    // 原始的 mig 对象不懂什么是"depth"，所以不能直接传给 depth_rewriting。
    // 我们必须用 depth_view 包装它。
    // 这个 view 不仅提供深度信息，还能把修改操作传回给底层的 mig。
    depth_view<mig_network> depth_mig(mig);

    // 将 view 传给算法，而不是原始 mig
    mig_algebraic_depth_rewriting(depth_mig, ps);
  }

  // 重置/重新加载
  void reset(std::string filename) {
    mig = mig_network(); // 清空旧网络

    // 这里也顺手修复了之前的 warning，检查返回值
    if (lorina::read_verilog(filename, verilog_reader(mig)) !=
        lorina::return_code::success) {
      throw std::runtime_error("Failed to reset Verilog file: " + filename);
    }
  }
};

// Pybind11 绑定代码
PYBIND11_MODULE(mig_core, m) {
  py::class_<MigManager>(m, "MigManager")
      .def(py::init<std::string>())
      .def("get_node_count", &MigManager::get_node_count)
      .def("get_depth", &MigManager::get_depth)
      .def("rewrite", &MigManager::rewrite)
      .def("reset", &MigManager::reset);
}
