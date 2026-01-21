// ---------------------------------------------------------
// src/mig_env.cpp - V10 (Compatibility Fix)
// ---------------------------------------------------------
#include <cstdint>
#include <pybind11/pybind11.h>

// IO
#include <mockturtle/io/aiger_reader.hpp>
#include <mockturtle/io/write_aiger.hpp>

// Networks
#include <mockturtle/networks/aig.hpp>
#include <mockturtle/networks/mig.hpp>

// Views
#include <mockturtle/views/depth_view.hpp>
#include <mockturtle/views/fanout_view.hpp>

// Algorithms
#include <mockturtle/algorithms/mig_resub.hpp>
#include <mockturtle/algorithms/node_resynthesis.hpp>
#include <mockturtle/algorithms/node_resynthesis/akers.hpp>
#include <mockturtle/algorithms/refactoring.hpp>

// 【关键修复 1】恢复使用 MIG 专用的重写头文件
// 如果这个文件也不存在，请尝试注释掉它，因为有些版本包含在 mig.hpp 中
#include <mockturtle/algorithms/mig_algebraic_rewriting.hpp>

// 【关键修复 2】Balancing
#include <mockturtle/algorithms/balancing.hpp>
#include <mockturtle/algorithms/balancing/sop_balancing.hpp>

#include <iostream>
#include <lorina/aiger.hpp>
#include <string>
#include <vector>

namespace py = pybind11;

class MigManager {
public:
  mockturtle::mig_network mig;
  std::vector<mockturtle::mig_network::signal> node_map;
  std::vector<bool> is_mapped;

  MigManager(std::string filename) {
    mockturtle::aig_network aig;
    if (lorina::read_aiger(filename, mockturtle::aiger_reader(aig)) !=
        lorina::return_code::success) {
      throw std::runtime_error("Failed to parse AIGER file");
    }

    uint32_t max_idx = 0;
    aig.foreach_node([&](auto n) {
      if (aig.node_to_index(n) > max_idx)
        max_idx = aig.node_to_index(n);
    });
    node_map.resize(max_idx + 1, mig.get_constant(false));
    is_mapped.resize(max_idx + 1, false);

    auto const_idx = aig.node_to_index(aig.get_node(aig.get_constant(false)));
    node_map[const_idx] = mig.get_constant(false);
    is_mapped[const_idx] = true;

    aig.foreach_pi([&](auto n) {
      auto mig_pi = mig.create_pi();
      node_map[aig.node_to_index(n)] = mig_pi;
      is_mapped[aig.node_to_index(n)] = true;
    });

    aig.foreach_po([&](auto f) {
      auto aig_node = aig.get_node(f);
      auto mig_signal = get_mig_signal(aig, aig.node_to_index(aig_node));
      if (aig.is_complemented(f))
        mig_signal = !mig_signal;
      mig.create_po(mig_signal);
    });

    std::cout << "[C++] Rebuilt. Gates: " << mig.num_gates() << std::endl;
  }

  mockturtle::mig_network::signal get_mig_signal(mockturtle::aig_network &aig,
                                                 uint32_t node_idx) {
    if (is_mapped[node_idx])
      return node_map[node_idx];

    auto n = aig.index_to_node(node_idx);
    std::vector<mockturtle::aig_network::signal> children;
    aig.foreach_fanin(n, [&](auto const &f) { children.push_back(f); });

    auto mig_f1 =
        get_mig_signal(aig, aig.node_to_index(aig.get_node(children[0])));
    auto mig_f2 =
        get_mig_signal(aig, aig.node_to_index(aig.get_node(children[1])));

    if (aig.is_complemented(children[0]))
      mig_f1 = !mig_f1;
    if (aig.is_complemented(children[1]))
      mig_f2 = !mig_f2;

    auto mig_node = mig.create_maj(mig_f1, mig_f2, mig.get_constant(false));
    node_map[node_idx] = mig_node;
    is_mapped[node_idx] = true;
    return mig_node;
  }

  // --- 动作空间适配 ---

  // 【Action 0】适配：使用编译器推荐的 depth_rewriting
  void rewrite() {
    // 你的版本中只有这个函数，它效果也很好，主要优化深度
    mockturtle::depth_view<mockturtle::mig_network> depth_mig(mig);
    mockturtle::mig_algebraic_depth_rewriting(depth_mig);
  }

  // 【Action 1】适配：Refactoring
  void rewrite_aggressive() {
    mockturtle::refactoring_params ps;
    ps.allow_zero_gain = true;
    mockturtle::akers_resynthesis<mockturtle::mig_network> resyn;
    mockturtle::refactoring(mig, resyn, ps);
  }

  // 【Action 2】适配：简化 Balancing 参数
  void balance() {
    // 1. 定义转换策略 (MIG <-> AIG)
    mockturtle::akers_resynthesis<mockturtle::aig_network> resyn_mig2aig;
    mockturtle::akers_resynthesis<mockturtle::mig_network> resyn_aig2mig;

    // 2. MIG -> AIG (为了使用成熟的 AIG balancing)
    auto aig = mockturtle::node_resynthesis<mockturtle::aig_network>(
        mig, resyn_mig2aig);

    // 3. 准备平衡参数
    mockturtle::balancing_params ps;
    ps.cut_enumeration_ps.cut_size = 4;

    // 【关键修复点】显式构造 rebalancing_function_t
    // 编译器无法自动将 sop_rebalancing 推导为 rebalancing_function_t (即
    // std::function) 我们必须手动声明这个类型，触发显式转换。
    mockturtle::rebalancing_function_t<mockturtle::aig_network> strategy =
        mockturtle::sop_rebalancing<mockturtle::aig_network>{};

    // 4. 执行平衡
    // 现在传入的 strategy 类型严格匹配，不会报错
    auto balanced_aig = mockturtle::balancing(aig, strategy, ps);

    // 5. AIG -> MIG (转回)
    mig = mockturtle::node_resynthesis<mockturtle::mig_network>(balanced_aig,
                                                                resyn_aig2mig);
  }

  // 【Action 3】Resubstitution
  void resub() {
    mockturtle::resubstitution_params ps;
    ps.max_inserts = 2;
    mockturtle::depth_view<mockturtle::mig_network> depth_mig(mig);
    mockturtle::fanout_view<mockturtle::depth_view<mockturtle::mig_network>>
        view(depth_mig);
    mockturtle::mig_resubstitution(view, ps);
  }

  void save(std::string filename) {
    mockturtle::akers_resynthesis<mockturtle::aig_network> resyn;
    auto aig =
        mockturtle::node_resynthesis<mockturtle::aig_network>(mig, resyn);
    mockturtle::write_aiger(aig, filename);
  }

  int get_node_count() { return mig.num_gates(); }
  int get_depth() {
    mockturtle::depth_view<mockturtle::mig_network> d(mig);
    return d.depth();
  }
  void reset(std::string filename) { *this = MigManager(filename); }
};

PYBIND11_MODULE(mig_core, m) {
  py::class_<MigManager>(m, "MigManager")
      .def(py::init<std::string>())
      .def("get_node_count", &MigManager::get_node_count)
      .def("get_depth", &MigManager::get_depth)
      .def("rewrite", &MigManager::rewrite)
      .def("rewrite_aggressive", &MigManager::rewrite_aggressive)
      .def("balance", &MigManager::balance)
      .def("resub", &MigManager::resub)
      .def("reset", &MigManager::reset)
      .def("save", &MigManager::save);
}