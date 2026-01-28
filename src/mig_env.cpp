// ---------------------------------------------------------
// mig tool box
// ---------------------------------------------------------
#include <cstdint>
#include <memory> 
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
#include <mockturtle/algorithms/mig_algebraic_rewriting.hpp>
#include <mockturtle/algorithms/balancing.hpp>
#include <mockturtle/algorithms/balancing/sop_balancing.hpp>
#include <mockturtle/algorithms/cleanup.hpp>

#include <iostream>
#include <lorina/aiger.hpp>
#include <string>
#include <vector>

namespace py = pybind11;

class MigManager {
public:
  std::unique_ptr<mockturtle::mig_network> mig;

  std::vector<mockturtle::mig_network::signal> node_map;
  std::vector<bool> is_mapped;

  MigManager(std::string filename) {
    load_file(filename);
  }

  void load_file(std::string filename) {
    mig.reset(); 
    node_map.clear();
    is_mapped.clear();

    mig = std::make_unique<mockturtle::mig_network>();

    mockturtle::aig_network aig;
    if (lorina::read_aiger(filename, mockturtle::aiger_reader(aig)) != lorina::return_code::success) {
      throw std::runtime_error("Failed to parse AIGER file: " + filename);
    }

    uint32_t max_idx = 0;
    aig.foreach_node([&](auto n) {
      if (aig.node_to_index(n) > max_idx) max_idx = aig.node_to_index(n);
    });
    
    node_map.resize(max_idx + 1, mig->get_constant(false));
    is_mapped.resize(max_idx + 1, false);

    auto const_idx = aig.node_to_index(aig.get_node(aig.get_constant(false)));
    node_map[const_idx] = mig->get_constant(false);
    is_mapped[const_idx] = true;

    aig.foreach_pi([&](auto n) {
      auto mig_pi = mig->create_pi();
      node_map[aig.node_to_index(n)] = mig_pi;
      is_mapped[aig.node_to_index(n)] = true;
    });

    aig.foreach_po([&](auto f) {
      auto aig_node = aig.get_node(f);
      auto mig_signal = get_mig_signal(aig, aig.node_to_index(aig_node));
      if (aig.is_complemented(f)) mig_signal = !mig_signal;
      mig->create_po(mig_signal);
    });
  }

  mockturtle::mig_network::signal get_mig_signal(mockturtle::aig_network &aig, uint32_t node_idx) {
    if (is_mapped[node_idx]) return node_map[node_idx];

    auto n = aig.index_to_node(node_idx);
    std::vector<mockturtle::aig_network::signal> children;
    aig.foreach_fanin(n, [&](auto const &f) { children.push_back(f); });

    auto mig_f1 = get_mig_signal(aig, aig.node_to_index(aig.get_node(children[0])));
    auto mig_f2 = get_mig_signal(aig, aig.node_to_index(aig.get_node(children[1])));

    if (aig.is_complemented(children[0])) mig_f1 = !mig_f1;
    if (aig.is_complemented(children[1])) mig_f2 = !mig_f2;

    auto mig_node = mig->create_maj(mig_f1, mig_f2, mig->get_constant(false));
    node_map[node_idx] = mig_node;
    is_mapped[node_idx] = true;
    return mig_node;
  }

  // action
  void rewrite() {
    mockturtle::depth_view<mockturtle::mig_network> depth_mig(*mig);
    mockturtle::mig_algebraic_depth_rewriting(depth_mig);
  }

  void refactor() {
    mockturtle::refactoring_params ps;
    ps.allow_zero_gain = true;
    mockturtle::akers_resynthesis<mockturtle::mig_network> resyn;
    mockturtle::refactoring(*mig, resyn, ps);
  }

  void balance() {
    mockturtle::akers_resynthesis<mockturtle::aig_network> resyn_mig2aig;
    mockturtle::akers_resynthesis<mockturtle::mig_network> resyn_aig2mig;

    bool is_huge = mig->num_gates() > 50000; 

    auto aig = mockturtle::node_resynthesis<mockturtle::aig_network>(*mig, resyn_mig2aig);

    mockturtle::balancing_params ps;
    if (is_huge) {
        ps.cut_enumeration_ps.cut_size = 4;   
        ps.only_on_critical_path = true;      
    } else {
        ps.cut_enumeration_ps.cut_size = 6; 
        ps.only_on_critical_path = false; 
    }

    mockturtle::rebalancing_function_t<mockturtle::aig_network> strategy = 
        mockturtle::sop_rebalancing<mockturtle::aig_network>{};
        
    auto balanced_aig = mockturtle::balancing(aig, strategy, ps);

    auto new_mig_obj = mockturtle::node_resynthesis<mockturtle::mig_network>(balanced_aig, resyn_aig2mig);
    mig = std::make_unique<mockturtle::mig_network>(std::move(new_mig_obj));

    if (!is_huge) {
        mockturtle::depth_view<mockturtle::mig_network> depth_mig(*mig);
        mockturtle::mig_algebraic_depth_rewriting(depth_mig);
    } else {
        auto cleaned_mig = mockturtle::cleanup_dangling(*mig);
        mig = std::make_unique<mockturtle::mig_network>(std::move(cleaned_mig));
    }
  }

  void resub() {
    mockturtle::resubstitution_params ps;
    ps.max_inserts = 1; 
    mockturtle::depth_view<mockturtle::mig_network> depth_mig(*mig);
    mockturtle::fanout_view<mockturtle::depth_view<mockturtle::mig_network>> view(depth_mig);
    mockturtle::mig_resubstitution(view, ps);
  }

  void save(std::string filename) {
    mockturtle::akers_resynthesis<mockturtle::aig_network> resyn;
    auto aig = mockturtle::node_resynthesis<mockturtle::aig_network>(*mig, resyn);
    mockturtle::write_aiger(aig, filename);
  }

  int get_node_count() { return mig->num_gates(); }
  int get_depth() {
    mockturtle::depth_view<mockturtle::mig_network> d(*mig);
    return d.depth();
  }

  // Weighted Switching Activity, WSA
  float get_switching_activity() {
    std::vector<double> probs(mig->size(), 0.0);
    
    // initialize PI prob = 0.5 (random input)
    mig->foreach_pi([&](auto n) {
        probs[mig->node_to_index(n)] = 0.5;
    });
    
    // constant 0 's prob = 0
    probs[mig->node_to_index(mig->get_node(mig->get_constant(false)))] = 0.0;

    // calc all nodes prob
    mig->foreach_node([&](auto n) {
        if (mig->is_constant(n) || mig->is_pi(n)) return;
        
        std::vector<double> child_probs;
        mig->foreach_fanin(n, [&](auto const& f) {
            auto child_node = mig->get_node(f);
            double p = probs[mig->node_to_index(child_node)];
            // Inverted : P -> (1-P)
            if (mig->is_complemented(f)) p = 1.0 - p;
            child_probs.push_back(p);
        });
        
        // MIG node is MAJ3 Gate
        if (child_probs.size() >= 3) {
            double pa = child_probs[0];
            double pb = child_probs[1];
            double pc = child_probs[2];
            // MAJ3 prob: ab + bc + ca - 2abc
            probs[mig->node_to_index(n)] = pa*pb + pb*pc + pa*pc - 2.0*pa*pb*pc;
        }
    });

    // calc total WSA
    mockturtle::fanout_view<mockturtle::mig_network> fanout_mig(*mig);
    double total_wsa = 0.0;
    
    mig->foreach_node([&](auto n) {
        // only logic gate
        if (mig->is_constant(n) || mig->is_pi(n)) return;
        
        double p = probs[mig->node_to_index(n)];
        // switching activity alpha = 2 * P * (1-P)
        double switching = 2.0 * p * (1.0 - p); 
        
        int fanout_count = fanout_mig.fanout_size(n);
        
        // WSA = switching activity * (intrinsic capacitance + load capacitance)
        total_wsa += switching * (1.0 + (double)fanout_count);
    });
    
    return (float)total_wsa;
  }

  void reset(std::string filename) {
    load_file(filename);
  }
};

PYBIND11_MODULE(mig_core, m) {
  py::class_<MigManager>(m, "MigManager")
      .def(py::init<std::string>())
      .def("get_node_count", &MigManager::get_node_count)
      .def("get_depth", &MigManager::get_depth)
      .def("get_switching_activity", &MigManager::get_switching_activity)

      .def("rewrite", &MigManager::rewrite, py::call_guard<py::gil_scoped_release>())
      .def("refactor", &MigManager::refactor, py::call_guard<py::gil_scoped_release>())
      .def("balance", &MigManager::balance, py::call_guard<py::gil_scoped_release>())
      .def("resub", &MigManager::resub, py::call_guard<py::gil_scoped_release>())
      
      .def("reset", &MigManager::reset) 
      .def("save", &MigManager::save);
}