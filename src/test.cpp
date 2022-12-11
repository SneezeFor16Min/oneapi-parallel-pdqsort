#ifndef _PDQ_TEST
#define _PDQ_TEST

#include "dpc_common.hpp"
#include "./util.cpp"
#include "./impl.cpp"

namespace test
{
using namespace sycl;
namespace rng = std::ranges;

int static n_threads = -1;

template <rng::random_access_range R>
void parallel_pdqsort_demo(R& arr) {
  size_t const len = rng::size(arr);
  std::cout << std::format("Array size: {}", len) << std::endl;

  try {
    queue q(cpu_selector{}, dpc_common::exception_handler);
    std::cout << std::format("Device(CPU): {}\nNum of cores: {}\n", q.get_device().get_info<info::device::name>(),
                             q.get_device().get_info<info::device::max_compute_units>());

    using T = rng::range_value_t<R>;
    usm_allocator<T, usm::alloc::shared> q_alloc{ q };
    std::vector<T, decltype(q_alloc)> usm_arr{ q_alloc };
    usm_arr.assign_range(arr);

    std::vector<T> sorted{ arr };
    {
      auto const t0 = tbb::tick_count::now();
      rng::sort(sorted);
      auto const t1 = tbb::tick_count::now();
      std::cout << std::format("Time usage for std::sort: {} sec\n", (t1 - t0).seconds());
      util::print(sorted);
    }

    tbb::task_arena arena;
    n_threads = arena.max_concurrency();
    std::cout << std::format("Num of workers: {}", n_threads) << std::endl;

    arena.execute([&] {
      std::cout << "Start pdqsort... ";
      auto const t0 = tbb::tick_count::now();
      impl::parallel_pdqsort(usm_arr);
      auto const t1 = tbb::tick_count::now();
      std::cout << "Complete\n";
      std::cout << std::format("Time usage for pdqsort: {} sec\n", (t1 - t0).seconds());
    });
    std::cout << std::format("Checking if array is the same as std::sort result... {}\n",
                             rng::equal(usm_arr, sorted) ? "Yes!" : "No?");

    arr.assign_range(usm_arr);
  } catch (exception const& e) {
    std::cerr << "An error happened: " << e.what() << std::endl;
    std::terminate();
  }
}
}  // namespace test

int main(int argc, char* argv[]) {
  size_t len = 0;
  while (std::cout << "Enter data size: ", std::cin >> len) {
    auto v = util::generate_vec(len, util::GenMode::Random);
    util::print(v);
    test::parallel_pdqsort_demo(v);
    util::print(v);
    std::cout << "====================" << std::endl;
  }

  return 0;
}

#endif