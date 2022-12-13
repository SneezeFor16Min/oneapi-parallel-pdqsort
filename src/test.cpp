#ifndef _PDQ_TEST
#define _PDQ_TEST

#include "dpc_common.hpp"
#include "./util.cpp"
#include "./impl.cpp"

namespace test
{
using namespace sycl;
namespace rng = std::ranges;

queue static q(cpu_selector{}, dpc_common::exception_handler);
tbb::task_arena arena;
int static n_threads = -1;

template <rng::random_access_range R>
double parallel_pdqsort_demo(R& arr) {
  try {
    using T = rng::range_value_t<R>;
    usm_allocator<T, usm::alloc::shared> q_alloc{ q };
    std::vector<T, decltype(q_alloc)> usm_arr{ q_alloc };
    usm_arr.assign_range(arr);

    tbb::tick_count::interval_t dt;
    arena.execute([&] {
      auto const t0 = tbb::tick_count::now();
      impl::parallel_pdqsort(usm_arr);
      auto const t1 = tbb::tick_count::now();
      dt = (t1 - t0);
    });
    arr.assign_range(usm_arr);

    return dt.seconds();
  } catch (exception const& e) {
    std::cerr << "An error happened: " << e.what() << std::endl;
    std::terminate();
  }
}
}  // namespace test

int main(int argc, char* argv[]) {
  using namespace sycl;
  namespace rng = std::ranges;

  std::cout << std::format("Device(CPU): {}\nNum of cores: {}\n", test::q.get_device().get_info<info::device::name>(),
                           test::q.get_device().get_info<info::device::max_compute_units>());

  test::n_threads = test::arena.max_concurrency();
  std::cout << std::format("Num of workers: {}\n", test::n_threads);
  size_t n = 2000, len = 100'000;
  std::cout << "\nPass (default 2000): ", std::cin >> n;
  std::cout << "\nSize (default 100000): ", std::cin >> len;

  std::vector<double> pdq_times, std_times;
  for (size_t i = 0; i < n; ++i) {
    std::cout << std::format("{}/{}\n", i, n);
    auto v = util::generate_vec(len);
    auto tmp = v;
    {
      auto const t0 = tbb::tick_count::now();
      rng::sort(tmp);
      auto const t1 = tbb::tick_count::now();
      std::cout << std::format("std::sort: {} sec\n", (t1 - t0).seconds());
      std_times.emplace_back((t1 - t0).seconds());
    }
    double dt = test::parallel_pdqsort_demo(v);
    std::cout << std::format("pdqsort: {} sec\n", dt);
    pdq_times.emplace_back(dt);
    if (i == n - 1) {
      std::cout << std::format("Sorted? {}\n", rng::equal(v, tmp) ? "Yes!" : "No??");
    }
  }
  auto r1 = util::stat(pdq_times);
  std::cout << std::format("\npdqsort:\nmin: {}\navg: {}\nmax: {}", r1.min, r1.avg, r1.max);
  auto r2 = util::stat(std_times);
  std::cout << std::format("\nstd::sort:\nmin: {}\navg: {}\nmax: {}", r2.min, r2.avg, r2.max);

  return 0;
}

#endif