#include <format>
#include <iostream>
#include <random>
#include <ranges>
#include <vector>
#include <CL/sycl.hpp>

#include "./util.cpp"
#include "dpc_common.hpp"
#include "tbb/tbb.h"
#undef min
#undef max

namespace impl
{
using namespace sycl;
using namespace tbb::detail;
namespace rng = std::ranges;

auto constexpr& _ = std::ignore;

/**
 * \brief Shift `end` to left within [begin, end] until it meets a smaller/equal
 * element.
 * \tparam It Iterator.
 */
template <std::forward_iterator It>
constexpr void _sort_left_shift(It begin, It end) {
  It j = end;
  while (j != begin && *end < *(j - 1))
    --j;
  if (j == end)
    return;
  std::iter_value_t<It> tmp{ std::move(*end) };
  for (It k = end; k != j; --k)
    *k = std::move(*(k - 1));
  *j = std::move(tmp);

  /*
  A more modernized version would be like this:
  ```
  It const iu = rng::upper_bound(begin, end, *end);
  _ = rng::rotate(iu, end, end + 1);
  ```
  However for short arrays this is slower than iterator-based implementation above.
  */
}

/**
 * \brief Shift `begin` to right within [begin, end] until it meets a
 * greater/equal element.
 * \tparam It Iterator.
 */
template <std::forward_iterator It>
constexpr void _sort_right_shift(It begin, It end) {
  It j = begin;
  while (j != end && *(j + 1) < *begin)
    ++j;
  if (j == begin)
    return;
  std::iter_value_t<It> tmp{ std::move(*begin) };
  for (It k = begin; k != j; ++k)
    *k = std::move(*(k + 1));
  *j = std::move(tmp);
  /*
  A more modernized version would be like this:
  ```
  It const il = rng::lower_bound(begin + 1, end + 1, *end);
  _ = rng::rotate(begin, begin, il + 1);
  ```
  However for short arrays this is slower than iterator-based implementation above.
  */
}

template <rng::forward_range R>
constexpr void ins_sort(R& v) {
  using _It = rng::iterator_t<R>;
  _It const ib = rng::begin(v), ie = rng::end(v);
  if (ib != ie)
    for (_It i = ib + 1; i != ie; ++i)
      _sort_left_shift(ib, i);
}

template <rng::random_access_range R>
constexpr void heap_sort(R& v) {
  rng::make_heap(v);
  rng::sort_heap(v);
}

int static n_threads = -1;

/**
 * \brief PDQ-sort on oneAPI TBB.
 * \tparam T - Data type
 * \tparam R - Container type
 * \tparam _It - Iterator type
 * \param v - A range of data.
 * \param limit - Number of imbalanced partitions before switching to `heap_sort`.
 * \param pred - Predecessor pivot.
 * \param balanced - `true` if the last partition was reasonably balanced.
 * \param partitioned - `true` if the last partition didn't shuffle elements (the slice was already partitioned).
 */
template <std::random_access_iterator _It, class T = std::iter_value_t<_It>>
void _tbb_pdqsort(rng::subrange<_It>&& v,
                  uint8_t limit,
                  std::optional<T> pred = {},
                  bool balanced = true,
                  bool partitioned = true) {
  tbb::task_group tasks;
  for (;;) {
    size_t const len = rng::size(v);
    size_t constexpr INS_SORT_LEN = 20;
    if (len <= INS_SORT_LEN) {
      ins_sort(v);
      break;
    }

    if (limit == 0) {
      heap_sort(v);
      break;
    }

    _It ib = rng::begin(v), ie = rng::end(v);
    size_t const l4 = len / 4;
    _It ip = ib + l4 * 2;
    if (!balanced) {
      /// Shuffle elements around the possible pivot, hopefully breaking patterns.
      auto break_patterns = [len, ip, ib] {
        if (len < 8)
          [[unlikely]] return;
        auto static dist = std::uniform_int_distribution<size_t>(0, std::numeric_limits<size_t>::max());

        for (auto i = -1; i <= 1; ++i) {
          size_t const other = dist(gen) % len;
          std::iter_swap(ip + i, ib + other);
        }
      };
      break_patterns();
      --limit;
    }

    auto maybe_sorted = true;
    /// Choose pivot using median-of-medians, and return whether the slice may
    /// have been sorted.
    auto choose_pivot = [len, l4, ib, ie, &maybe_sorted, &ip] {
      size_t constexpr SHORTEST_MEDIAN_OF_MEDIANS = 50;
      size_t constexpr MAX_SWAPS = 4 * 3;
      size_t n_swap = 0;
      _It const ip1 = ip - l4, ip3 = ip + l4;
      if (len >= 8) {
        auto sort2 = [&n_swap](_It a, _It b) {
          if (*b < *a) {
            std::iter_swap(a, b);
            ++n_swap;
          }
        };
        auto sort3 = [&sort2](_It a, _It b, _It c) { sort2(a, b), sort2(b, c), sort2(a, b); };
        if (len >= SHORTEST_MEDIAN_OF_MEDIANS) {
          auto sort_around = [&sort3](_It a) { sort3(a - 1, a, a + 1); };
          sort_around(ip1), sort_around(ip), sort_around(ip3);
        }
        sort3(ip1, ip, ip3);
      }

      if (n_swap < MAX_SWAPS) { maybe_sorted = (n_swap == 0); }
      else {
        std::reverse(ib, ie);
        ip = ie - 1 - l4 * 2;
      }
    };
    choose_pivot();
    if (balanced && partitioned && maybe_sorted) {
      /// Try identifying out-of-order elements and shifting them to correct
      /// positions.
      auto partial_ins_sort = [len, ib, ie]() -> bool /*wholly_sorted*/ {
        // max iterations of sort
        uint8_t constexpr MAX_STEPS = 5;
        // shortest len for sort
        size_t constexpr SHORTEST_SHIFTING_LEN = 50;
        _It it = ib + 1;
        for (uint8_t i = 0; i < MAX_STEPS; ++i) {
          // count adjacent out-of-order pairs
          while (it != ie && *(it - 1) <= *it)
            ++it;
          // reach end, which means all sorted
          if (it == ie)
            return true;
          // don't sort if too short
          if (len < SHORTEST_SHIFTING_LEN)
            return false;

          // shift `it-1`,`it`,`it+1`
          std::iter_swap(it - 1, it);
          _sort_left_shift(ib, it - 1);
          _sort_right_shift(it + 1, ie - 1);
        }
        return false;
      };
      if (partial_ins_sort()) {
        break;  // `v` is sorted!
      }
    }

    // If predecessor pivot == chosen pivot, then it's already the
    // smallest element.
    if (pred.has_value() && pred.value() >= *ip) {
      /// Bipartite `v` into elements == and > the pivot.
      auto partition_equal = [ib, ie, ip]() mutable {
        std::iter_swap(ib, ip);
        ip = ib;
        _It l = ++ib, r = ie;
        for (;;) {
          while (l != r && *l <= *ip)
            ++l;
          while (l != r && *ip < *(r - 1))
            --r;
          if (l == r)
            return --l;
          std::iter_swap(l++, --r);
        }
      };
      _It mid = partition_equal();
      v = { mid, ie };
      continue;
    }

    /// Bipartite `v` into elements <= and > the pivot.
    auto partition = [ib, ie, ip]() mutable {
      std::iter_swap(ib, ip);
      ip = ib;
      _It l = ++ib, r = ie;
      // TODO: implement block partitioning
      for (;;) {
        while (l != r && *l <= *ip)
          ++l;
        while (l != r && *ip < *(r - 1))
          --r;
        if (l == r)
          break;
        std::iter_swap(l++, --r);
      }
      // swap back
      std::iter_swap(ip, --l);
      return l;
    };
    _It const mid = partition();
    size_t const mid_pos = std::distance(ib, mid);
    balanced = std::min(mid_pos, len - mid_pos) >= len / 8;
    partitioned = true;  // TODO

    if (mid_pos < len - mid_pos - 1) {
      tasks.run([=] { _tbb_pdqsort<_It, T>({ ib, mid, mid_pos }, limit, pred, balanced, partitioned); });
      v = { mid + 1, ie }, pred = { *mid };
    }
    else {
      tasks.run([=] { _tbb_pdqsort<_It, T>({ mid + 1, ie }, limit, { *mid }, balanced, partitioned); });
      v = { ib, mid, mid_pos };
    }
  }
  tasks.wait();
}

template <rng::range R>
void parallel_pdqsort(R& v) {
  _tbb_pdqsort<rng::iterator_t<R>, rng::range_value_t<R>>(rng::subrange{ v }, d0::log2(rng::size(v)) + 1U);
}

template <rng::range R>
void parallel_pdqsort_demo(R& arr) {
  using T = rng::range_value_t<R>;
  size_t const len = rng::size(arr);
  std::cout << std::format("Array size: {}", len) << std::endl;

  try {
    queue q(cpu_selector{}, dpc_common::exception_handler);
    std::cout << std::format("Device(CPU): {}\nNum of cores: {}", q.get_device().get_info<info::device::name>(),
                             q.get_device().get_info<info::device::max_compute_units>()) << std::endl;

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
      parallel_pdqsort(usm_arr);
      auto const t1 = tbb::tick_count::now();
      std::cout << "Complete\n";
      std::cout << std::format("Time usage for pdqsort: {} sec\n", (t1 - t0).seconds());
    });
    std::cout << std::format("Checking if array is the same as std::sort result... {}\n",
                             rng::equal(usm_arr, sorted) ? "Yes!" : "No?");

    arr.assign_range(usm_arr);
  }
  catch (exception const& e) {
    std::cerr << "An error happened: " << e.what() << std::endl;
    std::terminate();
  }
}
}  // namespace impl

int main(int argc, char* argv[]) {
  size_t len = 0;
  while (std::cout << "Enter data size: ", std::cin >> len) {
    auto v = util::generate_vec(len, util::GenMode::Random);
    util::print(v);
    impl::parallel_pdqsort_demo(v);
    util::print(v);
    std::cout << "====================" << std::endl;
  }

  return 0;
}
