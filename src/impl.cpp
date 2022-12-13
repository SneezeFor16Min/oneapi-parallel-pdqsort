#ifndef _PDQ_IMPL
#define _PDQ_IMPL

#include <CL/sycl.hpp>

#include "tbb/tbb.h"
#undef min
#undef max

namespace impl
{
using namespace sycl;
namespace rng = std::ranges;

/**
 * @brief Shifts \c last to left within [\c first, \c last] until it meets a smaller/equal element.
 * @pre \c first < \c last.
 * @remark A more modernized version would be like this:
 * \code{.cpp}
 * It const iu = rng::upper_bound(first, last, *last);
 * rng::rotate(iu, last, last + 1);
 * \endcode
 * However, for short arrays this is slower than the low-level implementation.
 */
template <std::bidirectional_iterator It>
constexpr void _sort_left_shift(It first, It last) {
  It j1 = last, j2 = last;
  if (*j1 < *--j2) {
    std::iter_value_t<It> tmp = std::move(*j1);
    do {
      *j1 = std::move(*j2);
    } while (--j1 != first && tmp < *--j2);
    *j1 = std::move(tmp);
  }
}

/**
 * @brief Shifts \c first to right within [\c first, \c last] until it meets a greater/equal element.
 * @pre \c first < \c last.
 * @remark A more modernized version would be like this:
 * \code{.cpp}
 * It const il = rng::lower_bound(first + 1, last + 1, *first)
 * rng::rotate(first, first, il + 1);
 * \endcode
 * However, for short arrays this is slower.
 */
template <std::forward_iterator It>
constexpr void _sort_right_shift(It first, It last) {
  It j1 = first, j2 = first;
  if (*++j2 < *j1) {
    std::iter_value_t<It> tmp = std::move(*j1);
    do {
      *j1 = std::move(*j2);
    } while (++j1 != last && *++j2 < tmp);
    *j1 = std::move(tmp);
  }
}

/**
 * @brief Insertion sort on [\c begin, \c end) partially allowing limited shifts.
 * @tparam It Iterator.
 * @pre \c begin < \c end.
 */
template <std::bidirectional_iterator It>
constexpr bool _partial_ins_sort(It begin, It end) {
  uint8_t constexpr MAX_SHIFTS = 8;
  uint8_t limit = MAX_SHIFTS;
  It i = begin;
  for (++i; i != end; ++i) {
    It j1 = i, j2 = i;
    if (*j1 < *--j2) {
      std::iter_value_t<It> tmp{ std::move(*j1) };
      do {
        *j1 = std::move(*j2);
      } while (--limit && --j1 != begin && tmp < *--j2);
      *j1 = std::move(tmp);
    }
    if (limit == 0)
      return false;
  }
  return true;
}

template <rng::forward_range R>
  requires rng::common_range<R>
constexpr void ins_sort(R& v) {
  using It = rng::iterator_t<R>;
  It const ib = rng::begin(v), ie = rng::end(v);
  if (ib != ie) [[likely]]
    for (It i = ib + 1; i != ie; ++i)
      _sort_left_shift(ib, i);
}

template <rng::random_access_range R>
  requires rng::common_range<R>
constexpr void heap_sort(R& v) {
  rng::make_heap(v);
  rng::sort_heap(v);
}

/**
 * @brief Parallel pattern-defeating quick-sort on Intel oneAPI TBB.
 * @tparam It Iterator type
 * @param [in,out] tasks A TBB task group.
 * @param [in,out] v A range of data.
 * @param [in] limit Number of imbalanced partitions before switching to \c heap_sort.
 * @param [in] pred Predecessor pivot.
 * @param [in] balanced \c true iff the last partition was reasonably balanced.
 * @param [in] partitioned \c true iff the last partition didn't shuffle elements (i.e., the slice was already
 * partitioned).
 */
template <std::random_access_iterator It>
void _tbb_pdqsort(tbb::task_group& tasks,
                  rng::subrange<It>&& v,
                  uint8_t limit,
                  std::optional<std::iter_value_t<It>> pred = {},
                  bool balanced = true,
                  bool partitioned = true) {
  for (;;) {
    size_t const len = rng::size(v);
    size_t constexpr INS_SORT_LEN = 16;
    if (len <= INS_SORT_LEN) {
      ins_sort(v);
      break;
    }

    if (limit == 0) {
      heap_sort(v);
      break;
    }

    It ib = rng::begin(v), ie = rng::end(v);

    size_t const l4 = len / 4;
    It ip = ib + l4 * 2;
    if (!balanced) {
      /// Shuffle elements around the possible pivot, hopefully breaking patterns.
      auto break_patterns = [len, ip, ib] {
        auto gen_usize = [](uint32_t seed) constexpr -> size_t {
          auto gen_u32 = [seed]() mutable {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            return seed;
          };
          if constexpr (sizeof(size_t) <= 4) {
            return gen_u32();
          } else {
            return ((uint64_t)gen_u32() << 32) | (uint64_t)gen_u32();
          }
        };

        size_t seed = len;
        size_t const modulus = std::bit_ceil(len);
        for (int i = -1; i <= 1; ++i) {
          seed = gen_usize(seed);
          size_t other = seed & (modulus - 1);
          other = other >= len ? other - len : other;
          std::iter_swap(ip + i, ib + other);
        }
      };
      break_patterns();
      --limit;
    }

    auto maybe_sorted = true;  // optimistic
    /// Choose pivot using median-of-medians, and return whether the slice may have been sorted.
    auto choose_pivot = [len, l4, ib, ie, &maybe_sorted, &ip] {
      size_t constexpr SHORTEST_MEDIAN_OF_MEDIANS = 50;
      size_t constexpr MAX_SWAPS = 4 * 3;
      size_t n_swap = 0;
      It const ip1 = ip - l4, ip3 = ip + l4;
      {
        auto sort2 = [&n_swap](It a, It b) {
          if (*b < *a) {
            std::iter_swap(a, b);
            ++n_swap;
          }
        };
        auto sort3 = [&sort2](It a, It b, It c) { sort2(a, b), sort2(b, c), sort2(a, b); };
        if (len >= SHORTEST_MEDIAN_OF_MEDIANS) {
          auto sort_around = [&sort3](It a) { sort3(a - 1, a, a + 1); };
          sort_around(ip1), sort_around(ip), sort_around(ip3);
        }
        sort3(ip1, ip, ip3);
      }

      if (n_swap < MAX_SWAPS) {
        maybe_sorted = (n_swap == 0);
      } else {
        std::reverse(ib, ie);
        // maybe_sorted = true;
        ip = ie - 1 - l4 * 2;
      }
    };
    choose_pivot();
    if (balanced && partitioned && maybe_sorted) {
      if (_partial_ins_sort(ib, ip) && _partial_ins_sort(ip + 1, ie)) {
        break;  // `v` is sorted!
      }
    }

    // If predecessor pivot == chosen pivot, then it's already the smallest element.
    if (pred.has_value() && pred.value() >= *ip) {
      /// Bipartite `v` into elements == and > the pivot.
      auto partition_equal = [ib, ie, ip]() mutable {
        std::iter_swap(ib, ip);
        ip = ib;
        It l = ++ib, r = ie;
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
      It mid = partition_equal();
      v = { mid, ie };
      continue;
    }

    /// Bipartite `v` into elements <= and > the pivot.
    auto partition = [ib, ie, ip]() mutable {
      std::iter_swap(ib, ip);
      ip = ib;
      It l = ++ib, r = ie;
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
    It const mid = partition();
    size_t const mid_pos = std::distance(ib, mid);
    balanced = std::min(mid_pos, len - mid_pos) >= l4 / 2;
    partitioned = true;  // TODO

    if (mid_pos >= len - mid_pos - 1) {
      tasks.run([=, &tasks] { _tbb_pdqsort(tasks, { ib, mid, mid_pos }, limit, pred, balanced, partitioned); });
      v = { mid + 1, ie }, pred = { *mid };
    } else {
      tasks.run([=, &tasks] { _tbb_pdqsort(tasks, { mid + 1, ie }, limit, { *mid }, balanced, partitioned); });
      v = { ib, mid, mid_pos };
    }
  }
}

template <rng::random_access_range R>
void parallel_pdqsort(R& v) {
  tbb::task_group tasks;
  tasks.run_and_wait([&] { _tbb_pdqsort(tasks, rng::subrange{ v }, std::bit_width(rng::size(v))); });
}
}  // namespace impl

#endif