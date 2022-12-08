#include <iostream>
#include <format>
#include <ranges>
#include <random>
#include <numeric>

auto static gen = std::mt19937{ std::random_device{}() };

namespace util
{
namespace rng = std::ranges;
namespace sv = rng::views;

template <rng::range R>
void print(R const& r) {
  // display only first n and last n
  size_t constexpr n = 10;
  size_t const len = rng::size(r);
  std::cout << "[ ";
  if (len > 2 * n) {
    for (auto const& e : r | sv::take(n)) { std::cout << e << ' '; }
    std::cout << std::format("..<{} items>.. ", len - 2 * n);
    for (auto const& e : r | sv::reverse | sv::take(n) | sv::reverse) { std::cout << e << ' '; }
  }
  else { for (auto const& e : r) { std::cout << e << ' '; } }
  std::cout << "]" << std::endl;
}

enum class GenMode { Random, Sorted, RevSorted };

template <class T = int32_t>
auto generate_vec(size_t const len, GenMode const mode = GenMode::Random) {
  std::vector<T> v(len);
  switch (mode) {
    case GenMode::Random: {
      auto dist = std::uniform_int_distribution<T>(1, len);
      rng::generate(v, [&] { return dist(gen); });
      break;
    }
    case GenMode::Sorted: {
      rng::iota(v, 1);
      break;
    }
    case GenMode::RevSorted: {
      rng::iota(sv::reverse(v), 1);
      break;
    }
  }
  return v;
}
}
