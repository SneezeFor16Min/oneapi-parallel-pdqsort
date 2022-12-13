> **Note:** This project emerged from the final assignment of *High Performance Computing* course and was out of pure personal interest. PRs are welcomed.

A parallelized implementation of [Pattern-Defeating Quicksort (Orson Peters, 2021)](https://github.com/orlp/pdqsort) using Intel® oneAPI toolkit with support of C++20 features including `std::ranges` and `concept`s. This C++ implementation is based on that of Rust's `std::slice::sort_unstable` and Boost's `sort::pdqsort`. The project is still in progress.

## Performance

Test on Intel i5-12600K @ 3.70 Ghz (16 threads), Windows 11 22H2. Compiled with Intel® oneAPI DPC++ Compiler 2022.2.0.1, `/O2 /std:c++latest`.

||[`impl::parallel_pdqsort`](https://github.com/SneezeFor16Min/oneapi-parallel-pdqsort/blob/e0aeed3cbd3472c67708688a4e0e0166fb31b055/src/impl.cpp#L260)|`std::sort`|
|-|-:|-:|
|random_64|5.86μs±45.9%|**1.46**μs±**27.0%**|
|random_256|12.11μs±22.7%|**6.98**μs±**11.9%**|
|random_1024|36.29μs±**25.6%**|**32.4**μs±39.0%|
|random_4096|**82.34**μs±**22.9%**|160.09μs±64.6%|
|random_65536|**1.10**ms±12.6%|3.26ms±**1.5%**|

