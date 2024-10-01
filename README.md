# CFOR

### cfor is a C extension for python that allows for faster numerical operations. by wrapping the C for loop and optionally using OpenMP for parallelization and SIMD for vectorization.


### Benchmarks
```console
----------Numeric Benchmarks on 10000000 elements---------
Python loop (numeric) time: 17.3579 seconds
C extension with MP and SIMD (numeric) time: 12.7961 seconds, Speedup: 1.36x
C extension with MP no SIMD (numeric) time: 13.0290 seconds, Speedup: 1.33x
C extension without MP (numeric) time: 12.8194 seconds, Speedup: 1.35x
Pandas apply with lambda (numeric) time: 67.7576 seconds, Speedup: 0.26x
----------String Benchmarks on 10000000 elements----------
Python loop (string) time: 16.9410 seconds
C extension with MP (string) time: 18.3583 seconds, Speedup: 0.92x
C extension without MP (string) time: 19.3313 seconds, Speedup: 0.88x
Pandas apply with lambda (string) time: 37.4595 seconds, Speedup: 0.45x
```