import pandas as pd
import timeit
from src.cfor import c4


elements = 100000
# Numeric data to operate on
numeric_data = [1.0, 2.0, 3.0, 4.0, 5.0] * elements
threshold_numeric = 3.0

# String data to operate on
string_data = ["apple", "banana", "cherry", "date", "elderberry"] * elements
threshold_string = "cherry"

# Convert numeric data into a pandas DataFrame
df_numeric = pd.DataFrame({'values': numeric_data})

# Convert string data into a pandas DataFrame
df_string = pd.DataFrame({'values': string_data})

# Define the regular Python loop function for numeric data
def python_loop_numeric(data, threshold):
    result = []
    for value in data:
        if value > threshold:
            result.append(0.0)
        else:
            result.append(value)
    return result

# Define the regular Python loop function for string data
def python_loop_string(data, threshold):
    result = []
    for value in data:
        if value == threshold:
            result.append("")
        else:
            result.append(value)
    return result

# Define a function to apply in pandas using a lambda for numeric data
def pandas_apply_lambda_numeric(df, threshold):
    return df['values'].apply(lambda x: 0 if x > threshold else x)

# Define a function to apply in pandas using a lambda for string data
def pandas_apply_lambda_string(df, threshold):
    return df['values'].apply(lambda x: "" if x == threshold else x)


# Time the C extension with numeric data and multi-processing (MP enabled)
def benchmark_c4_numeric_mp():
    return c4(numeric_data, threshold_numeric, condition=">", use_mp=True)

# Time the C extension with string data and multi-processing (MP enabled)
def benchmark_c4_string_mp():
    return c4(string_data, threshold_string, condition="==", use_mp=True)

def benchmark_c4_numeric_mp_no_simd():
    return c4(numeric_data, threshold_numeric, condition=">", use_mp=True, use_simd=False)

def benchmark_c4_numeric_no_mp():
    return c4(numeric_data, threshold_numeric, condition=">", use_mp=False)

# Time the C extension with string data and multi-processing (MP enabled)
def benchmark_c4_string_no_mp():
    return c4(string_data, threshold_string, condition="==", use_mp=False)

# Time the regular Python loop for numeric data
def benchmark_python_loop_numeric():
    return python_loop_numeric(numeric_data, threshold_numeric)

# Time the regular Python loop for string data
def benchmark_python_loop_string():
    return python_loop_string(string_data, threshold_string)

# Time the pandas apply with lambda for numeric data
def benchmark_pandas_apply_numeric():
    return pandas_apply_lambda_numeric(df_numeric, threshold_numeric)

# Time the pandas apply with lambda for string data
def benchmark_pandas_apply_string():
    return pandas_apply_lambda_string(df_string, threshold_string)


# Run the benchmarks
python_loop_numeric_time = timeit.timeit(benchmark_python_loop_numeric, number=10)
python_loop_string_time = timeit.timeit(benchmark_python_loop_string, number=10)
c4_numeric_mp_time = timeit.timeit(benchmark_c4_numeric_mp, number=10)
c4_numeric_mp_no_simd_time = timeit.timeit(benchmark_c4_numeric_mp_no_simd, number=10)
c4_string_mp_time = timeit.timeit(benchmark_c4_string_mp, number=10)
pandas_apply_numeric_time = timeit.timeit(benchmark_pandas_apply_numeric, number=10)
pandas_apply_string_time = timeit.timeit(benchmark_pandas_apply_string, number=10)
c4_numeric_no_mp_time = timeit.timeit(benchmark_c4_numeric_no_mp, number=10)
c4_string_no_mp_time = timeit.timeit(benchmark_c4_string_no_mp, number=10)

# Calculate speedups for numeric data
numeric_mp_speedup = python_loop_numeric_time / c4_numeric_mp_time
numeric_mp_no_simd_speedup = python_loop_numeric_time / c4_numeric_mp_no_simd_time
numeric_no_mp_speedup = python_loop_numeric_time / c4_numeric_no_mp_time
pandas_numeric_speedup = python_loop_numeric_time / pandas_apply_numeric_time

# Calculate speedups for string data
string_mp_speedup = python_loop_string_time / c4_string_mp_time
string_no_mp_speedup = python_loop_string_time / c4_string_no_mp_time
pandas_string_speedup = python_loop_string_time / pandas_apply_string_time

# Output the results for numeric comparisons
print(f"----------Numeric Benchmarks on {elements} elements---------")
print(f"Python loop (numeric) time: {python_loop_numeric_time:.4f} seconds")
print(f"C extension with MP and SIMD (numeric) time: {c4_numeric_mp_time:.4f} seconds, Speedup: {numeric_mp_speedup:.2f}x")
print(f"C extension with MP no SIMD (numeric) time: {c4_numeric_mp_no_simd_time:.4f} seconds, Speedup: {numeric_mp_no_simd_speedup:.2f}x")
print(f"C extension without MP and with SIMD (numeric) time: {c4_numeric_no_mp_time:.4f} seconds, Speedup: {numeric_no_mp_speedup:.2f}x")
print(f"Pandas apply with lambda (numeric) time: {pandas_apply_numeric_time:.4f} seconds, Speedup: {pandas_numeric_speedup:.2f}x")

# Output the results for string comparisons
print(f"----------String Benchmarks on {elements} elements----------")
print(f"Python loop (string) time: {python_loop_string_time:.4f} seconds")
print(f"C extension with MP (string) time: {c4_string_mp_time:.4f} seconds, Speedup: {string_mp_speedup:.2f}x")
print(f"C extension without MP (string) time: {c4_string_no_mp_time:.4f} seconds, Speedup: {string_no_mp_speedup:.2f}x")
print(f"Pandas apply with lambda (string) time: {pandas_apply_string_time:.4f} seconds, Speedup: {pandas_string_speedup:.2f}x")