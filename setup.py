from setuptools import setup, Extension

module = Extension(
    'cfor',  # The name of the extension
    sources=['src/c/loopmodule.c'],  # The source file
    extra_compile_args=['-fopenmp'],  # Enable OpenMP support for parallel processing
    extra_link_args=['-fopenmp'],
)

setup(
    name='cfor',
    version='1.0',
    description='A C extension for applying loops with conditional checks and optional multi-processing',
    ext_modules=[module]
)

#python3 setup.py build_ext --inplace