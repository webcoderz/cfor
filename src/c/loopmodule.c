#include <Python.h>

#include <immintrin.h>  // For AVX, AVX2, and SSE intrinsics
#include <omp.h>
#include <string.h>
#include <cpuid.h>  // For detecting CPU SIMD capabilities

// Condition types
#define GREATER_THAN 0
#define LESS_THAN 1
#define EQUAL_TO 2
#define NO_CONDITION -1  // No condition provided
#define GREATER_THAN_EQUAL 3
#define LESS_THAN_EQUAL 4

// Function to check for AVX2 support
int supports_avx2() {
    unsigned int eax, ebx, ecx, edx;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);  // Use CPUID to check for AVX2
    return (ebx & (1 << 5)) != 0;  // Check bit 5 in EBX (AVX2 feature flag)
}

// Function to check for AVX support
int supports_avx() {
    unsigned int eax, ebx, ecx, edx;
    __cpuid(1, eax, ebx, ecx, edx);  // Use CPUID to check for AVX
    return (ecx & (1 << 28)) != 0;  // Check bit 28 in ECX (AVX feature flag)
}

// Function to check for SSE2 support
int supports_sse2() {
    unsigned int eax, ebx, ecx, edx;
    __cpuid(1, eax, ebx, ecx, edx);  // Use CPUID to check for SSE2
    return (edx & (1 << 26)) != 0;  // Check bit 26 in EDX (SSE2 feature flag)
}

// SIMD processing for numeric data (floats) with optional SIMD and OpenMP
void process_numeric_simd(float* data, Py_ssize_t length, float threshold, int condition_type, int use_mp, int use_simd) {
    if (use_simd && supports_avx2()) {
        __m256 threshold_vec = _mm256_set1_ps(threshold);  // Set threshold as AVX vector

        #pragma omp parallel for if (use_mp)  // Parallelize with OpenMP if enabled
        for (Py_ssize_t i = 0; i < length; i += 8) {  // Process 8 floats at a time
            __m256 data_vec = _mm256_loadu_ps(&data[i]);  // Load 8 floats into SIMD register
            __m256 mask;

            if (condition_type != NO_CONDITION) {
                switch (condition_type) {
                    case GREATER_THAN:
                        mask = _mm256_cmp_ps(data_vec, threshold_vec, _CMP_GT_OQ);  // Compare > threshold
                        break;
                    case LESS_THAN:
                        mask = _mm256_cmp_ps(data_vec, threshold_vec, _CMP_LT_OQ);  // Compare < threshold
                        break;
                    case EQUAL_TO:
                        mask = _mm256_cmp_ps(data_vec, threshold_vec, _CMP_EQ_OQ);  // Compare == threshold
                        break;
                    case LESS_THAN_EQUAL:
                        mask = _mm256_cmp_ps(data_vec, threshold_vec, _CMP_LE_OQ); // Compare <= threshold
                        break;
                    case GREATER_THAN_EQUAL:
                        mask = _mm256_cmp_ps(data_vec, threshold_vec, _CMP_GE_OQ); //Compare >= threshold
                        break;
                }
                __m256 result_vec = _mm256_blendv_ps(data_vec, _mm256_set1_ps(0.0), mask);  // Set values to 0 where condition is true
                _mm256_storeu_ps(&data[i], result_vec);  // Store 8 floats back into memory
            }
        }

        // Handle remaining elements (if length is not divisible by 8)
        for (Py_ssize_t i = (length / 8) * 8; i < length; i++) {
            if (condition_type == GREATER_THAN && data[i] > threshold) {
                data[i] = 0.0;
            } else if (condition_type == LESS_THAN && data[i] < threshold) {
                data[i] = 0.0;
            } else if (condition_type == EQUAL_TO && data[i] == threshold) {
                data[i] = 0.0;
            } else if (condition_type == GREATER_THAN_EQUAL && data[i] >= threshold) {
                data[i] = 0.0;
            } else if (condition_type == LESS_THAN_EQUAL && data[i] <= threshold) {
                data[i] = 0.0;
            } 
        }
    } else if (use_simd && supports_avx()) {
        __m128 threshold_vec = _mm_set1_ps(threshold);  // Set threshold as AVX vector (128-bit)

        #pragma omp parallel for if (use_mp)  // Parallelize with OpenMP if enabled
        for (Py_ssize_t i = 0; i < length; i += 4) {  // Process 4 floats at a time
            __m128 data_vec = _mm_loadu_ps(&data[i]);  // Load 4 floats into SIMD register
            __m128 mask;

            if (condition_type != NO_CONDITION) {
                switch (condition_type) {
                    case GREATER_THAN:
                        mask = _mm_cmp_ps(data_vec, threshold_vec, _CMP_GT_OQ);  // Compare > threshold
                        break;
                    case LESS_THAN:
                        mask = _mm_cmp_ps(data_vec, threshold_vec, _CMP_LT_OQ);  // Compare < threshold
                        break;
                    case EQUAL_TO:
                        mask = _mm_cmp_ps(data_vec, threshold_vec, _CMP_EQ_OQ);  // Compare == threshold
                        break;
                    case LESS_THAN_EQUAL:
                        mask = _mm_cmp_ps(data_vec, threshold_vec, _CMP_LE_OQ); // Compare <= threshold
                        break;
                    case GREATER_THAN_EQUAL:
                        mask = _mm_cmp_ps(data_vec, threshold_vec, _CMP_GE_OQ); //Compare >= threshold
                        break;
                }
                __m128 result_vec = _mm_blendv_ps(data_vec, _mm_set1_ps(0.0), mask);  // Set values to 0 where condition is true
                _mm_storeu_ps(&data[i], result_vec);  // Store 4 floats back into memory
            }
        }

        // Handle remaining elements (if length is not divisible by 4)
        for (Py_ssize_t i = (length / 4) * 4; i < length; i++) {
            if (condition_type == GREATER_THAN && data[i] > threshold) {
                data[i] = 0.0;
            } else if (condition_type == LESS_THAN && data[i] < threshold) {
                data[i] = 0.0;
            } else if (condition_type == EQUAL_TO && data[i] == threshold) {
                data[i] = 0.0;
            } else if (condition_type == GREATER_THAN_EQUAL && data[i] >= threshold) {
                data[i] = 0.0;
            } else if (condition_type == LESS_THAN_EQUAL && data[i] <= threshold) {
                data[i] = 0.0;
            }
        }
    } else if (use_simd && supports_sse2()) {
        __m128 threshold_vec = _mm_set1_ps(threshold);  // Set threshold as SSE2 vector (128-bit)

        #pragma omp parallel for if (use_mp)  // Parallelize with OpenMP if enabled
        for (Py_ssize_t i = 0; i < length; i += 4) {  // Process 4 floats at a time
            __m128 data_vec = _mm_loadu_ps(&data[i]);  // Load 4 floats into SIMD register
            __m128 mask;

            if (condition_type != NO_CONDITION) {
                switch (condition_type) {
                    case GREATER_THAN:
                        mask = _mm_cmpgt_ps(data_vec, threshold_vec);  // Compare > threshold
                        break;
                    case LESS_THAN:
                        mask = _mm_cmplt_ps(data_vec, threshold_vec);  // Compare < threshold
                        break;
                    case EQUAL_TO:
                        mask = _mm_cmpeq_ps(data_vec, threshold_vec);  // Compare == threshold
                        break;
                    case LESS_THAN_EQUAL:
                        mask = _mm_cmple_ps(data_vec, threshold_vec);  // Compare <= threshold
                        break;
                    case GREATER_THAN_EQUAL:
                        mask = _mm_cmpge_ps(data_vec, threshold_vec);  // Compare >= threshold
                        break;
                }
                __m128 result_vec = _mm_blendv_ps(data_vec, _mm_set1_ps(0.0), mask);  // Set values to 0 where condition is true
                _mm_storeu_ps(&data[i], result_vec);  // Store 4 floats back into memory
            }
        }

        // Handle remaining elements (if length is not divisible by 4)
        for (Py_ssize_t i = (length / 4) * 4; i < length; i++) {
            if (condition_type == GREATER_THAN && data[i] > threshold) {
                data[i] = 0.0;
            } else if (condition_type == LESS_THAN && data[i] < threshold) {
                data[i] = 0.0;
            } else if (condition_type == EQUAL_TO && data[i] == threshold) {
                data[i] = 0.0;
            } else if (condition_type == GREATER_THAN_EQUAL && data[i] >= threshold) {
                data[i] = 0.0;
            } else if (condition_type == LESS_THAN_EQUAL && data[i] <= threshold) {
                data[i] = 0.0;
            }
        }
    } else {
        // Scalar fallback (no SIMD or SIMD disabled)
        #pragma omp parallel for if (use_mp)  // Parallelize with OpenMP if enabled
        for (Py_ssize_t i = 0; i < length; i++) {
            if (condition_type == GREATER_THAN && data[i] > threshold) {
                data[i] = 0.0;
            } else if (condition_type == LESS_THAN && data[i] < threshold) {
                data[i] = 0.0;
            } else if (condition_type == EQUAL_TO && data[i] == threshold) {
                data[i] = 0.0;
            } else if (condition_type == GREATER_THAN_EQUAL && data[i] >= threshold) {
                data[i] = 0.0;
            } else if (condition_type == LESS_THAN_EQUAL && data[i] <= threshold) {
                data[i] = 0.0;
            }
        }
    }
}

// Python wrapper for processing data with SIMD or scalar fallback
static PyObject* c4(PyObject* self, PyObject* args) {
    PyObject* input_list;
    PyObject* threshold_py;
    int condition_type;
    int use_mp;
    int use_simd;  // Flag to control whether SIMD should be used

    if (!PyArg_ParseTuple(args, "OOiii", &input_list, &threshold_py, &condition_type, &use_mp, &use_simd)) {
        return PyErr_Format(PyExc_ValueError, "Expected a list, threshold (float or str), a condition type, use_mp flag, and use_simd flag");
    }

    Py_ssize_t length = PyList_Size(input_list);
    if (length < 1) {
        return PyErr_Format(PyExc_ValueError, "Input list must have at least one element");
    }

    // Detect if the input is numeric or string
    if (PyFloat_Check(PyList_GetItem(input_list, 0)) && PyFloat_Check(threshold_py)) {
        // Numeric data
        float* data = (float*)malloc(length * sizeof(float));
        if (!data) {
            return PyErr_NoMemory();
        }

        for (Py_ssize_t i = 0; i < length; i++) {
            PyObject* item = PyList_GetItem(input_list, i);
            data[i] = (float)PyFloat_AsDouble(item);
        }

        float threshold = (float)PyFloat_AsDouble(threshold_py);

        // Process the data with SIMD or scalar fallback
        process_numeric_simd(data, length, threshold, condition_type, use_mp, use_simd);

        // Convert the processed C array back to a Python list
        PyObject* transformed_list = PyList_New(length);
        if (!transformed_list) {
            free(data);
            return PyErr_NoMemory();
        }

        for (Py_ssize_t i = 0; i < length; i++) {
            PyList_SetItem(transformed_list, i, PyFloat_FromDouble(data[i]));
        }

        free(data);
        return transformed_list;

    } else if (PyUnicode_Check(PyList_GetItem(input_list, 0)) && PyUnicode_Check(threshold_py)) {
        // String data
        const char** data = (const char**)malloc(length * sizeof(char*));
        if (!data) {
            return PyErr_NoMemory();
        }

        for (Py_ssize_t i = 0; i < length; i++) {
            PyObject* item = PyList_GetItem(input_list, i);
            data[i] = PyUnicode_AsUTF8(item);
        }

        const char* threshold = PyUnicode_AsUTF8(threshold_py);

        // Process the string data (scalar since SIMD is not applicable to strings)
        #pragma omp parallel for if (use_mp)  // Use OpenMP for string data
        for (Py_ssize_t i = 0; i < length; i++) {
            if (condition_type == GREATER_THAN && strcmp(data[i], threshold) > 0) {
                data[i] = "";
            } else if (condition_type == LESS_THAN && strcmp(data[i], threshold) < 0) {
                data[i] = "";
            } else if (condition_type == EQUAL_TO && strcmp(data[i], threshold) == 0) {
                data[i] = "";
            } else if (condition_type == GREATER_THAN_EQUAL && strcmp(data[i], threshold) >= 0) {
                data[i] = "";
            } else if (condition_type == LESS_THAN_EQUAL && strcmp(data[i], threshold) <= 0) {
                data[i] = "";
            }
        }

        // Convert the processed C array back to a Python list
        PyObject* transformed_list = PyList_New(length);
        if (!transformed_list) {
            free(data);
            return PyErr_NoMemory();
        }

        for (Py_ssize_t i = 0; i < length; i++) {
            PyList_SetItem(transformed_list, i, PyUnicode_FromString(data[i]));
        }

        free(data);
        return transformed_list;
    }

    return PyErr_Format(PyExc_ValueError, "Data and threshold types must match and be either all floats or all strings");
}

// Method definitions
static PyMethodDef LoopMethods[] = {
    {"c4", c4, METH_VARARGS, "Apply a for loop with optional condition, optional SIMD optimization, and optional OpenMP"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef loopmodule = {
    PyModuleDef_HEAD_INIT,
    "loopmodule",
    NULL, -1, LoopMethods
};

// Module initialization
PyMODINIT_FUNC PyInit_cfor(void) {
    return PyModule_Create(&loopmodule);
}
