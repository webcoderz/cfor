#include <Python.h>
#include <string.h>
#include <omp.h>

// Condition types
#define GREATER_THAN 0
#define LESS_THAN 1
#define EQUAL_TO 2
#define NO_CONDITION -1  // Special value to indicate no condition

static PyObject* c4(PyObject* self, PyObject* args) {
    PyObject* input_list;
    PyObject* threshold_py;
    int condition_type;
    int use_mp;

    // Parse arguments: input list, threshold, condition type, use_mp flag
    if (!PyArg_ParseTuple(args, "OOii", &input_list, &threshold_py, &condition_type, &use_mp)) {
        return PyErr_Format(PyExc_ValueError, "Expected a list, threshold (float or str), a condition type, and an int use_mp flag");
    }

    Py_ssize_t length = PyList_Size(input_list);
    if (length < 1) {
        return PyErr_Format(PyExc_ValueError, "Input list must have at least one element");
    }

    // Check whether input list contains floats or strings
    int is_float = PyFloat_Check(PyList_GetItem(input_list, 0));
    int is_string = PyUnicode_Check(PyList_GetItem(input_list, 0));

    // Check whether the threshold is a float or a string
    int is_float_threshold = PyFloat_Check(threshold_py);
    int is_string_threshold = PyUnicode_Check(threshold_py);

    // Ensure that both input data and threshold are of the same type (either all floats or all strings)
    if ((is_float && !is_float_threshold) || (is_string && !is_string_threshold)) {
        return PyErr_Format(PyExc_ValueError, "Data and threshold types must match and be either all floats or all strings");
    }

    // Handle floats
    if (is_float && is_float_threshold) {
        float threshold = (float)PyFloat_AsDouble(threshold_py);

        // Convert Python list to C array of floats
        float* data = (float*)malloc(length * sizeof(float));
        if (!data) {
            return PyErr_NoMemory();
        }

        for (Py_ssize_t i = 0; i < length; i++) {
            PyObject* item = PyList_GetItem(input_list, i);
            if (!PyFloat_Check(item)) {
                free(data);
                return PyErr_Format(PyExc_ValueError, "All elements in the list must be floats");
            }
            data[i] = (float)PyFloat_AsDouble(item);
        }

        int error_flag = 0;

        // Process the data in parallel using OpenMP
        #pragma omp parallel for if (use_mp) shared(error_flag)
        for (Py_ssize_t i = 0; i < length; i++) {
            if (error_flag) continue;

            int condition_met = 1;  // Default to true for NO_CONDITION
            if (condition_type != NO_CONDITION) {
                switch (condition_type) {
                    case GREATER_THAN:
                        condition_met = data[i] > threshold;
                        break;
                    case LESS_THAN:
                        condition_met = data[i] < threshold;
                        break;
                    case EQUAL_TO:
                        condition_met = data[i] == threshold;
                        break;
                    default:
                        error_flag = 1;
                        continue;
                }
            }

            if (condition_met) {
                data[i] = 0.0;  // Example action: set the value to zero
            }
        }

        if (error_flag) {
            free(data);
            return PyErr_Format(PyExc_ValueError, "Invalid condition type");
        }

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

    // Handle strings
    } else if (is_string && is_string_threshold) {
        const char* threshold = PyUnicode_AsUTF8(threshold_py);

        // Preprocess the strings (convert Python strings to C strings)
        const char** data = (const char**)malloc(length * sizeof(const char*));
        if (!data) {
            return PyErr_NoMemory();
        }

        for (Py_ssize_t i = 0; i < length; i++) {
            PyObject* item = PyList_GetItem(input_list, i);
            if (!PyUnicode_Check(item)) {
                free(data);
                return PyErr_Format(PyExc_ValueError, "All elements in the list must be strings");
            }
            data[i] = PyUnicode_AsUTF8(item);  // Avoid strdup, directly reference
        }

        int error_flag = 0;

        // Process the data in parallel using OpenMP
        #pragma omp parallel for if (use_mp) shared(error_flag)
        for (Py_ssize_t i = 0; i < length; i++) {
            if (error_flag) continue;

            int condition_met = 1;  // Default to true for NO_CONDITION
            if (condition_type != NO_CONDITION) {
                switch (condition_type) {
                    case GREATER_THAN:
                        condition_met = strcmp(data[i], threshold) > 0;
                        break;
                    case LESS_THAN:
                        condition_met = strcmp(data[i], threshold) < 0;
                        break;
                    case EQUAL_TO:
                        condition_met = strcmp(data[i], threshold) == 0;
                        break;
                    default:
                        error_flag = 1;
                        continue;
                }
            }

            // No need for `strdup` and `free`, just update conditionally
            if (condition_met) {
                data[i] = "";  // Set to empty string directly
            }
        }

        if (error_flag) {
            free(data);
            return PyErr_Format(PyExc_ValueError, "Invalid condition type");
        }

        // Create the Python list from the modified C strings
        PyObject* transformed_list = PyList_New(length);
        if (!transformed_list) {
            free(data);
            return PyErr_NoMemory();
        }

        for (Py_ssize_t i = 0; i < length; i++) {
            PyObject* new_string = PyUnicode_FromString(data[i]);
            PyList_SetItem(transformed_list, i, new_string);  // Transfer ownership to the Python list
        }

        free(data);
        return transformed_list;
    }

    return PyErr_Format(PyExc_ValueError, "Data and threshold types must match and be either all floats or all strings");
}

// Method definitions
static PyMethodDef LoopMethods[] = {
    {"c4", c4, METH_VARARGS, "Apply a for loop with an optional condition and optional multi-processing for floats or strings"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef loopmodule = {
    PyModuleDef_HEAD_INIT,
    "loopmodule",
    NULL,  // No module documentation
    -1,
    LoopMethods
};

// Module initialization
PyMODINIT_FUNC PyInit_cfor(void) {
    return PyModule_Create(&loopmodule);
}
