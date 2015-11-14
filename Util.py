# General Utility Methods for Algorithms
import numpy as np


# James
def multiply_matrix(matrix_1, matrix_2):
    if matrix_1.shape[1] != matrix_2.shape[0]:
        return -1

    result = np.empty((matrix_1.shape[0], matrix_2.shape[1]))
    # We can use transpose & dot product library function.
    # Dot product of first rows of matrix_1 and matrix_2^t gives us first resulting number.of first row.
    # Dot product of first row of matrix_1 and second row of matrix_2^t gives us second resulting number of first row.
    matrix_2_t = matrix_2.transpose()
    for i in range(matrix_1.shape[0]):
        for j in range(matrix_2_t.shape[0]):
            result[i, j] = matrix_1[i].dot(matrix_2_t[j])

    print 'empty array:', result

    return result


def find_LU(matrix):
    return None


def find_QR(matrix):
    return None


# Emeke
def find_determinant(matrix):
    return None


# Emeke
def find_eigenvalues(matrix):
    return None


# Emeke
def find_eigenvectors(matrix):
    return None
