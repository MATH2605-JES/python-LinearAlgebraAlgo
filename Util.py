# General Utility Methods for Algorithms
import numpy as np


# James
def multiply_matrix(matrix_1, matrix_2):
    if matrix_1.shape[1] != matrix_2.shape[0]:
        return None

    result = np.empty((matrix_1.shape[0], matrix_2.shape[1]))
    # We can use transpose & dot product library function.
    # Dot product of first rows of matrix_1 and matrix_2^t gives us first resulting number.of first row.
    # Dot product of first row of matrix_1 and second row of matrix_2^t gives us second resulting number of first row.
    matrix_2_t = matrix_2.transpose()
    for i in range(matrix_1.shape[0]):
        for j in range(matrix_2_t.shape[0]):
            result[i, j] = matrix_1[i].dot(matrix_2_t[j])

    return result


# Seth
def find_LU(matrix):
    size = matrix.shape[0]
    L = np.identity(size, float)
    U = np.ndarray.astype(matrix, float)  # @TODO this is giving me warning (maybe try without float .. it auto casts?)
    for row in xrange(1, size):
        for col in xrange(0, row):
            L[row][col] = U[row][col] / U[col][col]
            U[row] -= L[row][col] * U[col]
    return L, U


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


# James
def matrix_trace(matrix):
    loop = min(matrix.shape[1], matrix.shape[0])
    sum = 0
    for i in range(loop):
        sum += matrix[i, i]
    return sum
