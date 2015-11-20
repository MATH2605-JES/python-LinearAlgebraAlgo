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
    U = np.ndarray.astype(matrix, float)
    for row in xrange(1, size):
        for col in xrange(0, row):
            L[row][col] = U[row][col] / U[col][col]
            U[row] -= L[row][col] * U[col]
    return L, U


def find_QR(matrix):
    return None


# Seth
def find_determinant(matrix):
    size = matrix.shape[0]
    if size == 1:
        return matrix[0][0]
    answer = 0
    modifier = 1
    for i in xrange(size):
        element = matrix[0][i]
        newMatrix = np.zeros((size - 1, size - 1))
        for row in xrange(1, size):
            newCol = 0
            for col in xrange(size):
                if col != i:
                    newMatrix[row - 1][newCol] = matrix[row][col]
                    newCol += 1
        print newMatrix
        answer += element * modifier * find_determinant(newMatrix)
        modifier *= -1
    return answer


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


# James
def vector_norm(vector):
    squared_sum = 0
    for i in range(len(vector)):
        squared_sum += vector[i] ** 2
    return np.sqrt(squared_sum)


# James
def house_holder_reconstruct(matrix, size):
    diff = size - matrix.shape[0]
    if diff < 1:
        return None
    arr = np.identity(size)
    for x in range(matrix.shape[1]):
        for y in range(matrix.shape[0]):
            arr[y + diff, x + diff] = matrix[y, x]

    return arr


# James
def house_holder_compute_h(matrix):
    v = np.array(matrix[:, 0]).astype(float)  # first column vector
    v[0] += vector_norm(v)  # construct u
    u_norm_squared = vector_norm(v) ** 2  # ||u||^2
    u = np.array([v])  # wrap to allow u^t
    h = np.identity(u.shape[1], float) - 2 / u_norm_squared * np.dot(u.T, u)  # @TODO will we get any nxn matrix
    return h


def house_holder(matrix):
    for i in range(matrix.shape[0] - 1):
        h = house_holder_compute_h(matrix)
        ha = multiply_matrix(h, matrix)


# James
# if        [ a b c
#            d e f
#            g h i ]
# return    [ e f
#            h i ]
#
def get_sub_matrix(matrix):
    arr = np.empty((matrix.shape[0] - 1, matrix.shape[1] - 1))
    for x in range(1, matrix.shape[1]):
        for y in range(1, matrix.shape[0]):
            arr[y - 1, x - 1] = matrix[y, x]
    return arr


if __name__ == '__main__':
    matrix = np.array([[3, 2, 2], [4, 1, 1], [0, 2, 5]])
    print house_holder_reconstruct((get_sub_matrix(matrix)), 4)
