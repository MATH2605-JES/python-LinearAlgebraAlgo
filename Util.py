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
def lu_fact(matrix):
    size = matrix.shape[0]
    L = np.identity(size, float)
    U = np.ndarray.astype(matrix, float)
    for row in xrange(1, size):
        for col in xrange(0, row):
            L[row][col] = U[row][col] / U[col][col]
            U[row] -= L[row][col] * U[col]

    error = matrix_error(multiply_matrix(L, U), matrix)
    return L, U, error


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


# James
def qr_fact_househ(matrix):
    matrix_size = matrix.shape[0]
    h = house_holder_compute_h(matrix)  # compute h
    Q = h  # start building Q from h_1
    ha = multiply_matrix(h, matrix)
    R = ha  # start building R from h_1 * A
    for i in range(matrix_size - 2):
        h = house_holder_reconstruct(house_holder_compute_h(get_sub_matrix(ha, i + 1)), matrix_size)  # get h_n
        ha = multiply_matrix(h, matrix)  # h_n * A @TODO this part ask
        Q = multiply_matrix(Q, h)  # (h_1 * ... h_(n-1)) * h_n
        R = multiply_matrix(h, R)  # h_n * (.... h_1 * A)

    error = matrix_error(multiply_matrix(Q, R), matrix)
    return Q.round(14), R.round(14), error


# James
# if        [ a b c
#            d e f
#            g h i ] , cut_size = 1
# return    [ e f
#            h i ] , will return same matrix of cut_size = 0
#
def get_sub_matrix(matrix, cut_size=1):
    m, n = matrix.shape
    if cut_size <= 0:
        return matrix
    arr = np.empty((m - cut_size, n - cut_size))
    for x in range(cut_size, n):
        for y in range(cut_size, m):
            arr[y - cut_size, x - cut_size] = matrix[y, x]
    return arr


# James
def qr_fact_givens(matrix):
    m, n = matrix.shape

    Q = np.identity(m)
    R = np.copy(matrix)

    for i in range(m):
        for j in range(min(i, n)):
            if R[i, j] != 0:
                x = R[j][j]
                y = R[i][j]
                norm = vector_norm([x, y])
                cos = x / norm
                sin = -y / norm

                G = np.identity(m)
                G[i, i] = cos
                G[i, j] = sin
                G[j, i] = -sin
                G[j, j] = cos

                Q = multiply_matrix(Q, G.T)
                R = multiply_matrix(G, R)

    error = matrix_error(multiply_matrix(Q, R), matrix)
    return Q.round(15), R.round(15), error


# James @TODO not sure if this is it
# untested
def matrix_error(matrix, original_matrix):
    if matrix.shape != original_matrix.shape:
        return None
    y, x = matrix.shape
    error_matrix = matrix - original_matrix
    # Allowed built-ins were iffy on this one, so didn't use np.sum(matrix-original_matrix, axis=1)
    sums = []
    for i in range(y):
        row_sum = 0
        for j in range(x):
            row_sum += abs(error_matrix[i, j])
        sums.append(row_sum)

    return max(sums)


# James
# Code reference: https://en.wikipedia.org/wiki/Triangular_matrix
# Note: This code is straightforward, and I also implemented matrix inverse function.
def solve_lu_b(matrix, b):
    m, n = matrix.shape
    if b.shape[0] != m:
        return None
    L, U, error = lu_fact(matrix)
    y = b
    # lower triangle loop, calculate y with dp, forward
    for i in range(m):
        for j in range(0, i):
            y[i] -= L[i, j] * y[j]
        y[i] /= L[i, i]
    # upper triangle loop, ,calculate x with dp, backward
    for i in reversed(range(m)):
        print i
        for j in range(i + 1, m):
            y[i] -= L[i, j] * y[j]
        y[i] /= L[i, i]
    return y


# James
# This beautiful code took 3.5 hours T_T
def matrix_cofactor(matrix):
    y, x = matrix.shape
    cofactor = np.empty([y, x])
    for i in range(y):
        flip = 1 if (i % 2 == 0) else -1
        for j in range(x):
            sub_matrix = np.delete(np.delete(matrix, j, 1), i, 0)
            cofactor[i, j] = flip * find_determinant(sub_matrix)
            flip *= -1

    return cofactor


def matrix_inverse(matrix):
    return 1.0 / find_determinant(matrix) * matrix_cofactor(matrix).T


if __name__ == '__main__':
    # matrix = np.array([[3, 2, 2], [4, 1, 1], [0, 2, 5]])
    matrix = np.array([[1, 1, 1, 1], [1, 2, 3, 4], [1, 3, 6, 10], [1, 4, 10, 20]])
    # matrix2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    # matrix = np.array([[4, 1, -2, 2], [1, 2, 0, 1], [-2, 0, 3, -2], [2, 1, -2, -1]])
    # q, r = house_holder(matrix)
    # print q
    # print r
    # q, r, erro = qr_fact_givens(matrix)
    l, u, erro = lu_fact(matrix)
    print solve_lu_b(matrix, np.array([1, 1 / 2, 1 / 3, 1 / 4]))
    # matrix = np.array([[4, 5], [0, 6]])
    # print find_determinant(matrix)
    # print l
    # print u
    # print erro
