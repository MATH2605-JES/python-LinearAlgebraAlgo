# General Utility Methods for Algorithms
import random as rand

import numpy as np
import numpy.matlib
import matplotlib.pyplot as pyplot
import matplotlib.colors as plotcolors


# James
def multiply_matrix(matrix_1, matrix_2):
    if matrix_1.shape[1] != matrix_2.shape[0]:
        return None

    result = np.empty((matrix_1.shape[0], matrix_2.shape[1]), dtype=float)
    # We can use transpose & dot product library function.
    # Dot product of first rows of matrix_1 and matrix_2^t gives us first resulting number.of first row.
    # Dot product of first row of matrix_1 and second row of matrix_2^t gives us second resulting number of first row.
    matrix_2_t = matrix_2.transpose()
    for i in range(matrix_1.shape[0]):
        for j in range(matrix_2_t.shape[0]):
            result[i, j] = matrix_1[i].dot(matrix_2_t[j])

    return result


# Emeke
# works n x m matrices
def multiply_matrix2(matrix_1, matrix_2):
    product = np.matlib.empty((matrix_1.shape[0], matrix_2.shape[1]))
    for i in range(product.shape[0]):
        for j in range(product.shape[1]):
            product[i, j] = matrix_1[i, :].dot(matrix_2[:, j])

    return product


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


# Seth
def vector_error(array):
    if len(array) == 0:
        return
    answer = np.absolute(array[0])
    for i in range(len(array)):
        if np.absolute(array[i]) > answer:
            answer = np.absolute(array[i])
    return answer


# Seth
def getDiag(matrix):
    diag = np.copy(matrix)
    for i in range(diag.shape[0]):
        for j in range(diag.shape[1]):
            if i != j:
                diag[i][j] = 0
    return diag


# Seth
def getLowerDiag(matrix):
    lower = np.copy(matrix)
    for i in range(lower.shape[0]):
        for j in range(lower.shape[1]):
            if i < j:
                lower[i][j] = 0
    return lower


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
def matrix_error(matrix, original_matrix):
    if matrix.shape != original_matrix.shape:
        return None
    y, x = matrix.shape
    error_matrix = matrix - original_matrix
    # Allowed built-ins were iffy on this one, so didn't use np.sum(matrix-original_matrix, axis=1)
    max = abs(error_matrix[0, 0])
    for i in range(y):
        for j in range(x):
            compared = abs(error_matrix[i, j])
            if max < compared:
                max = compared

    return max


# James
# This beautiful code took 3.5 hours T_T
def matrix_cofactor(matrix):
    y, x = matrix.shape
    cofactor = np.empty([y, x], dtype=float)
    for i in range(y):
        flip = 1.0 if (i % 2 == 0) else -1.0
        for j in range(x):
            sub_matrix = np.delete(np.delete(matrix, j, 1), i, 0)
            cofactor[i, j] = flip * find_determinant(sub_matrix)
            flip *= -1

    return cofactor


# James
def matrix_inverse(matrix):
    return 1.0 / find_determinant(matrix) * matrix_cofactor(matrix).T


# Emeke
def matrix_inverse_22(matrix):
    det = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    matrixB = np.matrix([[matrix[0, 0], - matrix[0, 1]], [-matrix[1, 0], matrix[1, 1]]])
    if det == 0:
        return None
    return (1.0 / det) * matrixB


""" Emeke
    Generates 1000 random 2x2 matrices
    Create a series of randomly generated matrices with uniformly distributed entries within a given range
    shape (tuple(int, int)): Desired shape of matrices.
    number (int): Requested number of matrices.
    lower (Real): Lower bound for random range.
    upper (Real): Upper bound for random range.
"""


def random_matrices(shape, number, lower, upper):
    series = tuple()

    while len(series) < number:
        mat = np.matlib.empty(shape)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                mat[i, j] = rand.uniform(lower, upper)

        series += (mat,)

    return series


# Emeke
def plot_colored(data, colors, color_label, xlabel, ylabel, title, xscale, yscale, cmap, fname):
    pyplot.clf()

    # Create colormap object if needed
    colormap = None if cmap is None else plotcolors.LinearSegmentedColormap.from_list('cplot', cmap)

    # Plot data
    pyplot.scatter(data[0], data[1], s=40, c=colors, cmap=colormap)

    # Create titles and legend, then render
    pyplot.colorbar().set_label(color_label)
    pyplot.title(title).set_size('xx-large')
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    pyplot.xlim(xscale)
    pyplot.ylim(yscale)
    pyplot.savefig(fname)
