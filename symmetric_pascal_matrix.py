import numpy as np
import matplotlib.pyplot as pyplot

import Util as util


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
    v[0] += util.vector_norm(v)  # construct u
    u_norm_squared = util.vector_norm(v) ** 2.0  # ||u||^2
    u = np.array([v])  # wrap to allow u^t
    h = np.identity(u.shape[1], float) - 2.0 / u_norm_squared * np.dot(u.T, u)
    return h


# James
def qr_fact_househ(matrix):
    y, x = matrix.shape

    Q = house_holder_compute_h(matrix)  # start building Q from h_1
    R = util.multiply_matrix(Q, matrix)  # start building R from h_1 * A
    for i in range(y - 2):
        h = house_holder_reconstruct(house_holder_compute_h(util.get_sub_matrix(R, i + 1)), y)  # get h_n
        Q = util.multiply_matrix(Q, h)  # (h_1 * ... h_(n-1)) * h_n
        R = util.multiply_matrix(h, R)  # h_n * (.... h_1 * A)

    error = util.matrix_error(util.multiply_matrix(Q, R), matrix)
    return Q, R, error


# James
def qr_fact_givens(matrix):
    m, n = matrix.shape

    Q = np.identity(m, dtype=float)
    R = np.copy(matrix)

    for i in range(m):
        for j in range(min(i, n)):
            if R[i, j] != 0:
                x = R[j][j]
                y = R[i][j]
                norm = util.vector_norm([x, y])
                cos = x * 1.0 / norm
                sin = -y * 1.0 / norm

                G = np.identity(m, dtype=float)
                G[i, i] = cos
                G[i, j] = sin
                G[j, i] = -sin
                G[j, j] = cos

                Q = util.multiply_matrix(Q, G.T)
                R = util.multiply_matrix(G, R)

    error = util.matrix_error(util.multiply_matrix(Q, R), matrix)
    return Q, R, error


# James
# Code reference: https://en.wikipedia.org/wiki/Triangular_matrix
# Note: This code is straightforward, and I also implemented matrix inverse function.
def solve_lu_b(l, u, b):
    m, n = l.shape
    if b.shape[0] != m:
        return None

    x = np.copy(b)

    # lower triangle loop, calculate y with dp, forward
    for i in range(m):
        for j in range(0, i):
            x[i] -= l[i, j] * x[j]
        x[i] /= l[i, i]
    # upper triangle loop, ,calculate x with dp, backward
    for i in reversed(range(m)):
        for j in range(i + 1, m):
            x[i] -= u[i, j] * x[j]
        x[i] /= u[i, i]

    return x


# James
def solve_qr_b(q, r, b):
    m, n = q.shape
    if b.shape[0] != m:
        return None

    x = util.multiply_matrix(q.T, b)  # x = Q^tb

    # upper triangle loop, ,calculate x with dp, backward
    for i in reversed(range(m)):
        for j in range(i + 1, m):
            x[i] -= r[i, j] * x[j]
        x[i] /= r[i, i]

    return x


# James
def generate_pascal_matrix(size):
    matrix = np.zeros([size, size])  # init matrix
    matrix[:, 0] = 1  # first col all 1
    matrix[0, :] = 1  # first row all 1
    for i in range(1, size):
        for j in range(1, size):
            matrix[i, j] = matrix[i - 1, j] + matrix[i, j - 1]

    return matrix


# James
def generate_b_matrix(size):
    matrix = np.zeros([size, 1])
    for i in range(1, size + 1):
        matrix[i - 1, 0] = 1.0 / i
    return matrix


# James
def problem_1d(n=(2, 13)):
    low_bound, high_bound = n

    file = open('problem_1d.txt', 'w')
    n, lu_error, lu_px_b_error, qr_househ_error, qr_px_b_househ_error, qr_givens_error, qr_px_b_givens_error, inverse_px_b_error = [], [], [], [], [], [], [], []

    for size in range(low_bound, high_bound):
        print >> file, '============[ n =', size, ']====================='
        n.append(size)
        p = generate_pascal_matrix(size).astype(int)
        print >> file, 'P ='
        print >> file, p
        b = generate_b_matrix(size)
        print >> file, 'b ='
        print >> file, b
        print >> file, '============LU factorization==============='
        l, u, error = util.lu_fact(p)
        print >> file, 'L ='
        print >> file, l.round(6)
        print >> file, 'U ='
        print >> file, u.round(6)
        print >> file, '||LU - P||_inf (error) =', error
        lu_error.append(error)
        x = solve_lu_b(l, u, b)
        print >> file, 'x_sol ='
        print >> file, x
        error = util.matrix_error(util.multiply_matrix(p, x), b)
        print >> file, '||Px - b||_inf (error) =', error
        lu_px_b_error.append(error)
        print >> file, '============QR factorization==============='
        print >> file, '(Householder)'
        q, r, error = qr_fact_househ(p)
        print >> file, 'Q ='
        print >> file, q.round(6)
        print >> file, 'R ='
        print >> file, r.round(6)
        print >> file, 'error =', error
        qr_househ_error.append(error)
        x = solve_qr_b(q, r, b)
        print >> file, 'x_sol ='
        print >> file, x
        error = util.matrix_error(util.multiply_matrix(p, x), b)
        print >> file, '||QR - P||_inf (error) =', error
        qr_px_b_househ_error.append(error)
        print >> file, '============QR factorization==============='
        print >> file, '(Givens)'
        q, r, error = qr_fact_givens(p)
        print >> file, 'Q ='
        print >> file, q.round(6)
        print >> file, 'R ='
        print >> file, r.round(6)
        print >> file, '||QR - P||_inf (error) =', error
        qr_givens_error.append(error)
        x = solve_qr_b(q, r, b)
        print >> file, 'x_sol ='
        print >> file, x
        error = util.matrix_error(util.multiply_matrix(p, x), b)
        print >> file, '||Px - b||_inf (error) =', error
        qr_px_b_givens_error.append(error)
        print >> file
        print >> file
        # ============Solve system Eq through inversion===============
        # Computationally long. Enable only if needed
        # x = multiply_matrix(matrix_inverse(p), b)
        # px_minus_b = multiply_matrix(p, x) - b
        # error = vector_error(px_minus_b)
        # print error
    f, ((ax1, ax2), (ax3, ax4)) = pyplot.subplots(2, 2, sharey=True, sharex=True)

    ax1.set_title('LU')
    ax1.plot(n, lu_error, label='LU Error', c='blue')
    ax1.plot(n, lu_px_b_error, label='LU Px_b Error', c='black')
    ax2.set_title('QR Householder')
    ax2.plot(n, qr_househ_error, label='QR Householder Error', c='blue')
    ax2.plot(n, qr_px_b_househ_error, label='QR Householder Px_b Error', c='black')
    ax3.set_title('QR Givens')
    ax3.plot(n, qr_givens_error, label='QR Givens Error', c='blue')
    ax3.plot(n, qr_px_b_givens_error, label='QR Givens Px_b Error', c='black')
    # Draw legend
    error_line, = ax4.plot([], label='Error (on P)', c='blue')
    px_b_error_line, = ax4.plot([], label='Error (on b)', c='black')
    ax4.legend(handles=[error_line, px_b_error_line])

    # Draw Error on P lines
    fig = pyplot.figure()
    pyplot.xlabel('size (n x n pascal matrix)')
    pyplot.ylabel('||LU or QR - P||_inf (error)')
    pyplot.title('Error on P')
    lue_plot, = pyplot.plot(n, lu_error, label='LU', c='red')
    qrhhe_plot, = pyplot.plot(n, qr_househ_error, label='QR Householder', c='green')
    qrge_plot, = pyplot.plot(n, qr_givens_error, label='QR Givens', c='blue')
    pyplot.legend(handles=[lue_plot, qrhhe_plot, qrge_plot])
    # Draw Error on b lines
    fig = pyplot.figure()
    pyplot.xlabel('size (n x n pascal matrix)')
    pyplot.ylabel('||Px - b||_inf (error)')
    pyplot.title('Error on b')
    lupxbe_plot, = pyplot.plot(n, lu_px_b_error, label='LU', c='red')
    qrhhpxbe_plot, = pyplot.plot(n, qr_px_b_househ_error, label='QR Householder', c='green')
    qrgpxbe_plot, = pyplot.plot(n, qr_px_b_givens_error, label='QR Givens', c='blue')
    pyplot.legend(handles=[lupxbe_plot, qrhhpxbe_plot, qrgpxbe_plot])
    pyplot.show()


if __name__ == '__main__':
    print 'Output file is generated after graphs are closed.'
    print 'Output as problem_1d.txt in the same directory.'
    problem_1d()
