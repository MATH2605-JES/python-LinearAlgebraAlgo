import random as rand

import numpy as np
import matplotlib.pyplot as plt

import Util

A0 = np.array([[1, .5, 1.0 / 3], [.5, 1, .25], [1.0 / 3, .25, 1]])
bvec0 = np.array([[.1], [.1], [.1]])
exact = np.array([[9.0 / 190], [28.0 / 475], [33.0 / 475]])


# Seth
def jacobi_iter(initData, A=A0, bvec=bvec0):  # x = -[D^-1](L+U)x + [D^-1]*b
    diagonal = Util.getDiag(A)
    lu = A - diagonal
    sInv = Util.matrix_inverse(diagonal)
    return iterate(initData, sInv, lu, bvec)


# Seth
def gs_iter(initData, A=A0, bvec=bvec0):  # x = -[(L+D)^-1]*Ux + [(L+D)^-1]*b
    ld = Util.getLowerDiag(A)
    upper = A - ld
    sInv = Util.matrix_inverse(ld)
    return iterate(initData, sInv, upper, bvec)


# Seth
def iterate(data, sInv, T, bVec, maxIt=100, tol=0.00005):
    for i in range(maxIt):
        previousData = data
        temp = Util.multiply_matrix(sInv, T * -1)
        data = Util.multiply_matrix(temp, previousData)
        data += Util.multiply_matrix(sInv, bVec)
        deltaData = np.absolute(previousData - data)
        if np.less_equal(deltaData, np.ones(data.shape) * tol).all():
            # print "interation: ", i, "\n", data
            return data, i
    return None


def part_b():
    totalIterationsJacobi = 0
    totalDataJacobi = np.array([[0.0], [0.0], [0.0]])
    x0marginOfError = []
    jacobiIterations = []
    gaussIterations = []
    totalIterationsGauss = 0
    totalDataGauss = np.array([[0.0], [0.0], [0.0]])
    ratio = 0
    num = 100.0
    for i in range(100):
        randomInit = np.array([[rand.uniform(-1, 1)], [rand.uniform(-1, 1)], [rand.uniform(-1, 1)]])

        initError = Util.vector_error(randomInit - exact)
        x0marginOfError.append(initError)

        data, iterations = jacobi_iter(randomInit)
        jacobiIterations.append(iterations)
        totalDataJacobi += data

        data, iterations1 = gs_iter(randomInit)
        gaussIterations.append(iterations1)
        totalDataGauss += data

        ratio += iterations / float(iterations1)
    print "Average solution for Jacobi: \n", totalDataJacobi / num
    print "Average solution for Gauss: \n", totalDataGauss / num
    print "ratio Jacobi:Gauss", ratio / 100.
    plt.title("Initial Error vs Number of Iterations")
    plt.xlabel('Initial Error')
    plt.ylabel('Number of Iterations')
    plt.scatter(x0marginOfError, gaussIterations, c='blue')
    plt.scatter(x0marginOfError, jacobiIterations, c="black")
    plt.show()


if __name__ == '__main__':
    x = np.array([[1], [1], [1]])
    answer, steps = jacobi_iter(x)
    print "intital vector : \n", x
    print"Jacobi answer: \n", answer, "in ", steps, "steps"
    answer2, steps2 = gs_iter(x)
    print "Gauss-Seidel answer: \n", answer2, "in ", steps2, "steps"
    print"================================================================================"
    print("part 2:")
    part_b()
