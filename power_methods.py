import Util
import numpy as np
import numpy.matlib


#Emeke
def power_method(matrix, v, epsilon, maxIter):
    itr = 1
    u = v
    w = v
    prev = 0
    while itr <= maxIter:
        uNext = Util.multiply_matrix2(matrix, u)
        cur = float(Util.multiply_matrix2(w.transpose(), uNext))/float(Util.multiply_matrix2(w.transpose(), u))
        u = uNext
        if abs(prev - cur) <= epsilon:
            return (cur, Util.vector_norm(u), itr)
        prev = cur
        itr += 1
    return None

if __name__ == '__main__':
    print 'Start 3.b\n'
    matrices = [[], [], []]

    for i in range(1000):
        mat = Util.random_matrices((2, 2), 1, -2, 2)[0]
        while ((mat[0,0]*mat[1,1] - mat[0,1]*mat[1,0]) == 0):
            mat = Util.random_matrices((2, 2), 1, -2, 2)[0]
        matrices[0].append(mat)
        matrices[1].append(mat[0,0]*mat[1,1] - mat[0,1]*mat[1,0])
        matrices[2].append(mat[0,0]+mat[1,1])

    #Run algorthim on matrices and their inverses, recording result
    dataset = [[],[],[]]
    dataset_inv = [[],[],[]]
    for i in range(len(matrices[0])):
        result = power_method(matrices[0][i], np.matrix('1;0', dtype=float), 0.00005, 100)
        if result != None:
            dataset[0].append(matrices[1][i])
            dataset[1].append(matrices[2][i])
            dataset[2].append(result[2]/float(100))
        result = power_method(Util.matrix_inverse_22(matrices[0][i]), np.matrix('1;0', dtype=float), 0.00005, 100)
        if result != None:
            dataset_inv[0].append(matrices[1][i])
            dataset_inv[1].append(matrices[2][i])
            dataset_inv[2].append(result[2]/float(100))

    #Plot data
    Util.plot_colored((dataset[0], dataset[1],), dataset[2], 'Iterations (x100)', 'Determinant', 'Trace', 'Standard', (-8, 8), (-4, 4), None, 'Standard.png')
    Util.plot_colored((dataset_inv[0], dataset_inv[1],), dataset_inv[2], 'Iterations (x100)', 'Determinant', 'Trace', 'Inverse', (-8, 8), (-4, 4), None, 'Inverse.png')
    print "Plots saved to Standard.png and Inverse.png"

