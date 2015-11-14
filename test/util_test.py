import unittest
import numpy as np
from Util import multiply_matrix


class TestUtil(unittest.TestCase):
    def test_multiply_matrix(self):
        """
        Identity matrix multiplication
        1   0   0       4   2       4   2
        0   1   0  dot  1   2   =   1   2
        0   0   1       3   3       3   3
        """
        identity = np.identity(3)
        matrix2 = np.array([[4, 2], [1, 2], [3, 3]])
        np.testing.assert_array_equal(identity.dot(matrix2), multiply_matrix(identity, matrix2))
        """
        Invalid multiplication
                        4   2       4   2
        empty      dot  1   2   =   1   2
                        3   3       3   3
        """
        matrix1 = np.array([[]])
        self.assertEqual(None, multiply_matrix(matrix1, matrix2))
        # 3x3 matrix multiplication
        matrix1 = np.array([[4, 2, 5], [1, 2, -2], [3, -5, 3]])
        matrix2 = np.array([[0, 0, 3], [1, 2, 2], [9, -9, -9]])
        print multiply_matrix(matrix1, matrix2)
        np.testing.assert_array_equal(matrix1.dot(matrix2), multiply_matrix(matrix1, matrix2))
        # unequal size multiplication
        matrix1 = np.array([[4, 2, 5], [1, 2, -2], [3, -5, 3]])
        matrix2 = np.array([[4, 0], [1, 2], [9, -9]])
        np.testing.assert_array_equal(matrix1.dot(matrix2), multiply_matrix(matrix1, matrix2))


if __name__ == '__main__':
    unittest.main()
