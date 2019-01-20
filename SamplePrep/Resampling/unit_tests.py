import resample
import numpy as np
import unittest


def round_list(my_list, places):
    if type(my_list[0]) == list:
        return [round_list(i, places) for i in my_list]
    if places > 0:
        return [round(i, places) for i in my_list]
    else:
        return [round(i) for i in my_list]


test_ECAL = np.array([[0, 1, 1, 0, 0],
                      [1, 3, 2, 1, 0],
                      [1, 3, 4, 2, 1],
                      [2, 3, 4, 3, 3],
                      [0, 1, 2, 3, 3]])

test_3D_ECAL = np.array([[[0, 1, 1, 0, 0],
                          [1, 3, 2, 1, 0],
                          [1, 3, 4, 2, 1],
                          [2, 3, 4, 3, 3],
                          [0, 1, 2, 3, 3]]])


class UnitTests(unittest.TestCase):

    def test_calculate_new_matrix_size_and_overhang(self):
        new_size = resample.get_smallest_new_size((5, 5, 5), (1.3, 2.5, 0.3))
        overhang = resample.calculate_overhang((5, 5, 5), (1.3, 2.5, 0.3), new_size)
        self.assertEqual(round_list(new_size, 0), [4, 2, 17])
        self.assertEqual(round_list(overhang, 2), [0.10, 0.00, 0.05])
        new_size = resample.get_largest_new_size((5, 5, 5), (1.3, 2.5, 0.3))
        overhang = resample.calculate_overhang((5, 5, 5), (1.3, 2.5, 0.3), new_size)
        self.assertEqual(round_list(new_size, 0), [3, 2, 16])
        self.assertEqual(round_list(overhang, 2), [-0.55, 0.00, -0.10])

    def test_get_cell_filling(self):
        self.assertEqual(resample.get_cell_filling(0, 2.5, -0.25), (-1, 0, 0.25, 0.75, 1.00))
        self.assertEqual(resample.get_cell_filling(2, 2.5, -0.25), (0, 1, 0.75, 0.25, 1.00))
        self.assertEqual(resample.get_cell_filling(2, 0.5, 0.0), (4, 5, 0.50, 0.50, 0.50))

    def test_one_d_resampling(self):
        one_d_resampling = resample.one_d_resampling(4, 8, 0.5, 0.0)
        correct_one_d_resampling = [[0, 1, 0.5, 0.5, 0.5],
                                    [2, 3, 0.5, 0.5, 0.5],
                                    [4, 5, 0.5, 0.5, 0.5],
                                    [6, 7, 0.5, 0.5, 0.5]]
        for i, j in zip(one_d_resampling, correct_one_d_resampling):
            self.assertListEqual(list(np.round(i, 1)), j)
        one_d_resampling = resample.one_d_resampling(8, 3, 2.5, -0.25)
        correct_one_d_resampling = [[-1, 0, 0.25, 0.75, 1.00],
                                    [0, 0, 1.00, 1.00, 1.00],
                                    [0, 1, 0.75, 0.25, 1.00],
                                    [1, 1, 1.00, 1.00, 1.00],
                                    [1, 1, 1.00, 1.00, 1.00],
                                    [1, 2, 0.25, 0.75, 1.00],
                                    [2, 2, 1.00, 1.00, 1.00],
                                    [2, 3, 0.75, 0.25, 1.00]]
        for i, j in zip(one_d_resampling, correct_one_d_resampling):
            self.assertListEqual(list(np.round(i, 2)), j)

    def test_get_flattened_index(self):
        matrix_shape = (4, 3, 8)
        self.assertEqual(resample.get_flattened_index(matrix_shape, (1, 2, 2)), 42)
        self.assertEqual(resample.get_flattened_index(matrix_shape, (3, 0, 5)), 77)
        self.assertEqual(resample.get_flattened_index(matrix_shape, (0, 1, 2)), 10)

    def test_get_resampling_fill_list(self):
        first_dim_resampling = resample.one_d_resampling(4, 8, 0.5, 0.0)
        second_dim_resampling = resample.one_d_resampling(8, 3, 2.5, -0.25)
        resamplers = [first_dim_resampling, second_dim_resampling]
        resampling_list = resample.get_resampling_fill_list((2, 5), resamplers, (8, 3))
        self.assertEqual(resampling_list, [((4, 1), 1/8), ((4, 2), 3/8), ((5, 1), 1/8), ((5, 2), 3/8)])

    def test_get_partial_resampling_matrix(self):
        resampling_matrix = resample.get_partial_resampling_matrix(test_ECAL.shape, (1.5, 1.5), (4, 4))
        resampled_matrix = np.matmul(test_ECAL.flatten(), resampling_matrix).reshape((4, 4))
        correct_matrix = np.array([[0, 1.5, 0.5, 0],
                                   [1.5, 6.5, 4, 0.5],
                                   [2.5, 7.5, 7, 3.5],
                                   [0, 2, 4, 3]])
        self.assertEqual(resampled_matrix.tolist(), correct_matrix.tolist())
        resampling_matrix = resample.get_partial_resampling_matrix(resampled_matrix.shape, (2/3, 2/3), (5, 5))
        resampled_matrix = np.matmul(resampled_matrix.flatten(), resampling_matrix).reshape((5, 5))
        correct_matrix = np.array([[0, 0.67, 0.44, 0.22, 0],
                                   [0.67, 2.89, 2.33, 1.78, 0.22],
                                   [0.89, 3.11, 2.78, 2.44, 0.89],
                                   [1.11, 3.33, 3.22, 3.11, 1.56],
                                   [0, 0.89, 1.33, 1.78, 1.33]])
        self.assertEqual(round_list(resampled_matrix.tolist(), 2), correct_matrix.tolist())

    def test_get_resampling_matrix(self):
        resampling_matrix = resample.get_full_resampling_matrix(test_ECAL.shape, (1.5, 1.5))
        resampled_matrix = np.matmul(test_ECAL.flatten(), resampling_matrix).reshape(test_ECAL.shape)
        correct_matrix = np.array([[0, 0.67, 0.44, 0.22, 0],
                                   [0.67, 2.89, 2.33, 1.78, 0.22],
                                   [0.89, 3.11, 2.78, 2.44, 0.89],
                                   [1.11, 3.33, 3.22, 3.11, 1.56],
                                   [0, 0.89, 1.33, 1.78, 1.33]])
        self.assertEqual(round_list(resampled_matrix.tolist(), 2), correct_matrix.tolist())
        pass

    def test_ATLAS_resampling(self):
        ATLAS_resampling_matrices = resample.get_ATLAS_resampling_matrices(test_3D_ECAL.shape)
        ATLAS_ECAL = resample.spoof_ATLAS_geometry(test_3D_ECAL, ATLAS_resampling_matrices)
        ATLAS_correct_ECAL = np.array([[[0.8025, 0.8025, 0.8025, 0.8025, 0.8025,
                                         0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                         0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                         0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                         0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                        [0.8025, 0.8025, 0.8025, 0.8025, 0.8025,
                                         0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                         0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                         0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                         0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                        [1.2176, 1.2176, 1.2176, 1.2176, 1.2176,
                                         0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                         0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                         0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                         0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                        [1.6326, 1.6326, 1.6326, 1.6326, 1.6326,
                                         0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                         0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                         0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                         0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                        [1.6326, 1.6326, 1.6326, 1.6326, 1.6326,
                                         0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                         0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                         0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                         0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]])
        self.assertTrue((np.round(ATLAS_ECAL, 4) == ATLAS_correct_ECAL).all())

    def test_CMS_resampling(self):
        CMS_resampling_matrix = resample.get_CMS_resampling_matrix(test_3D_ECAL.shape)
        CMS_ECAL = resample.spoof_CMS_geometry(test_3D_ECAL, CMS_resampling_matrix)
        CMS_correct_ECAL = np.array([[[0.3760, 0.3760, 0.3760, 0.3760, 0.3760],
                                      [0.3760, 0.3760, 0.3760, 0.3760, 0.3760],
                                      [0.5704, 0.5704, 0.5704, 0.5704, 0.5704],
                                      [0.7649, 0.7649, 0.7649, 0.7649, 0.7649],
                                      [0.7649, 0.7649, 0.7649, 0.7649, 0.7649]]])
        self.assertTrue((np.round(CMS_ECAL, 4) == CMS_correct_ECAL).all())

    def test(self):
        self.test_calculate_new_matrix_size_and_overhang()
        self.test_get_cell_filling()
        self.test_one_d_resampling()
        self.test_get_flattened_index()
        self.test_get_resampling_fill_list()
        self.test_get_partial_resampling_matrix()
        self.test_get_resampling_matrix()
        self.test_ATLAS_resampling()
        self.test_CMS_resampling()


if __name__ == "__main__":
    my_tests = UnitTests()
    my_tests.test()
