import unittest
import h5py as h5
import pathlib
import sys


class UnitTests(unittest.TestCase):

    def test_existence(self, sample):
        self.assertTrue(pathlib.Path(sample).exists())

    def test_readable(self, sample):
        readable = True
        try:
            h5.File(sample)
        except Exception:
            readable = False
        self.assertTrue(readable)

    def test_keys(self, sample):
        fixed_keys = ['ECAL', 'ECAL_E', 'ECAL_nHits', 'ECAL_ratioFirstLayerToSecondLayerE', 'ECAL_ratioFirstLayerToTotalE', 'ECALmomentX1', 'ECALmomentX2', 'ECALmomentX3', 'ECALmomentX4', 'ECALmomentX5', 'ECALmomentX6', 'ECALmomentY1', 'ECALmomentY2', 'ECALmomentY3', 'ECALmomentY4', 'ECALmomentY5', 'ECALmomentY6', 'ECALmomentZ1', 'ECALmomentZ2', 'ECALmomentZ3', 'ECALmomentZ4', 'ECALmomentZ5', 'ECALmomentZ6', 'HCAL', 'HCAL_E', 'HCAL_ECAL_ERatio', 'HCAL_ECAL_nHitsRatio', 'HCAL_nHits', 'HCAL_ratioFirstLayerToSecondLayerE', 'HCAL_ratioFirstLayerToTotalE', 'HCALmomentX1', 'HCALmomentX2', 'HCALmomentX3', 'HCALmomentX4', 'HCALmomentX5', 'HCALmomentX6', 'HCALmomentY1', 'HCALmomentY2', 'HCALmomentY3', 'HCALmomentY4', 'HCALmomentY5', 'HCALmomentY6', 'HCALmomentZ1', 'HCALmomentZ2', 'HCALmomentZ3', 'HCALmomentZ4', 'HCALmomentZ5', 'HCALmomentZ6', 'conversion', 'energy', 'openingAngle', 'pdgID']
        self.assertEqual(list(h5.File(sample).keys()), fixed_keys)

    def test_nevents(self, sample):
        self.assertEqual(h5.File(sample)['ECAL'].shape[0], 10000)

    def test(self, sample):
        self.test_existence(sample)
        self.test_readable(sample)
        self.test_keys(sample)
        self.test_nevents(sample)


if __name__ == "__main__":
    '''python generated_sample_unit_tests.py <file>'''
    my_tests = UnitTests()
    my_tests.test(sys.argv[1])
