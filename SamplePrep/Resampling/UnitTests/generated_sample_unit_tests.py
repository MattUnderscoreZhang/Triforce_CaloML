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
        fixed_keys = ['ECAL', 'HCAL', 'conversion', 'energy', 'openingAngle', 'pdgID']
        random_keys = ['ECAL', 'HCAL', 'conversion', 'energy', 'openingAngle', 'pdgID', 'recoEta', 'recoPhi', 'recoTheta']
        if "RandomAngle" in sample:
            self.assertEqual(list(h5.File(sample).keys()), random_keys)
        else:
            self.assertEqual(list(h5.File(sample).keys()), fixed_keys)

    def test_nevents(self, sample):
        self.assertEqual(h5.File(sample)['ECAL'].shape[0], 10000)

    def test(self, sample):
        print(f"Testing {sample}")
        self.test_existence(sample)
        self.test_readable(sample)
        self.test_keys(sample)
        self.test_nevents(sample)


if __name__ == "__main__":
    '''python generated_sample_unit_tests.py <file>'''
    my_tests = UnitTests()
    my_tests.test(sys.argv[1])
