'''Run with pytest. Asserts that all new files have the correct features.'''

import glob
import h5py as h5


def test_features():
    paths = ["/public/data/calo/FixedAngle/ATLAS/*/*.h5", "/public/data/calo/FixedAngle/CMS/*/*.h5", "/public/data/calo/RandomAngle/ATLAS/*/*.h5", "/public/data/calo/RandomAngle/CMS/*/*.h5"]

    correct_fixed_keys = ['ECAL', 'ECAL_E', 'ECAL_nHits', 'ECAL_ratioFirstLayerToSecondLayerE', 'ECAL_ratioFirstLayerToTotalE', 'ECALmomentX1', 'ECALmomentX2', 'ECALmomentX3', 'ECALmomentX4', 'ECALmomentX5', 'ECALmomentX6', 'ECALmomentY1', 'ECALmomentY2', 'ECALmomentY3', 'ECALmomentY4', 'ECALmomentY5', 'ECALmomentY6', 'ECALmomentZ1', 'ECALmomentZ2', 'ECALmomentZ3', 'ECALmomentZ4', 'ECALmomentZ5', 'ECALmomentZ6', 'HCAL', 'HCAL_E', 'HCAL_ECAL_ERatio', 'HCAL_ECAL_nHitsRatio', 'HCAL_nHits', 'HCAL_ratioFirstLayerToSecondLayerE', 'HCAL_ratioFirstLayerToTotalE', 'HCALmomentX1', 'HCALmomentX2', 'HCALmomentX3', 'HCALmomentX4', 'HCALmomentX5', 'HCALmomentX6', 'HCALmomentY1', 'HCALmomentY2', 'HCALmomentY3', 'HCALmomentY4', 'HCALmomentY5', 'HCALmomentY6', 'HCALmomentZ1', 'HCALmomentZ2', 'HCALmomentZ3', 'HCALmomentZ4', 'HCALmomentZ5', 'HCALmomentZ6', 'R9', 'conversion', 'energy', 'openingAngle', 'pdgID']

    correct_random_keys = ['ECAL', 'ECAL_E', 'ECAL_nHits', 'ECAL_ratioFirstLayerToSecondLayerE', 'ECAL_ratioFirstLayerToTotalE', 'ECALmomentX1', 'ECALmomentX2', 'ECALmomentX3', 'ECALmomentX4', 'ECALmomentX5', 'ECALmomentX6', 'ECALmomentY1', 'ECALmomentY2', 'ECALmomentY3', 'ECALmomentY4', 'ECALmomentY5', 'ECALmomentY6', 'ECALmomentZ1', 'ECALmomentZ2', 'ECALmomentZ3', 'ECALmomentZ4', 'ECALmomentZ5', 'ECALmomentZ6', 'HCAL', 'HCAL_E', 'HCAL_ECAL_ERatio', 'HCAL_ECAL_nHitsRatio', 'HCAL_nHits', 'HCAL_ratioFirstLayerToSecondLayerE', 'HCAL_ratioFirstLayerToTotalE', 'HCALmomentX1', 'HCALmomentX2', 'HCALmomentX3', 'HCALmomentX4', 'HCALmomentX5', 'HCALmomentX6', 'HCALmomentY1', 'HCALmomentY2', 'HCALmomentY3', 'HCALmomentY4', 'HCALmomentY5', 'HCALmomentY6', 'HCALmomentZ1', 'HCALmomentZ2', 'HCALmomentZ3', 'HCALmomentZ4', 'HCALmomentZ5', 'HCALmomentZ6', 'R9', 'conversion', 'energy', 'openingAngle', 'pdgID', 'recoEta', 'recoPhi', 'recoTheta']

    for path in paths:
        files = glob.glob(path)
        for one_file in files:
            data = h5.File(one_file)
            if "Random" in one_file:
                assert(list(data.keys()) == correct_random_keys), one_file
            else:
                assert(list(data.keys()) == correct_fixed_keys), one_file
