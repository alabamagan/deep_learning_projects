import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import pandas as pd
import SimpleITK as sitk
from sequence_identifier.io.preprocessing import flatten_dcm, make_datasheet

class TestPreprocessing(unittest.TestCase):

    def test_flatten_dcm(self):
        folder = Path('./test_data/ExampleDCM')

        # Call the function with the test data folder
        result = flatten_dcm(folder)

        # Assertions to verify the output
        # Check if result is of the correct type
        self.assertIsInstance(result, dict)
        self.assertIsInstance(next(iter(result.values())), sitk.Image)

    def test_flatten_dcm_mpi(self):
        import multiprocessing as mpi
        folder = Path('./test_data/ExampleDCM')

        # Call the function with the test data folder
        result = flatten_dcm(folder, num_workers=mpi.cpu_count() // 2)

        # Assertions to verify the output
        # Check if result is of the correct type
        self.assertIsInstance(result, dict)
        self.assertIsInstance(next(iter(result.values())), sitk.Image)


class TestMakeDatasheet(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

        # Mock NIfTI files
        self.files = [
            ('001_T1W.nii.gz', '001'),
            ('002_T2W.nii.gz', '002'),
            ('003_DWI.nii.gz', '003')
        ]

        # Create mock files in the temporary directory
        for filename, _ in self.files:
            with open(Path(self.test_dir) / filename, 'w') as f:
                f.write("")

    def tearDown(self):
        # Remove the temporary directory after the test
        shutil.rmtree(self.test_dir)

    def test_make_datasheet(self):
        # Call the function
        folder = Path(self.test_dir)
        df = make_datasheet(folder)

        # Verify the DataFrame
        expected_data = {
            '001_T1W.nii.gz': {'Study Number': '001', 'weight': 'T1W', 'plane': None, 'contrast': False, 'fat-suppression': False, 'technique': ''},
            '002_T2W.nii.gz': {'Study Number': '002', 'weight': 'T2W', 'plane': None, 'contrast': False, 'fat-suppression': False, 'technique': ''},
            '003_DWI.nii.gz': {'Study Number': '003', 'weight': 'DWI', 'plane': None, 'contrast': False, 'fat-suppression': False, 'technique': ''}
        }

        expected_df = pd.DataFrame.from_dict(expected_data, orient='index')
        expected_df.sort_index(inplace=True)
        expected_df.sort_index(axis=1, inplace=True)

        pd.testing.assert_frame_equal(df, expected_df)