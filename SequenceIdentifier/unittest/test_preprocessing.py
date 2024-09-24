import unittest
from sequence_identifier.io.preprocessing import flatten_dcm
from pathlib import Path
import SimpleITK as sitk

class TestPreprocessing(unittest.TestCase):

    def test_flatten_dcm(self):
        folder = Path('./test_data/ExampleDCM')

        # Call the function with the test data folder
        result = flatten_dcm(folder)

        # Assertions to verify the output
        # Check if result is of the correct type
        self.assertIsInstance(result, dict)
        self.assertIsInstance(next(iter(result.values())), sitk.Image)

        # You can add more assertions based on expected behavior and results
        # For example, check if certain IDs are present in the result (if expected)
        # self.assertIn('expected_id', result)

