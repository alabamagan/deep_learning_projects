import unittest
from npc_segment.img_proc import *
import SimpleITK as sitk
import numpy as np

def create_test_image(filename, size, spacing):
    """Create a test image with the specified size and spacing."""
    image = sitk.Image(size, sitk.sitkFloat32)
    image.SetSpacing(spacing)
    sitk.WriteImage(image, filename)

class TestImgProc(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create temporary directories for input, reference, and output
        cls.input_dir = tempfile.mkdtemp()
        cls.reference_dir = tempfile.mkdtemp()
        cls.output_dir = tempfile.mkdtemp()

        # Create test images in input and reference directories
        cls.input_image_path = os.path.join(cls.input_dir, 'test_image.mha')
        cls.reference_image_path = os.path.join(cls.reference_dir, 'test_image.mha')

        create_test_image(cls.input_image_path, size=[50, 50, 50], spacing=[2.0, 2.0, 2.0])
        create_test_image(cls.reference_image_path, size=[100, 100, 100], spacing=[1.0, 1.0, 1.0])

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary directories
        shutil.rmtree(cls.input_dir)
        shutil.rmtree(cls.reference_dir)
        shutil.rmtree(cls.output_dir)

    def test_resample_images(self):
        # Assuming resample_images is your function to test
        resample_images(self.input_dir, self.reference_dir, self.output_dir)

        # Check if the resampled image exists in the output directory
        resampled_image_path = os.path.join(self.output_dir, 'test_image.mha')
        self.assertTrue(os.path.exists(resampled_image_path))

        # Load the resampled image and reference image
        resampled_image = sitk.ReadImage(resampled_image_path)
        reference_image = sitk.ReadImage(self.reference_image_path)

        # Check if the resampled image has the same size and spacing as the reference image
        self.assertEqual(resampled_image.GetSize(), reference_image.GetSize())
        self.assertEqual(resampled_image.GetSpacing(), reference_image.GetSpacing())
