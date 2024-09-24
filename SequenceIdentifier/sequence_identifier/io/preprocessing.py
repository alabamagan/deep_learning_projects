import SimpleITK as sitk
import numpy as np
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from mnts.io.data_formatting import Dcm2NiiConverter
from mnts.io import dicom2nii, pydicom_read_series
from mnts.mnts_logger import MNTSLogger
import tempfile
import pprint

def flatten_dcm(folder: Path) -> Dict[str, sitk.Image]:
    """
    Converts DICOM series in a folder to flattened NIfTI images.

    Processes DICOM series found in the specified folder, converts them
    to NIfTI format, and applies a maximum intensity projection to flatten
    the images. The resulting images are stored in a dictionary with
    extracted IDs as keys.

    Args:
        folder (Path):
            Path to the folder containing DICOM series to be processed.

    Returns:
        Dict[str, sitk.Image]:
            A dictionary mapping extracted IDs to the corresponding
            flattened SimpleITK images.

    Raises:
        FileNotFoundError: If the specified folder does not exist.
        RuntimeError: If an error occurs during DICOM to NIfTI conversion.
    """

    output = {}
    id_globber = '\w*\d+'
    with (tempfile.TemporaryDirectory() as tmp_dir,
        MNTSLogger('flatten_dcm') as logger):
        logger.info(f"Working on {str(folder)}")
        # find all DCM series
        series = pydicom_read_series(folder)
        logger.info(f"Identified series: {pprint.pformat(series.keys())}")
        logger.info("Converting to NII")

        for series_id in series:
            converter = Dcm2NiiConverter(
                series[series_id][0].parent,
                tmp_dir,
                idglobber=id_globber,
                check_im_type=True,
                use_patient_id=True,
                add_scan_time=False,
                dump_meta_data=True
            )
            converter.Execute()

        # Iterate over all .nii.gz files in the specified folder
        for nii_file in Path(tmp_dir).rglob("*.nii.gz"):
            logger.info(f"Flattening with MIP: {nii_file.name}")
            # Extract the ID from the file name using a regular expression
            id = re.search(id_globber, nii_file.name).group()

            # Read the NIfTI image file
            im = sitk.ReadImage(str(nii_file))

            # Get the spacing of the image dimensions
            im_shape = im.GetSpacing()

            # Determine the index of the largest dimension
            max_dim = np.argmax(im_shape)

            # Perform maximum intensity projection along the largest dimension
            im_flattened = sitk.MaximumProjection(im, int(max_dim))

            # Store the flattened image in the output dictionary with the extracted ID as the key
            output[id] = im_flattened
    logger.debug(f"{output = }")
    return output



