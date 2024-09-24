import SimpleITK as sitk
import numpy as np
import re

from pathlib import Path
from typing import List, Tuple, Optional, Dict

import pandas as pd

from mnts.io.data_formatting import batch_dicom2nii
from mnts.io import dicom2nii, pydicom_read_series
from mnts.mnts_logger import MNTSLogger
from mnts.utils.sequence_check import unify_mri_sequence_name

import tempfile
import pprint


def flatten_img(img: sitk.Image) -> sitk.Image:
    """Flattens the image using mean intensity projection.

    Performs a mean intensity projection along the largest dimension
    of the image.

    Args:
        img (sitk.Image):
            The input image to be flattened.

    Returns:
        sitk.Image:
            The flattened image after mean intensity projection.

    """
    # Get the spacing of the image dimensions
    im_shape = img.GetSpacing()
    # Determine the index of the largest dimension
    max_dim = np.argmax(im_shape)
    # Perform maximum intensity projection along the largest dimension
    im_flattened = sitk.MeanProjection(img, int(max_dim))
    return im_flattened


def flatten_dcm(folder: Path,
                id_globber: Optional[str] = r'[A-Za-z]*[0-9]+',
                num_workers: Optional[int] = 1) -> Dict[str, sitk.Image]:
    """
    Converts DICOM series in a folder to flattened NIfTI images.

    Processes DICOM series found in the specified folder, converts them
    to NIfTI format, and applies an average intensity projection to flatten
    the images. The resulting images are stored in a dictionary with
    extracted IDs as keys.

    Args:
        folder (Path):
            Path to the folder containing DICOM series to be processed.
        id_globber (Optional[str]):
            Optional regex pattern to match IDs from the filenames. Defaults
            to r'[A-Za-z]*[0-9]+'.
        num_workers (Optional[int]):
            Number of workers to use for processing. Defaults to 1.

    Returns:
        Dict[str, sitk.Image]:
            A dictionary mapping extracted IDs to the corresponding
            flattened SimpleITK images.

    Raises:
        FileNotFoundError: If the specified folder does not exist.
        RuntimeError: If an error occurs during DICOM to NIfTI conversion.
    """
    output = {}
    with (tempfile.TemporaryDirectory() as tmp_dir,
        MNTSLogger('flatten_dcm') as logger):
        logger.info(f"Working on {str(folder)}")
        # find all DCM series
        series = pydicom_read_series(folder)
        logger.info(f"Identified series: {pprint.pformat(series.keys())}")
        logger.info("Converting to NII")

        batch_dicom2nii(
            [pathlist[0].parent for pathlist in series.values()],
            tmp_dir,
            workers=num_workers,
            idglobber=id_globber,
            check_im_type=True,
            use_patient_id=True,
            add_scan_time=False,
            dump_meta_data=True
        )


        # Iterate over all .nii.gz files in the specified folder
        for nii_file in Path(tmp_dir).rglob("*.nii.gz"):
            logger.info(f"Flattening with MIP: {nii_file.name}")
            # Extract the ID from the file name using a regular expression
            id = re.search(id_globber, nii_file.name).group()

            # Read the NIfTI image file
            im = sitk.ReadImage(str(nii_file))

            im_flattened = flatten_img(im)

            # Store the flattened image in the output dictionary with the extracted ID as the key
            output[id] = im_flattened
    logger.debug(f"{output = }")
    return output


def make_datasheet(folder: Path, id_globber: Optional[str] = r'[A-Za-z]*[0-9]+') -> pd.DataFrame:
    """Creates a datasheet from NIfTI files in a folder.

    Iterates over NIfTI files in the specified folder, extracting
    information and compiling it into a pandas DataFrame.

    Args:
        folder (Path):
            Path to the folder containing NIfTI files.
        id_globber (Optional[str]):
            Regular expression pattern to extract IDs from filenames.
            Defaults to r'[A-Za-z]*[0-9]+'

    Returns:
        pd.DataFrame:
            A DataFrame where each column represents metadata extracted
            from a NIfTI file, with IDs and unified MRI sequence names.
            Each row is the file processed.

    Raises:
        ValueError: If no files match the specified pattern.
    """
    # Find all NIfTI files in the folder
    nii_files = folder.rglob("*.nii.gz")

    # Initialize a list to store each data row
    rows = []

    # Iterate over each file in the folder
    for r in folder.iterdir():
        # Extract the ID from the filename using a regular expression
        id = re.search(id_globber, r.name).group()

        # Get a dictionary of MRI sequence information
        _, row_dict = unify_mri_sequence_name(r.name, return_glob_dict=True, glob_techniques=True)

        # Create a pandas Series from the dictionary, naming it after the file
        row = pd.Series(row_dict, name=r.name)

        # Add the extracted ID to the Series
        row['Study Number'] = id

        # Append the Series to the list of rows
        rows.append(row)

    # Concatenate all data rows into a single DataFrame, rows are filenames
    df_out = pd.concat(rows, axis=1).T
    df_out.sort_index(inplace=True)
    df_out.sort_index(axis=1, inplace=True)
    return df_out.infer_objects()