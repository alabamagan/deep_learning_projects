import SimpleITK as sitk
from tqdm.auto import tqdm
from mnts.mnts_logger import MNTSLogger
import cv2
import numpy as np

def mdprint(s: str):
    try:
        logger = MNTSLogger.get_global_logger()
        logger.info(s)
    except:
        print(s)

def verify_spacing_and_origin(out_mri_dir, out_seg_dir, total_reader_test_size):
    """
    Verify and fix spacing/origin mismatches between MRI and segmentation files

    Args:
        out_mri_dir (Path): Directory containing MRI files
        out_seg_dir (Path): Directory containing segmentation files
        total_reader_test_size (int): Total number of files to process

    Returns:
        pd.DataFrame: DataFrame containing spacing and origin verification results
    """
    # check if spacing of the segmentation and the mri are the same
    spacing_info = []
    for fn_mri, fn_seg in tqdm(zip(out_mri_dir.rglob('*.nii.gz'), out_seg_dir.rglob('*.nii.gz')),
                               total=total_reader_test_size):
        # Create readers for both MRI and segmentation
        mri_reader = sitk.ImageFileReader()
        seg_reader = sitk.ImageFileReader()

        # Set the file names
        mri_reader.SetFileName(str(fn_mri))
        seg_reader.SetFileName(str(fn_seg))

        # Read only the image information (header)
        mri_reader.ReadImageInformation()
        seg_reader.ReadImageInformation()

        # Get the spacing information
        mri_spacing = mri_reader.GetSpacing()
        seg_spacing = seg_reader.GetSpacing()

        # Get the origin information
        mri_origin = mri_reader.GetOrigin()
        seg_origin = seg_reader.GetOrigin()

        # Store spacing and origin info in a dict
        spacing_info.append({
            'filename': fn_mri.name,
            'mri_spacing': mri_spacing,
            'seg_spacing': seg_spacing,
            'mri_origin': mri_origin,
            'seg_origin': seg_origin,
            'spacing_match': np.allclose(mri_spacing, seg_spacing, rtol=0, atol=0.001),
            'origin_match': np.allclose(mri_origin, seg_origin, rtol=0, atol=0.001)
        })

    # Create dataframe with spacing information
    mdprint("# Spacing and Origin Verification")
    df_spacing = pd.DataFrame(spacing_info)
    display(df_spacing)

    # Check spacing matches
    if not df_spacing['spacing_match'].all():
        mdprint("## Spacing Mismatch")
        display(df_spacing[df_spacing['spacing_match'] == False])

        # For mismatched ones, resample the segmentation to match MRI spacing
        mismatched_files = df_spacing[~df_spacing['spacing_match']]
        for _, row in mismatched_files.iterrows():
            fn_mri = out_mri_dir / row['filename']
            fn_seg = out_seg_dir / row['filename']

            # Read the images
            mri = sitk.ReadImage(str(fn_mri), imageIO="NiftiImageIO")
            seg = sitk.ReadImage(str(fn_seg), imageIO="NiftiImageIO")

            # Setup resampling filter
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(mri)  # Use MRI as reference for spacing/size
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Use nearest neighbor for label images

            # Resample segmentation to match MRI spacing
            resampled_seg = resampler.Execute(seg)

            # Save resampled segmentation
            sitk.WriteImage(resampled_seg, str(fn_seg))

    else:
        mdprint("All spacing matches")

    # Check origin matches
    if not df_spacing['origin_match'].all():
        mdprint("## Origin Mismatch")
        display(df_spacing[df_spacing['origin_match'] == False])
    else:
        mdprint("All origins match")

    return df_spacing


def draw_grid_text(img, nrows, ncols, texts, text_coords, text_kwargs=None):
    """Draw text on image divided into grids

    Args:
        img (np.ndarray): Input image
        nrows (int): Number of rows to divide
        ncols (int): Number of columns to divide
        texts (list): List of strings to draw. Each string can contain newlines
        text_coords (list): List of (row,col) coordinates for each text
        text_kwargs (dict, optional): Text drawing parameters for cv2.putText. Defaults to None.

    Returns:
        np.ndarray: Image with text drawn in grids

    Raises:
        ValueError: If input parameters are invalid
    """
    # Input validation
    if img is None or len(img.shape) < 2:
        raise ValueError("Invalid image input")
    if not isinstance(nrows, int) or not isinstance(ncols, int) or nrows <= 0 or ncols <= 0:
        raise ValueError("nrows and ncols must be positive integers")
    if len(texts) != len(text_coords):
        raise ValueError("Number of texts must match number of coordinates")

    # Default text parameters
    default_text_kwargs = {
        'fontFace': cv2.FONT_HERSHEY_SIMPLEX,
        'fontScale': 1,
        'color': (255, 255, 255),
        'thickness': 2,
        'lineType': cv2.LINE_AA
    }
    if text_kwargs is not None:
        default_text_kwargs.update(text_kwargs)
    text_kwargs = default_text_kwargs

    h, w = img.shape[:2]
    cell_h, cell_w = h // nrows, w // ncols

    # Constants
    LINE_SPACING_RATIO = 0.1  # Percentage of cell height for line spacing

    # Make a copy to avoid modifying original
    result = img.copy()

    # Filter text parameters once
    text_size_kwargs = {k: text_kwargs[k] for k in ['fontFace', 'fontScale', 'thickness']}

    for text, (row, col) in zip(texts, text_coords):
        # Validate coordinates
        if not (0 <= row < nrows and 0 <= col < ncols):
            continue  # Skip invalid coordinates

        # Get cell boundaries
        x1 = col * cell_w
        y1 = row * cell_h

        # Handle empty text
        if not text.strip():
            continue

        # Split text into lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            continue

        # Get text sizes
        line_sizes = [cv2.getTextSize(line, **text_size_kwargs)[0] for line in lines]

        # Calculate vertical positions
        total_text_h = sum(h for w, h in line_sizes)
        line_spacing = int(cell_h * LINE_SPACING_RATIO)
        y_start = y1 + (cell_h - (total_text_h + (len(lines) - 1) * line_spacing)) // 2

        # Draw each line
        for line, (text_w, text_h) in zip(lines, line_sizes):
            # Ensure text fits within cell width
            scale = min(1.0, (cell_w * 0.9) / max(text_w, 1))
            if scale < 1.0:
                text_kwargs['fontScale'] *= scale
                # Recalculate text size with new scale
                text_w, text_h = cv2.getTextSize(line, **text_size_kwargs)[0]

            x = x1 + (cell_w - text_w) // 2  # Center horizontally
            y = max(y1, min(y1 + cell_h, y_start + text_h))  # Ensure y is within cell

            cv2.putText(result, line, (x, y), **text_kwargs)
            y_start += text_h + line_spacing

            # Reset font scale if it was modified
            if scale < 1.0:
                text_kwargs['fontScale'] /= scale

    return result