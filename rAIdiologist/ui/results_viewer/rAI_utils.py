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


def get_final_prediction(prob: float, segment: sitk.Image, tolerance: float = 0.1) -> int:
    r"""Classify a nasopharyngeal lesion into one of four diagnostic categories.

    Combines the DL malignancy probability with the physical size of the
    segmented region.  Small lesions (< 0.5 cm³) are classified by the DL
    score alone, while larger lesions include an additional uncertainty band
    around the decision threshold controlled by *tolerance*.

    Default thresholds were derived from ``analysis_v2.ipynb``.

    .. mermaid::
        graph TD
          A{Volume < 0.5 cm³?}
          A -->|yes| B{DL score < 0.5?}
          A -->|no|  C{abs DL score − 0.5 < tolerance?}
          B -->|yes| Norm([3: Normal nasopharynx])
          B -->|no|  Un1([4: Undetermined])
          C -->|yes| Un2([4: Undetermined])
          C -->|no|  D{DL score < 0.5?}
          D -->|yes| Benign([2: Benign hyperplasia])
          D -->|no|  NPC([1: NPC])

    Args:
        prob (float): DL malignancy probability in [0, 1].  Values closer to 1
            indicate higher likelihood of NPC.
        tolerance (float): Half-width of the uncertainty band around the 0.5
            decision boundary (applies to large-volume lesions only).  Predictions
            whose probability falls within ``[0.5 − tolerance, 0.5 + tolerance)``
            are reported as Undetermined (4).
        segment (sitk.Image): Binary or label segmentation mask in the
            resampled image space (isotropic 1 mm spacing assumed for physical
            volume calculation).

    Returns:
        int: Diagnostic category.

        +-------+---------------------------+------------------------------------------+
        | Value | Label                     | Condition                                |
        +=======+===========================+==========================================+
        | 1     | NPC                       | Large volume **and** prob ≥ 0.5 + tol    |
        +-------+---------------------------+------------------------------------------+
        | 2     | Benign hyperplasia        | Large volume **and** prob < 0.5 − tol    |
        +-------+---------------------------+------------------------------------------+
        | 3     | Normal nasopharynx        | Small volume **and** prob < 0.5          |
        +-------+---------------------------+------------------------------------------+
        | 4     | Undetermined              | Small volume + prob ≥ 0.5, **or**        |
        |       |                           | large volume + prob in uncertainty band  |
        +-------+---------------------------+------------------------------------------+
    """

    # Volume threshold: 0.5 cm³ = 500 mm³
    VOL_THR = 500.0   # mm³
    DL_THR  = 0.5

    label_statistics = sitk.LabelShapeStatisticsImageFilter()
    if isinstance(segment, str):
        segment = sitk.ReadImage(segment)
    label_statistics.Execute(segment > 0)

    volume_mm3 = 0.0
    if label_statistics.GetNumberOfLabels() > 0:
        volume_mm3 = label_statistics.GetPhysicalSize(1)

    small_volume = volume_mm3 < VOL_THR

    if small_volume:
        return 3 if prob < DL_THR else 4           # Normal  /  Undetermined

    # Large-volume path: check uncertainty band first
    if abs(prob - DL_THR) < tolerance:
        return 4                                    # Undetermined (borderline)

    return 1 if prob >= DL_THR else 2              # NPC  /  Benign hyperplasia


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