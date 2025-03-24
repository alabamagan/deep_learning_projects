from typing import *
import numpy as np
import SimpleITK as sitk
import rich
import streamlit as st
import logging
from visualization import make_grid, apply_jet_colormap, overlay_images
import cv2
from pathlib import Path
import tempfile

logger = logging.getLogger('streamlit.image_utils')


def check_image_metadata(img1: sitk.Image, img2: sitk.Image, tolerance=1e-3) -> bool:
    """Check if the metadata of two images match.

    Args:
        img1: First image
        img2: Second image
        tolerance: Tolerance value for floating point comparisons

    Returns:
        bool: Whether the metadata matches
    """
    spacing_match = np.all(np.isclose(img1.GetSpacing(), img2.GetSpacing(), atol=tolerance))
    direction_match = np.all(np.isclose(img1.GetDirection(), img2.GetDirection(), atol=tolerance))
    origin_match = np.all(np.isclose(img1.GetOrigin(), img2.GetOrigin(), atol=tolerance))
    size_match = np.array_equal(img1.GetSize(), img2.GetSize())

    if all([spacing_match, direction_match, origin_match, size_match]):
        logger.info("All metadata matches: spacing, direction, and origin.")
        return True
    else:
        if not spacing_match:
            st.error(f"Spacing does not match: {img1.GetSpacing() = } | {img2.GetSpacing() = }")
        if not direction_match:
            st.error(f"Direction does not match: {img1.GetDirection() = } | {img2.GetDirection() = }")
        if not origin_match:
            st.error(f"Origin does not match: {img1.GetOrigin() = } | {img2.GetOrigin() = }")
        if not size_match:
            st.error(f"Size does not match: {img1.GetSize() = } | {img2.GetSize() = }")
        return False


def rescale_intensity(image, lower=25, upper=99):
    """Rescale the intensity of an image to map the 5th and 95th percentiles to 0 and 255."""
    lower, upper = np.percentile(image, [lower, upper])
    if lower == upper:
        raise ValueError("Min point and Max point are the same")
    rescaled_image = np.clip((image - lower) / (upper - lower) * 255, 0, 255)
    return rescaled_image.astype(np.uint8)


@st.cache_data
def create_overlay_image(image_path: str,
                         window_range: Tuple[int, int],
                         attn_path: Optional[str] = None,
                         attn_threshold: Optional[Tuple[int, int]] = None,
                         alpha: Optional[float] = None,
                         head_settings: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Create an overlay image display with optional attention map and segmentation

    Args:
        image_path: str
            Path to the original image
        window_range: Tuple[int, int]
            Image window range (lower, upper)
        attn_path: Optional[str]
            Path to the attention map. If None, only original image will be displayed.
        attn_threshold: Optional[Tuple[int, int]]
            Attention map threshold range (min, max). Required if attn_path is provided.
        alpha: Optional[float]
            Opacity of attention map. Required if attn_path is provided.
        head_settings: Optional[Dict[str, Any]]
            Settings for attention heads. Required if attn_path is provided.

    Returns:
        Tuple containing:
        - overlayed: Final overlaid image
        - attn_map_target: Processed attention map (for histogram) or None if no attention map
    """
    # Validate required parameters for attention map
    if any(x is None for x in [attn_threshold, alpha, head_settings]):
        raise ValueError("attn_threshold, alpha, and head_settings are required when attn_path is provided")

    # Load and preprocess image
    image = sitk.ReadImage(str(image_path))
    image = sitk.DICOMOrient(image, 'LPS')

    # Convert to numpy array and create grid
    np_image = sitk.GetArrayFromImage(image)
    ncols = 5
    grid_image = rescale_intensity(make_grid(np_image, ncols=ncols),
                                   lower=window_range[0],
                                   upper=window_range[1])

    # If no attention map is provided, return the original image
    if attn_path is None:
        return image, None

    # Load and process attention map
    attn_map_ori = sitk.ReadImage(str(attn_path))
    attn_map = sitk.DICOMOrient(attn_map_ori, 'LPS')
    attn_map = sitk.Resample(attn_map, image)

    # Check metadata match
    same_spacial = check_image_metadata(image, attn_map)
    if not same_spacial:
        st.warning("Resampling...")
        attn_map = sitk.Resample(attn_map, image)

    # Convert to numpy array
    np_attn_map = sitk.GetArrayFromImage(attn_map)

    # Get number of attention heads
    num_heads = np_attn_map.shape[-1]
    st.session_state['num_heads'] = num_heads

    # Select attention map based on settings
    if head_settings['use_max']:
        attn_map_target = np_attn_map.max(axis=-1)
    elif head_settings['use_avg']:
        attn_map_target = np_attn_map.mean(axis=-1)
    else:
        attn_map_target = np_attn_map[..., head_settings['head_idx']]

    # Create grid for attention map
    attn_map_grid = make_grid(attn_map_target, ncols=ncols, normalize=False)

    # Normalize attention map
    attn_min_norm = attn_threshold[0] / 255.0
    attn_max_norm = attn_threshold[1] / 255.0

    # Scale attention map
    attn_min_val = np.min(attn_map_grid)
    attn_max_val = np.max(attn_map_grid)

    if attn_min_val >= attn_max_val:
        st.error("All attention values are the same")
        st.stop()

    # Normalize and threshold attention map
    attn_map_norm = (attn_map_grid - attn_min_val) / (attn_max_val - attn_min_val)
    attn_map_thresholded = np.clip(
        (attn_map_norm - attn_min_norm) / (attn_max_norm - attn_min_norm + 1e-8),
        0, 1
    )

    # Create final overlay
    colored_attn = apply_jet_colormap(attn_map_thresholded)
    overlayed = overlay_images(grid_image, colored_attn, alpha)

    return overlayed, attn_map_target, image


def save_image(image_path: str, min_val: float, max_val: float) -> str:
    """Save image to temporary directory with specified window settings.

    Args:
        image_path: Path to the source image
        min_val: Minimum window value
        max_val: Maximum window value

    Returns:
        str: Path to the saved image

    Raises:
        Exception: If image saving fails
    """
    try:
        # Read and process the image
        image = sitk.ReadImage(str(image_path))
        image = sitk.DICOMOrient(image, 'LPS')
        image = sitk.GetArrayFromImage(image)

        # Create grid and rescale intensity
        ncols = 5
        image = rescale_intensity(make_grid(image, ncols=ncols),
                                  lower=min_val,
                                  upper=max_val)

        # Create temporary directory if it doesn't exist
        temp_dir = Path(tempfile.gettempdir()) / "rAIdiologist"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique filename
        image_name = Path(image_path).stem
        output_path = temp_dir / f"{image_name}_processed.png"

        # Save the image
        cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        return str(output_path)

    except Exception as e:
        st.error(f"Failed to save image: {e}")
        raise