from typing import *
import numpy as np
import SimpleITK as sitk
import streamlit as st


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
        st.success("All metadata matches: spacing, direction, and origin.")
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


def create_overlay_image(
        image_path: str,
        attn_path: str,
        window_range: Tuple[int, int],
        attn_threshold: Tuple[int, int],
        alpha: float,
        head_settings: Dict[str, Any],
        grid_cols: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """Create an overlay image by combining the original image with attention map.

    Args:
        image_path: Path to the original image
        attn_path: Path to the attention map
        window_range: Image window range (lower, upper)
        attn_threshold: Attention map threshold range (min, max)
        alpha: Opacity of the attention map overlay
        head_settings: Attention head settings containing:
            - use_max: Whether to use maximum value
            - use_avg: Whether to use average value
            - head_idx: Selected attention head index
        grid_cols: Number of columns in the grid layout

    Returns:
        Tuple containing:
        - overlayed: Final overlayed image
        - attn_map_target: Processed attention map (for histogram)

    Raises:
        FileNotFoundError: If image files don't exist
        RuntimeError: If there's an error during image processing
    """
    try:
        # Load images
        image = sitk.ReadImage(str(image_path))
        attn_map_ori = sitk.ReadImage(str(attn_path))

        # Orient images to standard orientation
        image = sitk.DICOMOrient(image, 'LPS')
        attn_map = sitk.DICOMOrient(attn_map_ori, 'LPS')

        # Resample image to match attention map dimensions
        image = sitk.Resample(image, attn_map)

        # Verify metadata matches
        same_spacial = check_image_metadata(image, attn_map)
        if not same_spacial:
            st.warning("Resampling...")
            attn_map = sitk.Resample(attn_map, image)

        # Convert to numpy arrays for processing
        image = sitk.GetArrayFromImage(image)
        np_attn_map = sitk.GetArrayFromImage(attn_map)

        # Update number of attention heads in session state
        num_heads = np_attn_map.shape[-1]
        st.session_state['num_heads'] = num_heads

        # Select attention map based on user settings
        if head_settings['use_max']:
            attn_map_target = np_attn_map.max(axis=-1)
        elif head_settings['use_avg']:
            attn_map_target = np_attn_map.mean(axis=-1)
        else:
            attn_map_target = np_attn_map[..., head_settings['head_idx']]

        # Create visualization grid
        image = rescale_intensity(
            make_grid(image, ncols=grid_cols),
            lower=window_range[0],
            upper=window_range[1]
        )
        attn_map_grid = make_grid(attn_map_target, ncols=grid_cols, normalize=False)

        # Normalize attention map to 0-1 range
        attn_min_norm = attn_threshold[0] / 255.0
        attn_max_norm = attn_threshold[1] / 255.0

        # Get attention map value range
        attn_min_val = np.min(attn_map_grid)
        attn_max_val = np.max(attn_map_grid)

        # Handle case where all attention values are the same
        if attn_min_val >= attn_max_val:
            st.warning("All attention values are the same, using default visualization")
            return image, attn_map_target

        # Apply normalization and thresholding
        attn_map_norm = (attn_map_grid - attn_min_val) / (attn_max_val - attn_min_val)
        attn_map_thresholded = np.clip(
            (attn_map_norm - attn_min_norm) / (attn_max_norm - attn_min_norm + 1e-8),
            0, 1
        )

        # Create final visualization
        colored_attn = apply_jet_colormap(attn_map_thresholded)
        overlayed = overlay_images(image, colored_attn, alpha)

        return overlayed, attn_map_target

    except FileNotFoundError as e:
        st.error(f"Image file not found: {e}")
        raise
    except Exception as e:
        st.error(f"Error processing images: {e}")
        raise RuntimeError(f"Failed to create overlay image: {e}")


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
        # Implementation here
        pass
    except Exception as e:
        st.error(f"Failed to save image: {e}")
        raise