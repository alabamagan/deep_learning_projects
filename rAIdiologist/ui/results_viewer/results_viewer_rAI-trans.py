from pathlib import Path
from genericpath import isfile
from multiprocessing import Value
import sys
import re
from turtle import onrelease
import SimpleITK as sitk
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from visualization import *
from pathlib import Path
from mnts.utils import get_fnames_by_IDs, get_unique_IDs
from image_utils import create_overlay_image, check_image_metadata, save_image

import streamlit as st
from pprint import pprint, pformat
import plotly.express as px
import numpy as np
import json

from typing import *
import logging, time
import pydantic
import rich
from rich.logging import RichHandler
from rich.traceback import install
install()


class Configuration(pydantic.BaseModel):
    # Use ClassVar to mark class variables
    state_file: ClassVar[str] = ".session_state.json"  # File path to save/load the session state
    mapper: ClassVar[Dict[str, str]] = {
        'Image Directory': 'IMAGE_DIR',  # File path holding both predictions and attentions
        'Prediction CSV Directory': 'CSV_DIR',
        'ID Globber Regex': 'ID_GLOBBER'
    }
    GRID_COLS: ClassVar[int] = 5
    DEFAULT_WINDOW_RANGE: ClassVar[Tuple[int, int]] = (25, 99)
    DEFAULT_ATTN_THRESHOLD: ClassVar[Tuple[int, int]] = (15, 55)
    DEFAULT_OPACITY: ClassVar[float] = 0.5
    ATTN_MIN_VALUE: ClassVar[int] = 0
    ATTN_MAX_VALUE: ClassVar[int] = 255
    HIST_LOWER_PERCENTILE: ClassVar[int] = 2
    HIST_UPPER_PERCENTILE: ClassVar[int] = 98

    # File paths - instance variables
    IMAGE_DIR: str = '.'
    CSV_DIR: str = 'results.csv'
    ID_GLOBBER: str = r"\w{0,5}\d+"

    def save_to_json(self, filename: str):
        """Save the configuration to a JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump(self.model_dump(),
                          f)  # Convert model to dictionary using dict() method, then save using json.dump
                logger.info(f"Configuration saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

    @classmethod
    def load_from_json(cls, filename: str) -> 'Configuration':
        """Load the configuration from a JSON file."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)  # Load JSON data from file
                for k, v in data.items():
                    setattr(cls, k, v)
                return cls(**data)  # Use unpacking operator to convert dictionary to keyword arguments
        except Exception as e:
            # If loading fails, return default configuration
            logger.error(f"Failed to load configuration: {e}")
            return cls()


st.set_page_config(layout="wide")
st.write("# NPC Screening Results View")
st.write(
    """
    This is a small UI that is used to view the prediction results and the self attention the transformer is giving.
    You should have the results prepared in the  
    
    # Results Viewer (results_viewer_rAI-trans.py) Overview
    
    This Streamlit application is an interactive tool for visualizing and analyzing rAIdiologist predictions and transformer self-attention mechanisms.
    
    ## Main Features
    
    - **Image and Attention Map Visualization**: Displays medical images with overlaid transformer attention maps
    - **Prediction Analysis**: Shows model prediction outputs compared to ground truth labels
    - **Filtering Options**: Filter cases by true positives (TP), true negatives (TN), false positives (FP), or false negatives (FN)
    - **Flexible Attention Visualization**:
      - Select specific attention heads
      - Use maximum or average view across heads
      - Adjust window range and attention thresholds
      - Control attention map transparency
    """)
with st.expander("Technical Details"):
    st.write(
        """
        
        ## Technical Details
        
        - **Data Loading**: Loads image and attention map pairs from specified directories
        - **Statistical Analysis**: Displays metrics like accuracy, sensitivity, and specificity
        - **Interactive Elements**:
          - Intuitive UI controls for display settings
          - Navigation buttons for browsing cases
          - Histogram display of attention value distribution
          - Image saving functionality
        
        ## File Structure Example
        
        ```
        project/
        ├── results_viewer_rAI-trans.py    # Main application file
        ├── data/
        │   ├── images/                    # Original medical images
        │   ├── attention_maps/            # Generated attention maps
        │   └── predictions.csv            # Model prediction results
        ├── output/
        │   └── saved_visualizations/      # Saved overlay images
        └── utils/
            └── image_processing.py        # Helper functions for image manipulation
        ```
        
        ## Display Settings
               
        | Category | Feature | Description |
        |----------|---------|-------------|
        | **Display Settings** | **Window Level Controls** | |
        | | Window Width | Controls the contrast range |
        | | Window Center | Controls the brightness midpoint |
        | | **Attention Map Settings** | |
        | | Opacity | Controls the transparency of the attention overlay |
        | | Threshold | Sets minimum value for attention to be displayed |
        | | Colormap | Changes the color scheme of the attention visualization |
        | | Attention Head Selection | Choose which transformer attention head to display |
        | | **Display Mode Options** | |
        | | Raw Image | Shows only the original medical image |
        | | Attention Only | Shows only the attention map |
        | | Overlay | Combines both with adjustable parameters |
        | **Visualizations** | **Attention Histogram** | |
        | | X-axis | Attention intensity values |
        | | Y-axis | Frequency/count of pixels at each intensity |
        | | Vertical line | Current threshold setting |
        | | **Performance Metrics Panel** | |
        | | Confusion Matrix | Visualization of model prediction results |
        | | ROC Curve | Shows model performance |
        | | Precision-Recall Curve | Evaluates model precision |
        | | **Comparison View** | |
        | | Original Image | Shows unprocessed image |
        | | Attention Map | Shows attention distribution |
        | | Overlaid Result | Shows combined view |
                This tool is particularly useful for in-depth analysis of model attention mechanisms, understanding which image 
                regions the model focuses on when making diagnostic decisions, and evaluating model performance and explainability.
                """
    )

# -- inistilize states
st.session_state['last_confirmation'] = st.session_state.get("last_confirmation", False)


# -- Add rich handler if it's not already there:
def setup_logger(logger):
    # Check if the RichHandler is already added
    if not any(isinstance(handler, RichHandler) for handler in logger.handlers):
        # Remove all existing handlers
        for handler in logger.handlers:
            logger.removeHandler(handler)

        # Add RichHandler if it's not present
        rich_handler = RichHandler(console=False, rich_tracebacks=True, tracebacks_show_locals=True,
                                   locals_max_length=20)
        logger.addHandler(rich_handler)
        logger.setLevel(logging.INFO)  # Set the logging level if needed

        # Log a test message
        logger.info("Logger setup complete with RichHandler.")

    return logger


# * Adding this handler to streamlit
# First remove the error message in streamlit by default
logger = st.logger.get_logger("streamlit.error_util")
for handler in logger.handlers:
    logger.removeHandler(handler)
# Setup the logger
logger = st.logger.get_logger("App")
setup_logger(logger)


# Introduce my own error handling
def set_global_exception_handler(f):
    import sys
    error_util = sys.modules["streamlit.error_util"]
    error_util.handle_uncaught_app_exception = f


set_global_exception_handler(logger.error)


def _exception_hook(exctype, value, traceback):
    """Custom exception hook for logging uncaught exceptions."""
    if issubclass(exctype, KeyboardInterrupt):
        sys.__excepthook__(exctype, value, traceback)
        return
    logger.error("Uncaught exception", exc_info=(exctype, value, traceback))


sys.excepthook = _exception_hook


# -- Setup style
# Load the CSS file
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


load_css("./style.css")


# -- Define some useful functions
@st.cache_data
def load_image_attention_pairs(img_dir: Path, id_globber: str = r"\w+\d+"):
    """Load and pair image files with their corresponding attention maps
    Both files should be under img_dir and share the same ID pattern
    Images have 'image' in filename, attention maps have 'pb_pred'"""

    # Glob all nifti files
    all_files = list(img_dir.rglob("*nii.gz"))

    # Separate into images and attention maps
    image_files = {re.search(id_globber, f.name).group(): f
                   for f in all_files if 'image' in f.name.lower()}
    attn_files = {re.search(id_globber, f.name).group(): f
                  for f in all_files if 'pb_pred' in f.name.lower()}

    # Find IDs that have both image and attention map
    intersection = list(set(image_files.keys()) & set(attn_files.keys()))
    intersection.sort()

    # Create pairs dictionary
    paired = {sid: (image_files[sid], attn_files[sid]) for sid in intersection}
    return paired


def check_image_metadata(img1: sitk.Image, img2: sitk.Image, tolerance=1e-3):
    r"""This checks the meta information of the image and the segmentation to make sure they are the same."""
    # Check metadata with tolerance
    spacing_match = np.all(np.isclose(img1.GetSpacing(), img2.GetSpacing(), atol=tolerance))
    direction_match = np.all(np.isclose(img1.GetDirection(), img2.GetDirection(), atol=tolerance))
    origin_match = np.all(np.isclose(img1.GetOrigin(), img2.GetOrigin(), atol=tolerance))
    size_match = np.array_equal(img1.GetSize(), img2.GetSize())

    if all([spacing_match, direction_match, origin_match, size_match]):
        st.success("All metadata matches: spacing, direction, and origin.")
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


def build_configurations():
    conf_instance = Configuration.load_from_json(
        Configuration.state_file
    )
    try:
        loaded_state = conf_instance.dump()
    except:
        loaded_state = {}

    var_mapping = Configuration.mapper
    for k, v in var_mapping.items():
        setattr(st.session_state, v.lower(), getattr(conf_instance, v))

    with st.expander('Configurations', expanded=st.session_state.get('require_setup', True)):
        st.write("### Instructions")
        st.write("""
        - Insert the directories in absolute format or relative to the streamlit file
        - The ID globber is the regex that will be used to match the ID of the images and segmentations
        """)

        for k, v in var_mapping.items():
            # Get type of values
            var_type = type(getattr(conf_instance, v))

            # Build the fields for interactions
            if var_type == str:
                if 'DIR' in v:
                    # if it's a path, state is saved as a Path
                    setattr(st.session_state, v.lower(), Path(st.text_input(
                        k,
                        value=str(loaded_state.get(v.lower(), getattr(conf_instance, v)))
                    )))
                else:
                    # Otherwise, it's just regular text
                    setattr(st.session_state, v.lower(), loaded_state.get(v.lower(), getattr(conf_instance, v)))

        #  Add some buttons
        col1, col2, _ = st.columns([1, 1, 3])
        with col1:
            if st.button("Save States", use_container_width=True):
                # Dynamically get all attributes of the Configuration class and read corresponding values from session_state
                for field_name, field in Configuration.model_fields.items():
                    if field_name != 'state_file' and field_name != 'mapper':
                        session_key = field_name.lower()
                        if session_key in st.session_state:
                            # Convert Path objects to strings
                            value = st.session_state[session_key]
                            if isinstance(value, Path):
                                value = str(value)
                            setattr(conf_instance, field_name, value)
                conf_instance.save_to_json(Configuration.state_file)
                st.rerun()


@st.cache_data
def create_overlay_image(image_path: str,
                         attn_path: str,
                         window_range: Tuple[int, int],
                         attn_threshold: Tuple[int, int],
                         alpha: float,
                         head_settings: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Create an overlay image display

    Args:
        image_path: str
            Path to the original image
        attn_path: str
            Path to the attention map
        window_range: Tuple[int, int]
            Image window range (lower, upper)
        attn_threshold: Tuple[int, int]
            Attention map threshold range (min, max)
        alpha: float
            Opacity of attention map
        head_settings: Dict[str, Any]
            Settings for attention heads, including:
            - use_max: Whether to use maximum value
            - use_avg: Whether to use average value
            - head_idx: Index of selected attention head

    Returns:
        Tuple containing:
        - overlayed: Final overlaid image
        - attn_map_target: Processed attention map (for histogram)
    """
    # Load image
    image = sitk.ReadImage(str(image_path))
    attn_map_ori = sitk.ReadImage(str(attn_path))
    image = sitk.DICOMOrient(image, 'LPS')
    attn_map = sitk.DICOMOrient(attn_map_ori, 'LPS')

    # Resample image to match attention map
    image = sitk.Resample(image, attn_map)

    # Check metadata match
    same_spacial = check_image_metadata(image, attn_map)
    if not same_spacial:
        st.warning("Resampling...")
        attn_map = sitk.Resample(attn_map, image)

    # Convert to numpy array
    image = sitk.GetArrayFromImage(image)
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

    # Create grid
    ncols = 5
    image = rescale_intensity(make_grid(image, ncols=ncols),
                              lower=window_range[0],
                              upper=window_range[1])
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
    overlayed = overlay_images(image, colored_attn, alpha)

    return overlayed, attn_map_target


if 'initialized' not in st.session_state:
    try:
        loaded_state = Configuration.load_from_json(Configuration.state_file)
        st.session_state['initialized'] = True
    except:
        loaded_state = {}
    build_configurations()
else:
    build_configurations()

# Target ID list
with st.expander("Specify ID"):
    target_ids = st.text_input("CSV string", value="")
    if len(target_ids):
        target_ids = list(set(target_ids.split(',')))

# Load the pairs
image_dir = st.session_state.image_dir
id_globber = st.session_state.id_globber

if image_dir.is_dir():
    paired = load_image_attention_pairs(image_dir, id_globber)
    intersection = list(paired.keys())
    intersection.sort()

    # Filter by target_ids if specified
    if len(target_ids):
        intersection = set(intersection) & set(target_ids)
        if missing := set(target_ids) - set(intersection):
            st.warning(f"IDs specified but the following are missing: {','.join(missing)}")
        intersection = list(intersection)
    st.session_state.require_setup = False
else:
    st.error(f"Directory `{str(image_dir)}` does not exist!")
    st.stop()

# Load the csv file
csv_dir = st.session_state.csv_dir
if csv_dir.is_file():
    csv_data = pd.read_csv(csv_dir, index_col=0)
else:
    st.error(f"File `{str(csv_dir)}` does not exist!")
    st.stop()

# -- Streamlit app
st.title("Image and Attention Map Viewer")

# 初始化 session state
if 'selection_index' not in st.session_state:
    st.session_state.selection_index = 0
if 'filtered_intersection' not in st.session_state:
    st.session_state.filtered_intersection = intersection

# 選擇框 - 使用過濾後的列表
filtered_intersection = st.session_state.filtered_intersection
filtered_intersection.sort()
selected_index = st.selectbox("Slice image pair", range(len(filtered_intersection)),
                              format_func=lambda x: filtered_intersection[x],
                              index=min(st.session_state.selection_index,
                                        len(filtered_intersection) - 1) if filtered_intersection else 0)

if not selected_index == st.session_state.selection_index:
    st.session_state.selection_index = selected_index
    st.rerun()

# Use try-except to catch user input that doesn't exist
try:
    selected_pair = str(filtered_intersection[selected_index])
    # st.write(paired[selected_pair])
except:
    st.write("The selected ID is not in the record.")
    st.stop()

# If selected pair found
if selected_pair:
    with st.container(height=700):
        image_slot = st.empty()

    st.write("### Prediction Results")
    st.write("The prediction results are shown below. The first column is the ID of the image, "
             "and the second column is the raw prediction results (pre-sigmoid), the third column is the "
             "prediction results after sigmoid. The fourth column is the ground-truth label.")
    dataframe_slot = st.empty()
    with st.expander("📊 Full Prediction Results"):
        # Add filtering radio buttons for TP/TN/FP/FN
        filter_option = st.radio(
            "Filter prediction results:",
            ["All", "True Positives (TP)", "True Negatives (TN)", "False Positives (FP)", "False Negatives (FN)"],
            key="filter_option"
        )

        # Check if filter option has changed
        if 'previous_filter_option' not in st.session_state:
            st.session_state.previous_filter_option = filter_option

        filter_changed = st.session_state.previous_filter_option != filter_option
        if filter_changed:
            # Reset selection index when filter changes
            st.session_state.selection_index = 0
            st.session_state.previous_filter_option = filter_option

        # Create a filtered dataframe based on selection
        filtered_csv_data = csv_data.copy()
        filtered_ids = intersection.copy()

        # Calculate statistics for all categories
        if 'Decision_0' in csv_data.columns and 'Truth_0' in csv_data.columns:
            tp_count = len(csv_data[(csv_data['Decision_0'] == 1) & (csv_data['Truth_0'] == 1)])
            tn_count = len(csv_data[(csv_data['Decision_0'] == 0) & (csv_data['Truth_0'] == 0)])
            fp_count = len(csv_data[(csv_data['Decision_0'] == 1) & (csv_data['Truth_0'] == 0)])
            fn_count = len(csv_data[(csv_data['Decision_0'] == 0) & (csv_data['Truth_0'] == 1)])

            # Calculate metrics
            total = tp_count + tn_count + fp_count + fn_count
            accuracy = (tp_count + tn_count) / total if total > 0 else 0
            sensitivity = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
            specificity = tn_count / (tn_count + fp_count) if (tn_count + fp_count) > 0 else 0

            # Display metrics
            st.write(f"### Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("True Positives (TP)", tp_count)
                st.metric("False Positives (FP)", fp_count)
            with col2:
                st.metric("True Negatives (TN)", tn_count)
                st.metric("False Negatives (FN)", fn_count)
            with col3:
                st.metric("Accuracy", f"{accuracy:.2%}")
                st.metric("Sensitivity", f"{sensitivity:.2%}")
                st.metric("Specificity", f"{specificity:.2%}")

        # Apply filtering based on selection
        if filter_option != "All":
            # Let's determine TP/TN/FP/FN based on prediction and ground truth
            # Assuming the format is: raw prediction, probability, decision (0/1), ground truth (0/1)
            if 'Decision_0' in csv_data.columns and 'Truth_0' in csv_data.columns:
                if filter_option == "True Positives (TP)":
                    # Both prediction and truth are positive (1)
                    filtered_csv_data = csv_data[(csv_data['Decision_0'] == 1) & (csv_data['Truth_0'] == 1)]
                elif filter_option == "True Negatives (TN)":
                    # Both prediction and truth are negative (0)
                    filtered_csv_data = csv_data[(csv_data['Decision_0'] == 0) & (csv_data['Truth_0'] == 0)]
                elif filter_option == "False Positives (FP)":
                    # Prediction is positive (1) but truth is negative (0)
                    filtered_csv_data = csv_data[(csv_data['Decision_0'] == 1) & (csv_data['Truth_0'] == 0)]
                elif filter_option == "False Negatives (FN)":
                    # Prediction is negative (0) but truth is positive (1)
                    filtered_csv_data = csv_data[(csv_data['Decision_0'] == 0) & (csv_data['Truth_0'] == 1)]

                # Update the intersection list to only include IDs from the filtered dataframe
                filtered_ids = [id for id in intersection if id in filtered_csv_data.index]

                # Update the selection options if we're filtering
                if filtered_ids:
                    # Store filtered IDs in session state for access by selectbox
                    st.session_state.filtered_intersection = filtered_ids
                    if filter_changed:
                        st.rerun()  # Rerun to update the UI with the new filter
                else:
                    st.warning(f"No cases found for the selected filter: {filter_option}")
                    st.session_state.filtered_intersection = intersection
            else:
                st.warning("CSV data does not contain the required columns (Decision_0, Truth_0) for filtering")
                st.session_state.filtered_intersection = intersection
        else:
            # If "All" is selected, use the original intersection
            st.session_state.filtered_intersection = intersection
            if filter_changed:
                st.rerun()  # Rerun to update the UI when switching back to "All"

        # Show current filter status
        st.write(f"### Showing {len(filtered_csv_data)} cases for filter: {filter_option}")

        # Display the dataframe (either filtered or all)
        st.dataframe(filtered_csv_data)

    with st.expander("⚙️ Display settings"):
        st.write("### Image window-level and attention map settings")
        col1, spacer, col2 = st.columns([1, 0.1, 1])
        with col1:
            # Slider to control the window range of the image
            lower, upper = st.slider(
                'Image window range',
                min_value=0,
                max_value=99,
                value=st.session_state.get('image_window_range', Configuration.DEFAULT_WINDOW_RANGE)
            )
            st.session_state['image_window_range'] = (lower, upper)

            # Slider to control the threshold for the attention map
            attn_min, attn_max = st.slider(
                'Attention map threshold',
                min_value=Configuration.ATTN_MIN_VALUE,
                max_value=Configuration.ATTN_MAX_VALUE,
                value=st.session_state.get('attn_threshold', Configuration.DEFAULT_ATTN_THRESHOLD)
            )
            st.session_state['attn_threshold'] = (attn_min, attn_max)

            # Slider to control the opacity of the attention map
            alpha = st.slider(
                'Attention map opacity',
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get('attn_opacity', Configuration.DEFAULT_OPACITY)
            )
            st.session_state['attn_opacity'] = alpha

        with col2:
            st.write("### Attention Head Selection")
            # Add controls for attention head selection
            head_col1, head_col2 = st.columns([1, 1])
            with head_col1:
                use_max = st.checkbox("Use Max", value=False)
            with head_col1:
                use_avg = st.checkbox("Use Avg", value=False)
            with head_col1:
                head_idx = st.number_input(
                    'Select head',
                    min_value=0,
                    max_value=st.session_state.get('num_heads', 20),
                    value=0,
                    step=1,
                    disabled=(use_max or use_avg)
                )

    with st.spinner("Loading..."):
        img_path, attn_path = paired[selected_pair]

        # Settings
        head_settings = {
            'use_max': use_max,
            'use_avg': use_avg,
            'head_idx': head_idx
        }

        # Create the image
        overlayed, attn_map_target = create_overlay_image(
            img_path,
            attn_path,
            window_range=(lower, upper),
            attn_threshold=(attn_min, attn_max),
            alpha=alpha,
            head_settings=head_settings
        )

        # Show the image
        image_slot.image(overlayed, use_container_width=True)

        # Load the result dat and display them
        if selected_pair in filtered_csv_data.index:
            dataframe_slot.dataframe(filtered_csv_data.loc[selected_pair].to_frame().T)
        else:
            st.warning(f"Selected ID {selected_pair} is not found in the filtered dataset.")
            dataframe_slot.dataframe(pd.DataFrame())

    # Add histogram plot of attention map with threshold lines
    with st.expander("📊 Plots"):
        attn_values = attn_map_target.flatten()

        # Filter out extreme values for better visualization
        lower_percentile = Configuration.HIST_LOWER_PERCENTILE
        upper_percentile = Configuration.HIST_UPPER_PERCENTILE
        p1 = np.percentile(attn_values, lower_percentile)
        p2 = np.percentile(attn_values, upper_percentile)
        filtered_values = attn_values[(attn_values >= p1) & (attn_values <= p2)]

        # Create histogram using plotly
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=attn_values.flatten(),
            xbins=dict(start=max(2, attn_min - 40), end=min(255, attn_max + 40), size=1),
            histnorm='probability density',
            name='Distribution'
        ))

        # Add vertical lines for threshold values if applicable
        if attn_min < attn_max:
            # Convert threshold from 0-255 scale to actual data range
            actual_min = attn_min
            actual_max = attn_max

            fig.add_vline(
                x=actual_min,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Min ({attn_min})",
                annotation_position="top right"
            )
            fig.add_vline(
                x=actual_max,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Max ({attn_max})",
                annotation_position="top left"
            )

        fig.update_layout(
            title=f'Attention Map Distribution ({int(lower_percentile)}-{int(upper_percentile)}th percentile)',
            xaxis_title='Attention Value',
            yaxis_title='Density',
            xaxis_range=[max(2, attn_min - 40), min(255, attn_max + 40)],
            showlegend=False
        )

        st.plotly_chart(fig)

    # Previous button
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button('⬅️ Previous', use_container_width=True):
            current_index = selected_index
            previous_index = (current_index - 1) % len(filtered_intersection)
            st.session_state.selection_index = previous_index
            # No need to rerun becasue the frames update automatically

    # Next button
    with col2:
        if st.button('Next ➡️', use_container_width=True):
            current_index = selected_index
            next_index = (current_index + 1) % len(filtered_intersection) if filtered_intersection else 0
            st.session_state.selection_index = next_index
            # No need to rerun becasue the frames update automatically

    with col3:
        if st.button('Save Image', use_container_width=True):
            st.write("Saving...")
            try:
                # Get the image and attention map paths
                img_path, attn_path = paired[selected_pair]

                # Save the images to temp directory
                saved_img_path = save_image(img_path, lower, upper)
                saved_attn_path = save_image(attn_path, attn_min, attn_max)

                st.success(f"Images saved successfully to:\n{saved_img_path}\n{saved_attn_path}")
            except Exception as e:
                st.error(f"Failed to save images: {e}")
