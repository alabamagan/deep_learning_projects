import streamlit
from pathlib import Path
from genericpath import isfile
from multiprocessing import Value
import sys
import re
from turtle import onrelease
import SimpleITK as sitk
import numpy as np
import pandas as pd
import pydantic

from visualization import *
from pathlib import Path
from mnts.utils import get_fnames_by_IDs, get_unique_IDs

import streamlit as st
from pprint import pprint, pformat
import plotly.express as px
import numpy as np
import json

from typing import *
import logging, time
from rich.logging import RichHandler
from rich.traceback import install

class Configurations:
    IMAGE_DIR: str = '.'
    SEGMENT_DIR: str = '.'
    CSV_DIR: str = './results.csv'


st.set_page_config(layout="wide")
st.write("# NPC Screening Results View")
st.write(
    """
    This is a small UI that is 
    """
)


install()

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
def load_pair(MRI_DIR: Path, SEG_DIR: Path, id_globber: str = r"\w+\d+"):
    r"""This handles the matching between segmentation and images"""
    # Globbing files
    mri_files, seg_files = MRI_DIR.rglob("*nii.gz"), SEG_DIR.rglob("*nii.gz")
    mri_files = {re.search(id_globber, f.name).group(): f for f in mri_files}
    seg_files = {re.search(id_globber, f.name).group(): f for f in seg_files}

    # Get files with both segmentation and MRI
    intersection = list(set(mri_files.keys()).intersection(seg_files.keys()))
    intersection.sort()

    # Forming pairs
    paired = {sid: (mri_files[sid], seg_files[sid]) for sid in intersection}
    return paired


def load_dataframe(p: Path = None):
    p = p or Path(st.session_state.get('frame_path', None))
    if p.is_file():
        st.session_state.dataframe = pd.read_csv(p)


def check_image_metadata(img1: sitk.Image, img2: sitk.Image, tolerance=1e-3):
    r"""This checks the meta information of the image and the segmentation to make sure they are the same."""
    # Check metadata with tolerance
    spacing_match = np.all(np.isclose(mri_image.GetSpacing(), seg_image.GetSpacing(), atol=tolerance))
    direction_match = np.all(np.isclose(mri_image.GetDirection(), seg_image.GetDirection(), atol=tolerance))
    origin_match = np.all(np.isclose(mri_image.GetOrigin(), seg_image.GetOrigin(), atol=tolerance))
    size_match = np.array_equal(img1.GetSize(), img2.GetSize())

    if spacing_match and direction_match and origin_match:
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

    return all([spacing_match, direction_match, origin_match, size_match])


# -- Load the state if it exists
# Function to load state from a JSON file
def load_state(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load state: {e}")
        return None


# Function to save state to a JSON file
def save_state(file_path, state):
    try:
        with open(file_path, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        logging.error(f"Failed to save state: {e}")


# File path to save/load the session state
state_file = ".session_state.json"

if 'initialized' not in st.session_state:
    loaded_state = load_state(state_file)
    if loaded_state:
        st.session_state.mri_dir = Path(loaded_state.get('mri_dir', Configurations.IMAGE_DIR))
        st.session_state.seg_dir = Path(loaded_state.get('seg_dir', Configurations.SEGMENT_DIR))
        st.session_state.id_globber = loaded_state.get('id_globber', r"\w{0,5}\d+")
        st.session_state.frame_path = Path(loaded_state.get('frame_path', Configurations.CSV_DIR))
    else:
        # Initialize default settings if loading failed
        st.session_state.mri_dir = Path(Configurations.IMAGE_DIR)
        st.session_state.seg_dir = Path(Configurations.SEGMENT_DIR)
        st.session_state.id_globber = r"\w{0,5}\d+"
        st.session_state.frame_path = Path("Checked_Images.csv")
    st.session_state.initialized = True

with st.expander("Directory Setup", expanded=st.session_state.get("require_setup", False)):
    st.write("### Insert the local directory (where this app is running)")
    st.session_state.mri_dir = Path(st.text_input("<MRI_DIR>:", value=str(st.session_state.mri_dir)))
    st.session_state.seg_dir = Path(st.text_input("<SEG_DIR>:", value=str(st.session_state.seg_dir)))
    st.session_state.id_globber = st.text_input("Regex ID globber:",
                                                value=st.session_state.get('id_globber', r"\w{0,5}\d+"))
    st.session_state.frame_path = Path(
        st.text_input("Frame Path:", value=str(st.session_state.get('frame_path', "./Checked_Images.csv"))))

    col1, col2, _ = st.columns([1, 1, 3])
    with col1:
        if st.button("Save States", use_container_width=True):
            # Save the current state
            save_state(state_file, {
                'mri_dir': str(st.session_state.mri_dir),
                'seg_dir': str(st.session_state.seg_dir),
                'id_globber': st.session_state.id_globber,
                'frame_path': str(st.session_state.frame_path)
            })
            st.rerun()

    with col2:
        if st.button("Reload Dataframe", use_container_width=True, key="btn_reload_dataframe"):
            # Reload the dataframe from specified framepath
            load_dataframe(st.session_state.frame_path)
            st.rerun()

# * Setup paths
# Target ID list
with st.expander("Specify ID"):
    target_ids = st.text_input("CSV string", value="")
    if len(target_ids):
        target_ids = list(set(target_ids.split(',')))

mri_dir = st.session_state.mri_dir
seg_dir = st.session_state.seg_dir
id_globber = st.session_state.id_globber

# Get paired MRI and segmentation
if mri_dir.is_dir() and seg_dir.is_dir():
    paired = load_pair(mri_dir, seg_dir)
    intersection = list(paired.keys())
    intersection.sort()
    # further filtering if target_ids specified
    if len(target_ids):
        intersection = set(intersection) & set(target_ids)
        if missing := set(target_ids) - set(intersection):
            st.warning(f"IDs specified but the following are missing: {','.join(missing)}")
        intersection = list(intersection)
    st.session_state.require_setup = False
else:
    st.error(f"`{str(mri_dir)}` or `{str(seg_dir)}` not found!")
    st.stop()

# -- Streamlit app
st.title("MRI and segmentation viewer")

# Load Excel file into session state
frame_path = st.session_state.frame_path
if frame_path.is_file() and not st.session_state.last_confirmation:
    logging.info(f"Loading dataframe from file: {frame_path}")
    dataframe = pd.read_csv(frame_path, index_col=0)
else:
    st.error("Cannot load dataframe from csv.")
st.session_state.dataframe = dataframe

# Initialize session state
if 'selection_index' not in st.session_state:
    st.session_state.selection_index = 0

# Selection box
intersection.sort()
selected_index = st.selectbox("Select a pair", range(len(intersection)), format_func=lambda x: intersection[x],
                              index=st.session_state.selection_index)
if not selected_index == st.session_state.selection_index:
    # Need to trigger rerun here because the state change is not immediately reflected until next refresh
    st.session_state.selection_index = selected_index
    st.rerun()

# Use try-except to catch user input that doesn't exist
try:
    selected_pair = str(intersection[selected_index])
    st.write(paired[selected_pair])
except:
    st.write("Your selected ID does not match with the records.")
    st.stop()

if selected_pair:
    with st.container(height=700):
        image_slot = st.empty()

    # Sliders for window levels
    lower, upper = st.slider(
        'Window Levels',
        min_value=0,
        max_value=99,
        value=(25, 99)
    )

    with st.spinner("Running"):
        mri_path, seg_path = paired[selected_pair]
        result = dataframe.loc[selected_pair]

        # Load images
        mri_image = sitk.ReadImage(str(mri_path))
        seg_image = sitk.ReadImage(str(seg_path))

        # Check if the two images has the same spacing
        same_spacial = check_image_metadata(mri_image, seg_image)

        if not same_spacial:
            st.warning("Resampling")
            seg_image = sitk.Resample(seg_image, mri_image)

        try:
            mri_image, seg_image = crop_image_to_segmentation_sitk(mri_image, seg_image, 20)
        except ValueError as e:
            st.warning(f"Something wrong with the segmentation.")
            logger.error(e, exc_info=True)
        except IndexError as e:
            st.warning("The segmentation seems to be empty")
            logger.error(e, exc_info=True)

        mri_image = sitk.GetArrayFromImage(mri_image)
        seg_image = sitk.GetArrayFromImage(seg_image)

        # Rescale
        ncols = 5
        mri_image = rescale_intensity(make_grid(mri_image, ncols=ncols),
                                      lower=lower,
                                      upper=upper)
        seg_image = make_grid(seg_image, ncols=ncols).astype('int')

        try:
            mri_image = draw_contour(mri_image, seg_image, width=2)
        except ValueError:
            st.warning("Something wrong with the segmetnation.")

        # Display images
        image_slot.image(mri_image, use_column_width=True)

        # Display results
        st.dataframe(result.to_frame())

    # Button to go back one option
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button('⬅️', use_container_width=True):
            current_index = selected_index
            previous_index = (current_index - 1) % len(intersection)
            st.session_state.selection_index = previous_index
            st.rerun()

    # Button to load the next option
    with col2:
        if st.button('➡️ Checked and Next', use_container_width=True):
            current_index = selected_index
            next_index = min(current_index + 1, len(intersection))
            st.session_state.selection_index = next_index
            st.rerun()