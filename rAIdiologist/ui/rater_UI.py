import streamlit
import pydantic
from pathlib import Path
from typing import List, Union, Any

PathLike = Union[Path, str]

# Configurations
class Configurations(pydantic.BaseModel):
    DICOM_dir: PathLike = None
    ID_list: PathLike = None
    PW_File: PathLike = None
    SHOW_AI: bool = False

st.write("# NPC Screening Reader Study")
st.write(
    """
    ## Instruction
    
    > ⚠️ **Warning:** Once you click the "Submit" button, a timer will start.
    
    """
)

# Download DICOM

# Randomly shuffle backend data

# Save results with timer recording time

# Enable survey

# Save result

