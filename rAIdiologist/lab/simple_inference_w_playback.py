import torch
import torch.nn as nn
import torch.nn.functional as F
import os

import torchio
import torchio as tio
import matplotlib.pyplot as plt
from tqdm import tqdm

from einops import rearrange
from pathlib import Path

from rAIdiologist.config.network.rAIdiologist import *
from mnts.utils.filename_globber import get_fnames_by_IDs
from mnts.mnts_logger import MNTSLogger
from pytorch_med_imaging.pmi_data_loader.augmenter_factory import create_transform_compose

class Args:
    # image_data_dir = Path("/mnt/ftp_shared/NPC/NPC_screening_5_AI/reports_temp/normalized_images/NyulNormalizer/")
    image_data_dir = Path("/home/lwong/Source/Repos/NPC_Segmentation/NPC_Segmentation/60.Large-Study/HKU_data/NyulNormalizer")
    checkpoint_dir = Path("/home/lwong/Source/Repos/NPC_Segmentation/Backup/rAIdiologist_B01.pt")
    output_dir = Path("/mnt/ftp/Shared/temp")
    id_globber = '\w{0,5}\d+'
    id_list = [f"HKU{d:04d}" for d in range(11, 200)]
    inference_transform = "rAIdiologist/config/rAIdiologist_transform_inf.yaml"


# Get file names by IDs, filtering files in the specified directory
fname = get_fnames_by_IDs(
    Args.image_data_dir.glob("*nii.gz"),
    idlist=Args.id_list,
    globber=Args.id_globber,
    return_dict=True
)
print(fname)

# Create a dictionary of TorchIO ScalarImage objects from the file names
tio_images = {k: torchio.ScalarImage(f) for k, f in fname.items()}
# Create a transformation pipeline for inference
tio_transform = create_transform_compose(Args.inference_transform)
subjects = [tio.Subject(input=tioimg, sid=sid) for sid, tioimg in tio_images.items()]
# Create a TorchIO SubjectsDataset using the subjects and transformation
subjects_dataset = tio.SubjectsDataset(subjects, transform=tio_transform)

# Initialize the model
m: rAIdiologist_Transformer = create_rAIdiologist_v5_1()
m.set_mode(5)
m.load_state_dict(torch.load(Args.checkpoint_dir))
m = m.cuda()
m.eval()

# Enable recording of model operations
m.RECORD_ON = True


grid_size = {
    'w': 8,
    'h': 8,
    'z': 24
}

with (torch.no_grad(),
      MNTSLogger(".", "PlaybackTest", verbose=True, log_level='debug',
                 keep_file=False) as logger):
    for sub in tqdm(subjects_dataset):
        try:
            # Log the subject ID
            logger.info(f"{sub['sid'] = }")

            # Prepare input tensor for the model
            in_tensor = sub['input'][tio.DATA].float().cuda()
            sid = sub['sid']

            # Perform inference using the model
            x = m(in_tensor)

            # Retrieve and clean the model playback data
            playback, = m.get_playback()
            m.clean_playback()

            # Reshape the playback data for prediction and confidence
            # Revert the forward rearrange in rAIdiologist_Transformer
            sa_for_prediction = rearrange(
                playback[:, :, 0, 2:], 'b 1 (z w h) -> b 1 h w z', **grid_size
            )
            sa_for_confidence = rearrange(
                playback[:, :, 1, 2:], 'b 1 (z w h) -> b 1 h w z', **grid_size
            )

            # Interpolate reshaped playback data to match input tensor size
            sa_for_prediction_resized = F.interpolate(
                sa_for_prediction, size=(in_tensor.shape[-3], in_tensor.shape[-2], 24),
                mode='trilinear', align_corners=False
            ).float()
            sa_for_confidence_resized = F.interpolate(
                sa_for_confidence, size=(in_tensor.shape[-3], in_tensor.shape[-2], 24),
                mode='trilinear', align_corners=False
            ).float()

            # Create TorchIO images for input, prediction, and confidence
            img = tio.ScalarImage(tensor=in_tensor.cpu()[..., 1:])  # Remove the first slice
            play_back_prediction = tio.ScalarImage(tensor=sa_for_prediction_resized.squeeze().cpu())
            play_back_confidence = tio.ScalarImage(tensor=sa_for_confidence_resized.squeeze().cpu())

            # Save the images to the output directory
            img.save(Args.output_dir / f"{sid}.nii.gz")
            play_back_prediction.save(Args.output_dir / f"{sid}_heatmap.nii.gz")
            play_back_confidence.save(Args.output_dir / f"{sid}_confidence.nii.gz")
        except Exception as e:
            logger.error(e)
            continue

x = input('Press any key to continue...')