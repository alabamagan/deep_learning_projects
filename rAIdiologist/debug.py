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

# Get da
fname = get_fnames_by_IDs(Args.image_data_dir.glob("*nii.gz"),
                          idlist=Args.id_list, globber=Args.id_globber, return_dict=True)
print(fname)
tio_images = {k: torchio.ScalarImage(f) for k, f in fname.items()}
tio_transform = create_transform_compose(Args.inference_transform)
subjects = [tio.Subject(input=tioimg, sid=sid) for sid, tioimg in tio_images.items()]
subjects_dataset = tio.SubjectsDataset(subjects, transform=tio_transform)

m: rAIdiologist_Transformer = create_rAIdiologist_v5_1()
m.set_mode(5)
m.load_state_dict(torch.load(Args.checkpoint_dir))
m = m.cuda()
m.eval()
m.RECORD_ON = True
# m = nn.DataParallel(m)

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
            logger.info(f"{sub['sid'] = }")
            in_tensor = sub['input'][tio.DATA].float().cuda()
            sid = sub['sid']

            x = m(in_tensor)
            playback, = m.get_playback()
            m.clean_playback()

            # playback expected shape: (20 x 1 x [W+2]*[H+2], [W+2]*[H+2])
            # Revert the forward rearrange in rAIdiologist_Transformer
            sa_for_prediction = rearrange(playback[:, :, 0, 2:], 'b 1 (z w h) -> b 1 h w z', **grid_size)
            sa_for_confidence = rearrange(playback[:, :, 1, 2:], 'b 1 (z w h) -> b 1 h w z', **grid_size)
            # Interpolate sa_for_prediction to match in_tensor size
            sa_for_prediction_resized = F.interpolate(
                sa_for_prediction, size=(in_tensor.shape[-3], in_tensor.shape[-2], 24), mode='trilinear', align_corners=False
            ).float()
            sa_for_confidence_resized = F.interpolate(
                sa_for_confidence, size=(in_tensor.shape[-3], in_tensor.shape[-2], 24), mode='trilinear',
                align_corners=False
            ).float()

            img = tio.ScalarImage(tensor=in_tensor.cpu()[..., 1:]) # first slice removed
            play_back_prediction = tio.ScalarImage(tensor=sa_for_prediction_resized.squeeze().cpu())
            play_back_confidence = tio.ScalarImage(tensor=sa_for_confidence_resized.squeeze().cpu())

            img.save(Args.output_dir / f"{sid}.nii.gz")
            play_back_prediction.save(Args.output_dir / f"{sid}_heatmap.nii.gz")
            play_back_confidence.save(Args.output_dir / f"{sid}_confidence.nii.gz")
        except Exception as e:
            logger.error(e)
            continue

x = input('Press any key to continue...')