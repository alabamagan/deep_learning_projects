from pytorch_med_imaging.solvers import SegmentationSolver, SegmentationSolverCFG
from pytorch_med_imaging.controller import PMIController, PMIControllerCFG
from pytorch_med_imaging.networks.specialized.unet_loc_tex import UNetFCAttention_p
from pytorch_med_imaging.pmi_data_loader import PMIImageDataLoader

from npc_segment.network import UNetLocTexHistDeeper

data_loader_train = PMIImageDataLoader(
    input_dir = '',
    probmap_dir = '',
    target_dir = '',
    augmentation = '',
    id_globber = "^[a-zA-Z]{0,3}[0-9]+",
    sampler = 'weighted',
    sampler_kwargs = dict (
        patch_size = (128, 128, 1),
        samples_per_volume = 500,
    ),
    tio_queue_kwargs = dict(            # dict passed to ``tio.Queue``
        max_length             = 500,
        samples_per_volume     = 1,
        num_workers            = 8,
        shuffle_subjects       = True,
        shuffle_patches        = True,
        start_background       = True,
        verbose                = True,
    )
)

class SegmentCFG(SegmentationSolverCFG):
    net = UNetLocTexHistDeeper(1, 2, fc_inchan=204)

    unpack_key_forward   = [('input', 'feature'), 'gt']
    unpack_key_inference = ['input' , 'feature']

    gt_keys               = ['gt']
    sigmoid_params        = {'delay' : 5         , 'stretch': 1, 'cap': 0.2}

    class_weights       = [0.01, 1.0]
    num_of_epochs       = 1
    decay_rate_LR       = 1
    decay_on_plateau    = False

    lr_sche = 'ExpoentialLR'
    lr_scheduler_dict   = {'cooldown':100, 'patience':50}

    # How to scale class weights?