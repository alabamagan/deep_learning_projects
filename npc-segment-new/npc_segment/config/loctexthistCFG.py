from datetime import datetime
import torch
from pytorch_med_imaging.solvers import SegmentationSolver, SegmentationSolverCFG
from pytorch_med_imaging.inferencers import SegmentationInferencer
from pytorch_med_imaging.controller import PMIController, PMIControllerCFG
from pytorch_med_imaging.networks.specialized.unet_loc_tex import UNetFCAttention_p
from pytorch_med_imaging.pmi_data_loader import PMIImageDataLoader, PMIImageDataLoaderCFG

from npc_segment.network import UNetLocTexHistDeeper
from pytorch_med_imaging.pmi_data_loader.computations.queue_callback import loc_text_hist
import torch.nn as nn
from multiprocessing import Semaphore

# define a worker function for tio.CallBackQueue
sem = Semaphore(8)
def work_funct(*args, **kwargs):
    with sem:
        o = loc_text_hist(*args, **kwargs)
    return o



data_loader_shared_kwargs = dict(
    input_dir = '',
    probmap_dir = '',
    target_dir = '',
    augmentation = './assets/augmentation.yaml',
    id_globber = "^[a-zA-Z]{0,5}[0-9]+",
    sampler = 'weighted',
    sampler_kwargs = dict (
        patch_size = (128, 128, 1),
    )
)

data_loader_train = PMIImageDataLoaderCFG(
    tio_queue_kwargs = dict(            # dict passed to ``tio.Queue``
        max_length             = 1500,
        samples_per_volume     = 50,
        num_workers            = 8,
        shuffle_subjects       = True,
        shuffle_patches        = True,
        start_background       = True,
        verbose                = True,
    ),
    patch_sampling_callback = work_funct,
    create_new_attribute = 'feature',
    patch_sampling_callback_kwargs = {'nbins': 128, 'include': 'input'},
    inf_samples_per_vol = 550,
    **data_loader_shared_kwargs
)

data_loader_inf = PMIImageDataLoaderCFG(
    tio_queue_kwargs = dict(            # dict passed to ``tio.Queue``
        max_length             = 550,
        samples_per_volume     = 550,
        num_workers            = 8,
        shuffle_subjects       = False,
        shuffle_patches        = False,
        start_background       = True,
        verbose                = True,
    ),
    create_new_attribute = 'feature',
    inf_samples_per_vol = 550,
    patch_sampling_callback = work_funct,
    patch_sampling_callback_kwargs = {'nbins': 128, 'include': 'input'},
    **data_loader_shared_kwargs
)

class NPCSegmentSolverCFG(SegmentationSolverCFG):
    net = UNetLocTexHistDeeper(1, 1, fc_inchan=260)

    optimizer = 'Adam'

    unpack_key_forward   = [('input', 'feature'), 'gt']
    unpack_key_inference = ['input' , 'feature']

    gt_keys               = ['gt']
    sigmoid_params        = {'delay' : 5         , 'stretch': 1, 'cap': 0.2}

    class_weights       = [0.01, 1.0]
    num_of_epochs       = 1
    decay_rate_LR       = 1
    decay_on_plateau    = False

    init_lr = 1E-5
    batch_size = 40
    batch_size_val = 40

    lr_sche = 'ExpoentialLR'
    lr_sche_args = [0.99]
    lr_sche_kwargs   = {'cooldown':100, 'patience':50}

    # How to scale class weights?
    loss_function = nn.CrossEntropyLoss(torch.as_tensor(class_weights))


class NPCSegmentControllerCFG(PMIControllerCFG):
    run_mode = 'training'

    sequence = 'T2WFS'
    version = 'v1.0'

    # relative to main.py
    cp_load_dir = './assets/checkpoints/NPC_segment_{sequence}_{version}.pt'
    cp_save_dir = './assets/checkpoints/NPC_segment_{sequence}_{version}.pt'
    log_dir = f"./Log/npc-segment_{datetime.strftime(datetime.now(), '%Y-%m-%d')}.log"

    # Output directory is set during runtime
    output_dir = None

    _data_loader_cfg = data_loader_train # underscore for different loaders during training and testing
    _data_loader_inf_cfg = data_loader_inf
    data_loader_cls = PMIImageDataLoader

    solver_cfg = NPCSegmentSolverCFG()
    solver_cls = SegmentationSolver
    inferencer_cls = SegmentationInferencer

    debug_mode = False
    debug_validation = False
