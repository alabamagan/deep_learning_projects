import os

from pytorch_med_imaging.pmi_data import DataLabel
from pytorch_med_imaging.solvers import BinaryClassificationSolver, ClassificationSolver
from pytorch_med_imaging.inferencers import BinaryClassificationInferencer, ClassificationInferencer
from pytorch_med_imaging.controller import PMIControllerCFG
from pytorch_med_imaging.pmi_data_loader import PMITorchioDataLoader, PMITorchioDataLoaderCFG
from pytorch_med_imaging.solvers.earlystop import LossReferenceEarlyStop
from .network.scnet import SCDenseNet
from .loss.scdensenet_loss import DualLoss
from ..solvers.scdnetSolver import SCDenseNetInferencer, SCDenseNetSolver, SCDenseNetSolverCFG
from dataclasses import asdict
import copy
from datetime import datetime
from typing import *
import torch
import os

# Set project meta for neptune plot
os.environ['NEPTUNE_PROJECT'] = "CUHK-DIIR/NPC-Screening"

# For training
data_loader = PMITorchioDataLoaderCFG(
    input_data = {
        'input': './NPC_Segmentation/60.Large-Study/v1-All-Data/Normalized_2/T2WFS_TRA/01.NyulNormalized/',
        'probmap': './NPC_Segmentation/60.Large-Study/v1-All-Data/Normalized_2/T2WFS_TRA/00.HuangMask/',
        'gt': ('./NPC_Segmentation/99.Testing/NPC_Screening/v1/Datasheet_v2.csv', 'is_malignant'),
        'gt_seg': './NPC_Segmentation/60.Large-Study/v1-All-Data/Normalized_2_SCDensnet/T2WFS_TRA'
    },
    input_dtypes = {
        'probmap': 'uint8',
        'gt_seg': 'uint8'
    },
    master_data_key = 'gt',
    ignore_missing_ids = True,
    augmentation  = './SCDNet_transform_train.yaml',
    id_globber    = "^[a-zA-Z]{0,3}[0-9]+",
    sampler='weighted',  # Unset sampler to load the whole image
    sampler_kwargs=dict(
        patch_size=[384, 384, 16]
    ),
    tio_queue_kwargs = dict(            # dict passed to ``tio.Queue``
        max_length             = 240,
        samples_per_volume     = 2,
        num_workers            = min(12, os.cpu_count() * 3 // 4),
        shuffle_subjects       = True,
        shuffle_patches        = True,
        start_background       = True,
        verbose                = True,
    )
)

# For testing
data_loader_test = PMITorchioDataLoaderCFG(
    input_data = {
        'input': './NPC_Segmentation/60.Large-Study/v1-All-Data/Normalized_2/T2WFS_TRA/01.NyulNormalized/',
        'gt': ('./NPC_Segmentation/99.Testing/NPC_Screening/v1/Datasheet_v2.csv', 'is_malignant'),
        'gt_seg': './NPC_Segmentation/60.Large-Study/v1-All-Data/Normalized_2_SCDensnet/T2WFS_TRA'
    },
    input_dtypes = {
        'gt_seg': 'uint8'
    },
    master_data_key = 'gt',
    ignore_missing_ids = True,
    augmentation  = './SCDNet_transform_inf.yaml',
    target_column = 'is_malignant',
    id_globber    = "^[a-zA-Z]{0,3}[0-9]+",
    tio_queue_kwargs = dict(            # dict passed to ``tio.Queue``
        max_length             = 15,
        samples_per_volume     = 1,
        num_workers            = min(12, os.cpu_count() * 3 // 4),
        shuffle_subjects       = False,
        shuffle_patches        = True,
        start_background       = True,
        verbose                = True,
    )
)

class SCDenseNetSolverCFG(SCDenseNetSolverCFG):
    r"""This is created to cater for the configuration of rAIdiologist network"""
    net           = SCDenseNet()
    optimizer     = 'Adam'
    # init_mom      = 0.95
    init_lr       = 1E-4
    batch_size    = 8
    num_of_epochs = 200

    unpack_key_forward   = ['input', ('gt', 'gt_seg')]
    unpack_key_inference = ['input']

    early_stop        = 'loss_reference'
    early_stop_kwargs = {'warmup'       : 5, 'patience': 10}
    accumulate_grad   = 0

    loss_function = DualLoss()

id_list_dir = "./NPC_Segmentation/99.Testing/NPC_Screening/rai_v5.1/"
class SCDControllerCFG(PMIControllerCFG):
    run_mode    = 'training'
    fold_code   = 'B01'
    net_name    = 'SCDenseNet'
    id_list     = id_list_dir + "/{fold_code}.ini"
    id_list_val = id_list_dir + "/Validation.txt"
    output_dir  = './Results/{fold_code}'
    cp_load_dir = './Backup/{net_name}_{fold_code}.pt'
    cp_save_dir = './Backup/{net_name}_{fold_code}.pt'
    log_dir     = f"./Backup/Log/{net_name}_{datetime.strftime(datetime.now(), '%Y-%m-%d')}.log"
    rAI_pretrained_CNN = './Backup/{net_name}_{fold_code}_pretrain.pt'

    _data_loader_cfg     = data_loader
    _data_loader_inf_cfg = data_loader_test # inference need different dataloader
    data_loader_val_cfg  = data_loader_test # validation set comes from the same folder as testing set
    data_loader_cls      = PMITorchioDataLoader

    solver_cfg     = SCDenseNetSolverCFG()
    solver_cls     = SCDenseNetSolver
    inferencer_cls = SCDenseNetInferencer

    debug_validation = False
    compile_net = False

    # For plotting
    plotting        = True
    plotter_type = 'neptune'
    plotter_init_meta = {
        'description': "rAIdiologists training project.",
    }