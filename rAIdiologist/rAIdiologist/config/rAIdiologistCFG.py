import os

from pytorch_med_imaging.solvers import BinaryClassificationSolver, ClassificationSolver
from pytorch_med_imaging.inferencers import BinaryClassificationInferencer, ClassificationInferencer
from pytorch_med_imaging.controller import PMIControllerCFG
from pytorch_med_imaging.pmi_data_loader import PMIImageFeaturePairLoader, PMIImageFeaturePairLoaderCFG
from pytorch_med_imaging.solvers.earlystop import LossReferenceEarlyStop
from .network.rAIdiologist import rAIdiologist
from .network.slicewise_ran import SlicewiseAttentionRAN
from .loss.rAIdiologist_loss import ConfidenceBCELoss
from ..solvers.rAIdiologistSolver import rAIdiologistSolverCFG, rAIdiologistSolver
from ..solvers.rAIdiologistInferencer import *
from dataclasses import asdict
import copy
from datetime import datetime
import torch

# For training
data_loader = PMIImageFeaturePairLoaderCFG(
    input_dir     = './NPC_Segmentation/60.Large-Study/v1-All-Data/Normalized_2/T2WFS_TRA/01.NyulNormalized/',
    probmap_dir   = './NPC_Segmentation/60.Large-Study/v1-All-Data/Normalized_2/T2WFS_TRA/00.HuangMask/',
    target_dir    = './NPC_Segmentation/99.Testing/NPC_Screening/v1/Datasheet_v2.csv',
    augmentation  = './rAIdiologist_transform_train.yaml',
    target_column = 'is_malignant',
    id_globber    = "^[a-zA-Z]{0,3}[0-9]+",
    sampler       = 'weighted', # Unset sampler to load the whole image
    sampler_kwargs    = dict(
        patch_size = [320, 320, 25]
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

# For inference
data_loader_inf = PMIImageFeaturePairLoaderCFG(
    input_dir     = './NPC_Segmentation/60.Large-Study/v1-All-Data/Normalized_2/T2WFS_TRA/01.NyulNormalized/',
    probmap_dir   = './NPC_Segmentation/60.Large-Study/v1-All-Data/Normalized_2/T2WFS_TRA/00.HuangMask/',
    target_dir    = './NPC_Segmentation/99.Testing/NPC_Screening/v1/Datasheet_v2.csv',
    augmentation  = './rAIdiologist_transform_inf.yaml',
    target_column = 'is_malignant',
    id_globber    = "^[a-zA-Z]{0,3}[0-9]+",
    tio_queue_kwargs = dict(            # dict passed to ``tio.Queue``
        max_length             = 15,
        samples_per_volume     = 1,
        num_workers            = min(12, os.cpu_count() * 3 // 4),
        shuffle_subjects       = True,
        shuffle_patches        = True,
        start_background       = True,
        verbose                = True,
    )
)

class MySolverCFG(rAIdiologistSolverCFG):
    r"""This is created to cater for the configuration of rAIdiologist network"""
    net           = rAIdiologist(out_ch = 1, cnn_dropout= 0.2, rnn_dropout= 0.2)
    rAI_run_mode  = 1
    optimizer     = 'Adam'
    # init_mom      = 0.95
    init_lr       = 1E-4
    batch_size    = 8
    num_of_epochs = 200

    unpack_key_forward   = ['input'  , 'gt']
    unpack_key_inference = ['input']

    plot_to_tb        = True
    early_stop        = 'loss_reference'
    early_stop_kwargs = {'warmup'       : 150, 'patience': 15}
    accumulate_grad   = 0

    # lr_sche = 'ExponentialLR'
    # lr_sche_args = [0.99]
    # lr_sche = 'OneCycleLR'
    # lr_sche_args = "[]"
    # lr_sche_kwargs = "{'max_lr':1E-3,'total_steps':50,'cycle_momentum':True}"
    # lr_sche = 'CosineAnnealingWarmRestarts'
    # lr_sche_kwargs = "{'T_0': 10, 'T_mult': 2, 'eta_min': 0"
    rAI_inf_save_playbacks = True

    loss_function = ConfidenceBCELoss(pos_weight = torch.as_tensor([1.2]),
                                      conf_factor=1.2,
                                      conf_pos_weight=0.1)

id_list_dir = "./NPC_Segmentation/99.Testing/NPC_BM_LargeStudy/v3-3fold"
# id_list_dir = "./NPC_Segmentation/99.Testing/NPC_Screening/v3"
class MyControllerCFG(PMIControllerCFG):
    run_mode    = 'training'
    fold_code   = 'B00'
    id_list     = id_list_dir + "/{fold_code}.ini"
    id_list_val = id_list_dir + "/Validation.txt"
    output_dir  = './NPC_Segmentation/98.Output/NPC_Screening_old/{fold_code}'
    cp_load_dir = './Backup/rAIdiologist_{fold_code}.pt'
    cp_save_dir = './Backup/rAIdiologist_{fold_code}.pt'
    log_dir     = f"./Backup/Log/rAIdiologist_{datetime.strftime(datetime.now(), '%Y-%m-%d')}.log"
    rAI_pretrained_CNN = './Backup/rAIdiologist_{fold_code}_pretrain.pt'

    _data_loader_cfg     = data_loader
    _data_loader_inf_cfg = data_loader_inf
    data_loader_val_cfg  = data_loader_inf
    data_loader_cls      = PMIImageFeaturePairLoader

    solver_cfg     = MySolverCFG()
    solver_cls     = rAIdiologistSolver
    inferencer_cls = rAIdiologistInferencer

    debug_validation = False
    compile_net = False


class PretrainControllerCFG(MyControllerCFG):
    solver_cls     = BinaryClassificationSolver
    inferencer_cls = BinaryClassificationInferencer
    cp_load_dir    = MyControllerCFG.cp_load_dir.replace('.pt', '_pretrain.pt')
    cp_save_dir    = MyControllerCFG.cp_save_dir.replace('.pt', '_pretrain.pt')
    output_dir     = MyControllerCFG.output_dir + "_pretrain"
    # Override some settings
    solver_cfg     = MySolverCFG()
    solver_cfg.rAI_pretrain_mode = True