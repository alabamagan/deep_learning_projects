from pytorch_med_imaging.solvers import BinaryClassificationSolver
from pytorch_med_imaging.controller import PMIControllerCFG
from pytorch_med_imaging.pmi_data_loader import PMIImageFeaturePairLoader, PMIImageFeaturePairLoaderCFG
from pytorch_med_imaging.solvers.earlystop import LossReferenceEarlyStop
from .network.rAIdiologist import rAIdiologist
from .network.slicewise_ran import SlicewiseAttentionRAN
from ..solvers.rAIdiologistSolver import rAIdiologistSolverCFG, rAIdiologistSolver
from ..solvers.rAIdiologistInferencer import *
from dataclasses import asdict
import copy
from datetime import datetime

data_loader = PMIImageFeaturePairLoaderCFG(
    input_dir     = './NPC_Segmentation/60.Large-Study/v1-All-Data/Normalized_2/T2WFS_TRA/01.NyulNormalized/',
    probmap_dir   = './NPC_Segmentation/60.Large-Study/v1-All-Data/Normalized_2/T2WFS_TRA/00.HuangMask/',
    target_dir    = './NPC_Segmentation/60.Large-Study/v1-All-Data/v2-datasheet.csv',
    augmentation  = './v2_rAIdiologist_transform.yaml',
    target_column = 'is_malignant',
    id_globber    = "^[a-zA-Z]{0,3}[0-9]+",
    sampler       = 'weighted', # Unset sampler to load the whole image
    sampler_kwargs    = dict(
        patch_size = [325, 325, 25]
    ),
    tio_queue_kwargs = dict(            # dict passed to ``tio.Queue``
        max_length             = 15,
        samples_per_volume     = 1,
        num_workers            = 8,
        shuffle_subjects       = True,
        shuffle_patches        = True,
        start_background       = True,
        verbose                = True,
    )
)

data_loader_inf = PMIImageFeaturePairLoaderCFG(
    input_dir     = './NPC_Segmentation/60.Large-Study/v1-All-Data/Normalized_2/T2WFS_TRA/01.NyulNormalized/',
    probmap_dir   = './NPC_Segmentation/60.Large-Study/v1-All-Data/Normalized_2/T2WFS_TRA/00.HuangMask/',
    target_dir    = './NPC_Segmentation/60.Large-Study/v1-All-Data/v2-datasheet.csv',
    augmentation  = './v1_rAIdiologist_transform.yaml',
    target_column = 'is_malignant',
    id_globber    = "^[a-zA-Z]{0,3}[0-9]+",
    tio_queue_kwargs = dict(            # dict passed to ``tio.Queue``
        max_length             = 15,
        samples_per_volume     = 1,
        num_workers            = 8,
        shuffle_subjects       = True,
        shuffle_patches        = True,
        start_background       = True,
        verbose                = True,
    )
)

class MySolverCFG(rAIdiologistSolverCFG):
    r"""This is created to cater for the configuration of rAIdiologist network"""
    net          = rAIdiologist(out_ch = 1, dropout = 0.2, lstm_dropout = 0.2)
    rAI_run_mode = 1
    optimizer    = 'Adam'
    init_lr      = 1E-4
    batch_size = 8
    num_of_epochs = 200

    unpack_key_forward   = ['input'  , 'gt']
    unpack_key_inference = ['input']

    class_weights = [1] # class weight for NPC +ve

    plot_to_tb = True
    early_stop = 'loss_reference'
    early_stop_kwargs = {'warmup': 80, 'patience': 15}
    accumulate_grad = 4

    lr_sche = 'ExponentialLR'
    lr_sche_args = [0.99]
    rAI_inf_save_playbacks = True


class MyControllerCFG(PMIControllerCFG):
    run_mode     = 'training'
    fold_code     = 'B00'
    id_list       = "./NPC_Segmentation/99.Testing/NPC_BM_LargeStudy/v3-3fold/{fold_code}.ini"
    id_list_val   = "./NPC_Segmentation/99.Testing/NPC_BM_LargeStudy/v3-3fold/Validation.txt"
    output_dir    = './NPC_Segmentation/98.Testing/NPC_BM_rAI/{fold_code}'
    cp_load_dir   = './Backup/rAIdiologist_{fold_code}.pt'
    cp_save_dir   = './Backup/rAIdiologist_{fold_code}.pt'
    log_dir       = f"./Backup/Log/rAIdiologist_{datetime.strftime(datetime.now(), '%Y-%m-%d')}.log"

    _data_loader_cfg = data_loader
    _data_loader_inf_cfg = data_loader_inf
    data_loader_val_cfg = data_loader_inf
    data_loader_cls = PMIImageFeaturePairLoader

    solver_cfg = MySolverCFG()
    solver_cls = rAIdiologistSolver
    inferencer_cls = rAIdiologistInferencer

    debug_validation = False

