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

data_loader = PMIImageFeaturePairLoaderCFG(
    input_dir     = './NPC_Segmentation/60.Large-Study/v1-All-Data/Normalized_2/T2WFS_TRA/01.NyulNormalized/',
    probmap_dir   = './NPC_Segmentation/60.Large-Study/v1-All-Data/Normalized_2/T2WFS_TRA/00.HuangMask/',
    target_dir    = './NPC_Segmentation/60.Large-Study/v1-All-Data/v2-datasheet.csv',
    augmentation  = './v1_rAIdiologist_transform.yaml',
    target_column = 'is_malignant',
    id_globber    = "^[a-zA-Z]{0,3}[0-9]+",
    sampler = None, # Unset sampler to load the whole image
    tio_queue_kwargs = dict(            # dict passed to ``tio.Queue``
        max_length             = 15,
        samples_per_volume     = 1,
        num_workers            = 4,
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
    run_mode     = 'training'
    optimizer    = 'Adam'
    init_lr      = 1E-4
    batch_size = 8

    unpack_key_forward   = ['input'  , 'gt']
    unpack_key_inference = ['input']

    class_weights = [1.2] # class weight for NPC +ve

    plot_to_tb = True
    early_stop = 'loss_reference'
    early_stop_kwargs = {'warmup': 40, 'patience': 15}


class MyControllerCFG(PMIControllerCFG):
    fold_code = 'B00'
    id_list = "./NPC_Segmentation/99.Testing/NPC_BM_LargeStudy/v3-3fold/{fold_code}.ini"
    id_list_val = "./NPC_Segmentation/99.Testing/NPC_BM_LargeStudy/v3-3fold/Validation.txt"
    output_dir = './NPC_Segmentation/98.Testing/NPC_BM_rAI/{fold_code}'
    cp_load_dir = './Backup/rAIdiologist_{fold_code}.pt'
    cp_save_dir = './Backup/rAIdiologist_{fold_code}.pt'
    log_dir = "./Backup/Log/"

    data_loader_cfg = data_loader
    data_loader_cls = PMIImageFeaturePairLoader

    solver_cfg = MySolverCFG()
    solver_cls = rAIdiologistSolver
    inferencer_cls = rAIdiologistInferencer

    debug_validation = False

