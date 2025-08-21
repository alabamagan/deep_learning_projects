"""
Description:
    This files inference the rAIdiologist full network on a given input image or a batch of
    image in a folder, generating a set of heatmap and a csv file that contains the prediction


"""

import sys

from pytorch_med_imaging.controller import PMIController
from pytorch_med_imaging.solvers import BinaryClassificationSolver, ClassificationSolverCFG
from pytorch_med_imaging.inferencers import BinaryClassificationInferencer, ClassificationInferencer
from pytorch_med_imaging.pmi_data_loader import (PMIImageDataLoaderCFG, PMIImageDataLoader,
                                                 PMITorchioDataLoader, PMITorchioDataLoaderCFG)
from ..config import rAIdiologistCFG, SCDControllerCFG
from ..config.network import *
from ..config.loss import *
from ..rai_main import *
from ..rai_controller import rAIController

import yaml
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import click
import copy
from mnts.mnts_logger import MNTSLogger
from pathlib import Path

# This is now the network is created
global rai_options

# Default
DEFAULT_PARAMS = {
    'checkpoint': Path(__file__).parent / "../../assets/checkpoints/rAIdiologist_T2W-FS_c6b89d3596f441f9a4abaab5f47865cb.pt",
    'transform-inf-file': Path(__file__).parent / "../../assets/rAIdiologist_transform_inf.yaml",
}


@click.command()
@click.option('--inference-dir', type=click.Path(exists=True, dir_okay=True), required=False,
              help="Override inference directory.")
@click.option('--inference-output-dir', type=click.Path(exists=False, file_okay=False), required=False,
              help="Override the inference output directory")
@click.option('--inference-transform-file', type=click.Path(exists=True, file_okay=True), required=False,
              default=DEFAULT_PARAMS['transform-inf-file'],help="Override the inference transform file.")
@click.option('--checkpoint-dir', type=click.Path(exists=True, dir_okay=False), required=True,
              default=DEFAULT_PARAMS['checkpoint'], help="Override checkpoint directory.")
@click.option('--id-globber', type=str, default="\w+\d+",
              help="Override id-globber for inference.")
def main(inference_dir, inference_output_dir, inference_transform_file, checkpoint_dir, id_globber):
    # Setup logger
    logger_dict = logging.Logger.manager.loggerDict
    formatter = logging.Formatter(MNTSLogger.log_format)
    for logger_name, logger_instance in logger_dict.items():
        if isinstance(logger_instance, logging.Logger):
            for handlers in logger_instance.handlers:
                handlers.setFormatter(formatter)

    # * Load default CFG and then update the parameters
    cfg = rAIdiologistCFG.MyControllerCFG()
    cfg.run_mode = 'inference'
    cfg.verbose = False
    cfg.net_name = 'rai_v5.1'
    cfg.data_loader_cfg.id_globber = id_globber
    cfg.debug_mode = False

    # override original data directory setting if force inference instead of doing testing set evaluation
    if inference_dir is not None:
        inference_dir = Path(inference_dir)
        # Remove idlist limitation
        cfg.id_list = None
        cfg.id_list_val = None
        cfg.plotting = False
        cfg.data_loader_cfg.input_dir = str(inference_dir)
        cfg.data_loader_cfg.target_dir = None
        cfg.data_loader_cfg.probmap_dir = str(inference_dir.parent / 'HuangThresholding')
        cfg.data_loader_cfg.augmentation = str(inference_transform_file)

    if inference_output_dir is not None:
        # change output dir as well
        cfg.output_dir = inference_output_dir

    # specify checkpoint dir
    cfg.cp_load_dir = checkpoint_dir

    # -- Create the controller
    controller = rAIController(cfg)

    # * Set the network here as inferencer hasn't been created until exec()
    controller.solver_cfg.net = rai_options['networks'][controller.net_name]
    controller.solver_cfg.rAI_fixed_mode = 5

    # -- Run
    controller.exec()
