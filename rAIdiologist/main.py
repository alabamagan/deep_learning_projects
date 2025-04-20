import sys

from pytorch_med_imaging.controller import PMIController
from pytorch_med_imaging.solvers import BinaryClassificationSolver, ClassificationSolverCFG
from pytorch_med_imaging.inferencers import BinaryClassificationInferencer, ClassificationInferencer
from pytorch_med_imaging.pmi_data_loader import (PMIImageDataLoaderCFG, PMIImageDataLoader,
                                                 PMITorchioDataLoader, PMITorchioDataLoaderCFG)
from rAIdiologist.config import rAIdiologistCFG, SCDControllerCFG
from rAIdiologist.config.network import *
from rAIdiologist.config.loss import *
from rAIdiologist.rai_main import *
from rAIdiologist.rai_controller import rAIController

import yaml
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import click
import copy
from mnts.mnts_logger import MNTSLogger

# This is now the network is created
global rai_options

# Setup metadata for neptune
import neptune

@click.command()
@click.option('--inference', default = False, is_flag = True , help = "For guild operation")
@click.option('--ddp'      , default = False, is_flag = True , help = "For guild operation")
@click.option('--pretrain' , default = False, is_flag = True , help = "For guild operation")
@click.option('--inference-dir', type=click.Path(exists=True, dir_okay=True), required=False,
              help="Override inference directory.")
@click.option('--inference-output-dir', type=click.Path(exists=False, file_okay=False), required=False,
              help="Orveride the inference output directory")
@click.option('--id-globber', type=str, default=None,
              help="Override id-globber for inference. Ignored for training.")
@click.option('--flags-file', type=click.Path(exists=True, dir_okay=False), default='flags.yaml',
              help="Override the flags file.")
@click.option('--flags-hku-data', is_flag=True, default=False,
              help="Use HKU data for inference. Ignored if it's not inference model.")
@click.option('--model', default='rAI', type=click.Choice(['rAI', 'rAI-focused', 'scdense']),
              help="Choose between rAI and scdense.")
def main(inference, ddp, pretrain, inference_dir, inference_output_dir, id_globber, flags_file, flags_hku_data, model):
    if model == 'rAI':
        controller_cls = rAIController
        if not pretrain:
            cfg = rAIdiologistCFG.MyControllerCFG()
        else:
            cfg = rAIdiologistCFG.PretrainControllerCFG()
    if model == 'rAI-focused':
        controller_cls = rAIController
        if not pretrain:
            cfg = rAIdiologistCFG.rAIControllerFocusedCFG()
        else:
            cfg = rAIdiologistCFG.FocusedPretrainControllerCFG()
    elif model == 'scdense':
        cfg = SCDControllerCFG()
        controller_cls = PMIController

    # Guild is giving us trouble for double printing everything, but we can't suppress it's message because it
    # won't capture the scalar values if we do that. So we have to suppress our own logger instead...
    cfg.verbose = False

    # But first, Set the log format
    logger_dict = logging.Logger.manager.loggerDict
    formatter = logging.Formatter(MNTSLogger.log_format)
    for logger_name, logger_instance in logger_dict.items():
        if isinstance(logger_instance, logging.Logger):
            for handlers in logger_instance.handlers:
                handlers.setFormatter(formatter)

    # This sets the format for guild's output
    for handlers in logging.getLogger().handlers:
        handlers.setFormatter(formatter)

    if inference:
        if ddp:
            msg = "Inference mode can't run with DDP mode."
            raise ArithmeticError(msg)
        # Put mode into inference
        cfg.run_mode = 'inference'

        # override original data directory setting if force inference instead of doing testing set evaluation
        if inference_dir is not None:
            # Remove idlist limitation
            cfg.id_list = None
            cfg.data_loader_cfg.input_dir = str(inference_dir)

        if inference_output_dir is not None:
            # change output dir as well
            cfg.output_dir = inference_output_dir

        # create inference dataloader
        if flags_hku_data:
            hku_input_dir = './NPC_Segmentation/60.Large-Study/HKU_data/NyulNormalizer/'
            hku_probmap_dir = './NPC_Segmentation/60.Large-Study/HKU_data/Segmentation/Finesegment_output'
            hku_gt_dir = ('./NPC_Segmentation/60.Large-Study/HKU_data/datasheet.csv', 'is_malignant')
            if isinstance(cfg.data_loader_cfg, PMITorchioDataLoaderCFG):
                cfg.data_loader_cfg.input_data = {
                    'input': hku_input_dir,
                    'probmap': hku_probmap_dir,
                    'gt': hku_gt_dir
                }
            else:
                cfg.data_loader_cfg.input_dir = hku_input_dir
                cfg.data_loader_cfg.probmap_dir = hku_probmap_dir
                cfg.data_loader_cfg.gt_dir = hku_gt_dir
            cfg.data_loader_cfg.target_dir = None


            # Also remove the ID list
            cfg.id_list = None
            cfg.id_list_val = None


    # If pretrain, force mode open to 0, this was done in CFG already but just incase its not loaded properly
    if pretrain:
        # Change the flags file
        with open(flags_file, 'r') as f:
            loaded_flags = yaml.safe_load(f.read())
        loaded_flags['solver_cfg']['rAI_fixed_mode'] = 0
        with open(flags_file, 'w') as f:
            yaml.dump(loaded_flags, f)

    # If DDP mode is not on, simply execute one process
    if not ddp:
        controller = controller_cls(cfg)
        controller.override_cfg(flags_file)
        # Turn off verbosity so that we don't double print
        MNTSLogger.set_global_verbosity(False)

        # override network by guild flags
        if model.find('rAI') != -1:
            controller.solver_cfg.net = rai_options['networks'][controller.net_name]
            if controller.solver_cfg.rAI_classification:
                # When in pretrain mode (i.e., mode = 0), the solver needs to change,
                # if larger than 1, we can't use binary classification solver
                controller._logger.info("Forcing loss function to be CrossEntropyLoss")
                controller.solver_cfg.loss_function = ConfidenceCELoss(weight=torch.FloatTensor([0.5, 1.2]), lambda_1=0.05, lambda_2=0.)
        elif model == 'scdense':
            controller.solver_cfg.net = SCDenseNet()
        controller.exec()
    else:
        # run DDP
        world_size = torch.cuda.device_count()
        if world_size <= 1:
            msg = "DDP mode require more than one CUDA device."
            raise ArithmeticError(msg)

        mp.spawn(DDP_helper.ddp_helper, args=(world_size, copy.deepcopy(cfg), flags_file,), nprocs=world_size)

    # run inference after training
    if not inference:
        controller.cfg.run_mode = 'inference'
        previous_run = controller.plotter.run_id
        controller = controller_cls(controller.cfg)
        controller._plotter.add_tag(f"{model}")
        controller.exec()

if __name__ == '__main__':
    main()