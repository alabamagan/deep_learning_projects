import sys

from pytorch_med_imaging.controller import PMIController
from pytorch_med_imaging.pmi_data_loader import PMITorchioDataLoader, PMITorchioDataLoaderCFG
from rAIdiologist.config.SCDNetCFG import *
from rAIdiologist.config.network import *
from rAIdiologist.config.loss import *
from rAIdiologist.solvers.scdnetSolver import SCDenseNetSolver, SCDenseNetSolverCFG

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

# Setup print options
torch.set_printoptions(2, 20, 1, 120, sci_mode=False)

@click.command()
@click.option('--inference', default = False, is_flag = True , help = "For guild operation")
@click.option('--ddp'      , default = False, is_flag = True , help = "For guild operation")
@click.option('--focused'  , is_flag = True                  , help = "For guild operation")
@click.option('--inference-dir', type=click.Path(exists=True, dir_okay=True), required=False,
              help="Override inference directory.")
@click.option('--id-globber', type=str, default=None,
              help="Override id-globber for inference. Ignored for training.")
def main(inference, ddp, inference_dir, id_globber, focused):
    cfg = SCDControllerCFG()

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
            # create inference dataloader
            data_loader_inf = PMITorchioDataLoaderCFG(
                input_data = {
                    'input': str(inference_dir)
                },
                id_globber = id_globber or cfg._data_loader_inf_cfg.id_globber,
                augmentation  = cfg._data_loader_inf_cfg.augmentation
            )
            # Note that if both are set, inference assumes there's ground-truth data available
            # and will attempt to load it even when its not correct
            cfg._data_loader_cfg = data_loader_inf
            cfg._data_loader_inf_cfg = None
            cfg.data_loader_cls = PMITorchioDataLoader


    # If DDP mode is not on, simply execute one process
    if not ddp:
        controller = PMIController(cfg)
        controller.override_cfg('flags_scdense.yaml')
        controller.plotter.add_tag("SCDense")
        # Turn off verbosity so that we don't double print
        MNTSLogger.set_global_verbosity(False)

        # override network by guild flags
        controller.solver_cfg.net = SCDenseNet() # Settings are ignored here
        controller.exec()
    else:
        # run DDP
        world_size = torch.cuda.device_count()
        if world_size <= 1:
            msg = "DDP mode require more than one CUDA device."
            raise ArithmeticError(msg)
        raise NotImplementedError("DDP for SCDenseNet is not yet implemented")
        # mp.spawn(DDP_helper.ddp_helper, args=(world_size, copy.deepcopy(cfg), 'flags.yaml',), nprocs=world_size)

    # run inference after training
    if not inference:
        controller.cfg.run_mode = 'inference'
        previous_run = controller.plotter.run_id
        controller = PMIController(controller.cfg)
        controller._plotter.add_tag("SCDense")
        controller.exec()




if __name__ == '__main__':
    main()