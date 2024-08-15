from pytorch_med_imaging.controller import PMIController
from pytorch_med_imaging.solvers import BinaryClassificationSolver, ClassificationSolverCFG
from pytorch_med_imaging.inferencers import BinaryClassificationInferencer, ClassificationInferencer
from rAIdiologist.config.rAIdiologistCFG import *
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
@click.option('--inference', default=False, is_flag=True, help="For guild operation")
@click.option('--ddp', default=False, is_flag=True, help="For guild operation")
@click.option('--pretrain', default=False, is_flag=True, help="For guild operation")
def main(inference, ddp, pretrain):
    if not pretrain:
        cfg = MyControllerCFG()
    else:
        cfg = PretrainControllerCFG()

    # Guild is giving us trouble for double printing everything
    cfg.verbose = False
    logger_dict = logging.Logger.manager.loggerDict
    formatter = logging.Formatter(MNTSLogger.log_format)
    for logger_name, logger_instance in logger_dict.items():
        if isinstance(logger_instance, logging.Logger):
            for handlers in logger_instance.handlers:
                handlers.setFormatter(formatter)

    if inference:
        if ddp:
            msg = "Inference mode can't run with DDP mode."
            raise ArithmeticError(msg)
        cfg.run_mode = 'inference'

    # If pretrain, force mode open to 0, this was done in CFG already but just incase its not loaded properly
    if pretrain:
        # Change the flags file
        with open('flags.yaml', 'r') as f:
            loaded_flags = yaml.safe_load(f.read())
        loaded_flags['solver_cfg']['rAI_fixed_mode'] = 0
        with open('flags.yaml', 'w') as f:
            yaml.dump(loaded_flags, f)

    # If DDP mode is not on, simply execute one process
    if not ddp:
        controller = rAIController(cfg)
        controller.override_cfg('flags.yaml')

        # override network by guild flags
        controller.solver_cfg.net = rai_options['networks'][controller.net_name]
        if controller.solver_cfg.rAI_classification:
            # When in pretrain mode (i.e., mode = 0), the solver needs to change,
            # if larger than 1, we can't use binary classification solver
            controller._logger.info("Forcing loss function to be CrossEntropyLoss")
            controller.solver_cfg.loss_function = ConfidenceCELoss(weight=torch.FloatTensor([0.5, 1.2]), lambda_1=0.05, lambda_2=0.)
        controller.exec()
    else:
        # run DDP
        world_size = torch.cuda.device_count()
        if world_size <= 1:
            msg = "DDP mode require more than one CUDA device."
            raise ArithmeticError(msg)

        mp.spawn(DDP_helper.ddp_helper, args=(world_size, copy.deepcopy(cfg), 'flags.yaml',), nprocs=world_size)

    # run inference after training
    if not inference:
        controller.cfg.run_mode = 'inference'
        controller = rAIController(controller.cfg)
        controller.exec()



if __name__ == '__main__':
    main()