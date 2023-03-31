from pytorch_med_imaging.controller import PMIController
from rAIdiologist.config.rAIdiologistCFG import *
from rAIdiologist.config.network import *
from rAIdiologist.rai_main import *
from rAIdiologist.rai_controller import rAIController
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import click
import copy

global rai_options


@click.command()
@click.option('--inference', default=False, is_flag=True, help="For guild operation")
@click.option('--ddp', default=False, help="For guild operation")
@click.option('--pretrain', default=False, is_flag=True, help="For guild operation")
def main(inference, ddp, pretrain):
    if not pretrain:
        cfg = MyControllerCFG()
    else:
        cfg = PretrainControllerCFG()
    if inference:
        if ddp:
            msg = "Inference mode can't run with DDP mode."
            raise ArithmeticError(msg)
        cfg.run_mode = 'inference'

    # If DDP mode is not on, simply execute one process
    if not ddp:
        controller = rAIController(cfg)
        controller.override_cfg('flags.yaml')

        # override network by guild flags
        controller.solver_cfg.net = rai_options['networks'][controller.net_name]
        controller.exec()
    else:
        # run DDP
        world_size = torch.cuda.device_count()
        if world_size <= 1:
            msg = "DDP mode require more than one CUDA device."
            raise ArithmeticError(msg)

        mp.spawn(DDP_helper.ddp_helper, args=(world_size, copy.deepcopy(cfg), 'flags.yaml',), nprocs=world_size)


if __name__ == '__main__':
    main()