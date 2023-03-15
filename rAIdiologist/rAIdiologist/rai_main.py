import os
import copy
import torch.distributed as dist
from .config.network import *
from .rai_controller import rAIController
from pytorch_med_imaging.controller import PMIController, PMIControllerCFG

global rai_options
rai_options = {
    'networks': {
        'rai_v1': rAIdiologist(out_ch = 1, dropout = 0.2, lstm_dropout = 0.2),
        'rai_v2': rAIdiologist_v2(out_ch = 1, dropout = 0.2, lstm_dropout = 0.2),
        'rai_v2_drop': rAIdiologist_v2(out_ch = 1, dropout = 0.3, lstm_dropout = 0.2),
        'rai_v2_mean': rAIdiologist_v2(out_ch = 1, dropout = 0.2, lstm_dropout = 0.2, reduce_strats='mean'),
        'rai_v3': rAIdiologist_v3(out_ch = 1, dropout = 0.25, lstm_dropout = 0.2),
        'rai_v4': rAIdiologist_v4(out_ch = 1, dropout = 0.2, lstm_dropout = 0.2)
    }
}

class DDP_helper:
    r"""Guild helper function. Because guild wraps everything that is in the file main.py, a separate helper class
    is needed. This helper class allows passing instance templates as class attributes."""
    @classmethod
    def ddp_helper(cls, rank, world_size, cfg, flags):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "23455"
        dist.init_process_group("nccl", world_size=world_size, rank=rank)

        controller = rAIController(cfg)
        controller.override_cfg(flags)
        # Change the batch-size because each controller only has one GPU
        controller.solver_cfg.batch_size = controller.solver_cfg.batch_size // world_size
        # Override the network setting
        controller.solver_cfg.net = rai_options['networks'][controller.net_name]
        controller.exec()

        dist.barrier()
        dist.destroy_process_group()