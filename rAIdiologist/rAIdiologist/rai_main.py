import os
import copy
import torch.distributed as dist
from .config.network import *
from .config.network.old.old_swran import SlicewiseAttentionRAN_old
from .rai_controller import rAIController
from pytorch_med_imaging.controller import PMIController, PMIControllerCFG

global rai_options
rai_options = {
    'networks': {
        'rai_v1'      : create_rAIdiologist_v1(),
        'rai_v2'      : create_rAIdiologist_v2(),
        'rai_v3'      : create_rAIdiologist_v3(),
        'rai_v4'      : create_rAIdiologist_v4(),
        'rai_v41'     : create_rAIdiologist_v41(), # with grided ViT
        'rai_v42'     : create_rAIdiologist_v42(), # with grided ViT
        'rai_old'     : create_old_rAI(),
        'rai_old_mean': create_old_rAI_rmean(),
        'mean_swran': SlicewiseAttentionRAN_old(1, 1, reduce_by_mean=True),
        'maxvit': MaxViT(1, 1, 64, (2, 2, 5, 2), window_size=5),
        'old_swran': SlicewiseAttentionRAN_old(1, 1),
        'new_swran': SlicewiseAttentionRAN(1, 1, dropout = 0, reduce_strats='max')
    }
}

class DDP_helper:
    r"""A helper class for distributed data parallel (DDP) training using PyTorch. This class provides a function
    `ddp_helper` that initializes the DDP process group and creates a `rAIController` instance for each process.
    The `rAIController` instance is used to run the training loop on each process, with instance templates passed
    as class attributes. The batch size is adjusted for each process to ensure that the total batch size remains
    the same.
    """
    @classmethod
    def ddp_helper(cls, rank: int, world_size: int, cfg: dict, flags: PMIControllerCFG):
        r"""
        Args:
            rank (int):
                The rank of the current process.
            world_size (int):
                The total number of processes.
            cfg (dict):
                A dictionary containing the configuration options for the `rAIController` instance.
            flags (PMIControllerCFG):
                An object containing the command-line arguments for the training script.
        """
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