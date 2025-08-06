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
        'rai_v1'          : create_rAIdiologist_v1(),
        'rai_v2'          : create_rAIdiologist_v2(),
        'rai_v3'          : create_rAIdiologist_v3(),
        'rai_v4'          : create_rAIdiologist_v4(),
        'rai_v41'         : create_rAIdiologist_v41()         , # with grided ViT
        'rai_v42'         : create_rAIdiologist_v42()         , # with grided ViT
        'rai_v43'         : create_rAIdiologist_v43()         , # with grided ViT
        'rai_v5'          : create_rAIdiologist_v5()          , # with transformer
        'rai_v5.1'        : create_rAIdiologist_v5_1(),
        'rai_v5.1-focused': create_rAIdiologist_v5_1_focused(),
        'rai_old'         : create_old_rAI(),
        'rai_old_mean'    : create_old_rAI_rmean(),
        'mean_swran'      : SlicewiseAttentionRAN_old(1, 1, reduce_by_mean = True),
        'maxvit'          : MaxViT(1, 1, 64, (2, 2, 5, 2), window_size = 5),
        'old_swran'       : SlicewiseAttentionRAN_old(1, 1),
        'new_swran'       : SlicewiseAttentionRAN(1, 1, dropout=0, reduce_strats = 'max'),
        'scdnet'          : SCDenseNet(),
        'resnet3d101'     : get_ResNet3d_101(),     # Run in pretrain mode only
        'vgg16'           : get_vgg16(),            # Run in pretrain mode only
        'vgg11'           : get_vgg('11')
        'densenet3d121'   : get_densenet3d_121(),   # DenseNet3D-121
        'densenet3d169'   : get_densenet3d('169'),  # DenseNet3D-169
        'densenet3d201'   : get_densenet3d('201'),  # DenseNet3D-201
        'densenet3d264'   : get_densenet3d('264'),  # DenseNet3D-264
        'efficientnet3d_b0': get_efficientnet3d_b0(), # EfficientNet3D-B0
        'efficientnet3d_b1': get_efficientnet3d('efficientnet-b1'), # EfficientNet3D-B1
        'efficientnet3d_b2': get_efficientnet3d('efficientnet-b2'), # EfficientNet3D-B2
        'efficientnet3d_b3': get_efficientnet3d('efficientnet-b3'), # EfficientNet3D-B3
        'efficientnet3d_b4': get_efficientnet3d('efficientnet-b4'), # EfficientNet3D-B4
        'efficientnet3d_b5': get_efficientnet3d('efficientnet-b5'), # EfficientNet3D-B5
        'efficientnet3d_b6': get_efficientnet3d('efficientnet-b6'), # EfficientNet3D-B6
        'efficientnet3d_b7': get_efficientnet3d('efficientnet-b7'), # EfficientNet3D-B7
        'resnext3d50'     : get_resnext3d_50(),     # ResNeXt3D-50
        'resnext3d101'    : get_resnext3d('101'),   # ResNeXt3D-101
        'resnext3d152'    : get_resnext3d('152'),   # ResNeXt3D-152
        'mobilenet_v2_3d' : get_mobilenet_v2_3d(),  # MobileNetV2-3D
        'mobilenet_v3_3d' : get_mobilenet_v3_3d(),   # MobileNetV3-3D
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