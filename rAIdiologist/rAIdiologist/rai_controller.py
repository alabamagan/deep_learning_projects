from pytorch_med_imaging.controller import PMIController, PMIControllerCFG
from pytorch_med_imaging.solvers import BinaryClassificationSolver
from pytorch_med_imaging.inferencers import BinaryClassificationInferencer
from typing import Union, Optional
from pathlib import Path
import torch.nn as nn

PathLike = Union[str, Path]

class rAIController(PMIController):
    r""""""
    def override_cfg(self, override_file: PathLike):
        r"""Additional overrides that is specific with rAI configurations"""
        super(rAIController, self).override_cfg(override_file)


        # If MaxVit is used size required is [320, 320, 20]
        if self.net_name.find('maxvit') >= 0:
            self.data_loader_cfg.sampler_kwargs = dict(
                patch_size = [320,320,20]
            )

            self.solver_cls = BinaryClassificationSolver
            self.inferencer_cls = BinaryClassificationInferencer
            self._data_loader_inf_cfg.augmentation = './maxvit_inf_transform.yaml'

        # * Save some memory by using less batch-size during validation
        if self.solver_cfg.rAI_fixed_mode >= 3:
            self.solver_cfg.batch_size_val = self.solver_cfg.batch_size // 4

        # * Pass attribute from controller to solver
        # passing the network checkpoint, this is needed because guild flags are not optimized
        self.solver_cfg.rAI_pretrained_CNN = self.cfg.rAI_pretrained_CNN
        self.solver_cfg.net_name = self.net_name


    def exec(self):
        r"""Because the network might be redefined by guild after :func:`override_cfg` is called, the mode is explicitly
        set here again since mode 0 using `BinaryClassificationSolver` doesn't have the callback to set it during run.
        """
        if not self.net_name in ('old_swran', 'new_swran', 'mean_swran', 'maxvit'):
            self.solver_cfg.net.set_mode(self.solver_cfg.rAI_fixed_mode)
        super(rAIController, self).exec()
