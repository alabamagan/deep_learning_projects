from pytorch_med_imaging.controller import PMIController, PMIControllerCFG
from pytorch_med_imaging.solvers import BinaryClassificationSolver
from pytorch_med_imaging.inferencers import BinaryClassificationInferencer
from typing import Union, Optional
from pathlib import Path

PathLike = Union[str, Path]

class rAIController(PMIController):
    r""""""
    def override_cfg(self, override_file: PathLike):
        r"""Additional overrides that is specific with rAI configurations"""
        super(rAIController, self).override_cfg(override_file)
        #
        # if self.solver_cfg.rAI_fixed_mode >= 0:
        #     self.data_loader_cfg.augmentation = './v1_rAIdiologist_transform.yaml'
        #     self.data_loader_cfg.sampler = None
        #     self.data_loader_cfg.sampler_kwargs = None

        # If MaxVit is used size required is [320, 320, 20]
        if self.net_name.find('maxvit') >= 0:
            self.data_loader_cfg.sampler_kwargs = dict(
                patch_size = [320,320,20]
            )

            self.solver_cls = BinaryClassificationSolver
            self.inferencer_cls = BinaryClassificationInferencer
            self._data_loader_inf_cfg.augmentation = './maxvit_inf_transform.yaml'

