import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pytorch_med_imaging.solvers import BinaryClassificationSolver, ClassificationSolverCFG
from ..config.network.rAIdiologist import rAIdiologist
import gc

__all__ = ['rAIdiologistSolver']


class rAIdiologistSolverCFG(ClassificationSolverCFG):
    r"""
    Class Attributes:
        rAI_fixed_mode (int):
            Select from mode 1 - 5. Otherwise the mode will automatically progress based on the total epoch number
            and the current epoch number. Default to ``None``.
        rAI_pretrained_swran (str):
            If specified, the CNN portion will load the pre-trained SWRAN states. Otherwise, it is initialized using
            default initialization method. Default to empty string ``""``.
        rAI_classification (bool):
            If ``True``, the solver will run as a classification problem rather than binarQy classification problem.
            Default to ``False``.
        rAI_inf_save_playbacks (bool):
            If ``True``, play back will be saved to the ``output_dir``. Typically use with inference mode.

    """
    rAI_fixed_mode        : int  = None
    rAI_pretrained_swran  : str  = ""
    rAI_classification    : bool = False
    rAI_inf_save_playbacks: bool = False

    loss_function = nn.BCEWithLogitsLoss()
    net = rAIdiologist(1)

class rAIdiologistSolver(BinaryClassificationSolver):
    r"""This solver is written to train :class:`rAIdiologist` network. This mainly inherits :func:`._epoch_prehook` to
    set the network mode based on the current epoch number. This also customize :func:`_build_validation_df` to present
    the results better. """
    def __init__(self, cfg, *args, **kwargs):
        super(rAIdiologistSolver, self).__init__(cfg, *args, **kwargs)

        self._current_mode = None # initially, the mode is unsetted

        # Load fro mstored state
        if Path(self.rAI_pretrained_swran).is_file():
            self._logger.info(f"Loading pretrained SWRAN network from: {self.rAI_pretrained_swran}")
            result = self.net.load_pretrained_swran(self.rAI_pretrained_swran)
            if str(result) != "<All keys matched successfully>":
                self._logger.warning(f"Some keys were not loaded.")
                self._logger.warning(f"{result}")
        else:
            self._logger.warning(f"Pretrained SWRAN network specified ({self.rAI_pretrained_swran}) "
                                 f"but not loaded.")

        if os.getenv('CUBLAS_WORKSPACE_CONFIG') not in [":16:8", ":4096:2"]:
            self._logger.warning(f"Env variable CUBLAS_WORKSPACE_CONFIG was not set properly, which may invalidate"
                                 f" deterministic behavior of LSTM.")

        # Turn off record
        self.get_net()._RECORD_ON = False

    def prepare_lossfunction(self):
        if not self.rAI_classification:
            super(rAIdiologistSolver, self).prepare_lossfunction()
        else:
            super(BinaryClassificationSolver, self).prepare_lossfunction()

    def _build_validation_df(self, g, res, uid=None):
        r"""Tailored for rAIdiologist, model output were of shape (B x 3), where the first element is
        the prediction, the second element is the confidence and the third is irrelevant and only used
        by the network. In mode 0, the output shape is (B x 1)"""

        # res: (B x C)/(B x 1), g: (B x 1)
        chan = res.shape[-1] # if chan > 1, there is a value for confidence
        _data =np.concatenate([res.view(-1, chan).data.cpu().numpy(), g.data.view(-1, 1).cpu().numpy()], axis=-1)
        _df = pd.DataFrame(data=_data, columns=['res_%i'%i for i in range(chan)] + ['g'])
        _df['Verify_wo_conf'] = (_df['res_0'] >= 0) == (_df['g'] > 0)
        _df['Verify_wo_conf'].replace({True: "Correct", False: "Wrong"}, inplace=True)
        if chan > 1:
            _df['Verify_w_conf'] = ((_df['res_0'] >= 0) == (_df['g'] > 0)) == (_df['res_1'] >= 0)
            _df['Verify_w_conf'].replace({True: "Correct", False: "Wrong"}, inplace=True)

        # res: (B x C)/(B x 1)
        if chan > 1:
            dic = torch.zeros_like(res[..., :-1])
            dic = dic.type_as(res).int() # move to cuda if required
            dic[torch.where(res[..., :-1] >= 0)] = 1
        else:
            dic = (res >= 0).type_as(res).int()

        if uid is not None:
            try:
                _df.index = uid
            except:
                pass
        return _df, dic.view(-1, 1)

    def _align_g_res_size(self, g, res):
        # g: (B x 1), res is not important here

        g = g.squeeze()
        return g.view(-1, 1), res

    def _epoch_prehook(self, *args, **kwargs):
        r"""Update mode of network"""
        super(rAIdiologistSolver, self)._epoch_prehook(*args, **kwargs)
        current_epoch = self.plotter_dict.get('epoch_num', 0)
        total_epoch = self.num_of_epochs

        # Schedule mode of the network and findout if new mode is needed
        if self.rAI_fixed_mode is None:
            # mode is scheduled to occupy 25% of all epochs
            epoch_progress = current_epoch / float(total_epoch)
            current_mode = min(int(epoch_progress * 4) + 1, 4)
        else:
            current_mode = int(self.rAI_fixed_mode)

        # If new mode is needed, change mode
        if not current_mode == self._current_mode:
            self._logger.info(f"Setting rAIdiologist mode to {current_mode}")
            self._current_mode = current_mode
            if isinstance(self.net, torch.nn.DataParallel):
                self.net.get_submodule('module').set_mode(self._current_mode)
            else:
                self.net.set_mode(self._current_mode)

