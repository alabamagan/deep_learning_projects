import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from typing import Union, Any
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
        g = g.detach().cpu()
        res = res.detach().cpu()
        chan = res.shape[-1] # if chan > 1, there is a value for confidence
        _data =np.concatenate([res.view(-1, chan).data.numpy(), g.data.view(-1, 1).numpy()], axis=-1)
        _df = pd.DataFrame(data=_data, columns=['res_%i'%i for i in range(chan)] + ['g'])
        _df['Verify_wo_conf'] = (_df['res_0'] >= 0) == (_df['g'] > 0)
        _df['Verify_wo_conf'].replace({True: "Correct", False: "Wrong"}, inplace=True)
        if chan == 2:
            _df['Verify_w_conf'] = ((_df['res_0'] >= 0) == (_df['g'] > 0)) == (_df['res_1'] >= 0)
            _df['Verify_w_conf'].replace({True: "Correct", False: "Wrong"}, inplace=True)
        if chan == 4:
            rename_dict = {
                'res_0': 'overall_pred',
                'res_1': 'CNN_pred',
                'res_2': 'Weights',
                'res_3': 'LSTM_pred'
            }
            _df.rename(rename_dict, inplace=True, axis=1)
            _df['Same sign'] = (_df['CNN_pred'] / _df['CNN_pred'].abs()) \
                              == (_df['LSTM_pred'] / _df['LSTM_pred'].abs())

        # res: (B x C)/(B x 1)
        if chan > 1:
            dic = torch.zeros_like(res[..., 0])
            dic = dic.type_as(res).int() # move to cuda if required
            dic[torch.where(res[..., 0] >= 0)] = 1
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
            self._set_net_mode(current_mode)

    def _set_net_mode(self, mode):
        try:
            self.net.get_submodule('module').set_mode(mode)
            self.net.get_submodule('module').RECORD_ON = False
        except:
            self.net.set_mode(mode)
            self.net.RECORD_ON = False
        self._current_mode = mode

    def validation(self) -> list:
        original_mode = self.get_net()._mode.item()
        if original_mode > 3:
            self._set_net_mode(-1) # inference when not pretraining (i.e., mode = 0)
        super(rAIdiologistSolver, self).validation()
        self._set_net_mode(original_mode)

    def _validation_step_callback(self, g: torch.Tensor, res: torch.Tensor, loss: Union[torch.Tensor, float],
                                  uids=None) -> None:
        r"""Uses :attr:`perf` to store the dictionary of various data."""
        self.validation_losses.append(loss.item())
        if len(self.perfs) == 0:
            self.perfs.append({
                'dics'       : [],
                'gts'        : [],
                'predictions': [],
                'confidence' : [],
                'uids'       : []
            })
        store_dict = self.perfs[0]
        g, res = self._align_g_res_size(g, res)
        _df, dic = self._build_validation_df(g, res)

        # Decision were made by checking whether value is > 0.5 after sigmoid
        store_dict['dics'].extend(dic)
        store_dict['gts'].extend(g)
        if res.shape[1] > 1:
            store_dict['predictions'].extend(res[:, 0].flatten().tolist())
            store_dict['confidence'].extend(res[:, 1].flatten().tolist())
        else:
            store_dict['predictions'].extend(res.flatten().tolist())
        if isinstance(uids, (tuple, list)):
            store_dict['uids'].extend(uids)