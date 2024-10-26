import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from typing import Union, Any
from pytorch_med_imaging.solvers import BinaryClassificationSolver, ClassificationSolverCFG, ClassificationSolver
from ..config.network.rAIdiologist import rAIdiologist
from ..config.loss.rAIdiologist_loss import ConfidenceCELoss
import gc

__all__ = ['rAIdiologistSolver']


class rAIdiologistSolverCFG(ClassificationSolverCFG):
    r"""
    Attributes:
        rAI_fixed_mode (int):
            Select from mode 1 - 5. Otherwise the mode will automatically progress based on the total epoch number
            and the current epoch number. Default to ``None``.
        rAI_pretrained_CNN (str):
            If specified, the CNN portion will load the pre-trained SWRAN states. Otherwise, it is initialized using
            default initialization method. Default to empty string ``""``.
        rAI_classification (bool):
            If ``True``, the solver will run as a classification problem rather than binarQy classification problem.
            Default to ``False``.
        rAI_inf_save_playbacks (bool):
            If ``True``, play back will be saved to the ``output_dir``. Typically use with inference mode.
        rAI_pretrain_mode (bool):


    """
    rAI_fixed_mode        : int  = None
    rAI_pretrained_CNN    : str  = ""
    rAI_classification    : bool = False
    rAI_inf_save_playbacks: bool = False
    rAI_pretrain_mode     : bool = False

    loss_function = nn.BCEWithLogitsLoss()
    net = rAIdiologist(1)

class rAIdiologistSolver(BinaryClassificationSolver):
    r"""This solver is written to train :class:`rAIdiologist` network. This mainly inherits :func:`._epoch_prehook` to
    set the network mode based on the current epoch number. This also customize :func:`_build_validation_df` to present
    the results better.

    .. note::
        If `self.rAI_classification` is set to True, the solver will pass some function to ClassificationSolver because
        the output of the network is expect to be multi-class classification.
    """
    def __init__(self, cfg, *args, **kwargs):
        super(rAIdiologistSolver, self).__init__(cfg, *args, **kwargs)

        self._current_mode = None # initially, the mode is unsetted

        # Load from stored state
        if Path(self.rAI_pretrained_CNN).is_file():
            self._logger.info(f"Loading pretrained CNN network from: {self.rAI_pretrained_CNN}")
            result = self.net.load_pretrained_CNN(self.rAI_pretrained_CNN)
            if str(result) != "<All keys matched successfully>":
                self._logger.warning(f"Some keys were not loaded.")
                self._logger.warning(f"{result}")
        else:
            self._logger.warning(f"Pretrained CNN network specified ({self.rAI_pretrained_CNN}) "
                                 f"but not loaded.")

        if os.getenv('CUBLAS_WORKSPACE_CONFIG') not in [":16:8", ":4096:2"]:
            self._logger.warning(f"Env variable CUBLAS_WORKSPACE_CONFIG was not set properly, which may invalidate"
                                 f" deterministic behavior of LSTM.")

        # Turn off record
        self.get_net()._RECORD_ON = False
        self._validation_misclassification_record = {} # Doesn't know why sometimes this is not inherited from
                                                       # BinaryClassificationSolver

    def prepare_lossfunction(self):
        if not self.rAI_classification:
            super(rAIdiologistSolver, self).prepare_lossfunction()
        else:
            super(BinaryClassificationSolver, self).prepare_lossfunction()

    def _build_validation_df(self, g, res, uid=None):
        r"""Tailored for rAIdiologist, model output were of shape (B x 3), where the first element is
        the prediction, the second element is the confidence and the third is irrelevant and only used
        by the network. In mode 0, the output shape is (B x 1)"""

        # res: (B x S x C)/(B x S x 1)/B x (S x C), g: (B x 1)
        self._logger.debug(f"{g.shape = }; {res.shape = }")
        if self.rAI_classification:
            if isinstance(self.loss_function, ConfidenceCELoss) and isinstance(res, (list, tuple)):
                pred, conf = res
                _df = pd.DataFrame.from_dict({f'res_{d}': list(pred[:, d].numpy())
                                              for d in range(pred.shape[-1])})
                _df_gt = pd.DataFrame.from_dict({'gt': list(g.flatten().numpy())})
                _df_conf = pd.DataFrame.from_dict({'conf': list(conf.flatten().numpy())})
                _df = pd.concat([_df, _df_conf, _df_gt], axis=1)
                _df['predicted'] = torch.argmax(pred.squeeze(), dim=1).numpy()
                _df['eval'] = (_df['predicted'] == _df['gt']).replace({True: 'Correct', False: 'Wrong'})
                return _df, _df['predicted']
            else:
                return super(BinaryClassificationSolver, self)._build_validation_df(g, res, uid)
        else:
            g = g.detach().cpu()
            res = res.detach().cpu()

            chan = res.shape[1] # if chan > 1, there is a value for confidence
            _data =np.concatenate([res.view(-1, chan).data.numpy(), g.data.view(-1, 1).numpy()], axis=-1)
            _df = pd.DataFrame(data=_data, columns=['res_%i'%i for i in range(chan)] + ['g'])
            _df['Verify_wo_conf'] = (_df['res_0'] >= 0) == (_df['g'] > 0)
            _df.replace({'Verify_wo_conf':{True: "Correct", False: "Wrong"}}, inplace=True)
            if chan > 2 and chan < 4:
                _df['Verify_w_conf'] = ((_df['res_0'] >= 0) == (_df['g'] > 0)) == (_df['res_1'] >= 0)
                _df.replace({'Verify_w_conf': {True: "Correct", False: "Wrong"}}, inplace=True)
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
                res = res.squeeze(-1)
                if not res.dim() == 2:
                    raise ArithmeticError(f"Expect output dimension 2, got: {res.shape}")

                dic = torch.zeros_like(res[..., 0]).view([-1, 1])
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
        # g: (B x 1), res is either (B x S x C)) or (B x C)
        if self.rAI_classification:
            if isinstance(res, (tuple, list)):
                # in model > 1 output is expected to be a tuple
                res_pred, res_conf = res
                g, res_pred = super(BinaryClassificationSolver, self)._align_g_res_size(g, res_pred)
                return g, (res_pred, res_conf)
            else:
                return super(BinaryClassificationSolver, self)._align_g_res_size(g, res)
        else:
            g = g.squeeze()

            # make sure the dimension is correct
            if isinstance(res, (tuple, list)):
                res = [r.view(-1, 1) for r in res]
            elif res.dim() == 3:
                # Assume output is stack of prediction and confidence
                res = res.view(-1, 3, 1)
            else:
                res = res.view(-1,1)
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
            if isinstance(self.get_net(), rAIdiologist):
                self._logger.info(f"Setting rAIdiologist mode to {current_mode}")
                self._set_net_mode(current_mode)

    def _set_net_mode(self, mode):
        try:
            # When there's data parallel
            self.net.get_submodule('module').set_mode(mode)
            self.net.get_submodule('module').RECORD_ON = False
        except:
            self.net.set_mode(mode)
            self.net.RECORD_ON = False
        self._current_mode = mode

    def validation(self) -> list:
        # Save current train mode
        original_mode = self.get_net()._mode.item()
        if original_mode > 3:
            self._logger.info(f"Setting rAIdiologist mode to from {original_mode} -> -1 for validation.")
            self._set_net_mode(-1) # inference when not pretraining (i.e., mode = 0)
        super(rAIdiologistSolver, self).validation()
        self._logger.info(f"Setting rAIdiologist mode back to {original_mode}")
        self._set_net_mode(original_mode)

    def _validation_step_callback(self, g: torch.Tensor, res: torch.Tensor, loss: Union[torch.Tensor, float],
                                  uids=None) -> None:
        r"""Uses :attr:`perf` to store the dictionary of various data."""
        if self.rAI_classification:
            # Unpack the tuple first
            if isinstance(self.loss_function, ConfidenceCELoss) and isinstance(res, (list, tuple)):
                res, conf = res
            return super(BinaryClassificationSolver, self)._validation_step_callback(g, res, loss, uids)

        self.validation_losses.append(loss.item())
        # Initialize list
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
        self._update_misclassification_record(dic, g, uids)
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

    def _validation_callback(self) -> None:
        if self.rAI_classification:
            return super(BinaryClassificationSolver, self)._validation_callback()
        else:
            return super(rAIdiologistSolver,self)._validation_callback()

    def _loss_eval(self, *args):
        r"""Copy from :class:`ClassificationSolver` with some modificaiton to cater for the change in network output
        format, which is now a tuple, not just a tensor.
        """
        out, s, g = args

        s = self._match_type_with_network(s)
        g = self._match_type_with_network(g)

        g, out = self._align_g_res_size(g, out)

        if self.ordinal_class:
            if not isinstance(self.loss_function, CumulativeLinkLoss):
                msg = f"For oridinal_class mode, expects `CumulativeLinkLoss` as the loss function, got " \
                      f"{type(self.loss_function)} instead."
                raise AttributeError(msg)

        if self.ordinal_mse and not isinstance(self.loss_function, nn.SmoothL1Loss):
                msg = f"For oridinal_mse mode, expects `SmoothL1Loss` as the loss function, got " \
                      f"{type(self.loss_function)} instead."
                raise AttributeError(msg)

        # required dimension of CrossEntropy is either (B) or (B, num_class)
        if isinstance(self.loss_function, (nn.CrossEntropyLoss, ConfidenceCELoss)):
            # squeeze (B, 1) to (B)
            g = g.squeeze()

        # self._logger.debug(f"Output size out: {out.shape}({out.dtype}) g: {g.shape}({g.dtype})")
        # Cross entropy does not need any processing, just give the raw output
        loss = self.loss_function(out, g)
        return loss

    def _step_callback(self, s, g, out, loss, uid=None, step_idx=None) -> None:
        r"""Copy from :class:`ClassificationSolver` with some modificaiton to cater for the change in network output.

        See Also:
            * :class:`ClassificationSolver`
        """
        # Print step information
        _df, _ = self._build_validation_df(g, out, uid=uid)
        self._logger.debug('\n' + _df.to_string())

        # These are used for specific network and will be move to other places soon.
        if hasattr(self.net, 'module'):
            if hasattr(self.net.module, '_batch_callback'):
                self.net.module._batch_callback()
                self._logger.debug(f"LCL:{self.net.module.LCL.cutpoints}")
        elif hasattr(self.net, '_batch_callback'):
            self.net.module._batch_callback()
            self._logger.debug(f"LCL:{self.net.module.LCL.cutpoints}")