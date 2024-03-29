import torch
import json
from pathlib import Path
from pytorch_med_imaging.inferencers.BinaryClassificationInferencer import BinaryClassificationInferencer
from typing import Union, Iterable, Optional
from ..config.network.rAIdiologist import *

__all__ = ['rAIdiologistInferencer']


class rAIdiologistInferencer(BinaryClassificationInferencer):
    r"""Inferencer specifically for rAIdiologist net

    Attributes:
        rAI_inf_save_playbacks (bool):
            If `True`, the playbacks will be written out as a single JSON file.
        playbacks (list):
            A list of `torch.FloatTensor` with dim (3 x X)

    """
    def __init__(self, *args, **kwargs):
        super(rAIdiologistInferencer, self).__init__(*args, **kwargs)
        self.playbacks = []
        self._logger.debug(f"Mode: {self.rAI_fixed_mode}")
        if self.rAI_fixed_mode != 0:
            self.net.set_mode(-1) # -1 is the inference mode, which is actually mode 4 for now.
        else:
            self.net.set_mode(0)
            self._logger.info("Running inference for mode 0")

        self.net_handles = []
        if self.rAI_inf_save_playbacks and not self.net._mode == 0:
            self._logger.info("Registering save playback hooks...")
            self.net.RECORD_ON = True
            self.net_handles.append(self.net.register_forward_pre_hook(_playback_clean_hook))
            self.net_handles.append(self.net.register_forward_hook(self._forward_hook_gen()))
        else:
            self.net.RECORD_ON = False

    def _reshape_tensors(self,
                         out_list: Iterable[torch.FloatTensor],
                         gt_list: Iterable[torch.FloatTensor]):
        r"""rAIdiologist version of reshape prior to feeding the outputs into `_writter`

        Args:
            out_list:
                List of tensors with dimension (1 x C)
            gt_list:
                List of tensor with dimension (1 x 1) or (1 x C)

        Returns:
            out_tensor: (B x 3)
            gt_tensor: (B x 1)
        """
        if gt_list is None:
            raise AttributeError("gt_list should not be none, if there are no gt, it should be an empty list.")
        out_tensor = torch.cat(out_list, dim=0) #(NxC)
        if out_tensor.dim() < 2:
            out_tensor = out_tensor.unsqueeze(0)

        gt_tensor = torch.cat(gt_list, dim=0) if len(gt_list) > 0 else None
        if not gt_tensor is None:
            while gt_tensor.dim() < out_tensor.dim():
                gt_tensor = gt_tensor.unsqueeze(0)
        return out_tensor, gt_tensor

    def _writter(self,
                 out_tensor: torch.IntTensor,
                 uids: Iterable[Union[str, int]],
                 gt: Optional[torch.IntTensor] = None):
        r"""
        Playbacks should be a list with tensor elements of dimensions (S x 3), e.g., [(S x 3), (S x 3), ...]. The length
        of the playbacks should match that of the uids.

        Args:
             out_tensor:
             uids:
             gt:
        """
        dl = super(rAIdiologistInferencer, self)._writter(out_tensor[..., 0].view(-1, 1),
                                                          uids,
                                                          gt,
                                                          sig_out=True)
        try:
            dl._data_table['Conf_0'] = out_tensor[..., 1]
        except IndexError:
            pass
        try:
            # Sorting must be done after assigning the conf vector because the order of out_tensor
            # is not indexed.
            dl._data_table.set_index('IDs', inplace=True)
            dl._data_table.sort_index(inplace=True)
        except AttributeError or IndexError:
            self._logger.warning("IDs is not a column in the data table.")
        # Write again
        self._logger.info(f"Writing results to {self.output_dir}")
        dl.write(self.output_dir)

        if self.rAI_inf_save_playbacks:
            out_path = Path(self.output_dir).with_suffix('.json')
            self._logger.debug(f"playbacks: {self.playbacks}")
            self._logger.info(f"Writing playbacks to: {str(out_path)}")
            if len(uids) != len(self.playbacks):
                self._logger.warning("Playback does not match uids")
            out_dict = {u: l.tolist() for u, l in zip(uids, self.playbacks)}
            with out_path.open('w') as jf:
                json.dump(out_dict, jf, sort_keys=True)
        return dl

    def _forward_hook_gen(self):
        r"""This hook gen copies the playback from rAIdiologist after each forward run"""
        def copy_playback(module, input, output):
            if isinstance(module, rAIdiologist):
                playback = module.get_playback()
                if len(playback) == 0:
                    raise AttributeError("No playback is availble.")
                self.playbacks.extend(module.get_playback())
            return
        return copy_playback

def _playback_clean_hook(module, input):
    r"""This hook cleans the rAIdiologist playback list prior to running a mini-batch"""
    if isinstance(module, (rAIdiologist, rAIdiologist_v2, rAIdiologist_v3)):
        module.clean_playback()
    return
