import torch
import json
import einops
from tqdm.auto import tqdm
import torchio as tio
import SimpleITK as sitk

from pathlib import Path
from pytorch_med_imaging.inferencers.BinaryClassificationInferencer import BinaryClassificationInferencer
from typing import Union, Iterable, Optional, Tuple, List
from ..config.network.rAIdiologist import *
from ..config.network.lstm_rater import *

__all__ = ['rAIdiologistInferencer']


class rAIdiologistInferencer(BinaryClassificationInferencer):
    r"""Inferencer specifically for rAIdiologist net

    Attributes:
        rAI_inf_save_playbacks (bool):
            If `True`, the playbacks will be written out as a single JSON file.
        playbacks (list):
            A list of `torch.FloatTensor` with dim (3 x X)
        rAI_classification (bool):
            If true, some of the function will be ported to :class:`ClassificationInferencer`

    """


    def __init__(self, *args, **kwargs):
        super(rAIdiologistInferencer, self).__init__(*args, **kwargs)
        self.playbacks = []
        self._logger.debug(f"Mode: {self.rAI_fixed_mode}")
        if self.rAI_fixed_mode == 0:
            self.net.set_mode(self.rAI_fixed_mode)
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
        if self.rAI_classification:
            if isinstance(out_list, (list, tuple)):
                # For rAI_v5.1 classification version, the output is now a tuple of prediction and confidence.
                out_list, conf = out_list
            return super(BinaryClassificationInferencer, self)._reshape_tensors(out_list,
                                                                                gt_list)

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

    def _write_out(self):
        r"""This inherit passes the input image's directory to _writter
        """
        uids = []
        gt_tensor = []
        out_tensor = []
        last_batch_dim = 0
        with torch.no_grad():
            self.net = self.net.eval()
            # dataloader = DataLoader(self._inference_subjects, batch_size=self.batch_size, shuffle=False)
            dataloader = self.data_loader
            input_directories = []
            input_tensors = []
            for index, mb in enumerate(tqdm(dataloader.get_torch_data_loader(self.batch_size, exclude_augment=True),
                                            desc="Steps")):
                s = self._unpack_minibatch(mb, self.unpack_key_inference)
                s = self._match_type_with_network(s)
                input_directories.extend(mb['input']['path'])

                try:
                    self._logger.debug(f"Processing: {mb['uid']}")
                    _msg = f"s size: {s.shape if not isinstance(s, (list, tuple)) else [ss.shape for ss in s]}"
                    self._logger.debug(_msg)
                except:
                    pass

                # Squeezing output directly cause problem if the output has only one output channel.
                try:
                    if isinstance(s, (list, tuple)):
                        input_tensors.extend([ss.cpu() for ss in s[0]])  # ! Note that this might cause RAM problem
                        out = self.net(*s)
                    else:
                        input_tensors.extend([ss.cpu() for ss in s])  # ! Note that this might cause RAM problem
                        out = self.net(s)
                    out = self._prepare_network_output(out)
                except Exception as e:
                    if 'uid' in mb:
                        self._logger.error(f"Error when dealing with minibatch: {mb['uid']}")
                    raise e

                while ((out.dim() < last_batch_dim) or (out.dim() < 2)) and last_batch_dim != 0:
                    out = out.unsqueeze(0)
                    self._logger.log_print_tqdm('Unsqueezing last batch.' + str(out.shape))
                self._logger.debug(f"out size: {out.shape}")
                out_tensor.append(out.data.cpu())
                uids.extend(mb['uid'])
                if 'gt' in mb:
                    gt_tensor.append(mb['gt'])

                last_batch_dim = out.dim()
                del out

            out_tensor, gt_tensor = self._reshape_tensors(out_tensor, gt_tensor)
            dl = self._writter(out_tensor, uids, gt_tensor, input_tensors)
            self._logger.debug('\n' + dl._data_table.to_string())


    def _writter(self,
                 out_tensor: torch.IntTensor,
                 uids: Iterable[Union[str, int]],
                 gt: Optional[torch.IntTensor] = None,
                 input_tensors: Optional[List[torch.Tensor]] = None):
        r"""
        Playbacks should be a list with tensor elements of dimensions (S x 3), e.g., [(S x 3), (S x 3), ...]. The length
        of the playbacks should match that of the uids.

        Args:
             out_tensor:
             uids:
             gt:
        """
        if self.rAI_classification:
            dl = super(BinaryClassificationInferencer, self)._writter(out_tensor,
                                                                      uids,
                                                                      gt,
                                                                      sig_out=True)
            dl._data_table.sort_index(inplace=True)

            #TODO: Write playback
            return dl
        else:
            # Assume (B x 2 x C), where 2 is prediction and confidence
            if out_tensor.dim() == 3:
                out_tensor = einops.rearrange(out_tensor, 'b i c -> b c i').squeeze()

            # try to fix gt shape
            if gt is not None:
                gt = gt.view([-1, 1])

            dl = super(rAIdiologistInferencer, self)._writter(out_tensor[..., 0].view(-1, 1),
                                                              uids,
                                                              gt,
                                                              sig_out=True)
            try:
                dl._data_table['Conf_0'] = out_tensor[..., 2]
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
                out_path = Path(self.output_dir).parent / 'SelfAttention'
                out_path.mkdir(exist_ok=True, parents=True)
                if not out_path.is_dir():
                    raise FileExistsError(f"{out_path} is a file and not a directory")
                # self._logger.debug(f"playbacks: {self.playbacks}")
                self._logger.info(f"Writing playbacks to: {str(out_path)}")
                _debug = {
                    'type': [type(s) for s in self.playbacks],
                    'len': [len(s) for s in self.playbacks],
                    'shape': [s.shape for s in self.playbacks],
                    'input_tensor_shape': [s.shape for s in input_tensors],
                    'uids': uids,
                }
                self._logger.debug(f"{_debug}")
                if len(uids) != len(self.playbacks):
                    self._logger.warning("Playback does not match uids")
                else:
                    self._logger.info(f"Writing playback, this could take a while...")
                    for _uid, p, s in zip(uids, self.playbacks, input_tensors):
                        if isinstance(s, torch.Tensor):
                            # convert self-attention
                            grid_size = {
                                'h': self.net._grid_size[0],
                                'w': self.net._grid_size[0],
                                's': s.shape[-1] - 1
                            }
                            pb_pred, pb_conf = TransformerEncoderLayerWithAttn.sa_from_playback(p, s, grid_size)
                            # now it's very difficult to map the SA to original space, so I'll just save it like this
                            oname_pb_pred   = out_path / (_uid + '_pb_pred.nii.gz')
                            oname_in_tensor = out_path / (_uid + '_image.nii.gz')
                            pb_pred.save(oname_pb_pred)
                            tio.ScalarImage(tensor=s).save(oname_in_tensor)
                            self._logger.info(f"Written {_uid} to {out_path}")

            return dl

    def _forward_hook_gen(self):
        r"""This hook gen copies the playback from rAIdiologist after each forward run"""
        def copy_playback(module, input, output):
            if isinstance(module, rAIdiologist):
                playback = module.get_playback()
                if len(playback) == 0:
                    self._logger.warning("No playback is available.")
                self.playbacks.extend(module.get_playback())
            return
        return copy_playback

    def _prepare_output_dict(self, gt, out_tensor, sig_out, uids) -> dict:
        if self.rAI_classification:
            return super(BinaryClassificationInferencer, self)._prepare_output_dict(gt, out_tensor, sig_out, uids)
        else:
            return super()._prepare_output_dict(gt, out_tensor, sig_out, uids)

    def display_summary(self):
        if self.rAI_classification:
            return super(BinaryClassificationInferencer, self).display_summary()
        else:
            return super().display_summary()

    def _prepare_network_output(self, out: Union[Tuple, torch.FloatTensor]) -> torch.FloatTensor:
        r"""This is overriden to cater for network output that is sometimes a tuple."""
        if isinstance(out, (list, tuple)):
            out, conf = out
        return super(self.__class__, self)._prepare_network_output(out)


def _playback_clean_hook(module, input):
    r"""This hook cleans the rAIdiologist playback list prior to running a mini-batch"""
    if isinstance(module, (rAIdiologist)):
        module.clean_playback()
    return
