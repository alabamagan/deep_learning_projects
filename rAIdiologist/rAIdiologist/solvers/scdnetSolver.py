import os
from pathlib import Path
from typing import Union, Any, Tuple, Iterable, Optional

import numpy as np
import rich
from sklearn.metrics import roc_auc_score
import pandas as pd
import torch
import torch.nn as nn
import random
import pprint
import SimpleITK as sitk
from tqdm.auto import tqdm


from pytorch_med_imaging.inferencers import BinaryClassificationInferencer
from pytorch_med_imaging.solvers import BinaryClassificationSolver, ClassificationSolverCFG, ClassificationSolver
from pytorch_med_imaging.perf.classification_perf import *
from pytorch_med_imaging.perf.segmentation_perf import EVAL as seg_eval
from pytorch_med_imaging.utils.visualization.segmentation_vis import *
from pytorch_med_imaging.integration import NP_Plotter
from pytorch_med_imaging.pmi_data import DataLabel
from ..config.network.rAIdiologist import rAIdiologist
from ..config.loss.rAIdiologist_loss import ConfidenceCELoss
import gc

__all__ = ['SCDenseNetSolver']


class SCDenseNetSolverCFG(ClassificationSolverCFG):
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
    loss_function = nn.BCEWithLogitsLoss()
    net = rAIdiologist(1)

class SCDenseNetSolver(BinaryClassificationSolver):
    r"""This solver is written to train :class:`rAIdiologist` network. This mainly inherits :func:`._epoch_prehook` to
    set the network mode based on the current epoch number. This also customize :func:`_build_validation_df` to present
    the results better.

    .. note::
        If `self.rAI_classification` is set to True, the solver will pass some function to ClassificationSolver because
        the output of the network is expect to be multi-class classification.
    """
    def __init__(self, cfg, *args, **kwargs):
        super(SCDenseNetSolver, self).__init__(cfg, *args, **kwargs)
        self._validation_misclassification_record = {} # Doesn't know why sometimes this is not inherited from
                                                       # BinaryClassificationSolver

    def _build_validation_df(self, g, res, uid=None):
        r"""The output of the network consist of a segmentation and classification."""
        g_dic, g_seg = g
        res_dic, res_seg = res

        # calculate segmentation performance, assume binary prediction
        pred_seg = (res_seg > 0.5).int() # (B x 1 x H x W x D)
        seg_perf_df = seg_eval(pred_seg, g_seg.int())

        # populate classification predictions
        pred_dic = (res_dic > .5).int() # (B)
        # form dataframe
        _df = seg_perf_df
        _df['Pred_Float'] = res_dic.flatten().tolist()
        _df['Pred'] = pred_dic.flatten().tolist()
        _df['Truth'] = g_dic.flatten().tolist()
        if uid is not None:
            try:
                _df.index = uid
            except:
                pass
        return _df, pred_dic.view(-1, 1)

    def _align_g_res_size(self, g, res):
        r"""The output of the network consist of a segmentation and classification"""
        g_dic, g_seg = g
        res_dic, res_seg = res

        res_dic = res_dic.view_as(g_dic)
        res_seg = res_seg.view_as(g_seg)
        return (g_dic, g_seg), (res_dic, res_seg)

    def _validation_callback(self) -> None:
        r"""Override is need to cater the changes in data structure of :attr:`perfs`.

        Added parameters to log here:
        - Segmentation performance

        """
        store_dict  = self.perfs[0]
        validation_loss = np.mean(np.array(self.validation_losses).flatten())

        # df = {
        #     'DICE'      : pd.concat(DICE       , axis = 0).astype('float'),
        #     'prediction': pd.concat(predictions, axis = 0).astype('float'),
        #     'decision'  : pd.concat(dics       , axis = 0).astype('bool'),
        #     'truth'     : pd.concat(gts        , axis = 0).astype('bool')
        # }
        # Create the dataframe from dictionary of pd.Series
        df = {k: pd.concat(v) for k, v in store_dict.items() if len(v) if isinstance(v[0], pd.Series)}
        df = {k: v[~v.index.duplicated()] for k, v in df.items()}
        df = pd.DataFrame.from_dict(df)
        # Renmae columns for display
        df.rename(columns={'dics': 'Pred', 'gts': 'Truth', 'predictions': 'Pred_Float'}, inplace=True)
        df = df.astype({'Pred': int, 'Truth': int}, errors='ignore')
        if not len(store_dict['uids']):
            self._logger.warning("UIDs in store_dict may be incorrect.")
            df = df.reset_index()

        # TODO: ignore DICE of non-NPC by setting them to 0

        # Compute performance here
        acc, per_mean, res_table = self._compute_performance(torch.Tensor(df.dropna()['Pred']).bool(),
                                                             torch.Tensor(df.dropna()['Truth']).bool())
        _df = df.dropna()
        auc = roc_auc_score(_df['Truth'].values,_df['Pred_Float'].values)
        dice_mean = _df['DICE'].mean()
        df['Correct'] = df['Pred'] == df['Truth']
        self._logger.info(f"\n{df[[c for c in df.columns if c != 'seg_imgs']].to_string()}")
        # self._logger.debug("_val_perfs: \n%s"%res_table.T.to_string())
        self._logger.info("Validation Result - ACC: %.05f, VAL: %.05f, DICE: %.5f, AUC: %.5f"%(
            acc, validation_loss, dice_mean, auc
        ))
        self.plotter_dict['scalars']['val/loss'] = validation_loss
        self.plotter_dict['scalars']['val/performance/ACC'] = acc
        self.plotter_dict['scalars']['val/performance/DICE'] = dice_mean
        self.plotter_dict['scalars']['val/performance/AUC'] = auc
        for param, val in per_mean.items():
            self.plotter_dict['scalars']['val/performance/%s'%param] = val

        # Print the misclassification report
        if len(self._validation_misclassification_record) > 0:
            self._logger.info("Validation misclassification report: {}".format(
                pprint.pformat(self._validation_misclassification_record)
            ))

        # Push images to plotter
        if self.plotting and isinstance(self._plotter, NP_Plotter):
            if 'seg_imgs' in df:
                for _uid, _imgs in df['seg_imgs'].items():
                    if np.isnan(_imgs).all():
                        continue
                    self._plotter.add_image(f'val/segmentation/images/{_uid}', _imgs,
                                            name = _uid,
                                            step = self.current_epoch)


    def _validation_step_callback(self, g: Tuple[torch.Tensor], res: Tuple[torch.Tensor],
                                  loss: Union[torch.Tensor, float],uids: Optional[Iterable[str]] = None) -> None:
        r"""Uses :attr:`perf` to store the dictionary of various data from each validaiton step.

        .. note::
            Major changes here, results are revised from passing with dictionary to using pd.DataFrame instead.

        """
        self.validation_losses.append(loss.item())
        # Initialize list
        if len(self.perfs) == 0:
            self.perfs.append({
                'dics'       : [], # Naming is bad here, need to change but store_dict is used in many places
                'gts'        : [],
                'predictions': [],
                'confidence' : [],
                'uids'       : [],
                'DICE'       : [],
                'seg_imgs'   : []
            })
        store_dict = self.perfs[0]
        g, res = self._align_g_res_size(g, res)
        _df, dic = self._build_validation_df(g, res, uid=uids)

        # Put UIDs into the lists
        if isinstance(uids, (tuple, list)):
            store_dict['uids'].extend(uids)
        self._update_misclassification_record(_df['Pred'], _df['Truth'], uids)

        # Decision were made by checking whether value is > 0.5 after sigmoid
        store_dict['dics'].append(_df['Pred'])
        store_dict['gts'].append(_df['Truth'])
        store_dict['DICE'].append(_df['DICE'])
        store_dict['predictions'].append(_df['Pred_Float'])

        # Draw the segmentation
        if self.plotting or True:
            _, g_seg = g        # g_seg/res_seg: (B x 1 x H x W x D)
            _, res_seg = res

            # for each case
            vect_seg_imgs = pd.Series(name='seg_imgs')
            for idx, gg_seg in enumerate(g_seg):
                gg_seg = gg_seg.squeeze() != 0 # (H x W x D)
                res_seg = res_seg > .5 # result is already sigmoided

                # if segmentation prediction is not empty, get sample a random slice and display
                if torch.all(gg_seg == 0):
                    continue

                selected = random.choice(torch.argwhere(gg_seg.sum(axis=[0, 1]) > 0))
                img_selected = self._current_mb['input'][tio.DATA][idx].squeeze()
                for_display = draw_contour(
                    img_selected[..., selected].numpy().squeeze(),
                    res_seg[idx, ..., selected].numpy().squeeze(),
                    gt_seg = gg_seg[..., selected].numpy().squeeze(),
                    contour_alpha=0.2
                )

                if isinstance(uids, (tuple, list)) and len(uids):
                    vect_seg_imgs[uids[idx]] = for_display.T
                else:
                    vect_seg_imgs[idx] = for_display.T
            store_dict['seg_imgs'].append(vect_seg_imgs)

    def _loss_eval(self, out, *args):
        r"""For SCDNet, the ground-truth of both class and segmentation is required.
        """
        s, (g, g_seg) = args
        pred_cls, pred_seg = out

        g = self._match_type_with_network(g)
        g_seg = self._match_type_with_network(g_seg)

        g, pred = self._align_g_res_size((g, g_seg), (pred_cls, pred_seg))

        # self._logger.debug(f"Output size out: {out.shape}({out.dtype}) g: {g.shape}({g.dtype})")
        # Cross entropy does not need any processing, just give the raw output
        loss = self.loss_function(*pred, *g)
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


class SCDenseNetInferencer(BinaryClassificationInferencer):
    def _write_out(self):
        r"""Because segmentation is invovled, we cannot use the regular since a weighted sampler was used for training.
        Here, the inference would need to handle subject at a time, and use the aggregated queue that also carries the
        information of where the patches were extracted.

        Now SCDense net is not a patch based network, but we still took advantage of torchio to make sure the input
        is of uniform size. So the "number of patch" is 1 here, and the patch size is (384, 384, 16), which was
        specified in the original paper.
        """
        from pytorch_med_imaging.perf.segmentation_perf import EVAL
        # Creates the directory for holding output
        output_dir = Path(self.output_dir)
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True)

        uids = []
        seg_perf_df = []
        gt_series = pd.Series(name='Truth_0', dtype='int') # This convention is needed by display_summary
        pred_series = pd.Series(name='Prob_Class_0')       # This convention is needed by display_summary
        last_batch_dim = 0
        self._num_of_questions = 1 # For display_summary
        self._TARGET_DATASET_EXIST_FLAG = False
        with torch.no_grad():
            self.net = self.net.eval()


            # Do it subject by subject
            subjects = self.data_loader.get_subjects(exclude_transform=True)
            transform = self.data_loader.transform
            for mb in subjects:
                if 'uid' in mb:
                    self._logger.info(f"Dealing with {mb['uid']}")

                # mark original input so we can reproduce it's spacing
                ori_input = mb
                uid = mb.get('uid', None)

                # Note that the syntax here needs to use the forked torchio
                mb = transform(mb)

                # gets the input for network
                input, = self._unpack_minibatch(mb, self.unpack_key_inference) # this method return list
                input = input.unsqueeze(0) if input.dim() == 4 else input # tio returns 4D tensor, network wants 5D tensor
                input = self._match_type_with_network(input)

                # Inference with the network
                output_pred, output_seg = self.net(input)
                bin_output_seg = (output_seg > 0.5).cpu()

                # Reverse the transform and get tensor in original size
                inverse_transform = transform[-1].inverse()
                if not isinstance(inverse_transform, tio.Pad):
                    raise RuntimeError("Inverse transform is not correctly setup.")
                tio_bin_output_seg = tio.LabelMap(tensor=bin_output_seg.squeeze(0)) # tio likes 4D
                tio_inv_bin_output_seg = inverse_transform.apply_transform(tio.Subject(output=tio_bin_output_seg))
                tio_inv_bin_output_seg = tio_inv_bin_output_seg['output']

                # Save segmentation based on original image
                sitk_ori_im = sitk.ReadImage(ori_input['input-srcpath'])
                sitk_seg_output = tio_inv_bin_output_seg.as_sitk()
                sitk_seg_output = sitk.DICOMOrient(sitk_seg_output, ''.join(ori_input['input'].orientation))
                sitk_seg_output.CopyInformation(sitk_ori_im)

                # Write this to output
                output_seg_dir = output_dir / f"{uid}.nii.gz"
                self._logger.info(f"Writing segmentation result to: {output_seg_dir}")
                sitk.WriteImage(sitk_seg_output, output_seg_dir)

                # Add prediction result to table
                pred_series[uid] = output_pred.cpu().item()
                if 'gt' in ori_input:
                    gt_series[uid] = ori_input['gt'].cpu().item()

                if 'gt_seg' in ori_input:
                    gt_seg = ori_input['gt_seg']
                    # calculate DSC
                    seg_score = EVAL(tio_inv_bin_output_seg.tensor.numpy().flatten(), gt_seg.tensor.numpy().flatten())
                    seg_score = pd.Series(seg_score, name=uid)
                    seg_perf_df.append(seg_score)

        # Save prediction results
        output_csv_dir = output_dir / 'Prediction.csv'
        self._logger.info(f"Writing prediction results to: {str(output_csv_dir)}")
        out_df = pd.concat([gt_series, pred_series], axis=1)
        out_df['Decision_0'] = out_df['Prob_Class_0'] >= 0.5
        out_df.to_csv(output_csv_dir)
        if len(seg_perf_df):
            seg_perf_df = pd.concat(seg_perf_df, axis=1).T
            out_df = out_df.join(seg_perf_df)
            self._TARGET_DATASET_EXIST_FLAG = True
        self._logger.debug(f"out_df: \n{out_df.to_string()}")
        self._dl = DataLabel(out_df)


    def _prepare_network_output(self, out: Tuple) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        r"""Override to return a prediction result and a segmentation array with same batch size"""
        out_pred, out_seg = out
        return out_pred, out_seg

    def _writter(self):
        r"""This function is not used in this inferencer, everything is done within write_out"""
        pass

    def display_summary(self):
        self.output_dir = str(next(Path(self.output_dir).rglob('*.csv')))
        super().display_summary()

