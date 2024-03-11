import tempfile
from pathlib import Path
import yaml
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import click
import copy
import shutil
from shutil import copytree
from mnts.mnts_logger import MNTSLogger
from pytorch_med_imaging.controller import PMIController
from npc_segment.config.loctexthistCFG import *
from npc_segment.preprocess import NPCSegmentPreprocesser
from npc_segment.img_proc import *
import logging
from typing import Union, Any, List, Optional
import time

sequence_choice = ['T2WFS', 'T1W', 'CET1W', 'CET1WFS']

# @click.command()
# @click.argument('-i', '--input-dir', required=True, type=click.Path(), help="For overriding input directory setting.")
# @click.argument('-o', '--output-dir', required=True, type=click.Path(), help="For overriding output directory setting.")
# @click.option('--sequence', default='T2WFS', type=click.Choice(sequence_choice, case_sensitive=True),
#               help=f"Set the sequence. Chose from [{','.join(sequence_choice)}]")
# @click.option('--skip-norm', default=False, is_flag=True, help="If true, skip intensity normalization")
# @click.option('--inference', default=False, is_flag=True, help="For guild operation")
def main(input_dir, output_dir, sequence, skip_norm, inference, debug=False):
    main_logger = MNTSLogger('.', 'main', verbose=True, keep_file=False)
    t_start = time.time()
    main_logger.info("{:=^80}".format(" NPC auto Segmentation Running "))

    # Create controller
    cfg = NPCSegmentControllerCFG()
    cfg.sequence = sequence

    # Guild is giving us trouble for double printing everything
    cfg.verbose = False
    logger_dict = logging.Logger.manager.loggerDict
    formatter = logging.Formatter(MNTSLogger.log_format)
    for logger_name, logger_instance in logger_dict.items():
        if isinstance(logger_instance, logging.Logger):
            for handlers in logger_instance.handlers:
                handlers.setFormatter(formatter)

    # * Error checks
    if not inference:
        raise NotImplementedError("Training is not available yet!")
    else:
        cfg.run_mode = 'inference'

    # * Setup
    # Override sequence options
    cfg.sequence = sequence
    normalization_graph = Path("./assets/normalization_t2w.yaml")
    normalization_states = Path(f"./assets/norm_states/{sequence}")


    # * Perform normalization of inputs
    with tempfile.TemporaryDirectory() as normed_tempdir, \
            tempfile.TemporaryDirectory() as output_tempdir:
        # Put normalized images in a temp folder
        normed_tempdir_path = Path(normed_tempdir)
        # Create normalizer
        norm = NPCSegmentPreprocesser(normalization_graph, normalization_states)
        norm.input_dir = input_dir
        norm.output_dir = normed_tempdir_path
        # debug
        if debug:
            debug_path = tempfile.TemporaryDirectory()
            # copy 3 images to this path from the original input
            for i, r in enumerate(Path(input_dir).rglob('*nii.gz')):
                if i == 3:
                    break
                shutil.copy2(r, debug_path.name)
            norm.input_dir = debug_path.name

        norm.exec()
        if debug:
            debug_path.cleanup()


        # * Corase segmentation
        # Intermediate output directory for coarse segmentation
        output_tempdir_path = Path(output_tempdir)
        coarse_out_path = output_tempdir_path / "Coarse"

        run_inference(cfg,
                      normed_tempdir_path / "NyulNormalizer",
                      coarse_out_path,
                      normed_tempdir_path / "HuangThresholding")

        # grow the segmentation a git for better caturing the NPC, this function operates
        # inplace and overwrites the nii.gz labels
        grow_segmentation(str(coarse_out_path))


        # * Fine segmentation
        # Intermediate output directory
        fine_out_path = output_tempdir_path / "Fine"

        # Update the directires and redo the segmentation with coarse segment as probmap
        run_inference(cfg,
                      normed_tempdir_path / "NyulNormalizer",
                      fine_out_path,
                      coarse_out_path)


        # * Post-processing
        pp_out_path = output_tempdir_path / "PostProcessed"
        pp_out_path.mkdir(exist_ok=True)

        seg_post_main(fine_out_path, pp_out_path)

        # * Move results to output path
        output_dir = Path(output_dir)
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True, exist_ok=True)

        for f in pp_out_path.rglob("*nii.gz"):
            main_logger.info(f"Copying: {str(f)}")
            shutil.copy2(str(f.absolute()), str(output_dir.absolute()))


    main_logger.info("{:=^80}".format(f" Segmentation Done (Total: {time.time() - t_start:.01f}s) "))
    pass


def run_inference(cfg: PMIControllerCFG,
                  input_dir: Union[str, Path],
                  output_dir: Union[str, Path],
                  probmap_dir: Union[str, Path]) -> None:
    """Runs the inference process for NPC segmentation.

    This function overrides the input, output, and probability map directories specified
    in the configuration object with the provided directory paths, creates a
    PMIController with the updated configuration, and then executes the inference
    process. After execution, the PMIController instance is deleted.

    Args:
        cfg (NPCSegmentControllerCFG):
            The configuration object for NPC segmentation. This object will be copied and
            its directories will be overridden with the provided paths.
        input_dir (str):
            The directory path where input data is located.
        output_dir (str):
            The directory path where output data should be saved.
        probmap_dir (str):
            The directory path where probability maps are located.

    .. note::
            This function does not return any value.

    """
    # Override directories to a temp directory
    cfg: NPCSegmentControllerCFG = copy.copy(cfg)
    cfg.data_loader_cfg.input_dir = input_dir
    cfg.data_loader_cfg.probmap_dir = probmap_dir
    cfg.output_dir = output_dir # Note that output directory is controller's property because dataloader is only for
                                # loading data.

    # Create controller
    controller = PMIController(cfg)
    # Execute
    controller.exec()
    del controller


if __name__ == '__main__':
    input_dir = "/home/lwong/Source/Repos/deep_learning_projects/NPC_Segmentation/72.RecurrentData/StudyFilesBySequence/Pre/T2W-FS_TRA"
    output_dir ="/home/lwong/Source/Repos/deep_learning_projects/NPC_Segmentation/72.RecurrentData/StudyFilesBySequence/Segment/T2W-FS_TRA"
    main(input_dir, output_dir, 'T2WFS', False, True, debug=False)