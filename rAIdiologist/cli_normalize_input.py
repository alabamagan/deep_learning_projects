import re
import click
import mnts
from functools import partial
from datetime import datetime
from mnts.mnts_logger import MNTSLogger
from mnts.scripts.dicom2nii import dicom2nii
from mnts.filters.mpi_wrapper import mpi_wrapper, repeat_zip
from pathlib import Path
from typing import Optional
import pprint


@click.command(help='CLI for normalizing MRI images using the MNTS package.')
@click.option('--input-dir', '-i', type=click.Path(exists=True, dir_okay=True, path_type=Path), required=True,
              help='Directory where input data is located.')
@click.option('--output-dir', '-o',
              type=click.Path(dir_okay=True, file_okay=False, writable=True, path_type=Path), required=True,
              help='Directory where output data will be saved.')
@click.option('--mnts-state', '-s',
              type=click.Path(dir_okay=True, file_okay=True, readable=True, path_type=Path),
              help='Path to the MNTS state file required for normalization steps that utilize training.')
@click.option('--mnts-graph', '-g',
              type=click.Path(file_okay=True, dir_okay=False, readable=True, path_type=Path), required=True,
              help='Path to the MNTS graph file needed for normalization steps that involve training.')
@click.option('--num-workers', '-n', default=1, type=int, help='Number of workers to use.')
@click.option('--keep-log', is_flag=True, default=None,
              help='Flag to keep logs. Default log directory is ./[Date]-Normalization.log')
@click.option('--log-level', default='INFO', help='Logging level.')
@click.option('--verbose', '-v', is_flag=True, help="Output to STDOUT.")
def normalize_input(input_dir: Path,
                    output_dir: Path,
                    mnts_state: Optional[Path],
                    mnts_graph: Path,
                    num_workers: Optional[int],
                    keep_log: bool,
                    log_level: str,
                    verbose: bool):
    r"""Normalizes MRI images using the MNTS package.

    This command-line interface (CLI) tool processes MRI images located in
    the input directory and outputs the normalized images to the specified
    output directory. It utilizes the MNTS package for the normalization
    process, which may involve training steps defined by the MNTS state and
    graph files.

    Args:
        input_dir (Path):
            Directory where input data is located.
        output_dir (Path):
            Directory where output data will be saved.
        mnts_state (Optional[Path]):
            Path to the MNTS state file required for normalization steps
            that utilize training.
        mnts_graph (Path):
            Path to the MNTS graph file needed for normalization steps
            that involve training.
        num_workers (Optional[int]):
            Number of workers to use for parallel processing.
        keep_log (bool):
            Flag to keep logs. Default log directory is
            ./[Date]-Normalization.log.
        log_level (str):
            Logging level, e.g., 'INFO', 'DEBUG'.
        verbose (bool):
            If True, output is sent to STDOUT.

    Raises:
        FileNotFoundError: If the input directory or required files are not found.
        ValueError: If invalid values are provided for options.

    ..note::
        * For MNTS graph, it should be a YAML file, typically with nodes definitions
          like this:
          ```
          SpatialNorm:
              out_spacing: [0.4492, 0.4492, 4]

          HuangThresholding:
              closing_kernel_size: 10
              _ext:
                  upstream: 0
                  is_exit: True

          N4ITKBiasFieldCorrection:
              _ext:
                  upstream: [0, 1]

          NyulNormalizer:
              _ext:
                  upstream: [2, 1]
                  is_exit: True
          ```
        * For state, please make sure the state files are put in correct file tree.

    Example:
        $ python script.py --input-dir ./data/input --output-dir ./data/output \
        --mnts-graph ./mnts_graph.pb --num-workers 4 --verbose
    """
    # Get the current date
    current_date = datetime.now().strftime("%Y-%m-%d")
    # Create the log file name
    log_file_name = f"{current_date}-Normalization.log"
    with MNTSLogger(log_dir=log_file_name, keep_file=keep_log, log_level=log_level, verbose=verbose) as logger:
        logger.info("{:=^150}".format(" Initiating Normalization "))

        # creates the graph
        G = mnts.filters.MNTSFilterGraph.CreateGraphFromYAML(mnts_graph)
        logger.info(f"Created graph: {G}")

        # load states
        if G.requires_training:
            logger.info(f"Loading states from {mnts_state}")
            G.load_node_states(None, mnts_state)

        # * MPI normalization
        # First gather all input data
        input_fpaths = list(input_dir.rglob('*.nii.gz'))
        # Decide output directories
        output_prefix = [f.name for f in input_fpaths] # name of output files
        input_vector = [output_prefix, [output_dir], input_fpaths]
        logger.debug(f"{pprint.pformat(input_vector)}")
        job_length = len(list(repeat_zip(*input_vector)))

        # create progress bar with correct length
        G.set_progress_bar(job_length)

        # set it to the graph G
        if num_workers <= 1:
            logger.info("Start processing input without parallelization.")
            for row in repeat_zip(*input_vector):
                G.mpi_execute(*row)
        else:
            logger.info(f"Start processing input with {num_workers} workers.")
            mpi_wrapper(G.mpi_execute, input_vector, num_worker=num_workers)

        # close progressbar
        G.close_progress_bar()

if __name__ == '__main__':
    normalize_input()