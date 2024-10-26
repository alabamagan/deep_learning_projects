import pprint
from pathlib import Path

import click
import pandas as pd
from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException
from tqdm import tqdm

import torchio
import torchio as tio
from mnts.mnts_logger import MNTSLogger
from mnts.utils import get_unique_IDs
from mnts.utils.filename_globber import get_fnames_by_IDs
from pytorch_med_imaging.pmi_data_loader.augmenter_factory import create_transform_compose
from rAIdiologist.config.network.lstm_rater import TransformerEncoderLayerWithAttn
from rAIdiologist.config.network.rAIdiologist import *


class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return pprint.pformat(self.__dict__)

@click.command()
@click.option('--image-data-dir',
              type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
              default=Path("/home/lwong/Source/Repos/NPC_Segmentation/NPC_Segmentation/60.Large-Study/HKU_data/NyulNormalizer"),
              help='Directory for image data. All files with .nii.gz suffix are globbed.',
              required=True)
@click.option('--checkpoint-dir',
              type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
              default=Path("/home/lwong/Source/Repos/NPC_Segmentation/Backup/rAIdiologist_B01-c6b89d35.pt"),
              help='Path to the checkpoint file.',
              required=True)
@click.option('--output-dir',
              type=click.Path(file_okay=False, writable=True, dir_okay=True, path_type=Path),
              default=Path("/mnt/ftp/Shared/temp"),
              help='Directory for output files.',
              required=True)
@click.option('--id-globber',
              type=str,
              default=r'\w{0,5}\d+',
              help='ID globbing pattern.')
@click.option('--id-list',
              required=False,
              multiple=True,
              default=None,
              help='List of IDs.')
@click.option('--id-list-file',
              required=False,
              default=None,
              type=click.Path(file_okay=False, writable=True, dir_okay=True, path_type=Path),
              help='ID List file, if this is specified, id-list will be ignored.')
@click.option('--inference-transform',
              type=click.Path(file_okay=True, exists=True, path_type=Path),
              default="../rAIdiologist/config/rAIdiologist_transform_inf.yaml",
              help='Path to inference transformation configuration file.')
@click.option('--ground-truth',
              type=click.Path(file_okay=True, exists=True),
              default=None,
              help="Not implemented yet.",
              required=False)
@click.option('--debug',
              is_flag=True,
              help="For debugging.")
def main(**kwargs):
    args = Args(**kwargs)

    with (torch.no_grad(),
          MNTSLogger(".", "PlaybackTest", verbose=True, log_level='debug',
                     keep_file=False) as logger):
        logger.debug(f'f{args = }')

        # Load idlist
        if args.id_list_dir is not None:
            # if its a txt file, read line by line
            if args.id_list_dir.suffix == '.txt':
                args.id_list = [str(r.strip()) for r in args._id_list_dir.open('r').readlines()]
            # if it has .ini suffix
            elif args.id_list_dir.suffix == '.ini':
                from configparser import ConfigParser
                parser = ConfigParser()
                parser.read(str(args.id_list_dir))
                args.id_list = parser['FileList']['testing'].split(',')
            else:
                raise FileError("Input id_list_dir is incorrect, must be .txt or .ini")

        fname = get_fnames_by_IDs(
            args.image_data_dir.glob("*nii.gz"),
            idlist=args.id_list if len(args.id_list) else get_unique_IDs(args.image_data_dir.glob("*nii.gz"), args.id_globber),
            globber=args.id_globber,
            return_dict=True
        )

        # Create a dictionary of TorchIO ScalarImage objects from the file names
        tio_images = {k: torchio.ScalarImage(f) for k, f in fname.items() if len(f) == 1}
        # Check the length of the dictionary
        if not len(tio_images):
            raise RuntimeError("Cannot find any images from the files")
        # For output predictions

        # Create a transformation pipeline for inference
        tio_transform = create_transform_compose(args.inference_transform)
        subjects = [tio.Subject(input=tioimg, sid=sid) for sid, tioimg in tio_images.items()]
        # Create a TorchIO SubjectsDataset using the subjects and transformation
        subjects_dataset = tio.SubjectsDataset(subjects, transform=tio_transform)

        # Initialize the model
        m: rAIdiologist_Transformer = create_rAIdiologist_v5_1()
        m.set_mode(5)
        m.load_state_dict(torch.load(args.checkpoint_dir))
        m = m.cuda()
        m.eval()

        # Prepare excel output file
        out_df_fname = args.output_dir / 'predictions.csv'

        # Enable recording of model operations
        m.RECORD_ON = True
        grid_size = {
            'w': 8,
            'h': 8,
            's': 24
        }

        CONTINUE_FLAG = False
        if out_df_fname.exists():
            df_predictions = pd.read_csv(out_df_fname, index_col=0)
        else:
            df_predictions = pd.DataFrame(columns=['OverallPrediction', 'TransformerConfidence', 'TransformerPrediction'])
        for sub in tqdm(subjects_dataset):
            try:
                # Log the subject ID
                logger.info(f"{sub['sid'] = }")

                # Prepare input tensor for the model
                in_tensor = sub['input'][tio.DATA].float().cuda()
                sid = sub['sid']

                # Perform inference using the model
                x = m(in_tensor)

                # Record the prediction
                df_predictions.loc[sid] = x.flatten().cpu().tolist()

                # Retrieve and clean the model playback data
                playback = m.get_playback()
                m.clean_playback()

                playback = playback[0]
                play_back_prediction, play_back_confidence = TransformerEncoderLayerWithAttn.sa_from_playback(
                    playback, in_tensor, grid_size
                )

                # Save the images to the output directory
                img = tio.ScalarImage(
                    tensor=in_tensor.cpu()[..., 1:])  # Remove the first slice matching network's `forward`
                img.save(args.output_dir / f"{sid}.nii.gz")
                play_back_prediction.save(args.output_dir / f"{sid}_heatmap.nii.gz")
                play_back_confidence.save(args.output_dir / f"{sid}_confidence.nii.gz")
                logger.info(f"{sub['sid']} done.")

                if args.debug:
                    if df_predictions.shape[0] > 3:
                        logger.debug("Debug mode interruption.")
                        logger.debug("\n" + df_predictions.to_string())
                        continue
            except Exception as e:
                logger.error(e)
                logger.exception(e)
                if not CONTINUE_FLAG:
                    choice = click.prompt("Continue? a for always. (y/n/a)[y]",
                                          default='y',
                                          type=click.Choice(['y', 'n', 'a'], case_sensitive=False))
                    if choice == 'y':
                        continue
                    elif choice == 'n':
                        return
                    elif choice == 'a':
                        CONTINUE_FLAG = True
                        continue
                    else:
                        raise RuntimeError("How did you get here?")
                continue
        df_predictions.to_csv(out_df_fname, index=True)

if __name__ == '__main__':
    main()
    input('Press any key to continue...')