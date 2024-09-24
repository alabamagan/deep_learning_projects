import click
from sequence_identifier.io.preprocessing import flatten_dcm
from pathlib import Path
import SimpleITK as sitk

@click.command(help="Converts DICOM series in a folder to flattened NIfTI images.")
@click.argument('input-dir', type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path), required=True)
@click.argument('output-dir', type=click.Path(file_okay=False, dir_okay=True, path_type=Path), required=True)
@click.option('--id-globber', default=r'[A-Za-z]*[0-9]+', help='Regex pattern to match IDs from filenames.')
@click.option('--num-workers', default=1, help='Number of workers to use for processing.', type=int)
def cli_flatten_dcm(input_dir, output_dir, id_globber, num_workers):
    """Converts DICOM series in a folder to flattened NIfTI images.

    This tool processes a directory of DICOM files, converting each series
    into a NIfTI image. The results are stored in a flattened structure.


    Arguments:
        FOLDER: The directory containing DICOM files organized in subdirectories.

    Options:
        --id-globber: Regex pattern to identify unique IDs in filenames.
        --num-workers: Number of parallel workers for processing.

    The tool uses the specified regex pattern to extract IDs from filenames,
    allowing it to group and process DICOM series efficiently. The number
    of workers can be adjusted to optimize performance.
    """
    # Convert folder argument to a Path object
    folder_path = Path(input_dir)

    # Call the flatten_dcm function with the provided options
    for f in folder_path.iterdir():
        if not f.is_dir():
            click.echo("Skipping {}".format(f), err=False)
            continue
        result = flatten_dcm(f, id_globber, num_workers)

        for fname, img in result.items():
            outpath = output_dir / fname
            if not outpath.parent.exists():
                outpath.parent.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(img, str(outpath))


    # Output completion message and processed IDs
    click.echo("Flattening complete. Processed images:")
    for key in result:
        click.echo(f"ID: {key}")

if __name__ == '__main__':
    cli_flatten_dcm()