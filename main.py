from pathlib import Path
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import copick
import requests
import time
import torch
from tomotwin.modules.inference.argparse_embed_ui import EmbedConfiguration
from tomotwin.embed_main import make_embeddor, embed_subvolumes
from tomotwin.modules.inference.boxer import Boxer
from tomotwin.modules.tools.extract_reference import ExtractReference

MODEL_URL = "https://zenodo.org/records/8358240/files/tomotwin_latest.pth?download=1"
DATA_DIR_URL = "https://ftp.ebi.ac.uk/empiar/world_availability/10499/"


def get_data_path():
    return os.getcwd()


def download_file(url, destination):
    """Download a file from a URL to a destination path."""
    response = requests.get(url, stream=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def install():
    data_path = get_data_path()
    # Ensure the data path exists
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Download the model
    model_path = os.path.join(data_path, "tomotwin_latest.pth")
    if not os.path.exists(model_path):
        download_file(MODEL_URL, model_path)
    else:
        print("Model already exists")


def get_coords(points, voxel_spacing):
    """Convert points to voxel coordinates"""
    coords = []

    for point in points:
        coords.append(
            (
                point.location.x / voxel_spacing,
                point.location.y / voxel_spacing,
                point.location.z / voxel_spacing,
            )
        )
    return np.floor(coords).astype(int)

def get_picks(copick_run):
    """Get picks - catch retry error from CDP"""
    try: 
        return copick_run.picks
    except requests.exceptions.RetryError as retry_err:
        return None

def get_copick_run_coords(copick_run, voxel_spacing):
    """Get picks with retry and convert points to voxel coordinates"""
    protein_pick_coords = {}
    for _ in range(3):
        picks = get_picks(copick_run)
        if picks is not None:
            break
        else:
            print("Getting picks failed, retrying after 2 seconds")
            time.sleep(2)
    else:
        raise Exception("Getting picks failed")

    for pick in picks:
        protein_pick_coords[pick.pickable_object_name] = get_coords(
            pick.points, voxel_spacing
        )
    return protein_pick_coords

def make_config():
    """create configuration for tomotwin"""
    model_path = "tomotwin_latest.pth"
    conf = EmbedConfiguration(model_path, None, None, None, 2)
    conf.model_path = model_path
    conf.batchsize = 35 # should not be necessary
    conf.stride = [1, 1, 1] # should not be necessary
    conf.window_size = 37

    return conf


def process_run(copick_run, voxel_size, embedor, conf):
    """Process copick run"""
    print(copick_run.name)
    
    # Create run path
    run_path = Path("output") / copick_run.name
    run_path.mkdir(parents=True, exist_ok=True)

    # get run coords
    run_coords = get_copick_run_coords(copick_run, voxel_size)

    # get tomogram at 10.012 voxel size
    tomogram = (
        copick_run.get_voxel_spacing(voxel_size)
        .get_tomograms("wbp", portal_meta_query={"processing": "denoised"})[0]
        .numpy()
    )

    # 
    embeddings = []
    for protein, coords in run_coords.items():

        # create protein path 
        protein_path = run_path / protein
        protein_path.mkdir(exist_ok=True)

        # convert coords to dataframe
        coords_df = pd.DataFrame(coords)
        coords_df.columns = ["X", "Y", "Z"]

        # extract tomogram and save as a `mrc` file
        ExtractReference.extract_and_save(
            tomogram,
            coords_df,
            box_size=37,
            out_pth=protein_path,
            basename=protein,
            apix=voxel_size,
        )

        # get list of subvolumes
        subvolume_list = list(protein_path.glob("*.mrc"))
        if len(subvolume_list) == 0:
            continue

        # create embedding output directory
        output_path = protein_path / "output"
        output_path.mkdir(exist_ok=True)

        # set confiugration outputs
        conf.volumes_path = str(protein_path)
        conf.output_path = str(output_path)

        # embed subvolumes
        df = embed_subvolumes(subvolume_list, embedor, conf)

        # concat embeddings
        embeddings.append(df)

        # empty cuda cache
        torch.cuda.empty_cache()
    
    return pd.concat(embeddings)


def main(copick_config_path: str, voxel_size: float):
    """main runner function"""

    # download tomotwin model if not present
    install()

    # set up configs
    copick_root = copick.from_file(copick_config_path)
    conf = make_config()
    embedor = make_embeddor(conf, rank=None, world_size=1)


    for run in copick_root.runs:
        # create run output directory
        run_path = Path("output") / run.name
        if run_path.exists():
            print(f"Run {run.name} exists - skipping")
            continue

        # process run
        process_run(run, voxel_size, embedor, conf)


if __name__ == "__main__":
    copick_config_path = "copick.config"
    voxel_size = 10.012
    main(copick_config_path=copick_config_path, voxel_size=voxel_size)
