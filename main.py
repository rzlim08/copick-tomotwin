from pathlib import Path
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import copick
import requests
import math
from tomotwin.modules.inference.embedor import (
    Embedor,
)
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


def get_copick_run_coords(copick_run, voxel_spacing):
    protein_pick_coords = {}
    for picks in copick_run.picks:
        protein_pick_coords[picks.pickable_object_name] = get_coords(
            picks.points, voxel_spacing
        )
    return protein_pick_coords


def create_slices_from_coords(coords, window):
    boundary = math.ceil(window / 2) - 1
    slices = []
    for coord in coords:
        x, y, z = coord
        slice_x = slice(x - boundary, x + boundary)
        slice_y = slice(y - boundary, y + boundary)
        slice_z = slice(z - boundary, z + boundary)
        slices.append((slice_x, slice_y, slice_z))
    return slices


def extend_slices(slices, window):
    boundary = math.ceil(window / 2) - 1
    extended_slices = []
    for slice_x, slice_y, slice_z in slices:
        extended_slices.append(
            (
                slice(slice_x.start - boundary, slice_x.stop + boundary),
                slice(slice_y.start - boundary, slice_y.stop + boundary),
                slice(slice_z.start - boundary, slice_z.stop + boundary),
            )
        )
    return extended_slices


def sliding_window_embedding(
    tomo: np.array, boxer: Boxer, embedor: Embedor
) -> np.array:
    """
    Embeds the tomogram using a sliding window approach, placing embeddings in an array based on their positions.

    :param tomo: Tomogram as a numpy array.
    :param boxer: Box provider that generates positions for embedding.
    :param embedor: Embedor to embed the boxes extracted from the tomogram.
    :return: A numpy array with embeddings placed according to their positions.
    """
    boxes = boxer.box(tomogram=tomo)
    breakpoint()
    embeddings = embedor.embed(volume_data=boxes)
    if embeddings is None:
        return None

    # Assuming the shape of tomo is Z, Y, X and embeddings are in Z, Y, X, Embed_dim
    # Initialize an empty array for the embeddings with an additional dimension for the embedding vector
    embedding_array = np.zeros(
        tomo.shape + (embeddings.shape[-1],), dtype=embeddings.dtype
    )

    for i in range(embeddings.shape[0]):
        pos_z, pos_y, pos_x = boxes.get_localization(i).astype(int)
        embedding_array[pos_z, pos_y, pos_x, :] = embeddings[i]

    return embedding_array


def make_config():
    model_path = "tomotwin_latest.pth"
    conf = EmbedConfiguration(model_path, None, None, None, 2)
    conf.model_path = model_path
    conf.batchsize = 35
    conf.stride = [1, 1, 1]
    conf.window_size = 37

    return conf


def process_run(copick_run, voxel_size):
    print(copick_run.name)
    conf = make_config()
    embedor = make_embeddor(conf, rank=None, world_size=1)

    run_path = Path("output") / copick_run.name
    run_path.mkdir(parents=True, exist_ok=True)
    run_coords = get_copick_run_coords(copick_run, voxel_size)
    tomogram = (
        copick_run.get_voxel_spacing(voxel_size)
        .get_tomograms("wbp", portal_meta_query={"processing": "denoised"})[0]
        .numpy()
    )
    embeddings = []
    for protein, coords in run_coords.items():
        protein_path = run_path / protein
        protein_path.mkdir(exist_ok=True)
        coords_df = pd.DataFrame(coords)
        coords_df.columns = ["X", "Y", "Z"]
        ExtractReference.extract_and_save(
            tomogram,
            coords_df,
            box_size=37,
            out_pth=protein_path,
            basename=protein,
            apix=voxel_size,
        )
        subvolume_list = list(protein_path.glob("*.mrc"))
        conf.volumes_path = str(protein_path)
        output_path = protein_path / "output"
        output_path.mkdir(exist_ok=True)
        conf.output_path = str(output_path)
        df = embed_subvolumes(subvolume_list, embedor, conf)
        embeddings.append(df)
    
    return pd.concat(embeddings)


def main(copick_config_path: str, voxel_size: float):
    install()
    copick_root = copick.from_file(copick_config_path)
    embeddings = []
    for run in copick_root.runs:
        df = process_run(run, voxel_size)
        embeddings.append(df)

    full_df = pd.concat(embeddings)
    full_df.to_parquet('embeddings.parquet.gzip',
              compression='gzip')


if __name__ == "__main__":
    copick_config_path = "copick.config"
    voxel_size = 10.012
    main(copick_config_path=copick_config_path, voxel_size=voxel_size)
