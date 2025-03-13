from pathlib import Path

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import shutil
import matplotlib.pyplot as plt

from copy import deepcopy

import copick

import subprocess

import requests
import json
from cryoet_data_portal import Client, Tomogram
from scipy.ndimage import zoom
import math

MODEL_URL = "https://zenodo.org/records/8358240/files/tomotwin_latest.pth?download=1"
DATA_DIR_URL = "https://ftp.ebi.ac.uk/empiar/world_availability/10499/"

def get_data_path():
    return os.getcwd()
    
def download_file(url, destination):
    """Download a file from a URL to a destination path."""
    breakpoint()
    response = requests.get(url, stream=True)
    with open(destination, 'wb') as f:
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

def get_coords(
    points,
    voxel_spacing
):
    coords = []
    
    for point in points:
        coords.append((
            point.location.x / voxel_spacing,
            point.location.y / voxel_spacing,
            point.location.z / voxel_spacing,
        ))
    return np.floor(coords).astype(int)
    
def get_copick_run_coords(
    copick_run,
    voxel_spacing
):
    protein_pick_coords = {}
    for picks in copick_run.picks:
        protein_pick_coords[picks.pickable_object_name] = get_coords(picks.points, voxel_spacing)
    return protein_pick_coords

def create_slices_from_coords(coords, window):
    boundary = math.ceil(window / 2) - 1
    slices = []
    for coord in coords:
        x, y, z = coord
        slice_x = slice(x-boundary, x+boundary)
        slice_y = slice(y-boundary, y+boundary)
        slice_z = slice(z-boundary, z+boundary)
        slices.append((slice_x, slice_y, slice_z))
    return slices

def extend_slices(slices, window):
    boundary = math.ceil(window / 2) - 1
    extended_slices = []
    for slice_x, slice_y, slice_z in slices:
        extended_slices.append((
            slice(slice_x.start - boundary, slice_x.stop + boundary),
            slice(slice_y.start - boundary, slice_y.stop + boundary),
            slice(slice_z.start - boundary, slice_z.stop + boundary)
        ))
    return extended_slices
    
def process_run(copick_run, voxel_size):
    print(copick_run.name)
    run_coords = get_copick_run_coords(copick_run, voxel_size)
    tomogram = copick_run.get_voxel_spacing(voxel_size).get_tomograms("wbp", portal_meta_query={"processing": "denoised"})[0].numpy()
    tomogram_xyz = tomogram.transpose(2, 1, 0)
    for protein, coords in run_coords.items():
        slices = create_slices_from_coords(coords, window=37)
        extended_slices = extend_slices(slices, window=37)
        for extended_slice in extended_slices:
            subtomogram = tomogram_xyz[extended_slice]
            breakpoint()





def main(copick_config_path: str, voxel_size: float):
    install()
    copick_root = copick.from_file(copick_config_path)
    tomo_list = [run.name for run in copick_root.runs]
    for run in copick_root.runs:
        process_run(run, voxel_size)
        break
    print(tomo_list)
    print("hello world")


if __name__ == "__main__":
    copick_config_path = "copick.config"
    voxel_size = 10.012
    main(copick_config_path=copick_config_path, voxel_size=voxel_size)
