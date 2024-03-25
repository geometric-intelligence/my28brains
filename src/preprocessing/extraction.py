"""Input/Output utilities to extract meshes from nii brain images.

Structure_ID names, numbers, and colors:
----------------------------------------
1   255    0    0        1  1  1    "CA1"
2     0  255    0        1  1  1    "CA2+3"
3     0    0  255        1  1  1    "DG"
4   255  255    0        1  1  1    "ERC"
5     0  255  255        1  1  1    "PHC"
6   255    0  255        1  1  1    "PRC"
7    80  179  221        1  1  1    "SUB"
8   255  215    0        1  1  1    "AntHipp"
9   184  115   51        1  1  1    "PostHipp"
2, 6 are expected to grow in volume with progesterone
4, 5 are expected to shrink in volume with progesterone
"""

import os

import nibabel
import numpy as np
import skimage
import trimesh

from src.preprocessing import writing


def extract_meshes_from_nii_and_write(input_dir, output_dir, hemisphere, structure_id):
    """Write meshes for all substructures and all days.

    This function loads the segmentation images for a given structure/hemisphere,
    extracts the mesh, and saves the result.

    Parameters
    ----------
    input_dir : str
        Input directory.
        Here, directory of directories of days with .nii.
    output_dir str
        Output directory in my28brains/src/results/1_preprocess.
        Here, directory of meshes extracted from .nii.
    hemisphere : str, {'left', 'right'}
        Hemisphere to process.
    structure_id : int
        ID of the hippocampus anatomical structure to process.
        Possible indices are either -1 (entire structure) or any of the
        labels of the segmentation.
    """
    print(f"Looking into: {input_dir}")
    nii_paths = []
    valid_day_numbers = []
    for i_day, day_dir in enumerate(input_dir):
        file_found = False
        for file_name in os.listdir(day_dir):
            if file_name.startswith(hemisphere) and file_name.endswith(".nii.gz"):
                nii_paths.append(os.path.join(day_dir, file_name))
                file_found = True
                print(f"Found {file_name} in {day_dir}")
                valid_day_numbers.append(i_day + 1)
                break
        if not file_found:
            print(f"File not found in {day_dir}")
    valid_day_numbers = np.array(valid_day_numbers)

    print(
        f"\na. (Mesh) Found {len(nii_paths)} nii paths for hemisphere {hemisphere} in {input_dir}"
        f"\nValid day numbers: {valid_day_numbers}"
    )
    for path in nii_paths:
        print(path)

    for day, nii_path in zip(valid_day_numbers, nii_paths):
        ply_path = os.path.join(
            output_dir,
            f"{hemisphere}_structure_{structure_id}_day{day:02}.ply",
        )
        if os.path.exists(ply_path):
            print(f"File exists (no rewrite): {ply_path}")
            continue
        mesh = extract_mesh(nii_path=nii_path, structure_id=structure_id)
        writing.trimesh_to_ply(mesh=mesh, ply_path=ply_path)
        writing.save_colors_as_np_array(mesh, ply_path)


def _extract_mesh(img_fdata, structure_id):
    """Extract one surface mesh from the fdata of a segmented image.

    Parameters
    ----------
    img_fdata: array-like, shape = [n_x, n_y, n_z]. Voxels which are colored
        according to substructure assignment. For example, color of voxel
        (0, 0, 0) is an integer value that can be anywhere from 0-9.
    """
    print(f"Img fdata shape: {img_fdata.shape}")
    if structure_id == -1:
        img_mask = img_fdata != 0
    else:
        img_mask = img_fdata == structure_id

    masked_img_fdata = np.where(img_mask, img_fdata, 0)
    print(f"Masked img fdata shape: {masked_img_fdata.shape}")
    (
        vertices,
        faces,
        _,
        values,
    ) = skimage.measure.marching_cubes(  # omitted value is "normals"
        masked_img_fdata,
        level=0,
        step_size=1,
        allow_degenerate=False,
        method="lorensen",
    )
    print(f"Colors: {values.min()}, {values.max()}")
    print(f"Vertices.shape = {vertices.shape}")

    color_dict = {  # trimesh uses format [r, g, b, a] where a is alpha
        1: [255, 0, 0, 255],
        2: [0, 255, 0, 255],
        3: [0, 0, 255, 255],
        4: [255, 255, 0, 255],
        5: [0, 255, 255, 255],
        6: [255, 0, 255, 255],
        7: [80, 179, 221, 255],
        8: [255, 215, 0, 255],
        9: [184, 115, 51, 255],
    }

    colors = np.array([np.array(color_dict[value]) for value in values])
    print(f"Colors.shape = {colors.shape}")
    print("Colors:", colors)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)
    return mesh


def extract_mesh(nii_path, structure_id=-1):
    """Extract surface mesh(es) from a structure in the hippocampal formation.

    The nii_path should contain the segmentation of the hippocampal formation,
    with the following structures (given from their corresponding integer labels):

    From ITK Snap file:
    0     0    0    0        0  0  0    "Clear Label"
    1   255    0    0        1  1  1    "CA1"
    2     0  255    0        1  1  1    "CA2+3"
    3     0    0  255        1  1  1    "DG"
    4   255  255    0        1  1  1    "ERC"
    5     0  255  255        1  1  1    "PHC"
    6   255    0  255        1  1  1    "PRC"
    7    80  179  221        1  1  1    "SUB"
    8   255  215    0        1  1  1    "AntHipp"
    9   184  115   51        1  1  1    "PostHipp"
    Note that the label 0 denotes the background.

    Parameters
    ----------
    nii_path : str
        Path of the .nii.gz image with the segmented structures.
    structure_id : int or list
        ID of the hippocampus anatomical structure to
        mesh from the hippocampal formation.
        Possible indices are either -1 (entire structure) or any of the
        labels of the segmentation.

    Return
    ------
    mesh : trimesh.Trimesh or list of trimesh.Trimesh's
        Surface mesh (or list of meshes) of the anatomical structure.
    """
    print(f"Load image from {nii_path}")
    img = nibabel.load(nii_path)
    img_fdata = img.get_fdata()

    if isinstance(structure_id, int):
        return _extract_mesh(img_fdata, structure_id)

    meshes = []
    for one_structure_id in structure_id:
        one_mesh = _extract_mesh(img_fdata, one_structure_id)
        meshes.append(one_mesh)
    return meshes
