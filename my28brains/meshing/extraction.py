"""Input/Output utilities to read and write files."""

import os

import nibabel
import skimage
import trimesh

import my28brains.meshing.write as write


def extract_meshes_from_nii_and_write(input_dir, output_dir, hemisphere, structure_id):
    """Write meshes for all substructures and all days.

    This function loads the segmentation images for a given structure/hemisphere,
    extracts the mesh, and saves the result.

    Parameters
    ----------
    input_dir : str
        Input directory.
        Here, corresponding to directory of directories of days with .nii.
    output_dir str
        Output directory in my28brains/my28brains/results.
        Here storing meshes extracted from .nii.
    hemisphere : str
        Hemisphere to process. Either 'left' or 'right'.
    structure_id : int
        Structure ID to process.
    """
    print(f"Looking into: {input_dir}")
    nii_paths = []
    for day_dir in input_dir:
        for file_name in os.listdir(day_dir):
            if file_name.startswith(hemisphere) and file_name.endswith(".nii.gz"):
                nii_paths.append(os.path.join(day_dir, file_name))
                break

    print(f"Found {len(nii_paths)} nii paths for hemisphere {hemisphere}.")

    for i_path, nii_path in enumerate(nii_paths):
        day = i_path + 1
        ply_path = os.path.join(
            output_dir,
            f"{hemisphere}_structure_{structure_id}_day{day:02}.ply",
        )
        if os.path.exists(ply_path):
            print(f"File exists (no rewrite): {ply_path}")
            continue
        mesh = extract_mesh(nii_path=nii_path, structure_id=structure_id)

        write.trimesh_to_ply(mesh=mesh, ply_path=ply_path)


def _extract_mesh(img_fdata, structure_id):
    """Extract one surface mesh from the fdata of a segmented image."""
    if structure_id == -1:
        img_mask = img_fdata != 0
    else:
        img_mask = img_fdata == structure_id
    meshing_result = skimage.measure.marching_cubes(
        img_mask, level=0, step_size=1, allow_degenerate=False, method="lorensen"
    )
    mesh = trimesh.Trimesh(vertices=meshing_result[0], faces=meshing_result[1])
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
        Index or list of indices of the anatomical structure(s) to
        mesh from the hippocampal formation.
        Possible indices are either -1 (entire structure) or any of the
        labels of the segmentation.

    Return
    ------
    mesh : trimesh.Trimesh or list of trimesh.Trimesh's
        Surface mesh (or list of meshes) of the anatomical structure.
    """
    print(f"Loading image from {nii_path}...")
    img = nibabel.load(nii_path)
    img_fdata = img.get_fdata()

    if isinstance(structure_id, int):
        return _extract_mesh(img_fdata, structure_id)

    meshes = []
    for one_structure_id in structure_id:
        one_mesh = _extract_mesh(img_fdata, one_structure_id)
        meshes.append(one_mesh)
    return meshes
