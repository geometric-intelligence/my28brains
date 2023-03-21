"""Preprocessing of the 28brains dataset.

1. Segment the surfaces of the hippocampus and its substructurs.
2. Center the hippocampus and its substructures.

The processing is done with our `my28brains.meshing` module.

Meshed surfaces are stored into .ply files.

The meshes in the .ply files can be opened with:
- the `vscode-3d-preview` extension of VSCode
- [MeshLab](https://www.meshlab.net/)
- [Blender](https://www.blender.org/download/) with plugin [Stop-motion-OBJ]
"""

import glob
import itertools
import os

import default_config
import numpy as np
import trimesh

import my28brains.meshing

data_dir = default_config.data_dir
meshes_dir = default_config.meshes_dir
centered_dir = default_config.centered_dir

day_dirs = [os.path.join(data_dir, f"Day{i:02d}") for i in range(1, 61)]


def write_meshes(hemisphere, structure_id):
    """Write meshes for all substructures and all days.

    This function loads the segmentation images for a given structure/hemisphere,
    extracts the mesh, and saves the result.

    Parameters
    ----------
    hemisphere : str
        Hemisphere to process. Either 'left' or 'right'.
    structure_id : int
        Structure ID to process.
    """
    print(f"Looking into: {day_dirs}")
    nii_paths = []
    for i_day, day_dir in enumerate(day_dirs):
        for file_name in os.listdir(day_dir):
            if file_name.startswith(hemisphere) and file_name.endswith(".nii.gz"):
                nii_paths.append(os.path.join(day_dir, file_name))
                break

    print(
        f"Found {len(nii_paths)} nii paths for"
        f" hemisphere {hemisphere} and structure {structure_id}."
    )

    for i_path, nii_path in enumerate(nii_paths):
        day = i_path + 1
        ply_path = os.path.join(
            meshes_dir,
            f"{hemisphere}_structure_{structure_id}_day{day:02}.ply",
        )
        if os.path.exists(ply_path):
            print(f"File exists (no rewrite): {ply_path}")
            continue
        mesh = my28brains.meshing.extract_mesh(
            nii_path=nii_path, structure_id=structure_id
        )

        my28brains.meshing.write_trimesh_to_ply(mesh=mesh, ply_path=ply_path)


def write_whole_hippocampus_centered_mesh(hemisphere):
    """Write centered meshes for the whole hippocampus.

    This function loads the meshes for the whole hippocampus,
    centers them, and saves the result.

    Parameters
    ----------
    hemisphere : str
        Hemisphere to process. Either 'left' or 'right'.

    Returns
    -------
    hippocampus_centers : np.ndarray
        Array of shape (n_days, 3) containing the centers of the hippocampus
        for each day.
    """
    structure_id = -1
    string_base = os.path.join(
        meshes_dir, f"{hemisphere}_structure_{structure_id}**.ply"
    )
    paths = sorted(glob.glob(string_base))

    print(
        f"Found {len(paths)} ply files for"
        f" {hemisphere} hemisphere and anatomical structure {structure_id}:\n {paths}\n"
    )

    hippocampus_centers = []
    for path in paths:
        ply_path = os.path.join(centered_dir, os.path.basename(path))
        print(f"\tLoad mesh from path: {path}")
        mesh = trimesh.load(path)
        centered_mesh, hippocampus_center = my28brains.meshing.center_whole_hippocampus(
            mesh
        )
        hippocampus_centers.append(hippocampus_center)

        my28brains.meshing.write_trimesh_to_ply(mesh=centered_mesh, ply_path=ply_path)
    hippocampus_centers = np.array(hippocampus_centers)
    return hippocampus_centers


def write_substructure_centered_mesh(hemisphere, structure_id, hippocampus_centers):
    """Write centered meshes for a substructure.

    This function loads the meshes for a substructure,
    centers them, and saves the result.

    Parameters
    ----------
    hemisphere : str
        Hemisphere to process. Either 'left' or 'right'.
    structure_id : int
        Structure ID to process.
    hippocampus_centers : np.ndarray
        Array of shape (n_days, 3) containing the centers of the hippocampus
        for each day.
    """
    string_base = os.path.join(
        meshes_dir, f"{hemisphere}_structure_{structure_id}**.ply"
    )
    paths = sorted(glob.glob(string_base))

    print(
        f"Found {len(paths)} ply files for"
        f" {hemisphere} hemisphere and anatomical structure {structure_id}:\n {paths}\n"
    )

    for i_day, path in enumerate(paths):
        ply_path = os.path.join(centered_dir, os.path.basename(path))
        if os.path.exists(ply_path):
            print(f"File already exists (no rewrite): {ply_path}")
            continue
        print(f"\tLoad mesh from path: {path}")
        mesh = trimesh.load(path)
        centered_mesh = my28brains.meshing.center_substructure(
            mesh, hippocampus_centers[i_day]
        )
        my28brains.meshing.write_trimesh_to_ply(mesh=centered_mesh, ply_path=ply_path)


if __name__ == "__main__":
    """Parse the default_config file and run all preprocessing."""
    for hemisphere, structure_id in itertools.product(
        default_config.hemispheres, default_config.structure_ids
    ):
        write_meshes(hemisphere, structure_id)

    for hemisphere in default_config.hemispheres:
        hippocampus_centers = write_whole_hippocampus_centered_mesh(hemisphere)
        for structure_id in default_config.structure_ids:
            if structure_id != -1:
                write_substructure_centered_mesh(
                    hemisphere, structure_id, hippocampus_centers
                )
