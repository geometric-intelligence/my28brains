"""Center (and register) meshes."""

import glob
import os

import numpy as np
import trimesh

from my28brains.meshing import extraction


def center_whole_hippocampus_and_write(input_dir, output_dir, hemisphere):
    """Write centered meshes for the whole hippocampus.

    This function loads the meshes for the whole hippocampus,
    centers them, and saves the result.

    Parameters
    ----------
    input_dir : str
        Input directory.
        Here, corresponding to directory of meshes extracted from .nii.
    output_dir str
        Output directory in my28brains/my28brains/results.
        Here storing centered meshes.
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
        input_dir, f"{hemisphere}_structure_{structure_id}**.ply"
    )
    paths = sorted(glob.glob(string_base))

    print(
        f"Found {len(paths)} ply files for"
        f" {hemisphere} hemisphere and anatomical structure {structure_id}:\n {paths}\n"
    )

    hippocampus_centers = []
    for path in paths:
        ply_path = os.path.join(output_dir, os.path.basename(path))
        print(f"\tLoad mesh from path: {path}")
        mesh = trimesh.load(path)
        centered_mesh, hippocampus_center = center_whole_hippocampus(mesh)
        hippocampus_centers.append(hippocampus_center)

        extraction.write_trimesh_to_ply(mesh=centered_mesh, ply_path=ply_path)
    hippocampus_centers = np.array(hippocampus_centers)
    return hippocampus_centers


def center_substructure_and_write(
    input_dir, output_dir, hemisphere, structure_id, hippocampus_centers
):
    """Write centered meshes for a substructure.

    This function loads the meshes for a substructure,
    centers them, and saves the result.

    Parameters
    ----------
    input_dir : str
        Input directory.
        Here, corresponding to directory of meshes extracted from .nii.
    output_dir : str
        Output directory in my28brains/my28brains/results.
        Here storing centered meshes.
    hemisphere : str
        Hemisphere to process. Either 'left' or 'right'.
    structure_id : int
        Structure ID to process.
    hippocampus_centers : np.ndarray
        Array of shape (n_days, 3) containing the centers of the hippocampus
        for each day.
    """
    string_base = os.path.join(
        input_dir, f"{hemisphere}_structure_{structure_id}**.ply"
    )
    paths = sorted(glob.glob(string_base))

    print(
        f"Found {len(paths)} ply files for"
        f" {hemisphere} hemisphere and anatomical structure {structure_id}:\n {paths}\n"
    )

    for i_day, path in enumerate(paths):
        ply_path = os.path.join(output_dir, os.path.basename(path))
        if os.path.exists(ply_path):
            print(f"File already exists (no rewrite): {ply_path}")
            continue
        print(f"\tLoad mesh from path: {path}")
        mesh = trimesh.load(path)
        centered_mesh = center_substructure(mesh, hippocampus_centers[i_day])
        extraction.write_trimesh_to_ply(mesh=centered_mesh, ply_path=ply_path)


def center_whole_hippocampus(mesh):
    """Center a mesh by putting its barycenter at origin of the coordinates.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to center.

    Returns
    -------
    centered_mesh : trimesh.Trimesh
        Centered Mesh.
    hippocampus_center: coordinates of center of the mesh before centering
    """
    vertices = mesh.vertices
    hippocampus_center = np.mean(vertices, axis=0)
    centered_vertices = vertices - hippocampus_center
    return (
        trimesh.Trimesh(vertices=centered_vertices, faces=mesh.faces),
        hippocampus_center,
    )


def center_substructure(mesh, hippocampus_center):
    """Center a mesh by putting its barycenter at origin of the coordinates.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to center.
    center: coordinates of the vector that you would like to use to center your mesh
        i.e. in this case, this is the center of the whole brain, and mesh will
        be the meshes of the substructures of the brain.

    Returns
    -------
    centered_mesh : trimesh.Trimesh
        Centered Mesh.
    """
    vertices = mesh.vertices
    centered_vertices = vertices - hippocampus_center
    return trimesh.Trimesh(vertices=centered_vertices, faces=mesh.faces)


def register_mesh(mesh, base_mesh):
    """Register a mesh to a base mesh.

    Note that the rigid registration slightly un-centered the registered mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to register.

    Returns
    -------
    registered_mesh : trimesh.Trimesh
        Registered Mesh.
    """
    transform_mesh_to_base_mesh, cost = trimesh.registration.mesh_other(
        mesh=mesh, other=base_mesh, scale=False
    )
    print(f"Cost: {cost}")
    # Note: This modifies the original mesh in place
    registered_mesh = mesh.apply_transform(transform_mesh_to_base_mesh)
    return registered_mesh
