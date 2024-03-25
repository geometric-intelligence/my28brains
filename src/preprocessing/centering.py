"""Center (and register) meshes."""

import glob
import os

import numpy as np
import trimesh

from src.preprocessing import writing


def center_whole_hippocampus_and_write(input_dir, output_dir, hemisphere):
    """Write centered meshes for the whole hippocampus.

    This function loads the meshes for the whole hippocampus,
    centers them, and saves the result.

    Parameters
    ----------
    input_dir : str
        Input directory in my28brains/src/results/1_preprocess.
        Here, directory of meshes extracted from .nii.
    output_dir str
        Output directory in my28brains/src/results/1_preprocess.
        Here directory of centered meshes.
    hemisphere : str, {'left', 'right'}
        Hemisphere to process.

    Returns
    -------
    hippocampus_centers : np.ndarray, shape=[n_days, 3]
        Coordinates of the barycenter of the hippocampus for each day.
    """
    structure_id = -1
    string_base = os.path.join(
        input_dir, f"{hemisphere}_structure_{structure_id}**.ply"
    )
    paths = sorted(glob.glob(string_base))
    print(
        f"\nb. (Center) Found {len(paths)} .plys for ({hemisphere}, {structure_id}) in {input_dir}."
    )

    hippocampus_centers = []
    for path in paths:
        ply_path = os.path.join(output_dir, os.path.basename(path))
        if os.path.exists(ply_path):
            print(f"File exists (no rewrite): {ply_path}")
            continue
        print(f"Load mesh from {path}")
        mesh = trimesh.load(path)
        centered_mesh, hippocampus_center = center_whole_hippocampus(mesh)
        hippocampus_centers.append(hippocampus_center)

        writing.trimesh_to_ply(mesh=centered_mesh, ply_path=ply_path)
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
        Input directory in my28brains/src/results/1_preprocess.
        Directory of meshes extracted from .nii.
    output_dir : str
        Output directory in my28brains/src/results/1_preprocess.
        Here storing centered meshes.
    hemisphere : str, {'left', 'right'}
        Hemisphere to process.
    structure_id : int
        Structure ID to process.
        Possible indices are either -1 (entire structure) or any of the
        labels of the segmentation.
    hippocampus_centers : np.ndarray, shape=[n_days, 3]
        Coordinates of the barycenter of the hippocampus for each day.
    """
    string_base = os.path.join(
        input_dir, f"{hemisphere}_structure_{structure_id}**.ply"
    )
    paths = sorted(glob.glob(string_base))
    print(f"Found {len(paths)} .plys for {hemisphere} hemisphere, id {structure_id}.")

    for i_day, path in enumerate(paths):
        ply_path = os.path.join(output_dir, os.path.basename(path))
        if os.path.exists(ply_path):
            print(f"File exists (no rewrite): {ply_path}")
            continue
        print(f"\tLoad mesh from {path}")
        mesh = trimesh.load(path)
        centered_mesh = center_substructure(mesh, hippocampus_centers[i_day])
        writing.trimesh_to_ply(mesh=centered_mesh, ply_path=ply_path)


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
        trimesh.Trimesh(
            vertices=centered_vertices,
            faces=mesh.faces,
            vertex_colors=mesh.visual.vertex_colors,
        ),
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
    return trimesh.Trimesh(
        vertices=centered_vertices,
        faces=mesh.faces,
        vertex_colors=mesh.visual.vertex_colors,
    )


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
    transform_mesh_to_base_mesh, _ = trimesh.registration.mesh_other(
        mesh=mesh, other=base_mesh, scale=False
    )
    # Note: This modifies the original mesh in place
    registered_mesh = mesh.apply_transform(transform_mesh_to_base_mesh)
    return registered_mesh
