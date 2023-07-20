"""Input/Output utilities to read and write files."""

import os

import geomstats.backend as gs

# import nibabel
import numpy as np
import skimage
import trimesh

# from my28brains.discrete_surfaces import DiscreteSurfaces


def write_trimesh_to_ply(mesh, ply_path):
    """Write a mesh into a PLY file.

    Parameters
    ----------
    mesh : trimesh.Trimesh or list of trimesh.Trimesh's
        Surface mesh to write.
    ply_path : str
        Absolute path to write the mesh.
        Needs to end with .ply extension.
    """
    ply_text = trimesh.exchange.ply.export_ply(mesh, encoding="ascii")

    ply_dir = os.path.dirname(ply_path)
    if not os.path.exists(ply_dir):
        os.makedirs(ply_dir)

    print(f"Writing mesh at {ply_path}...")
    with open(ply_path, "wb") as f:
        f.write(ply_text)


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


def remove_degenerate_faces(vertices, faces, area_threshold=0.01, atol=gs.atol):
    """Remove degenerate faces of a surfaces.

    This returns a new surface with fewer vertices where the faces with area 0
    have been removed.

    A new DiscreteSurfaces object should be created afterwards,
    since one manifold corresponds to a predefined number of vertices and faces.
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    discrete_surfaces = DiscreteSurfaces(faces=faces)
    face_areas = discrete_surfaces.face_areas(vertices)
    face_mask = ~gs.less(face_areas, area_threshold)
    mesh.update_faces(face_mask)
    return mesh.vertices, mesh.faces
