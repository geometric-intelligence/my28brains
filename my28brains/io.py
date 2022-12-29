"""Input/Output utilities to read and write files."""

import os

import trimesh


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
