"""Functions to write meshes to files."""

import os

import numpy as np
import trimesh


def trimesh_to_ply(mesh, ply_path):
    """Write a mesh into a PLY file.

    Parameters
    ----------
    mesh : trimesh.Trimesh or list of trimesh.Trimesh's
        Surface mesh to write.
    ply_path : str
        Absolute path to write the mesh.
        Needs to end with .ply extension.
    """
    # ply_text = trimesh.exchange.ply.export_ply(mesh, encoding="ascii", include_attributes=True)
    ply_text = trimesh.exchange.ply.export_ply(
        mesh, encoding="binary", include_attributes=True
    )

    ply_dir = os.path.dirname(ply_path)
    if not os.path.exists(ply_dir):
        os.makedirs(ply_dir)

    print(f"- Write mesh to {ply_path}")
    with open(ply_path, "wb") as f:
        f.write(ply_text)


def save_colors_as_np_array(mesh, ply_path):
    """Save the colors of a mesh into a .txt file.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Surface mesh to write.
    ply_path : str
        Absolute path to write the colors.
        Needs to end with .txt extension.
    """
    arr_path = ply_path.replace(".ply", "_colors.npy")
    print(f"- Write colors to {arr_path}")
    print(f"mesh.visual.vertex_colors: {mesh.visual.vertex_colors}")
    colors = mesh.visual.vertex_colors
    np.save(arr_path, colors)

    # with open(txt_path, "w") as f:
    #     for color in mesh.visual.vertex_colors:
    #         f.write(f"{color[0]} {color[1]} {color[2]} {color[3]}\n")
