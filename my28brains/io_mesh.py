"""Input/Output utilities to read and write files."""

import os

import geomstats.backend as gs
import trimesh

from my28brains.my28brains.discrete_surfaces import DiscreteSurfaces


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


def remove_degenerate_faces(vertices, faces, atol=gs.atol):
    """Remove degenerate faces of a surfaces.

    This returns a new surface with fewer vertices where the faces with area 0
    have been removed.

    A new DiscreteSurfaces object should be created afterwards,
    since one manifold corresponds to a predefined number of vertices and faces.
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    discrete_surfaces = DiscreteSurfaces(faces=faces)
    face_areas = discrete_surfaces.face_areas(vertices)
    # face_mask = ~gs.isclose(face_areas, 0.0, atol=atol)
    face_mask = ~gs.less(face_areas, 0.2)
    mesh.update_faces(face_mask)
    return mesh.vertices, mesh.faces
