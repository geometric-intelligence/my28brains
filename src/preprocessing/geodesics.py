"""Computation of unparameterized geodesics."""

import glob
import inspect
import os
import time

import geomstats.backend as gs
import numpy as np
import torch
import trimesh
from geomstats.geometry.discrete_surfaces import DiscreteSurfaces

import H2_SurfaceMatch.H2_match  # noqa: E402
import H2_SurfaceMatch.utils.input_output as h2_io  # noqa: E402
import H2_SurfaceMatch.utils.utils  # noqa: E402
import src.import_project_config as pc
import src.preprocessing.writing as write

# from src.regression.discrete_surfaces import DiscreteSurfaces


def remove_degenerate_faces(vertices, faces, vertex_colors, area_threshold=0.01):
    """Remove degenerate faces of a surfaces.

    This returns a new surface with fewer vertices where the faces with area 0
    have been removed.

    A new DiscreteSurfaces object should be created afterwards,
    since one manifold corresponds to a predefined number of vertices and faces.
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors)

    discrete_surfaces = DiscreteSurfaces(faces=faces)
    face_areas = discrete_surfaces.face_areas(gs.array(vertices))
    face_mask = ~gs.less(face_areas, area_threshold)
    mesh.update_faces(face_mask)
    return mesh.vertices, mesh.faces, mesh.visual.vertex_colors


def remove_degenerate_faces_and_write(
    input_dir, output_dir, hemisphere, structure_id, area_threshold
):
    """Write centered nondegenerate meshes for a given structure/hemisphere.

    This function loads the centered meshes for a given structure/hemisphere,
    removes degenerate faces, and saves the result.

    Parameters
    ----------
    input_dir : str
        Input directory in my28brains/src/results.
        Here storing meshes.
    output_dir str
        Input directory in my28brains/src/results.
        Here storing meshes without degenerated faces.
    hemisphere : str
        Hemisphere to process. Either 'left' or 'right'.
    structure_id : int
        Structure ID to process.
    """
    string_base = os.path.join(
        input_dir,
        f"{hemisphere}_structure_{structure_id}**.ply",
    )
    paths = sorted(glob.glob(string_base))
    print(
        f"\nc. (Remove degenerate) Found {len(paths)} .plys for ({hemisphere}, {structure_id}) in {input_dir}"
    )

    for path in paths:
        ply_path = os.path.join(
            output_dir,
            os.path.basename(path).split(".")[0] + f"_at_{area_threshold}.ply",
        )
        if os.path.exists(ply_path):
            print(f"File exists (no rewrite): {ply_path}")
            continue
        print(f"Load mesh from {path}")
        mesh = trimesh.load(path)
        new_vertices, new_faces, new_colors = remove_degenerate_faces(
            mesh.vertices, mesh.faces, mesh.visual.vertex_colors, area_threshold
        )
        new_mesh = trimesh.Trimesh(
            vertices=new_vertices, faces=new_faces, vertex_colors=new_colors
        )
        write.trimesh_to_ply(new_mesh, ply_path)


def scale_decimate(path, project_dir):
    """Scale and decimate a mesh.

    Parameters
    ----------
    path : str
        Path to the mesh.
    config : object
        Config object containing:
        - scaling_factor: float
            Scaling factor to apply to the mesh.
        - initial_decimation_fact: int
            Initial decimation factor to apply to the mesh.

    Returns
    -------
    list
        List containing:
        - vertices: np.ndarray
            Vertices of the mesh.
        - faces: np.ndarray
            Faces of the mesh.
    """
    project_config = pc.import_default_config(project_dir)
    vertices, faces, colors = h2_io.loadData(path)
    vertices = vertices / project_config.scaling_factor  # was / 10
    n_faces_after_decimation = int(
        faces.shape[0] / project_config.initial_decimation_fact
    )
    (
        vertices_after_decimation,
        faces_after_decimation,
        colors_after_decimation,
    ) = H2_SurfaceMatch.utils.utils.decimate_mesh(
        vertices, faces, n_faces_after_decimation, colors=colors
    )

    return [vertices_after_decimation, faces_after_decimation, colors_after_decimation]


def scale_decimate_and_compute_geodesic(
    start_path, end_path, project_dir, template_path=None, gpu_id=1
):
    """Compute the geodesic between two meshes, after scaling and decimation of both.

    Parameters
    ----------
    start_path : str
        Path to the source mesh.
    end_path : str
        Path to the target mesh.
    template_path : str
        Path to the template mesh.
    gpu_id : int
        ID of the GPU to use.
    config : object
        Config object containing parameters of the experiment.
    """
    project_config = pc.import_default_config(project_dir)
    device = torch.device(
        f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id is not None else "cpu"
    )
    start_time = time.time()

    source = scale_decimate(
        path=start_path, project_dir=project_dir
    )  # [vertices, faces, colors]
    target = scale_decimate(path=end_path, project_dir=project_dir)

    template = None
    if template_path:
        template = scale_decimate(path=template_path, project_dir=project_dir)

    # decimation also happens at the start of h2_match.H2multires
    geod, F0, color0 = H2_SurfaceMatch.H2_match.H2MultiRes(
        source=source,
        target=target,
        a0=project_config.a0,
        a1=project_config.a1,
        b1=project_config.b1,
        c1=project_config.c1,
        d1=project_config.d1,
        a2=project_config.a2,
        resolutions=project_config.resolutions,
        start=template,
        paramlist=project_config.paramlist,
        device=device,
    )
    comp_time = time.time() - start_time
    print(f"Geodesic computation took: {comp_time / 60:.2f} minutes.")
    return geod, F0, color0


def interpolate_with_geodesic(input_paths, output_dir, i_pair, gpu_id, config):
    """Auxiliary function that will be run in parallel on different GPUs.

    This creates a geodesic with n_geodesic_time between a pair (start, end) of meshes.

    First: source, target (+ template) are preprocessed: scaled and decimated.
    Then, geodesics are computed between source & target,
    using parameterization of template if template is provided.

    Note the decimation of faces: decrease the resolution of the mesh,
    ie the number of face and vertices.

    Parameters
    ----------
    i_pair : int
        Index of the pair of meshes to process.
    i_template : int
        Index of the template mesh which defines the vertices
        and faces that all mesh should have.
    paths : list
        List of paths to the meshes.
    gpu_id : int
        ID of the GPU to use.
    """
    project_dir = config.project_dir
    project_config = pc.import_default_config(project_dir)
    start_path, end_path = input_paths[i_pair], input_paths[i_pair + 1]

    basename = os.path.splitext(os.path.basename(start_path))[0]
    ply_prefix = os.path.join(output_dir, basename)

    file_ply = f"{ply_prefix}0.ply"
    if os.path.exists(os.path.join(output_dir, file_ply)):
        print(f"Geodesic for pair {i_pair} exists. Skipping to next pair.")
        return

    geod, F0 = scale_decimate_and_compute_geodesic(
        start_path=start_path, end_path=end_path, gpu_id=gpu_id, project_dir=project_dir
    )

    for i_geodesic_time in range(geod.shape[0]):
        h2_io.plotGeodesic(
            [geod[i_geodesic_time]],
            F0,
            stepsize=project_config.stepsize,  # open3d plotting parameter - unused
            file_name=f"{ply_prefix}{i_geodesic_time}",
            axis=[0, 1, 0],
            angle=-1 * np.pi / 2,
        )
    print(f"\tGeodesic interpolation {i_pair} saved to: " f"{output_dir}.")


def reparameterize_with_geodesic(
    input_paths, output_dir, i_path, project_dir, gpu_id=None
):
    """Auxiliary function that will be run in parallel on different GPUs.

    The start path is the path whose parameterization is used as reference.
    The end path is the path that needs to be reparameterized

    We use the end mesh as the basename for the ply files becuase that is the
    mesh that we are re-parameterizing.

    Note the decimation of faces: decrease the resolution of the mesh,
    ie the number of face and vertices.

    Parameters
    ----------
    i_pair : int
        Index of the pair of meshes to process.
    i_template : int
        Index of the template mesh which defines the vertices
        and faces that all mesh should have.
    paths : list
        List of paths to the meshes.
    gpu_id : int
        ID of the GPU to use.
    """
    print("i_path: ", i_path)

    project_config = pc.import_default_config(project_dir)
    start_path = input_paths[project_config.template_day_index]
    end_path = input_paths[i_path]
    ply_path = os.path.join(output_dir, os.path.basename(end_path))

    if os.path.exists(ply_path):
        print(f"File exists (no rewrite): {ply_path}")
        return

    geod, F0, color0 = scale_decimate_and_compute_geodesic(
        start_path=start_path, end_path=end_path, gpu_id=gpu_id, project_dir=project_dir
    )

    h2_io.plotGeodesic(  # if this does not work, then might have to make a specific color thing in utils.makeGeodMeshes instead of using Rho
        [geod[-1]],
        F0,
        color0,
        stepsize=project_config.stepsize[
            "menstrual_mesh"
        ],  # open3d plotting parameter - unused
        file_name=os.path.splitext(ply_path)[0],  # remove .ply extension
        axis=[0, 1, 0],
        angle=-1 * np.pi / 2,
    )
    print(f"- Write mesh to {ply_path}.")
