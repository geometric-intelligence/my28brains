'''This file calculates the "path energy" of the geodesic between two shapes.'''
import glob
import itertools
import multiprocessing
import os
import sys
import time
import warnings

import numpy as np
import torch
import trimesh

import my28brains.default_config as default_config
import my28brains.meshing as meshing

# from multiprocessing import cpu_count

# from joblib import Parallel, delayed

sys_dir = os.path.dirname(default_config.work_dir)
sys.path.append(sys_dir)
sys.path.append(default_config.h2_dir)

import H2_SurfaceMatch.H2_match  # noqa: E402
import H2_SurfaceMatch.enr.H2 # noqa: E402
import H2_SurfaceMatch.utils.input_output  # noqa: E402
import H2_SurfaceMatch.utils.utils  # noqa: E402

warnings.filterwarnings("ignore")

centered_dir = default_config.centered_dir
centered_nondegenerate_dir = default_config.centered_nondegenerate_dir
energies_dir = os.path.join(os.getcwd(), "my28brains", "results", "meshes_energies")

# Choose the pair of geodesics for which you would like to calculate
# (1) a geodesic between
# (2) The energy of this geodesic
hemisphere = "left"
structure_id = -1
area_threshold = '0.0'
days = np.array([1,10])

centered_nondegenerate_dir = os.path.join(
    os.getcwd(), "my28brains", "results", "meshes_centered_nondegenerate"
)
print(f"centered_nondegenerate_dir: {centered_nondegenerate_dir}")

# write paths to each day
paths = []
for day in days:
    string_base = os.path.join(
        centered_nondegenerate_dir,
        f"{hemisphere}_structure_{structure_id}_day{day:02d}_at_{area_threshold}.ply",
    )
    # paths = sorted(glob.glob(string_base))
    paths.append(string_base)

print(
    f"Found {len(paths)} ply files for {hemisphere} hemisphere"
    f" and anatomical structure {structure_id}:"
)
for path in paths:
    print(path)

# template copied from main
def _interpolate_with_geodesic(paths, gpu_id=0):
    """Auxiliary function that will be run in parallel on different GPUs.

    Note the decimation of faces: decrease the resolution of the mesh,
    ie the number of face and vertices.

    Parameters
    ----------
    i_pair : int
        Index of the pair of meshes to process.
    i_template : int
        Index of the template mesh which defines the vertices and faces that all mesh should have.
    paths : list
        List of paths to the meshes.
    gpu_id : int
        ID of the GPU to use.
    """
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    start_path = paths[0]
    end_path = paths[1]

    # We use the start mesh as the basename for the ply files
    ply_prefix = os.path.join(energies_dir, os.path.basename(start_path))

    # check to see if complete geodesic already has been written
    # geodesic_exists = True
    # for i_geodesic_time in range(n_geodesic_time):
    #     file_name = ply_prefix + "{}".format(i_geodesic_time)
    #     file_ply = file_name + ".ply"
    #     geodesic_exists = os.path.exists(os.path.join(energies_dir, file_ply))

    # if geodesic_exists:
    #     print(f"Geodesic for pair {i_pair} already exists. Skipping to next pair.")
    #     return

    # Source preprocessing
    [
        vertices_source,
        faces_source,
        FunS,
    ] = H2_SurfaceMatch.utils.input_output.loadData(start_path)
    vertices_source = vertices_source  # was / 10

    print(f"Verticies Source shape: {vertices_source.shape}")

    # Initial decimation for source
    n_faces_after_decimation = int(
        faces_source.shape[0] / default_config.initial_decimation_fact
    )
    [
        vertices_source,
        faces_source,
    ] = H2_SurfaceMatch.utils.utils.decimate_mesh(  # noqa E231
        vertices_source, faces_source, n_faces_after_decimation
    )

    sources = [[vertices_source, faces_source]]

    # Target preprocessing
    [
        vertices_target,
        faces_target,
        FunT,
    ] = H2_SurfaceMatch.utils.input_output.loadData(end_path)
    vertices_target = vertices_target  # was / 10

    print(f"Verticies Target shape: {vertices_target.shape}")

    # Initial decimation for target
    n_faces_after_decimation = int(
        faces_target.shape[0] / default_config.initial_decimation_fact
    )
    [
        vertices_target,
        faces_target,
    ] = H2_SurfaceMatch.utils.utils.decimate_mesh(  # noqa E231
        vertices_target, faces_target, n_faces_after_decimation
    )
    targets = [[vertices_target, faces_target]]

    # Template preprocessing
    # [
    #     vertices_template,
    #     faces_template,
    #     FunTemplate,
    # ] = H2_SurfaceMatch.utils.input_output.loadData(template_path)
    # vertices_template = vertices_template  # was / 10

    # # Initial decimation for template
    # n_faces_after_decimation = int(
    #     faces_template.shape[0] / default_config.initial_decimation_fact
    # )
    # [
    #     vertices_template,
    #     faces_template,
    # ] = H2_SurfaceMatch.utils.utils.decimate_mesh(  # noqa E231
    #     vertices_template, faces_template, n_faces_after_decimation
    # )
    # template = [[vertices_template, faces_template]]sss

    source = sources[0]
    target = targets[0]
    # template = template[0]

    # decimation also happens at the start of h2_match.H2multires
    geod, F0 = H2_SurfaceMatch.H2_match.H2MultiRes(
        source=source,
        target=target,
        a0=default_config.a0,
        a1=default_config.a1,
        b1=default_config.b1,
        c1=default_config.c1,
        d1=default_config.d1,
        a2=default_config.a2,
        resolutions=default_config.resolutions,
        start=None,
        paramlist=default_config.paramlist,
        device=device,
    )
    comp_time = time.time() - start_time
    print(f"Geodesic interpolation took: {comp_time / 60:.2f} minutes.")

    for i_geodesic_time in range(geod.shape[0]):
        file_name = ply_prefix + "{}".format(i_geodesic_time)
        H2_SurfaceMatch.utils.input_output.plotGeodesic(
            [geod[i_geodesic_time]],
            F0,
            stepsize=default_config.stepsize,  # open3d plotting parameter - unused
            file_name=file_name,
            axis=[0, 1, 0],
            angle=-1 * np.pi / 2,
        )
        print(f"Geodesic interpolation saved to: " f"{energies_dir}.")
    
    # path_energy = H2.getPathEnergyH2(geod, a0=default_config.a0,
    #     a1=default_config.a1,
    #     b1=default_config.b1,
    #     c1=default_config.c1,
    #     d1=default_config.d1,
    #     a2=default_config.a2, F_sol, stepwise=False, device=None)


_interpolate_with_geodesic(paths, gpu_id=0)