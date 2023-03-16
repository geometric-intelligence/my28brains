"""Mesh and compute deformations of brain time series."""

import glob
import os
import sys
import time
import warnings

import numpy as np

# import torch
import trimesh

import my28brains.io_mesh as io_mesh
import my28brains.my28brains_config as my28brains_config

# from multiprocessing import cpu_count

# from joblib import Parallel, delayed

sys_dir = os.path.dirname(my28brains_config.WORK_DIR)
sys.path.append(sys_dir)
H2_SURFACEMATCH_DIR = os.path.join(my28brains_config.WORK_DIR, "H2_SurfaceMatch")
sys.path.append(H2_SURFACEMATCH_DIR)

import H2_SurfaceMatch.H2_match  # noqa: E402
import H2_SurfaceMatch.utils.input_output  # noqa: E402
import H2_SurfaceMatch.utils.utils  # noqa: E402

warnings.filterwarnings("ignore")

# use_cuda = 1
# torchdeviceId = torch.device("cuda:0") if use_cuda else "cpu"
# torchdtype = torch.float32

# print(torchdeviceId)

# CENTERED_MESHES_DIR = os.path.join(os.getcwd(), "data", "centered_meshes")

# # build path to centered_nondegenerate meshes, and create directory if
# # it does not exist
# CENTERED_NONDEGENERATE_MESHES_DIR = os.path.join(
#     os.getcwd(), "data", "centered_nondegenerate_meshes"
# )
# print("CENTERED_NONDEGENERATE_MESHES_DIR: ", CENTERED_NONDEGENERATE_MESHES_DIR)
# if not os.path.exists(CENTERED_NONDEGENERATE_MESHES_DIR):
#     os.makedirs(CENTERED_NONDEGENERATE_MESHES_DIR)


def write_centered_nondegenerate_meshes(hemisphere, structure_id):
    """Write centered nondegenerate meshes for a given structure/hemisphere."""
    string_base = os.path.join(
        my28brains_config.CENTERED_MESHES_DIR,
        f"{hemisphere}_structure_{structure_id}**.ply",
    )
    paths = sorted(glob.glob(string_base))

    print(
        f"Found {len(paths)} ply files for {hemisphere}"
        + f"hemisphere and anatomical structure {structure_id}:"
    )
    for path in paths:
        print(path)

    for path in paths:
        ply_path = os.path.join(
            my28brains_config.CENTERED_NONDEGENERATE_MESHES_DIR,
            os.path.basename(path),
        )

        if not os.path.exists(ply_path):
            print(f"\tLoad mesh from path: {path}")
            mesh = trimesh.load(path)
            new_vertices, new_faces = io_mesh.remove_degenerate_faces(
                mesh.vertices, mesh.faces
            )
            new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)

            print(ply_path)
            io_mesh.write_trimesh_to_ply(new_mesh, ply_path)

        else:
            print(f"File already exists. Not re-writing for : {ply_path}")


for hemisphere in my28brains_config.hemispheres:
    for structure_id in my28brains_config.structure_ids:
        write_centered_nondegenerate_meshes(hemisphere, structure_id)


def _geodesic_interp(i_pair, structure_id, start_paths, end_paths, device_id=-1):
    """Auxiliary function that will be run in parallelon different GPUs.

    note the decimation of faces: decrease the reoltuion of the mesh,
    ie the number of face ans vertice.
    """
    print(
        f"\n\n -------> Geodesic interpolation for [{hemisphere} hemisphere], "
        f"[structure {structure_id}] , pair:"
        f"{i_pair}/{len(start_paths)}"
    )

    start_time = time.time()
    start_path = start_paths[i_pair]
    end_path = end_paths[i_pair]

    # We use the start mesh as the basename for the ply files
    ply_prefix = os.path.join(
        my28brains_config.GEODESICS_DIR, os.path.basename(start_path)
    )

    # check to see if complete geodesic already has been written
    geodesic_exists = True
    for i_time in range(5):
        file_name = ply_prefix + "{}".format(i_time)
        file_ply = file_name + ".ply"
        geodesic_exists = os.path.exists(
            os.path.join(my28brains_config.GEODESICS_DIR, file_ply)
        )

    if geodesic_exists:
        print(f"Geodesic for pair {i_pair} already exists. Skipping to next pair.")
    else:

        [
            vertices_source,
            faces_source,
            FunS,
        ] = H2_SurfaceMatch.utils.input_output.loadData(start_path)
        vertices_source = vertices_source / 1  # was 10
        [
            vertices_source,
            faces_source,
        ] = H2_SurfaceMatch.utils.utils.decimate_mesh(  # noqa E231
            vertices_source, faces_source, int(faces_source.shape[0] / 4)
        )
        sources = [[vertices_source, faces_source]]

        [
            vertices_target,
            faces_target,
            FunT,
        ] = H2_SurfaceMatch.utils.input_output.loadData(end_path)
        vertices_target = vertices_target / 1  # was 10
        [
            vertices_target,
            faces_target,
        ] = H2_SurfaceMatch.utils.utils.decimate_mesh(  # noqa E231
            vertices_target, faces_target, int(faces_target.shape[0] / 4)
        )
        targets = [[vertices_target, faces_target]]

        source = sources[0]
        target = targets[0]

        # decimation also happens at the beginning of the code within
        # h2_match.H2multires
        geod, F0 = H2_SurfaceMatch.H2_match.H2MultiRes(
            source,
            target,
            my28brains_config.a0,
            my28brains_config.a1,
            my28brains_config.b1,
            my28brains_config.c1,
            my28brains_config.d1,
            my28brains_config.a2,
            resolutions=2,
            paramlist=my28brains_config.paramlist,
            device_id=device_id,
        )
        comp_time = time.time() - start_time
        print(f"Geodesic interpolation {i_pair} took: {comp_time / 60:.2f} minutes.")

        # # We use the start mesh as the basename for the ply files
        # ply_prefix = os.path.join(my28brains_config.GEODESICS_DIR,
        # os.path.basename(start_path))

        for i_time in range(geod.shape[0]):
            file_name = ply_prefix + "{}".format(i_time)
            H2_SurfaceMatch.utils.input_output.plotGeodesic(
                [geod[i_time]],
                F0,
                stepsize=5,
                file_name=file_name,
                axis=[0, 1, 0],
                angle=-1 * np.pi / 2,
            )
            print(
                f"Geodesic interpolation {i_pair} saved to: "
                f"{my28brains_config.GEODESICS_DIR}."
            )


def write_substructure_geodesic(hemisphere, structure_id, device_id):
    """Write a geodesic for one structure."""
    # use_cuda = device_id
    # torchdeviceId = torch.device(f"cuda:{use_cuda}") if use_cuda else "cpu"
    # torchdtype = torch.float32
    string_base = os.path.join(
        my28brains_config.CENTERED_NONDEGENERATE_MESHES_DIR,
        f"{hemisphere}_structure_{structure_id}**.ply",
    )
    paths = sorted(glob.glob(string_base))

    print(
        f"Found {len(paths)} ply files for {hemisphere} hemisphere"
        + f"and anatomical structure {structure_id}:"
    )
    for path in paths:
        print(path)

    start_paths = paths[:-1]
    end_paths = paths[1:]

    GEODESICS_DIR = os.path.join(os.getcwd(), "data", "geodesics")
    print("GEODESICS_DIR: ", GEODESICS_DIR)
    if not os.path.exists(GEODESICS_DIR):
        os.makedirs(GEODESICS_DIR)

    # Parallel(n_jobs=int(cpu_count()))(
    # delayed(_geodesic_interp)(i_pair, structure_id, start_paths, end_paths, device_id)
    #     for i_pair in range(len(start_paths))
    # )

    for i_pair in range(len(start_paths)):
        print(
            f"\n\n -------> Geodesic interpolation for [{hemisphere} hemisphere], "
            f"[structure {structure_id}] , pair:"
            f"{i_pair}/{len(start_paths)}"
        )
        _geodesic_interp(i_pair, structure_id, start_paths, end_paths, device_id)


# we have 10 GPUs, naming starts at 0.
for hemisphere in my28brains_config.hemispheres:
    for structure_id in my28brains_config.structure_ids:
        if structure_id == -1:
            device_id = 0
        else:
            device_id = structure_id

        # NOTE: when ready to fully parallelize, change from device_id =0
        write_substructure_geodesic(hemisphere, structure_id, device_id=0)


# for hemisphere in my28brains_config.hemispheres:
#     for structure_id in my28brains_config.structure_ids:
#         write_substructure_geodesic(hemisphere, structure_id)
