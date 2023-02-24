"""Mesh and compute deformations of brain time series."""

import glob
import os
import sys
import time
import warnings

import numpy as np
import torch
import trimesh

import my28brains.io_mesh as io_mesh
import my28brains.my28brains_config as my28brains_config

sys_dir = os.path.dirname(my28brains_config.WORK_DIR)
sys.path.append(sys_dir)
H2_SURFACEMATCH_DIR = os.path.join(my28brains_config.WORK_DIR, "H2_SurfaceMatch")
sys.path.append(H2_SURFACEMATCH_DIR)

import H2_SurfaceMatch.H2_match  # noqa: E402
import H2_SurfaceMatch.utils.input_output  # noqa: E402
import H2_SurfaceMatch.utils.utils  # noqa: E402

warnings.filterwarnings("ignore")

use_cuda = 1
torchdeviceId = torch.device("cuda:0") if use_cuda else "cpu"
torchdtype = torch.float32

print(torchdeviceId)

# CENTERED_MESHES_DIR = os.path.join(os.getcwd(), "data", "centered_meshes")

# # build path to centered_nondegenerate meshes, and create directory if
# # it does not exist
# CENTERED_NONDEGENERATE_MESHES_DIR = os.path.join(
#     os.getcwd(), "data", "centered_nondegenerate_meshes"
# )
# print("CENTERED_NONDEGENERATE_MESHES_DIR: ", CENTERED_NONDEGENERATE_MESHES_DIR)
# if not os.path.exists(CENTERED_NONDEGENERATE_MESHES_DIR):
#     os.makedirs(CENTERED_NONDEGENERATE_MESHES_DIR)


for hemisphere in my28brains_config.hemispheres:
    for structure_id in my28brains_config.structure_ids:

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
            print(f"\tLoad mesh from path: {path}")
            mesh = trimesh.load(path)
            new_vertices, new_faces = io_mesh.remove_degenerate_faces(
                mesh.vertices, mesh.faces
            )
            new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)

            ply_path = os.path.join(
                my28brains_config.CENTERED_NONDEGENERATE_MESHES_DIR,
                os.path.basename(path),
            )
            print(ply_path)
            io_mesh.write_trimesh_to_ply(new_mesh, ply_path)


def _geodesic_interp(i_pair, device_id=-1):
    """Auxiliary function that will be run in parallelon different GPUs.

    note the decimation of faces: decrease the reoltuion of the mesh,
    ie the number of face ans vertice.
    """
    if device_id == -1:
        device_id = 0

    start_time = time.time()
    start_path = start_paths[i_pair]
    end_path = end_paths[i_pair]

    [VS, FS, FunS] = H2_SurfaceMatch.utils.input_output.loadData(start_path)
    VS = VS / 10
    [VS, FS] = H2_SurfaceMatch.utils.input_output.decimate_mesh(
        VS, FS, int(FS.shape[0] / 4)
    )
    sources = [[VS, FS]]

    [VT, FT, FunT] = H2_SurfaceMatch.utils.input_output.loadData(end_path)
    VT = VT / 10
    [VT, FT] = H2_SurfaceMatch.utils.input_output.decimate_mesh(
        VT, FT, int(FT.shape[0] / 4)
    )
    targets = [[VT, FT]]

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

    # We use the start mesh as the basename for the ply files
    ply_prefix = os.path.join(GEODESICS_DIR, os.path.basename(start_path))

    for i_time in range(geod.shape[0]):
        H2_SurfaceMatch.utils.input_output.plotGeodesic(
            [geod[i_time]],
            F0,
            stepsize=5,
            file_name=ply_prefix + "{}".format(i_time),
            axis=[0, 1, 0],
            angle=-1 * np.pi / 2,
        )
    print(f"Geodesic interpolation {i_pair} saved to: {GEODESICS_DIR}.")


for hemisphere in my28brains_config.hemispheres:
    for structure_id in my28brains_config.structure_ids:
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

        for i_pair in range(len(start_paths)):
            print(
                f"\n\n -------> Geodesic interpolation for pair:"
                f"{i_pair}/{len(start_paths)}"
            )
            device_id = structure_id
            _geodesic_interp(i_pair, device_id)
