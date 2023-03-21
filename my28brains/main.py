"""Mesh and compute deformations of brain time series."""

import glob
import os
import sys
import time
import warnings

import numpy as np
import torch
import trimesh

import my28brains.default_config as default_config
import my28brains.io_mesh as io_mesh

# from multiprocessing import cpu_count

# from joblib import Parallel, delayed

sys_dir = os.path.dirname(default_config.work_dir)
sys.path.append(sys_dir)
sys.path.append(default_config.h2_dir)

import H2_SurfaceMatch.H2_match  # noqa: E402
import H2_SurfaceMatch.utils.input_output  # noqa: E402
import H2_SurfaceMatch.utils.utils  # noqa: E402

warnings.filterwarnings("ignore")

geodesics_dir = default_config.geodesics_dir
centered_nondegenerate_dir = default_config.centered_nondegenerate_dir
print("geodesics_dir: ", geodesics_dir)
print("centered_nondegenerate_dir: ", centered_nondegenerate_dir)


def write_centered_nondegenerate_meshes(hemisphere, structure_id):
    """Write centered nondegenerate meshes for a given structure/hemisphere.

    This function loads the centered meshes for a given structure/hemisphere,
    removes degenerate faces, and saves the result.

    Parameters
    ----------
    hemisphere : str
        Hemisphere to process. Either 'left' or 'right'.
    structure_id : int
        Structure ID to process.
    """
    string_base = os.path.join(
        default_config.centered_dir,
        f"{hemisphere}_structure_{structure_id}**.ply",
    )
    paths = sorted(glob.glob(string_base))

    print(
        f"Found {len(paths)} ply files for {hemisphere}"
        + f"hemisphere and anatomical structure {structure_id}:"
    )
    for path in paths:
        ply_path = os.path.join(
            centered_nondegenerate_dir,
            os.path.basename(path),
        )
        if not os.path.exists(ply_path):
            print(f"\tLoad mesh from: {path}")
            mesh = trimesh.load(path)
            new_vertices, new_faces = io_mesh.remove_degenerate_faces(
                mesh.vertices, mesh.faces
            )
            new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)

            print(f"Write to: {ply_path}")
            io_mesh.write_trimesh_to_ply(new_mesh, ply_path)
        else:
            print(f"File already exists (no rewrite): {ply_path}")


def _geodesic_interp(i_pair, structure_id, start_paths, end_paths, device_id=0):
    """Auxiliary function that will be run in parallel on different GPUs.

    Note the decimation of faces: decrease the resolution of the mesh,
    ie the number of face and vertices.
    """
    print(
        f"\n\n -------> Geodesic interpolation for [{hemisphere} hemisphere], "
        f"[structure {structure_id}] , pair:"
        f"{i_pair}/{len(start_paths)}"
    )
    device = torch.device(f"cuda:{device_id}" if device_id is not None else "cpu")
    start_time = time.time()
    start_path = start_paths[i_pair]
    end_path = end_paths[i_pair]

    # We use the start mesh as the basename for the ply files
    ply_prefix = os.path.join(geodesics_dir, os.path.basename(start_path))

    # check to see if complete geodesic already has been written
    geodesic_exists = True
    for i_time in range(default_config.n_times):
        file_name = ply_prefix + "{}".format(i_time)
        file_ply = file_name + ".ply"
        geodesic_exists = os.path.exists(os.path.join(geodesics_dir, file_ply))

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
            source=source,
            target=target,
            a0=default_config.a0,
            a1=default_config.a1,
            b1=default_config.b1,
            c1=default_config.c1,
            d1=default_config.d1,
            a2=default_config.a2,
            resolutions=2,
            paramlist=default_config.paramlist,
            device=device,
        )
        comp_time = time.time() - start_time
        print(f"Geodesic interpolation {i_pair} took: {comp_time / 60:.2f} minutes.")

        # # We use the start mesh as the basename for the ply files
        # ply_prefix = os.path.join(geodesics_dir,
        # os.path.basename(start_path))

        for i_time in range(geod.shape[0]):
            file_name = ply_prefix + "{}".format(i_time)
            H2_SurfaceMatch.utils.input_output.plotGeodesic(
                [geod[i_time]],
                F0,
                stepsize=2,  # spacing within the geodesic on the plot (?)
                file_name=file_name,
                axis=[0, 1, 0],
                angle=-1 * np.pi / 2,
            )
            print(f"Geodesic interpolation {i_pair} saved to: " f"{geodesics_dir}.")


def write_substructure_geodesic(hemisphere, structure_id):
    """Write a geodesic for one structure.

    This function will write a geodesic for one structure, for all pairs of
    consecutives meshes in the centered_nondegenerate_dir
    corresponding to that structure.

    Parameters
    ----------
    hemisphere : str
        Hemisphere of the structure.
    structure_id : int
        ID of the structure.
    """
    string_base = os.path.join(
        centered_nondegenerate_dir,
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

    # Parallel(n_jobs=int(cpu_count()))(
    # delayed(_geodesic_interp)(i_pair, structure_id, start_paths, end_paths, device_id)
    #     for i_pair in range(len(start_paths))
    # )

    for i_pair in range(len(start_paths)):
        device_id = (i_pair + 3) % 10
        print(
            f"\n\n -------> Geodesic interpolation for [{hemisphere} hemisphere], "
            f"[structure {structure_id}] , pair:"
            f"{i_pair}/{len(start_paths)}"
        )
        _geodesic_interp(i_pair, structure_id, start_paths, end_paths, device_id)


for hemisphere in default_config.hemispheres:
    for structure_id in default_config.structure_ids:
        write_centered_nondegenerate_meshes(hemisphere, structure_id)

# we have 10 GPUs, naming starts at 0.
for hemisphere in default_config.hemispheres:
    for structure_id in default_config.structure_ids:
        # if structure_id == -1:
        #     device_id = 0
        # else:
        #     device_id = structure_id

        # NOTE: when ready to fully parallelize, change from device_id =0
        write_substructure_geodesic(hemisphere, structure_id)


# for hemisphere in default_config.hemispheres:
#     for structure_id in default_config.structure_ids:
#         write_substructure_geodesic(hemisphere, structure_id)
