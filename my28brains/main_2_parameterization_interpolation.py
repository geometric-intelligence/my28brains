"""Mesh and compute deformations of brain time series.

Note that the preprocessing should be done before running this script.

Run:
>>> python main_preprocessing.py && python main.py

Note on pykeops:
If main cannot run because of:
EOFError: Ran out of input
This is probably because of a pykeops issue.

To fix this, run the following commands:
- remove: rm -rf ~/.cache/keops*/build
- rebuild from python console.
    >>> import pykeops
"""

import os

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import glob
import itertools
import multiprocessing
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
import H2_SurfaceMatch.utils.input_output  # noqa: E402
import H2_SurfaceMatch.utils.utils  # noqa: E402

warnings.filterwarnings("ignore")

centered_dir = default_config.centered_dir
centered_nondegenerate_dir = default_config.centered_nondegenerate_dir
geodesics_dir = default_config.geodesics_dir
parameterized_meshes_dir = default_config.parameterized_meshes_dir


def interpolate_with_geodesic(args_with_queue):
    """Interpolate geodesics for a given pair of meshes.

    Parameters
    ----------
    args_with_queue : tuple
        Tuple containing the following:
        - i_pair : int
            Index of the pair of meshes to interpolate.
        - paths : list
            List of paths to the meshes to interpolate.
        - n_geodesic_time : int
            Number of time points along each interpolating geodesic.
        - queue : multiprocessing.Queue
            Queue containing the GPU IDs to use.
    """
    i_pair, i_template, paths, n_geodesic_time, queue = args_with_queue
    gpu_id = queue.get()
    try:
        ident = multiprocessing.current_process().ident
        print("{}: starting process on GPU {}".format(ident, gpu_id))
        _interpolate_with_geodesic(i_pair, i_template, paths, n_geodesic_time, gpu_id)
        # if default_config.interpolate_geodesics:
        #     _interpolate_with_geodesic(
        #         i_pair, i_template, paths, n_geodesic_time, gpu_id
        #     )
        # elif default_config.generate_parameterized_data:
        #     _generate_parameterized_data(
        #         i_pair, i_template, paths, n_geodesic_time, gpu_id
        #     )

        print("{}: finished".format(ident))
    finally:
        queue.put(gpu_id)


def write_centered_nondegenerate_meshes(hemisphere, structure_id, area_threshold):
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
        + f" hemisphere and anatomical structure {structure_id}:"
    )
    for path in paths:
        ply_path = os.path.join(
            centered_nondegenerate_dir,
            os.path.basename(path).split(".")[0] + f"_at_{area_threshold}.ply",
        )
        if not os.path.exists(ply_path):
            print(f"\tLoad mesh from: {path}")
            mesh = trimesh.load(path)
            new_vertices, new_faces = meshing.remove_degenerate_faces(
                mesh.vertices, mesh.faces, area_threshold
            )
            new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)

            print(f"Write to: {ply_path}")
            meshing.write_trimesh_to_ply(new_mesh, ply_path)
        else:
            print(f"File already exists (no rewrite): {ply_path}")


# def _interpolate_with_geodesic(i_pair, i_template, paths, n_geodesic_time, gpu_id):
#     """Auxiliary function that will be run in parallel on different GPUs.

#     Note the decimation of faces: decrease the resolution of the mesh,
#     ie the number of face and vertices.

#     Parameters
#     ----------
#     i_pair : int
#         Index of the pair of meshes to process.
#     i_template : int
#         Index of the template mesh which defines the vertices and faces that all mesh should have.
#     paths : list
#         List of paths to the meshes.
#     gpu_id : int
#         ID of the GPU to use.
#     """
#     device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
#     start_time = time.time()
#     start_path = paths[i_pair]
#     end_path = paths[i_pair + 1]
#     template_path = paths[i_template]

#     # CHANGE ALERT: using os.path.splitext to extract basename without extension
#     # We use the start mesh as the basename for the ply files
#     ply_prefix = os.path.join(
#         geodesics_dir, os.path.splitext(os.path.basename(start_path))[0]
#     )

#     # ply_prefix =  os.path.join(geodesics_dir, os.path.basename(start_path))

#     # check to see if complete geodesic already has been written
#     geodesic_exists = True
#     for i_geodesic_time in range(n_geodesic_time):
#         file_name = ply_prefix + "{}".format(i_geodesic_time)
#         file_ply = file_name + ".ply"
#         geodesic_exists = os.path.exists(os.path.join(geodesics_dir, file_ply))

#     if geodesic_exists:
#         print(f"Geodesic for pair {i_pair} already exists. Skipping to next pair.")
#         return

#     # Source preprocessing
#     [
#         vertices_source,
#         faces_source,
#         FunS,
#     ] = H2_SurfaceMatch.utils.input_output.loadData(start_path)
#     vertices_source = vertices_source  # was / 10

#     # Initial decimation for source
#     n_faces_after_decimation = int(
#         faces_source.shape[0] / default_config.initial_decimation_fact
#     )
#     [
#         vertices_source,
#         faces_source,
#     ] = H2_SurfaceMatch.utils.utils.decimate_mesh(  # noqa E231
#         vertices_source, faces_source, n_faces_after_decimation
#     )

#     sources = [[vertices_source, faces_source]]

#     # Target preprocessing
#     [
#         vertices_target,
#         faces_target,
#         FunT,
#     ] = H2_SurfaceMatch.utils.input_output.loadData(end_path)
#     vertices_target = vertices_target  # was / 10

#     # Initial decimation for target
#     n_faces_after_decimation = int(
#         faces_target.shape[0] / default_config.initial_decimation_fact
#     )
#     [
#         vertices_target,
#         faces_target,
#     ] = H2_SurfaceMatch.utils.utils.decimate_mesh(  # noqa E231
#         vertices_target, faces_target, n_faces_after_decimation
#     )
#     targets = [[vertices_target, faces_target]]

#     # Template preprocessing
#     [
#         vertices_template,
#         faces_template,
#         FunTemplate,
#     ] = H2_SurfaceMatch.utils.input_output.loadData(template_path)
#     vertices_template = vertices_template  # was / 10

#     # Initial decimation for template
#     n_faces_after_decimation = int(
#         faces_template.shape[0] / default_config.initial_decimation_fact
#     )
#     [
#         vertices_template,
#         faces_template,
#     ] = H2_SurfaceMatch.utils.utils.decimate_mesh(  # noqa E231
#         vertices_template, faces_template, n_faces_after_decimation
#     )
#     template = [[vertices_template, faces_template]]

#     source = sources[0]
#     target = targets[0]
#     template = template[0]

#     # decimation also happens at the start of h2_match.H2multires
#     geod, F0 = H2_SurfaceMatch.H2_match.H2MultiRes(
#         source=source,
#         target=target,
#         a0=default_config.a0,
#         a1=default_config.a1,
#         b1=default_config.b1,
#         c1=default_config.c1,
#         d1=default_config.d1,
#         a2=default_config.a2,
#         resolutions=default_config.resolutions,
#         start=None,
#         paramlist=default_config.paramlist,
#         device=device,
#     )
#     comp_time = time.time() - start_time
#     print(f"Geodesic interpolation {i_pair} took: {comp_time / 60:.2f} minutes.")

#     for i_geodesic_time in range(geod.shape[0]):
#         file_name = ply_prefix + "{}".format(i_geodesic_time)
#         H2_SurfaceMatch.utils.input_output.plotGeodesic(
#             [geod[i_geodesic_time]],
#             F0,
#             stepsize=default_config.stepsize,  # open3d plotting parameter - unused
#             file_name=file_name,
#             axis=[0, 1, 0],
#             angle=-1 * np.pi / 2,
#         )
#         print(f"Geodesic interpolation {i_pair} saved to: " f"{geodesics_dir}.")


def _interpolate_with_geodesic(i_pair, i_template, paths, n_geodesic_time, gpu_id):
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
    end_path = paths[i_pair + 1]
    template_path = paths[i_template]

    geodesic_exists = True

    if default_config.interpolate_geodesics:
        start_path = paths[i_pair]
        ply_prefix = os.path.join(
            geodesics_dir, os.path.splitext(os.path.basename(start_path))[0]
        )

        for i_geodesic_time in range(n_geodesic_time):
            file_name = ply_prefix + "{}".format(i_geodesic_time)
            file_ply = file_name + ".ply"
            geodesic_exists = os.path.exists(os.path.join(geodesics_dir, file_ply))

    elif default_config.generate_parameterized_data:
        start_path = paths[0]

        ply_prefix = os.path.join(
            parameterized_meshes_dir, os.path.splitext(os.path.basename(end_path))[0]
        )
        file_name = ply_prefix + "_parameterized"
        file_ply = file_name + ".ply"
        geodesic_exists = os.path.exists(
            os.path.join(parameterized_meshes_dir, file_ply)
        )

    if geodesic_exists:
        print(f"Geodesic for pair {i_pair} already exists. Skipping to next pair.")
        return

    # Source preprocessing
    [
        vertices_source,
        faces_source,
        FunS,
    ] = H2_SurfaceMatch.utils.input_output.loadData(start_path)
    vertices_source = vertices_source / default_config.scaling_factor  # was / 10

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
    vertices_target = vertices_target / default_config.scaling_factor  # was / 10

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
    [
        vertices_template,
        faces_template,
        FunTemplate,
    ] = H2_SurfaceMatch.utils.input_output.loadData(template_path)
    vertices_template = vertices_template / default_config.scaling_factor  # was / 10

    # Initial decimation for template
    n_faces_after_decimation = int(
        faces_template.shape[0] / default_config.initial_decimation_fact
    )
    [
        vertices_template,
        faces_template,
    ] = H2_SurfaceMatch.utils.utils.decimate_mesh(  # noqa E231
        vertices_template, faces_template, n_faces_after_decimation
    )
    template = [[vertices_template, faces_template]]

    source = sources[0]
    target = targets[0]
    template = template[0]

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
    print(f"Geodesic interpolation {i_pair} took: {comp_time / 60:.2f} minutes.")

    if default_config.interpolate_geodesics:
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
            print(f"Geodesic interpolation {i_pair} saved to: " f"{geodesics_dir}.")
    elif default_config.generate_parameterized_data:
        # save the last point in the geodesic
        H2_SurfaceMatch.utils.input_output.plotGeodesic(
            [geod[-1]],
            F0,
            stepsize=default_config.stepsize,  # open3d plotting parameter - unused
            file_name=file_name,
            axis=[0, 1, 0],
            angle=-1 * np.pi / 2,
        )
        print(
            f"Geodesic interpolation {i_pair} saved to: " f"{parameterized_meshes_dir}."
        )


def _generate_parameterized_data(i_pair, i_template, paths, n_geodesic_time, gpu_id):
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
    # make the start path the first mesh in the list
    start_path = paths[0]
    # make the end path the mesh that needs to be re-parameterized
    end_path = paths[i_pair + 1]
    print(f"end_path: {end_path}")
    # template_path = paths[i_template]

    # We use the end mesh as the basename for the ply files becuase that is the
    # mesh that we are re-parameterizing
    print("basename: ", os.path.basename(end_path))
    print("split: ", os.path.splitext(os.path.basename(end_path)))
    ply_prefix = os.path.join(
        parameterized_meshes_dir, os.path.splitext(os.path.basename(end_path))[0]
    )
    # ply_prefix = os.path.split(os.path.join(parameterized_meshes_dir, os.path.basename(end_path)))[0]
    print(f"ply_prefix: {ply_prefix}")
    file_name = ply_prefix + "_parameterized"
    print(f"file_name: {file_name}")
    file_ply = file_name + ".ply"

    # check to see if complete geodesic already has been written
    geodesic_exists = True
    geodesic_exists = os.path.exists(os.path.join(parameterized_meshes_dir, file_ply))
    if geodesic_exists:
        print(f"Geodesic for pair {i_pair} already exists. Skipping to next pair.")
        return

    # for i_geodesic_time in range(n_geodesic_time):
    #     file_name = ply_prefix + "{}".format(i_geodesic_time)
    #     file_ply = file_name + ".ply"
    #     geodesic_exists = os.path.exists(os.path.join(parameterized_meshes_dir, file_ply))

    # Source preprocessing
    [
        vertices_source,
        faces_source,
        FunS,
    ] = H2_SurfaceMatch.utils.input_output.loadData(start_path)
    vertices_source = vertices_source  # was / 10

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

    # Initial decimation for template
    # n_faces_after_decimation = int(
    #     faces_template.shape[0] / default_config.initial_decimation_fact
    # )
    # [
    #     vertices_template,
    #     faces_template,
    # ] = H2_SurfaceMatch.utils.utils.decimate_mesh(  # noqa E231
    #     vertices_template, faces_template, n_faces_after_decimation
    # )
    # template = [[vertices_template, faces_template]]

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
    print(f"Geodesic interpolation {i_pair} took: {comp_time / 60:.2f} minutes.")

    # save the last point in the geodesic
    H2_SurfaceMatch.utils.input_output.plotGeodesic(
        [geod[-1]],
        F0,
        stepsize=default_config.stepsize,  # open3d plotting parameter - unused
        file_name=file_name,
        axis=[0, 1, 0],
        angle=-1 * np.pi / 2,
    )
    print(f"Geodesic interpolation {i_pair} saved to: " f"{parameterized_meshes_dir}.")

    # for i_geodesic_time in range(geod.shape[0]):
    #     file_name = ply_prefix + "{}".format(i_geodesic_time)
    #     H2_SurfaceMatch.utils.input_output.plotGeodesic(
    #         [geod[i_geodesic_time]],
    #         F0,
    #         stepsize=default_config.stepsize,  # open3d plotting parameter - unused
    #         file_name=file_name,
    #         axis=[0, 1, 0],
    #         angle=-1 * np.pi / 2,
    #     )
    #     print(
    #         f"Geodesic interpolation {i_pair} saved to: " f"{parameterized_meshes_dir}."
    #     )


if __name__ == "__main__":
    """Parse the default_config file and launch all experiments.

    This launches experiments with different:
    - hippocampal substructure,
    - hemisphere,
    - number of time points along the geodesic,
    - area threshold for removing degenerate faces.
    """
    for hemisphere, structure_id, n_geodesic_time, area_threshold in itertools.product(
        default_config.hemispheres,
        default_config.structure_ids,
        default_config.n_geodesic_times,
        default_config.area_thresholds,
    ):
        write_centered_nondegenerate_meshes(hemisphere, structure_id, area_threshold)

        string_base = os.path.join(
            centered_nondegenerate_dir,
            f"{hemisphere}_structure_{structure_id}**_at_{area_threshold}.ply",
        )
        paths = sorted(glob.glob(string_base))

        print(
            f"Found {len(paths)} ply files for {hemisphere} hemisphere"
            f" and anatomical structure {structure_id}:"
        )
        for path in paths:
            print(path)

        multiprocessing.set_start_method("spawn", force=True)
        manager = multiprocessing.Manager()
        queue = manager.Queue()
        for gpu_id in range(default_config.n_gpus):
            queue.put(gpu_id)

        pool = multiprocessing.Pool(processes=default_config.n_gpus)
        i_pairs = list(range(default_config.day_range[0], default_config.day_range[1]))

        if default_config.interpolate_geodesics:
            print(f"LAUNCHING GEODESIC INTERPOLATION.")
        elif default_config.generate_parameterized_data:
            print(f"LAUNCHING PARAMETERIZED DATA GENERATION.")
        else:
            raise ValueError(
                "No action specified. Please check the config file and make sure that either interpolate_geodesics or generate_parameterized_data is set to True."
            )

        args_with_queue = [
            (i_pair, default_config.i_template, paths, n_geodesic_time, queue)
            for i_pair in i_pairs
        ]

        for _ in pool.imap_unordered(interpolate_with_geodesic, args_with_queue):
            pass
        pool.close()
        pool.join()
