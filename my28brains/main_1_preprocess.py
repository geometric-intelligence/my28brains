"""Preprocessing of the 28brains dataset.

1. Mesh: Segment the surfaces of the hippocampus and its substructures.
2. Center: Center the hippocampus and its substructures.
3. Remove degenerate faces
4. Compute geodesic, either for interpolation or for reparameterization.
5. Sort the meshes by hormone levels.


Interpolation: take e a mesh at time t and a mesh at time t+1, and interpolate
--> outputs in geodesics_dir
OR
Reparameterizations of the meshes.
--> outputs in parameterized_dir

Note on pykeops:
----------------
If main cannot run because of:
EOFError: Ran out of input
This is probably because of a pykeops issue.

To fix this, run the following commands:
- remove: rm -rf ~/.cache/keops*/build
- rebuild from python console.
    >>> import pykeops

The processing is done with our `my28brains.meshing` module.

Meshed surfaces are stored into .ply files.

The meshes in the .ply files can be opened with:
- the `vscode-3d-preview` extension of VSCode,
- [MeshLab](https://www.meshlab.net/),
- [Blender](https://www.blender.org/download/) with plugin [Stop-motion-OBJ].
"""
import glob
import itertools
import multiprocessing
import os
import warnings

import my28brains.default_config as default_config
import my28brains.preprocessing.centering as centering
import my28brains.preprocessing.extraction as extraction
import my28brains.preprocessing.geodesics as geodesics
import my28brains.preprocessing.sorting as sorting

warnings.filterwarnings("ignore")
raw_dir = default_config.raw_dir
meshes_dir = default_config.meshes_data_dir
centered_dir = default_config.centered_dir
centered_nondegenerate_dir = default_config.centered_nondegenerate_dir
geodesics_dir = default_config.geodesics_dir
parameterized_dir = default_config.parameterized_dir
sorted_parameterized_dir = default_config.sorted_parameterized_dir

days_dir = [os.path.join(raw_dir, f"Day{i:02d}") for i in range(1, 61)]


def run_func_in_parallel_with_queue(func_args_queue):
    """Fun a function in parallel using a multiprocessing.Queue.

    Parameters
    ----------
    func_args_queue : tuple
        Tuple containing, in this order:
        - func: callable
            Function to run in parallel.
        - args: the args of func
            Tuple of args.
        - queue : multiprocessing.Queue
            Queue containing the GPU IDs to use.
    """
    args = func_args_queue[1:-1]
    func, queue = func_args_queue[0], func_args_queue[-1]
    gpu_id = queue.get()
    try:
        ident = multiprocessing.current_process().ident
        print(f"{ident}: starting process on GPU {gpu_id}.")
        func(*args, gpu_id)
        print(f"{ident}: finished")
    finally:
        queue.put(gpu_id)


if __name__ == "__main__":
    # 1. Extract meshes from the .nii files and write them to .ply files.
    for hemisphere, structure_id in itertools.product(
        default_config.hemisphere, set(default_config.structure_ids + [-1])
    ):
        # Add the whole hippocampus id=[-1] to list of structure ids,
        # in order to be able to compute its center and center the substructures.
        extraction.extract_meshes_from_nii_and_write(
            input_dir=days_dir,
            output_dir=meshes_dir,
            hemisphere=hemisphere,
            structure_id=structure_id,
        )

    # 2. Center the meshes and write them to .ply files.
    for hemisphere in default_config.hemisphere:
        hippocampus_centers = centering.center_whole_hippocampus_and_write(
            input_dir=meshes_dir, output_dir=centered_dir, hemisphere=hemisphere
        )
        for structure_id in default_config.structure_ids:
            if structure_id != -1:
                centering.center_substructure_and_write(
                    input_dir=meshes_dir,
                    output_dir=centered_dir,
                    hemisphere=hemisphere,
                    structure_id=structure_id,
                    hippocampus_centers=hippocampus_centers,
                )

    # 3. Remove degenerate faces
    for hemisphere, structure_id, area_threshold in itertools.product(
        default_config.hemisphere,
        default_config.structure_ids,
        default_config.area_thresholds,
    ):
        geodesics.remove_degenerate_faces_and_write(
            input_dir=centered_dir,
            output_dir=centered_nondegenerate_dir,
            hemisphere=hemisphere,
            structure_id=structure_id,
            area_threshold=area_threshold,
        )

    # 4. Compute geodesic to perform reparameterization to first mesh
    for hemisphere, structure_id, area_threshold in itertools.product(
        default_config.hemisphere,
        default_config.structure_ids,
        default_config.area_thresholds,
    ):
        string_base = os.path.join(
            centered_nondegenerate_dir,
            f"{hemisphere}_structure_{structure_id}**_at_{area_threshold}.ply",
        )
        input_paths = sorted(glob.glob(string_base))
        print(
            f"Found {len(input_paths)} .plys for {hemisphere} hemisphere"
            f" and anatomical structure {structure_id}:"
        )
        for path in input_paths:
            print(path)
        output_dir = parameterized_dir

        multiprocessing.set_start_method("spawn", force=True)
        queue = multiprocessing.Manager().Queue()
        for gpu_id in range(default_config.n_gpus):
            queue.put(gpu_id)

        pool = multiprocessing.Pool(processes=default_config.n_gpus)
        i_paths = list(range(default_config.day_range[0], default_config.day_range[1]))

        func = geodesics.generate_parameterized_data
        func_args_queue = [
            (func, input_paths, output_dir, i_path, queue) for i_path in i_paths
        ]

        for _ in pool.imap_unordered(run_func_in_parallel_with_queue, func_args_queue):
            pass
        pool.close()
        pool.join()

    # 5. Geodesic interpolation: interpolate between t and t+1
    for hemisphere, structure_id, area_threshold in itertools.product(
        default_config.hemisphere,
        default_config.structure_ids,
        default_config.area_thresholds,
    ):
        string_base = os.path.join(
            centered_nondegenerate_dir,
            f"{hemisphere}_structure_{structure_id}**_at_{area_threshold}.ply",
        )
        input_paths = sorted(glob.glob(string_base))
        print(
            f"Found {len(input_paths)} .plys for {hemisphere} hemisphere"
            f" and anatomical structure {structure_id}:"
        )
        for path in input_paths:
            print(path)
        output_dir = geodesics_dir

        multiprocessing.set_start_method("spawn", force=True)
        queue = multiprocessing.Manager().Queue()
        for gpu_id in range(default_config.n_gpus):
            queue.put(gpu_id)

        pool = multiprocessing.Pool(processes=default_config.n_gpus)
        i_pairs = list(range(default_config.day_range[0], default_config.day_range[1]))

        func = geodesics.interpolate_with_geodesic
        func_args_queue = [
            (func, input_paths, output_dir, i_pair, queue) for i_pair in i_pairs
        ]

        for _ in pool.imap_unordered(run_func_in_parallel_with_queue, func_args_queue):
            pass
        pool.close()
        pool.join()

    # 6. Sort meshes by hormone levels
    for hemisphere, structure_id, area_threshold in itertools.product(
        default_config.hemisphere,
        default_config.structure_ids,
        default_config.area_thresholds,
    ):
        sorting.sort_meshes_by_hormones_and_write(
            input_dir=parameterized_dir,
            output_dir=sorted_parameterized_dir,
            hemisphere=hemisphere,
            structure_id=structure_id,
            area_threshold=area_threshold,
        )
