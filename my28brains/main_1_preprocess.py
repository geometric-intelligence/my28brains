"""Preprocessing of the my28brains dataset.

The processing is done:
- with the `my28brains.preprocessing` module.
- on the 60 days for steps a-c.
- on the days specified in default_config.day_range for steps d-f.

a. Mesh by segmenting surfaces of the hippocampus and its substructures.
-> outputs in meshed_dir

b. Center meshes by putting whole hippocampus barycenter at 0.
-> outputs in centered_dir

c. Remove degenerate faces using area thresholds
-> outputs in nondegenerate_dir

d. Reparameterize meshes: use parameterization of the first mesh
-> outputs in reparameterized_dir

e. Sort meshes by hormone levels
-> outputs in sorted_dir
NOTE: Sorting is not necessary for regression.
NOTE: make sure day_range = [2, 11] in default_config.py

f. (Optional) Interpolate between t and t+1 with a geodesic
-> outputs in interpolated_dir

Note on pykeops:
----------------
If main cannot run because of:
EOFError: Ran out of input
This is probably because of a pykeops issue.

To fix this, run the following commands:
- remove: rm -rf ~/.cache/keops*/build
- rebuild from python console.
    >>> import pykeops

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
day_dirs = [os.path.join(raw_dir, f"Day{i:02d}") for i in range(1, 61)]

meshed_dir = default_config.meshed_dir
centered_dir = default_config.centered_dir
nondegenerate_dir = default_config.nondegenerate_dir
reparameterized_dir = default_config.reparameterized_dir
sorted_dir = default_config.sorted_dir
interpolated_dir = default_config.interpolated_dir

day_range = default_config.day_range
day_range_index = [day_range[0] - 1, day_range[1] - 1]


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
    # a. Mesh by segmenting surfaces of the hippocampus and its substructures.
    for hemisphere, structure_id in itertools.product(
        default_config.hemisphere, set(default_config.structure_ids + [-1])
    ):
        # Add the whole hippocampus id=[-1] to list of structure ids,
        # in order to be able to compute its center and center the substructures.
        extraction.extract_meshes_from_nii_and_write(
            input_dir=day_dirs,
            output_dir=meshed_dir,
            hemisphere=hemisphere,
            structure_id=structure_id,
        )

    # b. Center meshes by putting whole hippocampus barycenter at 0.
    for hemisphere in default_config.hemisphere:
        hippocampus_centers = centering.center_whole_hippocampus_and_write(
            input_dir=meshed_dir, output_dir=centered_dir, hemisphere=hemisphere
        )
        for structure_id in default_config.structure_ids:
            if structure_id != -1:
                centering.center_substructure_and_write(
                    input_dir=meshed_dir,
                    output_dir=centered_dir,
                    hemisphere=hemisphere,
                    structure_id=structure_id,
                    hippocampus_centers=hippocampus_centers,
                )

    # c. Remove degenerate faces using area thresholds
    for hemisphere, structure_id, area_threshold in itertools.product(
        default_config.hemisphere,
        default_config.structure_ids,
        default_config.area_thresholds,
    ):
        geodesics.remove_degenerate_faces_and_write(
            input_dir=centered_dir,
            output_dir=nondegenerate_dir,
            hemisphere=hemisphere,
            structure_id=structure_id,
            area_threshold=area_threshold,
        )

    # d. Reparameterize meshes: use parameterization of the first mesh
    for hemisphere, structure_id, area_threshold in itertools.product(
        default_config.hemisphere,
        default_config.structure_ids,
        default_config.area_thresholds,
    ):
        string_base = os.path.join(
            nondegenerate_dir,
            f"{hemisphere}_structure_{structure_id}**_at_{area_threshold}.ply",
        )
        input_paths = sorted(glob.glob(string_base))
        print(
            f"\nd. (Reparameterize) Found {len(input_paths)} .plys for ({hemisphere}, {structure_id}) in {nondegenerate_dir}"
        )
        output_dir = reparameterized_dir

        multiprocessing.set_start_method("spawn", force=True)
        queue = multiprocessing.Manager().Queue()
        for gpu_id in range(default_config.n_gpus):
            queue.put(gpu_id)

        pool = multiprocessing.Pool(processes=default_config.n_gpus)
        i_paths = list(range(day_range_index[0], day_range_index[1] + 1))

        func = geodesics.reparameterize_with_geodesic
        func_args_queue = [
            (func, input_paths, output_dir, i_path, queue) for i_path in i_paths
        ]

        for _ in pool.imap_unordered(run_func_in_parallel_with_queue, func_args_queue):
            pass
        pool.close()
        pool.join()

    # e. Sort meshes by hormone levels
    for hemisphere, structure_id, area_threshold in itertools.product(
        default_config.hemisphere,
        default_config.structure_ids,
        default_config.area_thresholds,
    ):
        sorting.sort_meshes_by_hormones_and_write(
            input_dir=reparameterized_dir,
            output_dir=sorted_dir,
            hemisphere=hemisphere,
            structure_id=structure_id,
            area_threshold=area_threshold,
        )

    if not default_config.run_interpolate:
        print("\nf. Skipping interpolation. Preprocessing done!")
        exit()
    # f. (Optional) Geodesic interpolation between t and t+1
    for hemisphere, structure_id, area_threshold in itertools.product(
        default_config.hemisphere,
        default_config.structure_ids,
        default_config.area_thresholds,
    ):
        string_base = os.path.join(
            nondegenerate_dir,
            f"{hemisphere}_structure_{structure_id}**_at_{area_threshold}.ply",
        )
        input_paths = sorted(glob.glob(string_base))
        print(
            f"\nF. Found {len(input_paths)} .plys for {hemisphere} hemisphere, id {structure_id}"
        )
        output_dir = interpolated_dir

        multiprocessing.set_start_method("spawn", force=True)
        queue = multiprocessing.Manager().Queue()
        for gpu_id in range(default_config.n_gpus):
            queue.put(gpu_id)

        pool = multiprocessing.Pool(processes=default_config.n_gpus)
        i_pairs = list(range(day_range_index[0], day_range_index[1]))

        func = geodesics.interpolate_with_geodesic
        func_args_queue = [
            (func, input_paths, output_dir, i_pair, queue) for i_pair in i_pairs
        ]

        for _ in pool.imap_unordered(run_func_in_parallel_with_queue, func_args_queue):
            pass
        pool.close()
        pool.join()
