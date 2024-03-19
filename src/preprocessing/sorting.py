"""Sort meshes in meshes_parameterized by hormone level.

It outputs to meshes_parameterized_sorted_by_hormone
"""

import glob
import inspect
import os

import geomstats.backend as gs
import numpy as np
import pandas as pd

import H2_SurfaceMatch.utils.input_output as h2_io  # noqa: E402
import src.import_project_config as pc


def sort_meshes_by_hormones_and_write(
    input_dir, output_dir, hemisphere, structure_id, area_threshold, project_dir
):
    """Sort meshes in meshes_parameterized by hormone level.

    WARNING: THIS IS FOR MENSTRUAL DATA

    Parameters
    ----------
    input_dir : str
        Input directory in my28brains/src/results/1_preprocess.
        Here storing reparameterized meshes.
    output_dir : str
        Input directory in my28brains/src/results/1_preprocess.
        Here storing reparameterized meshes sorted by hormone level.
    hemisphere : str, {'left', 'right'}
        Hemisphere to process.
    structure_id : int
        Structure ID to process.
    area_threshold : float
        Area threshold to process.
    project_config : project config.
        Config object.
    """
    project_config = pc.import_default_config(project_dir)
    string_base = os.path.join(
        input_dir, f"{hemisphere}_structure_{structure_id}**.ply"
    )
    paths = sorted(glob.glob(string_base))
    print(
        f"\ne. (Sort) Found {len(paths)} .plys for ({hemisphere}, {structure_id}) in {input_dir}"
    )

    hormones_path = os.path.join(project_config.data_dir, "hormones.csv")
    df = pd.read_csv(hormones_path, delimiter=",")
    days_used = df[df["dayID"] < project_config.day_range[1] + 1]
    days_used = days_used[days_used["dayID"] > project_config.day_range[0] - 1]

    print(days_used)
    hormone_levels = days_used["Prog"]

    # Load meshes
    mesh_sequence_vertices, mesh_sequence_faces = [], []
    first_day = int(project_config.day_range[0])
    last_day = int(project_config.day_range[1])
    for day in range(first_day, last_day + 1):
        mesh_path = os.path.join(
            input_dir,
            f"{hemisphere}_structure_{structure_id}_day{day:02d}"
            f"_at_{area_threshold}.ply",
        )
        vertices, faces, _ = h2_io.loadData(mesh_path)
        mesh_sequence_vertices.append(vertices)
        mesh_sequence_faces.append(faces)
        print(vertices.shape)
    mesh_sequence_vertices = gs.array(mesh_sequence_vertices)

    for faces in mesh_sequence_faces:
        if (faces != mesh_sequence_faces[0]).all():
            raise ValueError("Meshes are not parameterized")
    mesh_faces = gs.array(mesh_sequence_faces[0])

    # Combine the two lists into a list of tuples
    combined_list = list(zip(hormone_levels, mesh_sequence_vertices))

    # Sort the combined list based on the hormone levels
    sorted_list = sorted(combined_list)

    # Extract the sorted object list
    sorted_meshes = [mesh for (_, mesh) in sorted_list]
    sorted_hormone_levels = [level for (level, _) in sorted_list]

    # Save the sorted meshes
    i_mesh = 0
    for hormone_level, mesh in zip(sorted_hormone_levels, sorted_meshes):
        # day = i_mesh + 1
        # ply_path = os.path.join(
        #     output_dir,
        #     f"{hemisphere}_structure_{structure_id}_day{day:02d}"
        #     f"_at_{area_threshold}.ply",
        # )
        ply_path = os.path.join(
            output_dir,
            f"{hemisphere}_structure_{structure_id}_mesh{i_mesh:02d}"
            f"_hormone_level{hormone_level}.ply",
        )
        if os.path.exists(ply_path):
            print(f"File exists (no rewrite): {ply_path}")
            continue
        print(f"- Write mesh to {ply_path}")
        h2_io.save_data(
            os.path.splitext(ply_path)[0],  # remove .ply extension
            ".ply",
            gs.array(mesh).numpy(),
            gs.array(mesh_faces).numpy(),
        )
        i_mesh += 1

    # Save the sorted hormone levels with numpy
    sorted_hormone_levels_path = os.path.join(output_dir, "sorted_hormone_levels.npy")
    if os.path.exists(sorted_hormone_levels_path):
        print(f"File exists (no rewrite): {sorted_hormone_levels_path}")
    else:
        np.savetxt(sorted_hormone_levels_path, sorted_hormone_levels, delimiter=",")
