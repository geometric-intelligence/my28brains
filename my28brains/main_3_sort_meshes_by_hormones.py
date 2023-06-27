"""sorts meshes in meshes_parameterized by hormone level. outputs to meshes_parameterized_sorted_by_hormone"""

import os
import sys
import my28brains.default_config as default_config
import pandas as pd
os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs
import H2_SurfaceMatch.utils.input_output
import numpy as np

sorted_parameterized_meshes_dir = default_config.sorted_parameterized_meshes_dir
parameterized_meshes_dir = default_config.parameterized_meshes_dir

hormones_path = os.path.join(default_config.data_dir, "hormones.csv")
df = pd.read_csv(hormones_path, delimiter=",")
days_used = df[df["dayID"] < default_config.day_range[1] + 1]
days_used = days_used[days_used["dayID"] > default_config.day_range[0] - 1]

print(days_used)

hormone_levels = days_used["Prog"]

# Load meshes

mesh_dir = default_config.parameterized_meshes_dir
mesh_sequence_vertices = []
mesh_sequence_faces = []
first_day = int(default_config.day_range[0])
last_day = int(default_config.day_range[1])
for i_mesh in range(first_day, last_day + 1):
    mesh_path = os.path.join(
        default_config.parameterized_meshes_dir,
        f"{default_config.hemisphere[0]}_structure_-1_day{i_mesh:02d}_at_0.0_parameterized.ply",
    )
    [
        vertices,
        faces,
        Fun,
    ] = H2_SurfaceMatch.utils.input_output.loadData(mesh_path)
    mesh_sequence_vertices.append(vertices)
    mesh_sequence_faces.append(faces)
mesh_sequence_vertices = gs.array(mesh_sequence_vertices)

# parameterized = all(faces == mesh_sequence_faces[0] for faces in mesh_sequence_faces)
for faces in mesh_sequence_faces:
    if (faces != mesh_sequence_faces[0]).all():
        raise ValueError("Meshes are not parameterized")

mesh_faces = gs.array(mesh_sequence_faces[0])


# Example object list
parameterized_meshes = mesh_sequence_vertices

# Combine the two lists into a list of tuples
combined_list = list(zip(hormone_levels, parameterized_meshes))

# Sort the combined list based on the hormone levels
sorted_list = sorted(combined_list)

# Extract the sorted object list
sorted_meshes = [mesh for (level, mesh) in sorted_list]
sorted_hormone_levels = [level for (level, mesh) in sorted_list]

# Print the sorted object list
print(sorted_meshes)
print(sorted_hormone_levels)

# Save the sorted meshes
for i_mesh, mesh in enumerate(sorted_meshes):
    sorted_mesh_path = os.path.join(sorted_parameterized_meshes_dir, 
                                    # f"parameterized_mesh{i_mesh:02d}_hormone_level{sorted_hormone_levels[i_mesh]}")
                                    f"parameterized_mesh{i_mesh:02d}")

                                    
    H2_SurfaceMatch.utils.input_output.save_data(
        sorted_mesh_path,
        ".ply",
        gs.array(mesh).numpy(),
        gs.array(mesh_faces).numpy(),
    )

# Save the sorted hormone levels with numpy
sorted_hormone_levels_path = os.path.join(sorted_parameterized_meshes_dir, "sorted_hormone_levels.npy")
np.savetxt(sorted_hormone_levels_path, sorted_hormone_levels, delimiter=",")
