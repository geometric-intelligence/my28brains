"""Utils to import data."""

import os

import default_config

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs
import numpy as np
import trimesh

import H2_SurfaceMatch.utils.input_output
import my28brains.datasets.synthetic as synthetic


def load(config):  # , device = "cuda:0"):
    """Load data according to values in config file."""
    if config.dataset_name == "synthetic":
        print("Using synthetic data")
        data_dir = default_config.synthetic_data_dir
        start_shape = config.start_shape
        end_shape = config.end_shape
        n_times = config.n_times

        start_shape_dir = os.path.join(
            data_dir,
            f"{start_shape}_subs{config.n_subdivisions}_ell{config.ellipse_dimensions}",
        )
        end_shape_dir = os.path.join(
            data_dir,
            f"{end_shape}_subs{config.n_subdivisions}_ell{config.ellipse_dimensions}",
        )
        mesh_dir = os.path.join(
            data_dir,
            f"geodesic_{start_shape}_{end_shape}_{n_times}_subs{config.n_subdivisions}"
            f"_ell{config.ellipse_dimensions}",
        )

        start_vertices_path = os.path.join(start_shape_dir, "vertices.npy")
        start_faces_path = os.path.join(start_shape_dir, "faces.npy")
        end_vertices_path = os.path.join(end_shape_dir, "vertices.npy")
        end_faces_path = os.path.join(end_shape_dir, "faces.npy")

        if not os.path.exists(mesh_dir):
            print(
                "Creating synthetic geodesic with"
                f" start_mesh: {start_shape}, end_mesh: {end_shape},"
                f" and n_times: {n_times}"
            )

            if not os.path.exists(start_shape_dir):
                print(f"Creating {start_shape} mesh in {start_shape_dir}")
                start_mesh = synthetic.generate_synthetic_mesh(
                    start_shape, config.n_subdivisions, config.ellipse_dimensions
                )
                start_mesh_vertices = start_mesh.vertices
                start_mesh_faces = start_mesh.faces

                os.makedirs(start_shape_dir)
                np.save(start_vertices_path, start_mesh_vertices)
                np.save(start_faces_path, start_mesh_faces)
            else:
                print(f"{start_shape} mesh exists in {start_shape_dir}. Loading now.")
                start_mesh_vertices = np.load(start_vertices_path)
                start_mesh_faces = np.load(start_faces_path)
                start_mesh = trimesh.Trimesh(
                    vertices=start_mesh_vertices, faces=start_mesh_faces
                )

            if not os.path.exists(end_shape_dir):
                print(f"Creating {end_shape} mesh in {end_shape_dir}")
                end_mesh = synthetic.generate_synthetic_mesh(
                    end_shape, config.n_subdivisions, config.ellipse_dimensions
                )
                end_mesh_vertices = end_mesh.vertices
                end_mesh_faces = end_mesh.faces

                os.makedirs(end_shape_dir)
                np.save(end_vertices_path, end_mesh_vertices)
                np.save(end_faces_path, end_mesh_faces)
            else:
                print(f"{end_shape} mesh exists in {end_shape_dir}. Loading now.")
                end_mesh_vertices = np.load(end_vertices_path)
                end_mesh_faces = np.load(end_faces_path)
                end_mesh = trimesh.Trimesh(
                    vertices=end_mesh_vertices, faces=end_mesh_faces
                )

            (
                mesh_sequence_vertices,
                mesh_faces,
                times,
                true_intercept,
                true_coef,
            ) = synthetic.generate_synthetic_parameterized_geodesic(
                start_mesh,
                end_mesh,
                n_times,
            )

            print("Original mesh_sequence vertices: ", mesh_sequence_vertices.shape)
            print("Original mesh faces: ", mesh_faces.shape)
            print("Times: ", times.shape)

            os.makedirs(mesh_dir)
            np.save(
                os.path.join(mesh_dir, "mesh_sequence_vertices.npy"),
                mesh_sequence_vertices,
            )
            np.save(os.path.join(mesh_dir, "mesh_faces.npy"), mesh_faces)
            np.save(os.path.join(mesh_dir, "times.npy"), times)
            np.save(
                os.path.join(mesh_dir, "true_intercept.npy"),
                true_intercept,
            )
            np.save(os.path.join(mesh_dir, "true_coef.npy"), true_coef)
            return mesh_sequence_vertices, mesh_faces, times, true_intercept, true_coef

        else:
            print(
                "Synthetic geodesic exists with"
                f" start mesh {start_shape}, end mesh {end_shape},"
                f" and n_times {n_times}. Loading now."
            )
            mesh_sequence_vertices = gs.array(
                np.load(os.path.join(mesh_dir, "mesh_sequence_vertices.npy"))
            )
            mesh_faces = gs.array(np.load(os.path.join(mesh_dir, "mesh_faces.npy")))
            times = gs.array(np.load(os.path.join(mesh_dir, "times.npy")))
            true_intercept = gs.array(
                np.load(os.path.join(mesh_dir, "true_intercept.npy"))
            )
            true_coef = gs.array(np.load(os.path.join(mesh_dir, "true_coef.npy")))
            return mesh_sequence_vertices, mesh_faces, times, true_intercept, true_coef

    elif config.dataset_name == "real":
        print("Using real data")
        mesh_dir = default_config.sorted_parameterized_meshes_dir
        mesh_sequence_vertices = []
        mesh_sequence_faces = []
        first_day = int(default_config.day_range[0])
        last_day = int(default_config.day_range[1])
        # times = gs.arange(0, 1, 1/(last_day - first_day + 1))

        hormone_levels_path = os.path.join(
            default_config.sorted_parameterized_meshes_dir, "sorted_hormone_levels.npy"
        )
        hormone_levels = np.loadtxt(hormone_levels_path, delimiter=",")
        times = gs.array(hormone_levels)
        print("times: ", times)

        # for i_mesh in range(first_day, last_day + 1):
        for i_mesh in range(last_day - first_day + 1):
            # mesh_path = os.path.join(
            #     default_config.sorted_parameterized_meshes_dir,
            #     f"{config.hemisphere}_structure_-1_day{i_mesh:02d}_at_0.0_parameterized.ply",
            # )
            # file_name = f"parameterized_mesh{i_mesh:02d}_hormone_level****.ply"
            file_name = f"parameterized_mesh{i_mesh:02d}.ply"

            mesh_path = os.path.join(
                default_config.sorted_parameterized_meshes_dir, file_name
            )
            [
                vertices,
                faces,
                Fun,
            ] = H2_SurfaceMatch.utils.input_output.loadData(mesh_path)
            mesh_sequence_vertices.append(vertices)
            mesh_sequence_faces.append(faces)
        mesh_sequence_vertices = gs.array(mesh_sequence_vertices)

        # parameterized = all(
        # faces == mesh_sequence_faces[0] for faces in mesh_sequence_faces)
        for faces in mesh_sequence_faces:
            if (faces != mesh_sequence_faces[0]).all():
                raise ValueError("Meshes are not parameterized")

        mesh_faces = gs.array(mesh_sequence_faces[0])
        true_intercept = gs.array(mesh_sequence_vertices[0])
        true_coef = gs.array(mesh_sequence_vertices[1] - mesh_sequence_vertices[0])
        print(mesh_dir)
        return mesh_sequence_vertices, mesh_faces, times, true_intercept, true_coef
    else:
        raise ValueError(f"Unknown dataset name {config.dataset_name}")


def mesh_diameter(mesh_vertices):
    """Compute the diameter of a mesh."""
    max_distance = 0
    for i_vertex in range(mesh_vertices.shape[0]):
        for j_vertex in range(i_vertex + 1, mesh_vertices.shape[0]):
            distance = gs.linalg.norm(mesh_vertices[i_vertex] - mesh_vertices[j_vertex])
            if distance > max_distance:
                max_distance = distance
    return max_distance


def add_noise(mesh_sequence_vertices, noise_factor):
    """Add noise to mesh_sequence_vertices."""
    noise_sd = noise_factor * mesh_diameter(mesh_sequence_vertices[0])
    for i_mesh in range(len(mesh_sequence_vertices)):
        mesh_sequence_vertices[i_mesh] += gs.random.normal(
            loc=0.0, scale=noise_sd, size=mesh_sequence_vertices[i_mesh].shape
        )
    return mesh_sequence_vertices


# in progress...
# when i do this, i will most likely change main_2_mesh_parameterization
# to take in a list of meshes
# and then call it here.
