"""Utils to import data."""

import os

import geomstats.backend as gs
import numpy as np
import trimesh

import my28brains.datasets.synthetic as synthetic


def load(config):
    """Load data according to values in config file."""
    if config.data_type == "synthetic":
        print("Using synthetic data")
        synthetic_mesh_sequence_dir = config.synthetic_mesh_sequence_dir
        start_shape_dir = config.start_shape_dir
        end_shape_dir = config.end_shape_dir
        start_shape = config.start_shape
        end_shape = config.end_shape
        n_times = config.n_times

        if not os.path.exists(synthetic_mesh_sequence_dir):
            print(
                "Creating synthetic geodesic with"
                f" start_mesh: {start_shape}, end_mesh: {end_shape},"
                f" and n_times: {n_times}"
            )

            if not os.path.exists(start_shape_dir):
                print(f"Creating {start_shape} mesh in {start_shape_dir}")
                start_mesh = synthetic.generate_synthetic_mesh(start_shape)
                start_mesh_vertices = start_mesh.vertices
                start_mesh_faces = start_mesh.faces

                os.makedirs(start_shape_dir)
                np.save(
                    os.path.join(start_shape_dir, "vertices.npy"), start_mesh_vertices
                )
                np.save(os.path.join(start_shape_dir, "faces.npy"), start_mesh_faces)
            else:
                print(f"{start_shape} mesh exists in {start_shape_dir}. Loading now.")
                start_mesh_vertices = np.load(
                    os.path.join(start_shape_dir, "vertices.npy")
                )
                start_mesh_faces = np.load(os.path.join(start_shape_dir, "faces.npy"))
                start_mesh = trimesh.Trimesh(
                    vertices=start_mesh_vertices, faces=start_mesh_faces
                )

            if not os.path.exists(end_shape_dir):
                print(f"Creating {end_shape} mesh in {end_shape_dir}")
                end_mesh = synthetic.generate_synthetic_mesh(end_shape)
                end_mesh_vertices = end_mesh.vertices
                end_mesh_faces = end_mesh.faces

                os.makedirs(end_shape_dir)
                np.save(os.path.join(end_shape_dir, "vertices.npy"), end_mesh_vertices)
                np.save(os.path.join(end_shape_dir, "faces.npy"), end_mesh_faces)
            else:
                print(f"{end_shape} mesh exists in {end_shape_dir}. Loading now.")
                end_mesh_vertices = np.load(os.path.join(end_shape_dir, "vertices.npy"))
                end_mesh_faces = np.load(os.path.join(end_shape_dir, "faces.npy"))
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
                start_mesh, end_mesh, n_times
            )
            print("Original mesh_sequence vertices: ", mesh_sequence_vertices.shape)
            print("Original mesh faces: ", mesh_faces.shape)
            print("Times: ", times.shape)

            os.makedirs(synthetic_mesh_sequence_dir)
            np.save(
                os.path.join(synthetic_mesh_sequence_dir, "mesh_sequence_vertices.npy"),
                mesh_sequence_vertices,
            )
            np.save(
                os.path.join(synthetic_mesh_sequence_dir, "mesh_faces.npy"), mesh_faces
            )
            np.save(os.path.join(synthetic_mesh_sequence_dir, "times.npy"), times)
            np.save(
                os.path.join(synthetic_mesh_sequence_dir, "true_intercept.npy"),
                true_intercept,
            )
            np.save(
                os.path.join(synthetic_mesh_sequence_dir, "true_coef.npy"), true_coef
            )
            return mesh_sequence_vertices, mesh_faces, times, true_intercept, true_coef

        else:
            print(
                "Synthetic geodesic exists with"
                f" start mesh {start_shape}, end mesh {end_shape},"
                f" and n_times {n_times}. Loading now."
            )
            mesh_sequence_vertices = gs.array(
                np.load(
                    os.path.join(
                        synthetic_mesh_sequence_dir, "mesh_sequence_vertices.npy"
                    )
                )
            )
            mesh_faces = gs.array(
                np.load(os.path.join(synthetic_mesh_sequence_dir, "mesh_faces.npy"))
            )
            times = gs.array(
                np.load(os.path.join(synthetic_mesh_sequence_dir, "times.npy"))
            )
            true_intercept = gs.array(
                np.load(os.path.join(synthetic_mesh_sequence_dir, "true_intercept.npy"))
            )
            true_coef = gs.array(
                np.load(os.path.join(synthetic_mesh_sequence_dir, "true_coef.npy"))
            )
            return mesh_sequence_vertices, mesh_faces, times, true_intercept, true_coef

    elif config.data_type == "real":
        print("Using real data")
        mesh_dir = config.parameterized_meshes_dir
        print(mesh_dir)
        raise NotImplementedError


# in progress...
# when i do this, i will most likely change main_2_mesh_parameterization
# to take in a list of meshes
# and then call it here.
