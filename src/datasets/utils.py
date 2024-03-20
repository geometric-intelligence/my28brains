"""Utils to import data."""

import glob
import inspect
import os

import geomstats.backend as gs
import numpy as np
import pandas as pd
import torch
import trimesh
from geomstats.geometry.discrete_surfaces import (
    DiscreteSurfaces,
    ElasticMetric,
    _ExpSolver,
)
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean, variance

import H2_SurfaceMatch.utils.input_output as h2_io
import src.datasets.synthetic as synthetic
import src.import_project_config as pc

# from src.regression.discrete_surfaces import DiscreteSurfaces, ElasticMetric, _ExpSolver
from src.regression.geodesic_regression import RiemannianGradientDescent


def get_optimizer(use_cuda, n_vertices, max_iter=100, tol=1e-5):
    """Determine Optimizer based on use_cuda.

    If we are running on GPU, we use RiemannianGradientDescent.

    Parameters
    ----------
    use_cuda : bool. Whether to use GPU.
    n_vertices : int
    max_iter : int
    tol : float
    """
    if use_cuda:
        embedding_space_dim = 3 * n_vertices
        print("embedding_space_dim", embedding_space_dim)
        embedding_space = Euclidean(dim=embedding_space_dim)
        optimizer = RiemannianGradientDescent(
            max_iter=max_iter,
            init_step_size=0.1,
            tol=tol,
            verbose=False,
            space=embedding_space,
        )
    else:
        optimizer = None
    return optimizer


def load_synthetic_data(config):
    """Load synthetic data according to values in config file."""
    if config.device_id is None:
        torchdeviceId = torch.device("cuda:0") if config.use_cuda else "cpu"
    else:
        torchdeviceId = (
            torch.device(f"cuda:{config.device_id}") if config.use_cuda else "cpu"
        )

    project_dir = config.project_dir
    project_config = pc.import_default_config(project_dir)
    if config.dataset_name == "synthetic_mesh":
        print("Using synthetic mesh data")
        data_dir = project_config.synthetic_data_dir
        start_shape, end_shape = config.start_shape, config.end_shape
        n_X = config.n_X
        n_subdivisions = config.n_subdivisions
        noise_factor = config.noise_factor
        project_linear_noise = config.project_linear_noise
        linear_noise = config.linear_noise

        mesh_dir = os.path.join(
            data_dir,
            f"geodesic_{start_shape}_{end_shape}_{n_X}_subs{n_subdivisions}"
            f"_noise{noise_factor}_projected{project_linear_noise}_linear_noise{linear_noise}",
        )

        mesh_sequence_vertices_path = os.path.join(
            mesh_dir, "mesh_sequence_vertices.npy"
        )
        mesh_faces_path = os.path.join(mesh_dir, "mesh_faces.npy")
        X_path = os.path.join(mesh_dir, "X.npy")
        true_intercept_path = os.path.join(mesh_dir, "true_intercept.npy")
        true_coef_path = os.path.join(mesh_dir, "true_coef.npy")

        noiseless_mesh_dir = os.path.join(
            data_dir,
            f"geodesic_{start_shape}_{end_shape}_{n_X}_subs{n_subdivisions}"
            f"_noise{0.0}",
        )

        noiseless_mesh_sequence_vertices_path = os.path.join(
            noiseless_mesh_dir, "mesh_sequence_vertices.npy"
        )
        noiseless_mesh_faces_path = os.path.join(noiseless_mesh_dir, "mesh_faces.npy")
        noiseless_X_path = os.path.join(noiseless_mesh_dir, "X.npy")
        noiseless_true_intercept_path = os.path.join(
            noiseless_mesh_dir, "true_intercept.npy"
        )
        noiseless_true_coef_path = os.path.join(noiseless_mesh_dir, "true_coef.npy")

        if os.path.exists(noiseless_mesh_dir):
            print(f"Noiseless geodesic exists in {mesh_dir}. Loading now.")
            noiseless_mesh_sequence_vertices = gs.array(
                np.load(noiseless_mesh_sequence_vertices_path)
            )
            mesh_faces = gs.array(np.load(noiseless_mesh_faces_path))
            X = gs.array(np.load(noiseless_X_path))
            true_intercept = gs.array(np.load(noiseless_true_intercept_path))
            true_coef = gs.array(np.load(noiseless_true_coef_path))
        else:
            print(
                f"Noiseless geodesic does not exist in {noiseless_mesh_dir}. Creating one."
            )
            start_mesh = load_mesh(start_shape, n_subdivisions, config)
            end_mesh = load_mesh(end_shape, n_subdivisions, config)

            (
                noiseless_mesh_sequence_vertices,
                mesh_faces,
                X,
                true_intercept,
                true_coef,
            ) = synthetic.generate_parameterized_mesh_geodesic(
                start_mesh, end_mesh, config, n_X, config.n_steps
            )

            os.makedirs(noiseless_mesh_dir)
            np.save(
                noiseless_mesh_sequence_vertices_path, noiseless_mesh_sequence_vertices
            )
            np.save(noiseless_mesh_faces_path, mesh_faces)
            np.save(noiseless_X_path, X)
            np.save(noiseless_true_intercept_path, true_intercept)
            np.save(noiseless_true_coef_path, true_coef)

        y_noiseless = noiseless_mesh_sequence_vertices

        faces = gs.array(mesh_faces)
        print("config.use_cuda: ", config.use_cuda)
        print("config.torch_dtype: ", config.torch_dtype)
        print("config.torchdeviceId: ", torchdeviceId)
        if config.use_cuda:
            faces = faces.to(torchdeviceId)

        space = DiscreteSurfaces(faces=gs.array(mesh_faces))
        elastic_metric = ElasticMetric(
            space=space,
            a0=project_config.a0,
            a1=project_config.a1,
            b1=project_config.b1,
            c1=project_config.c1,
            d1=project_config.d1,
            a2=project_config.a2,
        )
        optimizer = get_optimizer(
            config.use_cuda, n_vertices=len(y_noiseless[0]), max_iter=100, tol=1e-5
        )
        elastic_metric.exp_solver = _ExpSolver(
            n_steps=config.n_steps, optimizer=optimizer
        )
        space.metric = elastic_metric

        if os.path.exists(mesh_dir):
            print(f"Synthetic geodesic exists in {mesh_dir}. Loading now.")
            mesh_sequence_vertices = gs.array(np.load(mesh_sequence_vertices_path))
            mesh_faces = gs.array(np.load(mesh_faces_path))
            X = gs.array(np.load(X_path))
            true_intercept = gs.array(np.load(true_intercept_path))
            true_coef = gs.array(np.load(true_coef_path))

            y = mesh_sequence_vertices
            return space, y, y_noiseless, X, true_intercept, true_coef

        print(f"No noisy synthetic geodesic found in {mesh_dir}. Creating one.")
        # projecting linear noise does not apply to meshes
        # project_linear_noise = config.project_linear_noise
        # mesh_sequence_vertices = synthetic.add_linear_noise(
        #     space,
        #     noiseless_mesh_sequence_vertices,
        #     config.dataset_name,
        #     project_linear_noise,
        #     noise_factor=noise_factor,
        # )

        space = DiscreteSurfaces(faces=gs.array(mesh_faces))
        print(f"space faces: {space.faces.shape}")
        elastic_metric = ElasticMetric(
            space=space,
            a0=project_config.a0,
            a1=project_config.a1,
            b1=project_config.b1,
            c1=project_config.c1,
            d1=project_config.d1,
            a2=project_config.a2,
        )
        optimizer = get_optimizer(
            config.use_cuda, n_vertices=len(y_noiseless[0]), max_iter=100, tol=1e-5
        )
        elastic_metric.exp_solver = _ExpSolver(
            n_steps=config.n_steps, optimizer=optimizer
        )
        space.metric = elastic_metric

        if config.linear_noise:
            print(f"noise factor: {config.noise_factor}")
            print(f"dataset name: {config.dataset_name}")
            print(f"y noiseless shape: {y_noiseless.shape}")
            mesh_sequence_vertices = synthetic.add_linear_noise(
                space,
                noiseless_mesh_sequence_vertices,
                config.dataset_name,
                config.project_linear_noise,
                noise_factor=config.noise_factor,
            )
        else:
            mesh_sequence_vertices = synthetic.add_geodesic_noise(
                space,
                noiseless_mesh_sequence_vertices,
                config.dataset_name,
                noise_factor=config.noise_factor,
            )

        print("Noisy mesh_sequence vertices: ", mesh_sequence_vertices.shape)
        print("Noisy mesh faces: ", mesh_faces.shape)
        print("X: ", X.shape)

        os.makedirs(mesh_dir)
        np.save(mesh_sequence_vertices_path, mesh_sequence_vertices)
        np.save(mesh_faces_path, mesh_faces)
        np.save(X_path, X)
        np.save(true_intercept_path, true_intercept)
        np.save(true_coef_path, true_coef)

        y = mesh_sequence_vertices
        return space, y, y_noiseless, X, true_intercept, true_coef

    elif config.dataset_name in ["hyperboloid", "hypersphere"]:
        print(f"Creating synthetic dataset on {config.dataset_name}")
        if config.dataset_name == "hyperboloid":
            space = Hyperbolic(dim=config.space_dimension, coords_type="extrinsic")
        else:
            space = Hypersphere(dim=config.space_dimension)

        # X, y_noiseless, y_noisy, true_intercept, true_coef = synthetic.generate_noisy_benchmark_data(space = space, linear_noise=config.linear_noise, dataset_name=config.dataset_name, n_samples=config.n_X, noise_factor=config.noise_factor)
        X, y_noiseless, true_intercept, true_coef = synthetic.generate_general_geodesic(
            space, config.n_X, config.synthetic_tan_vec_length
        )
        if config.linear_noise:
            print(f"noise factor: {config.noise_factor}")
            print(f"dataset name: {config.dataset_name}")
            print(f"space dimension: {config.space_dimension}")
            print(f"y noiseless shape: {y_noiseless.shape}")
            y_noisy = synthetic.add_linear_noise(
                space,
                y_noiseless,
                config.dataset_name,
                config.project_linear_noise,
                noise_factor=config.noise_factor,
            )
        else:
            y_noisy = synthetic.add_geodesic_noise(
                space,
                y_noiseless,
                config.dataset_name,
                noise_factor=config.noise_factor,
            )
        return space, y_noisy, y_noiseless, X, true_intercept, true_coef
    else:
        raise ValueError(f"Unknown dataset name {config.dataset_name}")


def load_real_data(config):
    """Load real brain meshes according to values in config file."""
    print(config)
    project_dir = config.project_dir
    project_config = pc.import_default_config(project_dir)

    # load meshes
    mesh_sequence_vertices = []
    mesh_sequence_faces = []
    if project_config.sort:
        days_to_ignore = None
        print("Using menstrual mesh data (from progesterone sorted directory)")
        mesh_dir = project_config.sorted_dir

        sorted_hormone_levels_path = os.path.join(mesh_dir, "sorted_hormone_levels.npy")
        sorted_hormone_levels = np.loadtxt(sorted_hormone_levels_path, delimiter=",")

        for i, hormone_level in enumerate(sorted_hormone_levels):
            file_suffix = f"hormone_level{hormone_level}.ply"

            # List all files in the directory
            files_in_directory = os.listdir(mesh_dir)

            # Filter files that end with the specified format
            matching_files = [
                file for file in files_in_directory if file.endswith(file_suffix)
            ]

            # Construct the full file paths using os.path.join
            mesh_paths = [os.path.join(mesh_dir, file) for file in matching_files]

            # Print the result
            for mesh_path in mesh_paths:
                print(f"Mesh Path {i + 1}: {mesh_path}")
                vertices, faces, _ = h2_io.loadData(mesh_path)
                mesh_sequence_vertices.append(vertices)
                mesh_sequence_faces.append(faces)
    else:
        print("Using menstrual mesh data (from reparameterized directory)")
        mesh_dir = project_config.reparameterized_dir

        # make sure there are meshes in the directory
        mesh_string_base = os.path.join(
            mesh_dir, f"{config.hemisphere}_structure_{config.structure_id}**.ply"
        )
        mesh_paths = sorted(glob.glob(mesh_string_base))
        print(
            f"\ne. (Sort) Found {len(mesh_paths)} .plys for ({config.hemisphere}, {config.structure_id}) in {mesh_dir}"
        )

        # load meshes
        mesh_sequence_vertices, mesh_sequence_faces = [], []
        first_day = int(project_config.day_range[0])
        last_day = int(project_config.day_range[1])

        days_to_ignore = []
        for day in range(first_day, last_day + 1):
            mesh_path = os.path.join(
                mesh_dir,
                f"{config.hemisphere}_structure_{config.structure_id}_day{day:02d}"
                f"_at_{config.area_threshold}.ply",
            )
            if not mesh_path.exists():
                print(f"Day {day} has no data. Skipping.")
                print(f"DayID not to use: {day}")
                days_to_ignore.append(day)
                continue

            vertices, faces, _ = h2_io.loadData(mesh_path)
            if vertices.shape[0] == 0:
                print(f"Day {day} has no data. Skipping.")
                print(f"DayID not to use: {day}")
                days_to_ignore.append(day)
                continue
            mesh_sequence_vertices.append(vertices)
            print("vertices.shape ", vertices.shape)
            mesh_sequence_faces.append(faces)
        days_to_ignore = gs.array(days_to_ignore)

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

    space = DiscreteSurfaces(faces=mesh_faces)
    elastic_metric = ElasticMetric(
        space=space,
        a0=project_config.a0,
        a1=project_config.a1,
        b1=project_config.b1,
        c1=project_config.c1,
        d1=project_config.d1,
        a2=project_config.a2,
    )
    optimizer = get_optimizer(
        config.use_cuda, n_vertices=len(true_intercept), max_iter=100, tol=1e-5
    )
    elastic_metric.exp_solver = _ExpSolver(n_steps=config.n_steps, optimizer=optimizer)
    space.metric = elastic_metric

    y = mesh_sequence_vertices

    if project_config.dataset_name == "menstrual_mesh":
        # load all hormones
        hormones_path = os.path.join(project_config.data_dir, "hormones.csv")
        df = pd.read_csv(hormones_path, delimiter=",")
    if project_config.dataset_name == "pregnancy_mesh":
        hormones_path = os.path.join(project_config.data_dir, "28Baby_Hormones.csv")
        df = pd.read_csv(hormones_path, delimiter=",")
        df["dayID"] = [int(entry.split("-")[1]) for entry in df["sessionID"]]
        df = df[df["dayID"] != 27]  # sess 27 is a repeat of sess 26

    days_used = df[df["dayID"] < project_config.day_range[1] + 1]
    days_used = days_used[days_used["dayID"] > project_config.day_range[0] - 1]
    if days_to_ignore is not None:
        for day in days_to_ignore:
            day = int(day)
            days_used = days_used[days_used["dayID"] != day]
            print("Hormones excluded from day: ", day)
    print(days_used)
    all_hormone_levels = days_used

    print(f"space faces: {space.faces.shape}")
    print(f"y shape: {y.shape}")
    print(f"X shape: {all_hormone_levels.shape}")
    print(f"true intercept shape: {true_intercept.shape}")
    print(f"true coef shape: {true_coef.shape}")

    return space, y, all_hormone_levels, true_intercept, true_coef


def load_mesh(mesh_type, n_subdivisions, config):
    """Load a mesh from the synthetic dataset.

    If the mesh does not exist, create it.

    Parameters
    ----------
    mesh_type : str, {"sphere", "ellipsoid", "pill", "cube"}
    """
    project_dir = config.project_dir
    project_config = pc.import_default_config(project_dir)
    data_dir = project_config.synthetic_data_dir
    shape_dir = os.path.join(data_dir, f"{mesh_type}_subs{n_subdivisions}")
    vertices_path = os.path.join(shape_dir, "vertices.npy")
    faces_path = os.path.join(shape_dir, "faces.npy")

    if os.path.exists(shape_dir):
        print(f"{mesh_type} mesh exists in {shape_dir}. Loading now.")
        vertices = np.load(vertices_path)
        faces = np.load(faces_path)
        return trimesh.Trimesh(vertices=vertices, faces=faces)

    print(f"Creating {mesh_type} mesh in {shape_dir}")
    mesh = synthetic.generate_mesh(mesh_type, n_subdivisions)
    os.makedirs(shape_dir)
    np.save(vertices_path, mesh.vertices)
    np.save(faces_path, mesh.faces)
    return mesh


def mesh_diameter(mesh_vertices):
    """Compute the diameter of a mesh."""
    max_distance = 0
    for i_vertex in range(mesh_vertices.shape[0]):
        for j_vertex in range(i_vertex + 1, mesh_vertices.shape[0]):
            distance = gs.linalg.norm(mesh_vertices[i_vertex] - mesh_vertices[j_vertex])
            if distance > max_distance:
                max_distance = distance
    return max_distance


# def add_noise(mesh_sequence_vertices, noise_factor):
#     """Add noise to mesh_sequence_vertices.

#     Note that this function modifies the input mesh_sequence_vertices,
#     which is overwritten by its noisy version.

#     For example, after running:
#     noisy_mesh = data_utils.add_noise(
#         mesh_sequence_vertices=[mesh],
#         noise_factor=10
#     )
#     the mesh has become noisy_mesh as well.
#     """
#     diameter = mesh_diameter(mesh_sequence_vertices[0])
#     noise_sd = noise_factor * diameter
#     for i_mesh in range(len(mesh_sequence_vertices)):
#         mesh_sequence_vertices[i_mesh] += gs.random.normal(
#             loc=0.0, scale=noise_sd, size=mesh_sequence_vertices[i_mesh].shape
#         )
#     return mesh_sequence_vertices
