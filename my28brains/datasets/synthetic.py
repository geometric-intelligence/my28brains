"""Generate meshes and synthetic geodesics.

If we need to remove degenerate faces on these meshes, it can be done with:
>>> area_threshold = 0.0005
>>> face_areas = SURFACE_SPACE.face_areas(torch.tensor(ellipsoid.vertices))
>>> face_mask = ~gs.less(face_areas, area_threshold)
>>> ellipsoid.update_faces(face_mask)
"""

import geomstats.backend as gs
import numpy as np
import torch
import trimesh
from geomstats.geometry.discrete_surfaces import (
    DiscreteSurfaces,
    ElasticMetric,
    _ExpSolver,
)
from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean, variance

import H2_SurfaceMatch.H2_match  # noqa: E402
import H2_SurfaceMatch.utils.input_output  # noqa: E402
import H2_SurfaceMatch.utils.utils  # noqa: E402
import my28brains.datasets.utils as data_utils
import my28brains.default_config as default_config


def generate_mesh(mesh_type, n_subdivisions=None, ellipsoid_dims=[2, 2, 3]):
    """Generate a synthetic mesh.

    Parameters
    ----------
    mesh_type : str, {"sphere", "ellipsoid", "pill", "cube"}
        Type of mesh to generate.
    n_subdivisions : int
        How many X to subdivide the mesh (from trimesh).
        Note that the number of faces will grow as function of 4 ** subdivisions,
        so you probably want to keep this under ~5.
    ellipsoid_dims : list
        List of integers representing the dimensions of the ellipse.
        Example: ellipsoid_dims=[2, 2, 3].

    Returns
    -------
    mesh : trimesh.Trimesh
        The generated mesh.
    """
    if mesh_type == "sphere":
        return generate_sphere_mesh(subdivisions=n_subdivisions)
    if mesh_type == "ellipsoid":
        return generate_ellipsoid_mesh(
            subdivisions=n_subdivisions, ellipsoid_dims=ellipsoid_dims
        )
    if mesh_type == "pill":
        return trimesh.creation.capsule(height=1.0, radius=1.0, count=None)
    if mesh_type == "cube":
        return generate_cube_mesh()
    raise ValueError(f"mesh_type {mesh_type} not recognized")


def generate_sphere_mesh(subdivisions=3):
    """Create a sphere trimesh."""
    subdivisions = subdivisions
    radius = 30**subdivisions
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    return sphere


def generate_ellipsoid_mesh(subdivisions=3, ellipsoid_dims=[2, 2, 3]):
    """Create an ellipsoid trimesh."""
    subdivisions = subdivisions
    ellipsoid_dims = ellipsoid_dims
    radius = 30**subdivisions
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    # Create a scaling matrix for the semi-axes lengths
    scales = np.array(ellipsoid_dims)
    scale_matrix = np.diag(scales)
    scale_matrix = gs.array(scale_matrix)
    # Apply the scaling transformation to the mesh vertices
    scaled_vertices = sphere.vertices.dot(scale_matrix)
    # Create a new mesh with the scaled vertices
    ellipsoid = trimesh.Trimesh(vertices=scaled_vertices, faces=sphere.faces)
    return ellipsoid


def generate_cube_mesh():
    """Create the cube mesh used in geomstats unit tests.

    See: geomstats/datasets/data/cube_meshes/cube_mesh_diagram.jpeg.
    """
    vertices = np.array(
        [
            [1, 1, 1],
            [1, -1, 1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, 1],
            [-1, -1, 1],
            [-1, -1, -1],
            [-1, 1, -1],
        ]
    )

    faces = np.array(
        [
            [0, 1, 4],
            [1, 4, 5],
            [0, 3, 4],
            [3, 4, 7],
            [1, 2, 3],
            [0, 1, 3],
            [1, 2, 5],
            [2, 5, 6],
            [5, 6, 7],
            [4, 5, 7],
            [2, 6, 7],
            [2, 3, 7],
        ]
    )

    return trimesh.Trimesh(vertices=vertices, faces=faces)


def generate_parameterized_geodesic(start_mesh, end_mesh, n_X=5, n_steps=3):
    """Generate a synthetic geodesic between two parameterized meshes.

    Importantly, start_mesh and end_mesh must have the same number
    of vertices and faces.

    This function generates a geodesic without noise.

    More precisely, this function generates a geodesic with:
    - initial mesh (intercept): start_mesh,
    - initial tangent vector (coef): end_mesh - start_mesh.

    Parameters
    ----------
    start_mesh : trimesh.Trimesh
        Mesh that represents the start of the geodesic.
    end_mesh : trimesh.Trimesh
        Mesh that represents the start of the geodesic.
    n_X : int
        Number of points to sample along the geodesic.
    n_steps : int
        Number of steps to use in the exponential map.

    Returns
    -------
    geodesic_points : torch.tensor, shape=[n_X, n_vertices, 3]
    faces : array-like, shape=[n_faces, 3]
    true_intercept : torch.tensor, shape=[n_vertices, 3]
    true_coef: torch.tensor, shape=[n_vertices, 3]

    Notes
    -----
    true_intercept and true_coef are useful for evaluating the
    performance of a regression model on this synthetic data.
    """
    SURFACE_SPACE = DiscreteSurfaces(faces=gs.array(start_mesh.faces))
    METRIC = ElasticMetric(
        space=SURFACE_SPACE,
        a0=default_config.a0,
        a1=default_config.a1,
        b1=default_config.b1,
        c1=default_config.c1,
        d1=default_config.d1,
        a2=default_config.a2,
    )
    METRIC.exp_solver = _ExpSolver(n_steps=n_steps)
    X = gs.arange(0, 1, 1 / n_X)

    initial_point = torch.tensor(start_mesh.vertices)
    end_point = torch.tensor(end_mesh.vertices)
    true_intercept = initial_point
    true_coef = initial_point - end_point

    geodesic = METRIC.geodesic(
        initial_point=true_intercept, initial_tangent_vec=true_coef
    )
    print("Geodesic function created. Computing points along geodesic...")
    geod = geodesic(X)
    print("Done.")
    return geod, start_mesh.faces, X, true_intercept, true_coef


def generate_unparameterized_geodesic(start_mesh, end_mesh, gpu_id=1):
    """Generate a synthetic geodesic between two unparameterized meshes.

    Parameters
    ----------
    start_mesh : trimesh.Trimesh
        Mesh that represents the start of the geodesic.
    end_mesh : trimesh.Trimesh
        Mesh that represents the end of the geodesic.
    gpu_id : int
        GPU to use for the computation.

    Returns
    -------
    geod : torch.tensor, shape=[n_X, n_vertices, 3]
        The geodesic. n_X is given through paramlist.
    F0 : torch.tensor, shape=[n_faces, 3]
        The faces of each mesh along the geodesic.
    """
    source = [start_mesh.vertices, start_mesh.faces]
    target = [end_mesh.vertices, end_mesh.faces]
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
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
        start=source,
        paramlist=default_config.paramlist,
        device=device,
    )
    return geod, F0


def add_geodesic_noise(space, y, dataset_name, noise_factor=0.01):
    """Generate synthetic data on the hypersphere or hyperboloid.

    Parameters
    ----------
    space: manifold on which the data lie
    y: points on manifold, torch.tensor, shape=[n_samples, dim of data]
    noise_std: float, standard deviation of noise
    """
    n_samples = len(y)

    if dataset_name == "synthetic_mesh":
        mesh_sequence_vertices = y
        diameter = data_utils.mesh_diameter(mesh_sequence_vertices[0])
        noise_std = noise_factor * diameter
    else:
        noise_std = noise_factor
        # Note: noise factor is a percentage of the "radius" of the manifold
        # which is 1 for hypersphere.

    # Generate normal noise
    normal_noise = np.random.normal(
        size=(n_samples, space.embedding_space.dim), scale=noise_std
    )

    normal_noise = gs.array(normal_noise)
    noise = space.to_tangent(normal_noise, base_point=y) / gs.pi / 2

    # Add noise
    y_noisy = space.metric.exp(noise, y)

    return y_noisy


def add_linear_noise(space, y, dataset_name, project_linear_noise, noise_factor=0.01):
    """Generate synthetic data on the hypersphere or hyperboloid.

    Note: data is "random".
    """
    if dataset_name == "synthetic_mesh":
        mesh_sequence_vertices = y
        diameter = data_utils.mesh_diameter(mesh_sequence_vertices[0])
        noise_std = noise_factor * diameter
    else:
        noise_std = abs(y[0] - y[-1]) * noise_factor

    # Generate normal noise
    normal_noise = np.random.normal(loc=gs.array(0.0), scale=noise_std, size=y.shape)

    y_noisy = y + normal_noise

    if project_linear_noise:
        y_noisy = space.projection(y_noisy)

    return y_noisy


def generate_benchmark_data(space, n_samples=1):
    """Generate a pair of random points on the hypersphere or hyperboloid.

    Parameters
    ----------
    X: torch.tensor, shape=[n_samples]
    y: points on manifold, torch.tensor, shape=[n_samples, dim]
    intercept: torch.tensor, shape=[dim]. "intercept" of distribution
        (named for purpose of regression)
    coef: torch.tensor, shape=[dim]. "coef" of distribution
    """
    gs.random.seed(0)  # TODO: make sure this creates same dataset every time
    # TODO: add this to funciton above

    # X = gs.random.rand(n_samples)
    X = gs.linspace(0, 1, n_samples)
    X -= gs.mean(X)

    random_euclidean_point = gs.random.rand(space.embedding_space.dim)
    intercept = space.projection(random_euclidean_point)
    print(f"intercept shape: {intercept.shape}")
    vector = 5.0 * gs.random.rand(space.embedding_space.dim)
    vector_norm = gs.linalg.norm(vector)
    if vector_norm < 1e-6:
        vector = vector + 0.5
    vector = vector / vector_norm
    coef = space.to_tangent(vector, base_point=intercept)
    print(f"coef shape: {coef.shape}")
    y = space.metric.exp(X[:, None] * coef, base_point=intercept)
    return X, y, intercept, coef


def fixed_hypersphere_data():
    """Return 50 X,y pairs on the hypersphere."""
    y = gs.array(
        [
            [0.9211, 0.3862, 0.0488],
            [0.7714, -0.4459, 0.4540],
            [0.2010, -0.5103, 0.8362],
            [0.6079, 0.1532, -0.7791],
            [0.6236, -0.4175, 0.6609],
            [0.9893, -0.1224, 0.0788],
            [-0.8530, -0.4718, -0.2231],
            [-0.2829, -0.7977, 0.5326],
            [0.4476, -0.5307, 0.7197],
            [-0.4776, -0.6260, 0.6164],
            [0.0258, -0.5652, 0.8246],
            [-0.7137, -0.6746, -0.1885],
            [-0.7635, -0.6181, 0.1872],
            [0.7201, 0.2879, 0.6313],
            [-0.4812, -0.5523, 0.6807],
            [-0.2552, -0.6861, 0.6813],
            [-0.7560, -0.5013, 0.4210],
            [-0.7903, -0.6120, 0.0280],
            [0.8168, -0.1109, 0.5661],
            [0.6022, -0.4212, 0.6782],
            [-0.5588, -0.7810, 0.2789],
            [0.2766, -0.2307, 0.9329],
            [-0.9762, -0.0191, -0.2161],
            [0.0685, -0.8817, 0.4667],
            [-0.4751, -0.8053, 0.3548],
            [0.4926, -0.4342, 0.7542],
            [0.2780, -0.4693, 0.8381],
            [0.1316, 0.2232, 0.9658],
            [0.8317, -0.2540, 0.4937],
            [-0.7069, -0.6559, -0.2646],
            [0.9435, -0.1042, 0.3147],
            [0.1821, -0.5250, 0.8314],
            [-0.4741, -0.5589, 0.6803],
            [-0.7416, -0.1336, -0.6574],
            [0.3860, -0.7009, 0.5998],
            [0.6603, 0.3948, 0.6389],
            [0.2312, -0.3444, 0.9099],
            [0.2332, -0.5857, 0.7763],
            [0.8866, 0.1957, 0.4190],
            [-0.7670, -0.1580, 0.6219],
            [0.0574, -0.9091, 0.4126],
            [-0.9239, -0.0320, -0.3813],
            [0.7252, 0.1223, 0.6776],
            [-0.2445, -0.7969, 0.5525],
            [-0.1327, -0.3877, 0.9122],
            [0.0704, -0.2249, 0.9718],
            [0.8862, 0.0940, 0.4537],
            [-0.8635, -0.3169, -0.3923],
            [0.9083, 0.3467, 0.2343],
            [0.7324, 0.4971, 0.4652],
        ]
    )
    X = gs.array(
        [
            0.4575,
            0.1953,
            -0.0532,
            0.4082,
            0.1325,
            0.2786,
            -0.3340,
            -0.1615,
            0.0688,
            -0.2243,
            -0.0597,
            -0.3358,
            -0.1573,
            0.1093,
            -0.0307,
            -0.0718,
            -0.1053,
            -0.3071,
            0.1525,
            0.2723,
            -0.3022,
            0.1642,
            -0.4028,
            0.0112,
            -0.2865,
            0.0457,
            0.0750,
            0.1694,
            0.2328,
            -0.2849,
            0.2973,
            0.1180,
            -0.3167,
            -0.3598,
            -0.0311,
            0.4049,
            0.0595,
            -0.0317,
            0.2835,
            -0.1446,
            -0.2404,
            -0.4711,
            0.2361,
            -0.1708,
            0.0257,
            -0.0462,
            0.3509,
            -0.4715,
            0.4847,
            0.3671,
        ]
    )
    return X, y


def fixed_hyperboloid_data():
    """Return 50 X,y pairs on the hyperboloid."""
    y = gs.array(
        [
            [1.2597, 0.6713, 0.3693],
            [1.1301, 0.4422, 0.2856],
            [1.0746, 0.3115, 0.2404],
            [1.2251, 0.6160, 0.3486],
            [1.1123, 0.4040, 0.2721],
            [1.1595, 0.5006, 0.3064],
            [1.0424, 0.2093, 0.2067],
            [1.0596, 0.2679, 0.2258],
            [1.0972, 0.3692, 0.2601],
            [1.0525, 0.2452, 0.2184],
            [1.0736, 0.3087, 0.2395],
            [1.0422, 0.2087, 0.2065],
            [1.0601, 0.2695, 0.2264],
            [1.1065, 0.3909, 0.2676],
            [1.0783, 0.3214, 0.2437],
            [1.0718, 0.3036, 0.2377],
            [1.0669, 0.2898, 0.2331],
            [1.0446, 0.2177, 0.2094],
            [1.1176, 0.4157, 0.2762],
            [1.1570, 0.4958, 0.3047],
            [1.0450, 0.2192, 0.2099],
            [1.1209, 0.4227, 0.2787],
            [1.0372, 0.1888, 0.2001],
            [1.0857, 0.3406, 0.2503],
            [1.0464, 0.2243, 0.2115],
            [1.0924, 0.3574, 0.2560],
            [1.0986, 0.3724, 0.2612],
            [1.1224, 0.4259, 0.2798],
            [1.1424, 0.4673, 0.2945],
            [1.0466, 0.2248, 0.2117],
            [1.1673, 0.5152, 0.3116],
            [1.1087, 0.3957, 0.2693],
            [1.0438, 0.2147, 0.2084],
            [1.0403, 0.2014, 0.2041],
            [1.0782, 0.3212, 0.2437],
            [1.2231, 0.6126, 0.3473],
            [1.0953, 0.3644, 0.2584],
            [1.0781, 0.3209, 0.2436],
            [1.1615, 0.5043, 0.3077],
            [1.0617, 0.2743, 0.2280],
            [1.0509, 0.2396, 0.2165],
            [1.0327, 0.1699, 0.1941],
            [1.1436, 0.4696, 0.2953],
            [1.0585, 0.2644, 0.2247],
            [1.0884, 0.3476, 0.2527],
            [1.0757, 0.3145, 0.2414],
            [1.1924, 0.5605, 0.3281],
            [1.0327, 0.1698, 0.1940],
            [1.2821, 0.7055, 0.3822],
            [1.2009, 0.5753, 0.3336],
        ]
    )
    X = gs.array(
        [
            0.4575,
            0.1953,
            -0.0532,
            0.4082,
            0.1325,
            0.2786,
            -0.3340,
            -0.1615,
            0.0688,
            -0.2243,
            -0.0597,
            -0.3358,
            -0.1573,
            0.1093,
            -0.0307,
            -0.0718,
            -0.1053,
            -0.3071,
            0.1525,
            0.2723,
            -0.3022,
            0.1642,
            -0.4028,
            0.0112,
            -0.2865,
            0.0457,
            0.0750,
            0.1694,
            0.2328,
            -0.2849,
            0.2973,
            0.1180,
            -0.3167,
            -0.3598,
            -0.0311,
            0.4049,
            0.0595,
            -0.0317,
            0.2835,
            -0.1446,
            -0.2404,
            -0.4711,
            0.2361,
            -0.1708,
            0.0257,
            -0.0462,
            0.3509,
            -0.4715,
            0.4847,
            0.3671,
        ]
    )
    return X, y
