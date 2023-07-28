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

import H2_SurfaceMatch.H2_match  # noqa: E402
import H2_SurfaceMatch.utils.input_output  # noqa: E402
import H2_SurfaceMatch.utils.utils  # noqa: E402
import my28brains.default_config as default_config


def generate_mesh(mesh_type, n_subdivisions=None, ellipsoid_dims=[2, 2, 3]):
    """Generate a synthetic mesh.

    Parameters
    ----------
    mesh_type : str, {"sphere", "ellipsoid", "pill", "cube"}
        Type of mesh to generate.
    n_subdivisions : int
        How many times to subdivide the mesh (from trimesh).
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


def generate_parameterized_geodesic(start_mesh, end_mesh, n_times=5, n_steps=3):
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
    n_times : int
        Number of points to sample along the geodesic.
    n_steps : int
        Number of steps to use in the exponential map.

    Returns
    -------
    geodesic_points : torch.tensor, shape=[n_times, n_vertices, 3]
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
    times = gs.arange(0, 1, 1 / n_times)

    initial_point = torch.tensor(start_mesh.vertices)
    end_point = torch.tensor(end_mesh.vertices)
    true_intercept = initial_point
    true_coef = initial_point - end_point

    geodesic = METRIC.geodesic(
        initial_point=true_intercept, initial_tangent_vec=true_coef
    )
    print("Geodesic function created. Computing points along geodesic...")
    geod = geodesic(times)
    print("Done.")
    return geod, start_mesh.faces, times, true_intercept, true_coef


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
    geod : torch.tensor, shape=[n_times, n_vertices, 3]
        The geodesic. n_times is given through paramlist.
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
