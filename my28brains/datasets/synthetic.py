"""Generate meshes and synthetic geodesics."""

import os
import subprocess
import sys

import geomstats.backend as gs
import numpy as np
import torch
import trimesh

work_dir = os.getcwd()
my28brains_dir = os.path.join(work_dir, "my28brains")
h2_dir = os.path.join(work_dir, "H2_SurfaceMatch")
sys_dir = os.path.dirname(work_dir)
sys.path.append(sys_dir)
sys.path.append(h2_dir)
sys.path.append(my28brains_dir)

from geomstats.geometry.discrete_surfaces import (DiscreteSurfaces,
                                                  ElasticMetric)

import H2_SurfaceMatch.H2_match  # noqa: E402
import H2_SurfaceMatch.utils.input_output  # noqa: E402
import H2_SurfaceMatch.utils.utils  # noqa: E402
import my28brains.default_config as default_config


def generate_synthetic_mesh(mesh_type):
    """Generate a synthetic mesh.

    appropriate mesh types:
        sphere
        ellipsoid
        pill
    """
    if mesh_type == "sphere":
        return generate_sphere_mesh()
    elif mesh_type == "ellipsoid":
        return generate_ellipsoid_mesh()
    elif mesh_type == "pill":
        return generate_pill_mesh()
    else:
        raise ValueError(f"mesh_type {mesh_type} not recognized")


def generate_sphere_mesh():
    """Create a sphere trimesh."""
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=30.0)
    return sphere


def generate_ellipsoid_mesh():
    """Create an ellipsoid trimesh."""
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=30.0)
    # Create a scaling matrix for the semi-axes lengths
    scales = np.array([2, 2, 3])
    scale_matrix = np.diag(scales)
    scale_matrix = gs.array(scale_matrix)
    # Apply the scaling transformation to the mesh vertices
    scaled_vertices = sphere.vertices.dot(scale_matrix)
    # Create a new mesh with the scaled vertices
    ellipsoid = trimesh.Trimesh(vertices=scaled_vertices, faces=sphere.faces)
    return ellipsoid


def generate_pill_mesh():
    """Create a pill trimesh.

    Note that this mesh is not parameterized the same way as the other meshes.
    (i.e. sphere and ellipsoid are parameterized the same way, but pill is not)
    """
    pill = trimesh.creation.capsule(height=30.0, radius=10.0)
    return pill


def generate_synthetic_parameterized_geodesic(start_mesh, end_mesh, n_times=5):
    """Generate a synthetic geodesic between two parameterized meshes.

    Parameters
    ----------
    mesh1: a trimesh object
    mesh2: a trimesh object
    n_times: the number of points to sample along the geodesic
    note: mesh1 and mesh2 must have the same number of vertices and faces

    Returns
    -------
    geodesic_points: a torch tensor of shape (n_times, n_vertices, 3).
    true_intercept: a torch tensor of shape (n_vertices, 3)
    true_slope: a torch tensor of shape (n_vertices, 3)
    note: true_intercept and true_slope are useful for evaluating the
    performance of a regression model on this synthetic data.
    """
    SURFACE_SPACE = DiscreteSurfaces(faces=start_mesh.faces)
    METRIC = ElasticMetric(
        space=SURFACE_SPACE,
        a0=default_config.a0,
        a1=default_config.a1,
        b1=default_config.b1,
        c1=default_config.c1,
        d1=default_config.d1,
        a2=default_config.a2,
    )
    dim = 3
    n_vertices = start_mesh.vertices.shape[0]
    geodesic_points = gs.zeros(n_times, n_vertices, dim)
    times = gs.arange(0, 1, 1 / n_times)
    initial_point = gs.array(start_mesh.vertices)
    end_point = gs.array(end_mesh.vertices)
    geodesic = METRIC.geodesic(initial_point=initial_point, end_point=end_point)
    geodesic_points = geodesic(times)
    true_intercept = initial_point
    true_slope = METRIC.log(end_point, initial_point)
    return geodesic_points, start_mesh.faces, times, true_intercept, true_slope


def generate_unparameterized_synthetic_geodesic(start_mesh, end_mesh, n_times=5):
    """Generate a synthetic geodesic between two unparameterized meshes.

    Parameters
    ----------
    start_mesh: a trimesh object
    end_mesh: a trimesh object
    n_times: the number of points to sample along the geodesic
    """
    vertices_source = start_mesh.vertices
    faces_source = start_mesh.faces
    vertices_target = end_mesh.vertices
    faces_target = end_mesh.faces
    source = [vertices_source, faces_source]
    target = [vertices_target, faces_target]
    gpu_id = 1
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
    return geod
