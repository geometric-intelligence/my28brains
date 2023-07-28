"""Evaluate whether the shape manifold is euclidean at a given scale."""

import os

os.environ["GEOMSTATS_BACKEND"] = "pytorch"  # noqa: E402
import geomstats.backend as gs

import my28brains.datasets.utils as data_utils


def subspace_test(mesh_sequence_vertices, times, tol_factor=0.001):
    """Test whether the manifold subspace where the data lie is euclidean.

    For 10 random pairs of meshes, we calculate 1) the linear distance
    between them, and 2) the geodesic distance between them.
    If the manifold is euclidean, these two distances should be the same.
    If the manifold is not euclidean, they will be different.

    We calculate the median of the ratio of the two distances and use it
    to determine whether the manifold is approximately euclidean.

    If the manifold is approximately euclidean, linear regression
    will return a reasonable result.

    Parameters
    ----------
    mesh_sequence_vertices: vertices of mesh sequence
    mesh_sequence_faces: faces of mesh sequence

    Returns
    -------
    euclidean_subspace_via_ratio: boolean, whether or not the manifold is euclidean
        based on the ratio of linear distance to geodesic distance
    euclidean_subspace_via_diffs: boolean, whether or not the manifold is euclidean
        based on the difference between linear distance and geodesic distance,
        compared to a tolerance that utilizes the size of the mesh and number of
        vertices.
    """
    # HACK ALERT: assumes that data sequence is a geodesic
    geodesic = gs.array(mesh_sequence_vertices)

    line = gs.array(
        [
            t * mesh_sequence_vertices[0] + (1 - t) * mesh_sequence_vertices[-1]
            for t in times
        ]
    )

    mesh_sequence_diff = abs(geodesic - line)
    summed_mesh_sequence_diffs = sum(sum(sum(mesh_sequence_diff)))
    print(f"mesh_sequence_diff.shape", geodesic.shape)
    print(f"summed_mesh_sequence_diffs.shape", summed_mesh_sequence_diffs.shape)

    n_vertices = mesh_sequence_vertices[0].shape[0]
    normalized_mesh_sequence_diff = summed_mesh_sequence_diffs / (
        n_vertices * len(times) * 3
    )

    diff_tolerance = tol_factor * data_utils.mesh_diameter(mesh_sequence_vertices[0])
    euclidean_subspace = normalized_mesh_sequence_diff < diff_tolerance
    return euclidean_subspace, diff_tolerance
