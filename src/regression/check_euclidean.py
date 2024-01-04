"""Evaluate whether the shape manifold is euclidean at a given scale."""

import os

os.environ["GEOMSTATS_BACKEND"] = "pytorch"  # noqa
import geomstats.backend as gs

import src.datasets.utils as data_utils


def subspace_test(mesh_sequence_vertices, X, tol_factor=0.001):
    """Test whether the manifold subspace where the data lie is euclidean.

    DISCLAIMER: NOT USED IN PAPER.

    For 10 random pairs of meshes, we calculate:
    - 1) the linear distance between them, and
    - 2) the geodesic distance between them.
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
            for t in X
        ]
    )

    mesh_sequence_diff = abs(geodesic - line)
    summed_mesh_sequence_diffs = sum(sum(sum(mesh_sequence_diff)))
    print("mesh_sequence_diff.shape", geodesic.shape)
    print("summed_mesh_sequence_diffs.shape", summed_mesh_sequence_diffs.shape)

    n_vertices = mesh_sequence_vertices[0].shape[0]
    normalized_mesh_sequence_diff = summed_mesh_sequence_diffs / (
        n_vertices * len(X) * 3
    )

    diff_tolerance = tol_factor * data_utils.mesh_diameter(mesh_sequence_vertices[0])
    euclidean_subspace = normalized_mesh_sequence_diff < diff_tolerance
    return euclidean_subspace, diff_tolerance


def euclidean_spread_rmsd(space, y, dataset_name):
    """Test whether the manifold is euclidean across the spread of the data.

    DISCLAIMER: NOT USED IN PAPER.

    Useful for testing whether linear regression will produce reasonable results
    as an approximation to geodesic regression.

    Parameters
    ----------
    space: the manifold on which the data lie.
    y: array of shape (n_samples, n_features)
        The data to be fit by linear regression.
    dataset_name: str
    n_comparison_points: int
        The number of points to use to compare the line and geodesic.

    Returns
    -------
    rmsd: float
        The root mean squared deviation between a line and geodesic between
        the first and last points in the data.

    Assumptions:
    - The first and last points in the data are the endpoints of a geodesic.
    """
    start_point = y[0]
    end_point = y[-1]

    true_coef = end_point - start_point
    geodesic_fn = space.metric.geodesic(
        initial_point=start_point, initial_tangent_vec=true_coef
    )  # ivp gives better results than bvp.

    n_comparison_points = 5
    geodesic = geodesic_fn(gs.linspace(0, 1, n_comparison_points))
    end_point = geodesic[-1]

    line = gs.array(
        [
            t * end_point + (1 - t) * start_point
            for t in gs.linspace(0, 1, n_comparison_points)
        ]
    )

    rmsd = gs.linalg.norm(line - geodesic) / n_comparison_points

    if dataset_name == "synthetic_mesh":
        diameter = data_utils.mesh_diameter(start_point)
        n_vertices = len(start_point)
        rmsd = rmsd / (diameter * n_vertices)

    return rmsd


def euclidean_noise_rmsd(space, y_noisy, y_noiseless, dataset_name):
    """Test whether the manifold is euclidean on the scale of the dataset noise.

    DISCLAIMER: NOT USED IN PAPER.

    Useful for testing whether geodesic regression with linear residuals will produce
    reasonable results as an approximation to geodesic regression.

    Parameters
    ----------
    space: the manifold on which the data lie.
    y_noisy: array of shape (n_samples, n_features)
        Synthetic data with noise added.
    y_noiseless: array of shape (n_samples, n_features)
        Synthetic data without noise added.
    dataset_name: str
    n_comparison_points: int
        The number of points to use to compare the line and geodesic.


    Returns
    -------
    summed_line_geod_rmsds: float
        The root mean squared deviation between a line and geodesic between
        a noisy datapoint and a noiseless datapoint, summed over all samples.
    """
    n_comparison_points = 5
    summed_line_geod_rmsds = 0

    if y_noiseless is None:
        start_point = y_noisy[0]
        end_point = y_noisy[-1]

        true_coef = end_point - start_point
        geodesic_fn = space.metric.geodesic(
            initial_point=start_point, initial_tangent_vec=true_coef
        )  # ivp gives better results than bvp.
        geodesic = geodesic_fn(gs.linspace(0, 1, n_comparison_points))
        y_noiseless = geodesic

    for i_point in range(len(y_noisy)):
        noisy_point = y_noisy[i_point]
        noiseless_point = y_noiseless[i_point]

        rmsd = euclidean_spread_rmsd(
            space, gs.array([noisy_point, noiseless_point]), dataset_name
        )

        summed_line_geod_rmsds += rmsd

    normalized_summed_line_geod_rmsds = summed_line_geod_rmsds / len(y_noisy)
    return normalized_summed_line_geod_rmsds
