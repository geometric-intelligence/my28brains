"""Functions for parameterized regression."""

import os

import numpy as np

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs
from geomstats.geometry.discrete_surfaces import DiscreteSurfaces, ElasticMetric
from sklearn.linear_model import LinearRegression

import H2_SurfaceMatch.utils.input_output  # noqa: E402
import H2_SurfaceMatch.utils.utils  # noqa: E402
import my28brains.default_config as default_config
from my28brains.geodesic_regression import GeodesicRegression

my28brains_dir = default_config.my28brains_dir
synthetic_data_dir = default_config.synthetic_data_dir
parameterized_meshes_dir = default_config.parameterized_meshes_dir
data_dir = default_config.data_dir


def save_regression_results(
    dataset_name,
    sped_up,
    true_intercept_vertices,
    true_intercept_faces,
    true_coef,
    regression_intercept,
    regression_coef,
    duration_time,
    regression_dir,
):
    """Save regression results to file.

    Parameters
    ----------
    dataset_name: string, either "synthetic" or "real"
    sped_up: boolean, whether or not the data was sped up
    true_intercept: numpy array, the true intercept
    true_coef: numpy array, the true slope
    regression_intercept: numpy array, the intercept calculated via regression
    regression_coef: numpy array, the slope calculated via regression
    duration_time: float, the duration of the regression
    """
    true_intercept_file_name = (
        "true_intercept_" + dataset_name + "_sped_up_" + str(sped_up)
    )
    true_slope_file_name = "true_slope_" + dataset_name + "_sped_up_" + str(sped_up)
    regression_intercept_file_name = (
        "regression_intercept_" + dataset_name + "_sped_up_" + str(sped_up)
    )
    regression_slope_file_name = (
        "regression_slope_" + dataset_name + "_sped_up_" + str(sped_up)
    )

    true_intercept_path = os.path.join(regression_dir, true_intercept_file_name)
    true_slope_path = os.path.join(regression_dir, true_slope_file_name)
    regression_intercept_path = os.path.join(
        regression_dir, regression_intercept_file_name
    )
    regression_slope_path = os.path.join(regression_dir, regression_slope_file_name)

    true_intercept_path = os.path.join(regression_dir, true_intercept_file_name)
    true_slope_path = os.path.join(regression_dir, true_slope_file_name)
    regression_intercept_path = os.path.join(
        regression_dir, regression_intercept_file_name + str(duration_time)
    )
    regression_slope_path = os.path.join(
        regression_dir, regression_slope_file_name + str(duration_time)
    )

    # NOTE: is t = 0 the intercept? let's check this if things aren't working.

    H2_SurfaceMatch.utils.input_output.save_data(
        true_intercept_path,
        ".ply",
        gs.array(true_intercept_vertices).numpy(),
        gs.array(true_intercept_faces).numpy(),
    )
    H2_SurfaceMatch.utils.input_output.save_data(
        regression_intercept_path,
        ".ply",
        gs.array(regression_intercept).numpy(),
        gs.array(true_intercept_faces).numpy(),
    )

    np.savetxt(true_slope_path, true_coef)
    np.savetxt(regression_slope_path, regression_coef)


def create_decimated_mesh_sequence_list(
    original_mesh_sequence_vertices, original_mesh_faces
):
    """Create a list of decimated meshes from a list of original meshes.

    The original mesh sequence is first in this list. The second mesh is slightly
    decimated, and the last mesh is very decimated (very few vertices).
    """
    decimated_geodesics_list = (
        []
    )  # all the decimated geodesics for the geod regr. (0 = original mesh)
    mesh_faces_list = (
        []
    )  # all the decimated mesh faces for the geod regr. (0 = original mesh)

    # TEMP CHANGE
    decimated_geodesics_list.append(original_mesh_sequence_vertices)

    # mesh_seq_dict = {
    #     f"/{i_decimation}": my_decimation_function(mesh, i_decimation),
    #     for i_decimation in range(1, default_config.n_decimations+1)
    # }

    # TODO: implement mesh dictionary so that i don't have to keep track of order
    # TODO: change geodesic --> mesh_sequence
    # mesh_seq_dict = {}

    # TODO: print "we are going to decimate the mesh by a factor of 2, 4, 8, ..."

    for i_decimation in range(
        1, default_config.n_decimations + 1
    ):  # reverse(range(n_decimations))
        n_faces_after_decimation = int(
            original_mesh_faces.shape[0]
            / (i_decimation**default_config.regression_decimation_factor_step)
        )

        assert n_faces_after_decimation > 2
        one_decimated_geodesic = []
        for one_mesh in original_mesh_sequence_vertices:
            [
                one_decimated_mesh_vertices,
                decimated_faces,
            ] = H2_SurfaceMatch.utils.utils.decimate_mesh(  # noqa E231
                one_mesh, original_mesh_faces, n_faces_after_decimation
            )
            one_decimated_geodesic.append(one_decimated_mesh_vertices)
        one_decimated_geodesic = gs.array(one_decimated_geodesic)
        decimated_geodesics_list.append(one_decimated_geodesic)
        mesh_faces_list.append(decimated_faces)

        # mesh_seq_dict[f"/{i_decimation}"] = one_decimated_geodesic

    # NOTE: moved mesh_faces_list.append(original_mesh_faces) to after the loop

    # Note: decimated_mesh_sequences must remain a list. It is not a numpy array.

    return decimated_geodesics_list, mesh_faces_list


def geodesic_regression(
    mesh_sequence,
    mesh_faces,
    times,
    tol,
    intercept_hat_guess,
    coef_hat_guess,
    initialization="warm_start",
):
    """Perform regression on parameterized meshes.

    Parameters
    ----------
    mesh_sequence: list of vertices of meshes.
    EACH MESH is a numpy array of shape (n, 3)
    mesh_faces: numpy array of shape (m, 3)
    where m is the number of faces
    times: list of times corresponding to mesh_sequence
    intercept_hat_guess: initial guess for intercept of regression fit
    coef_hat_guess: initial guess for slope of regression fit

    Returns
    -------
    intercept_hat: intercept of regression fit
    coef_hat: slope of regression fit
    """
    SURFACE_SPACE = DiscreteSurfaces(faces=mesh_faces)

    METRIC = ElasticMetric(
        space=SURFACE_SPACE,
        a0=default_config.a0,
        a1=default_config.a1,
        b1=default_config.b1,
        c1=default_config.c1,
        d1=default_config.d1,
        a2=default_config.a2,
    )

    # maxiter was 100
    # method was riemannian
    gr = GeodesicRegression(
        SURFACE_SPACE,
        metric=METRIC,
        center_X=False,
        method="extrinsic",
        max_iter=5,
        init_step_size=0.1,
        tol=tol,
        verbose=True,
        initialization=initialization,
    )

    if intercept_hat_guess is None:
        intercept_hat_guess = mesh_sequence[0]
    elif intercept_hat_guess.shape != mesh_sequence[0].shape:
        raise ValueError(
            "intercept_hat_guess must be None or "
            "have the same shape as mesh_sequence[0]"
        )

    if coef_hat_guess is None:
        coef_hat_guess = mesh_sequence[1] - mesh_sequence[0]

    # NOTE: THIS IS BUGGING on second iteration
    # coeff_hat_guess = METRIC.log(mesh_sequence[1], mesh_sequence[0])

    gr.intercept_ = intercept_hat_guess
    gr.coef_ = coef_hat_guess

    print("Intercept guess: ", gr.intercept_.shape)
    print("Coef guess: ", gr.coef_.shape)

    # times = gs.reshape(times, (len(times), 1))

    gr.fit(times, mesh_sequence, compute_training_score=False)

    intercept_hat, coef_hat = gr.intercept_, gr.coef_

    return intercept_hat, coef_hat


def linear_regression(mesh_sequence_vertices, times):
    """Perform linear regression on parameterized meshes.

    Parameters
    ----------
    mesh_sequence_vertices: vertices of mesh sequence to be fit
    times: list of times corresponding to mesh_sequence_vertices

    Returns
    -------
    intercept_hat: intercept of regression fit
    coef_hat: slope of regression fit
    """
    original_mesh_shape = mesh_sequence_vertices[0].shape

    mesh_sequence_vertices = mesh_sequence_vertices.reshape((len(times), -1))

    times = np.reshape(times, (len(times), 1))

    lr = LinearRegression()

    lr.fit(times, mesh_sequence_vertices)

    intercept_hat, coef_hat = lr.intercept_, lr.coef_

    intercept_hat = intercept_hat.reshape(original_mesh_shape)
    coef_hat = coef_hat.reshape(original_mesh_shape)

    intercept_hat = gs.array(intercept_hat)
    coef_hat = gs.array(coef_hat)

    return intercept_hat, coef_hat
