"""Functions for parameterized regression."""

import os

import numpy as np

os.environ["GEOMSTATS_BACKEND"] = "pytorch"  # noqa: E402
import geomstats.backend as gs
from geomstats.geometry.discrete_surfaces import DiscreteSurfaces
from sklearn.linear_model import LinearRegression

import H2_SurfaceMatch.utils.input_output as h2_io  # noqa: E402
import my28brains.default_config as default_config
from my28brains.regression.geodesic_regression import GeodesicRegression


def save_regression_results(
    dataset_name,
    y,
    space,
    true_coef,
    regr_intercept,
    regr_coef,
    duration_time,
    results_dir,
    model,
    linear_residuals,
    y_hat=None,
):
    """Save regression results to files.

    Parameters
    ----------
    dataset_name: string, either "synthetic_mesh" or "real_mesh"
    y: input data given to regression (points on manifold)
    true_intercept: numpy array, the true intercept
    true_coef: numpy array, the true slope
    regr_intercept: numpy array, the intercept calculated via regression
    regr_coef: numpy array, the slope calculated via regression
    duration_time: float, the duration of the regression
    model: linear regression or geodesic regression
    linear_residuals: boolean, whether geodesic regression was performed
        with linear residuals
    results_directory: string, the directory in which to save the results
    y_hat: numpy array, the y values predicted by the regression model.
    """
    if model == "linear":
        suffix = f"{dataset_name}_lr"
    if linear_residuals:
        suffix = f"{dataset_name}_gr_linear_residuals"
    else:
        suffix = f"{dataset_name}_gr_geodesic_residuals"
    true_intercept_path = os.path.join(results_dir, f"true_intercept_{suffix}")
    true_coef_path = os.path.join(results_dir, f"true_coef_{suffix}")
    regr_intercept_path = os.path.join(
        results_dir, f"regr_intercept_{suffix}_{duration_time}"
    )
    regr_coef_path = os.path.join(results_dir, f"regr_coef_{suffix}_{duration_time}")
    y_path = os.path.join(results_dir, f"y_{suffix}")
    y_hat_path = os.path.join(results_dir, f"y_hat_{suffix}")

    if dataset_name == "synthetic_mesh" or dataset_name == "real_mesh":
        faces = gs.array(space.faces).numpy()
        mesh_sequence_vertices = y
        h2_io.save_data(
            true_intercept_path,
            ".ply",
            gs.array(mesh_sequence_vertices[0]).numpy(),
            faces,
        )
        h2_io.save_data(
            regr_intercept_path,
            ".ply",
            gs.array(regr_intercept).numpy(),
            faces,
        )

        # HACK ALERT: uses the plotGeodesic function to plot
        # the original mesh sequence, which is not a geodesic
        h2_io.plotGeodesic(
            geod=gs.array(mesh_sequence_vertices).detach().numpy(),
            F=faces,
            stepsize=default_config.stepsize[dataset_name],
            file_name=y_path,
        )

        if y_hat is not None:
            h2_io.plotGeodesic(
                geod=gs.array(y_hat).detach().numpy(),
                F=faces,
                stepsize=default_config.stepsize[dataset_name],
                file_name=y_hat_path,
            )

    np.savetxt(true_coef_path, true_coef)
    np.savetxt(regr_coef_path, regr_coef)


def fit_geodesic_regression(
    y,
    space,
    X,
    tol,
    intercept_hat_guess,
    coef_hat_guess,
    initialization="warm_start",
    linear_residuals=False,
    # device = "cuda:0",
):
    """Perform regression on parameterized meshes or benchmark data.

    Parameters
    ----------
    y:
        for meshes- list of vertices of meshes.
        for benchmark- list of points
    EACH MESH is a numpy array of shape (n, 3)
    space: space on which to perform regression
    X: list of X corresponding to y
    intercept_hat_guess: initial guess for intercept of regression fit
    coef_hat_guess: initial guess for slope of regression fit
    tol: tolerance for geodesic regression. If none logged, value 0.001.

    Returns
    -------
    intercept_hat: intercept of regression fit
    coef_hat: slope of regression fit
    """
    print(f"initialization: {initialization}")
    print(f"linear_residuals: {linear_residuals}")

    # maxiter was 100
    # method was riemannian
    gr = GeodesicRegression(
        space,
        center_X=False,
        method="extrinsic",
        # tol=tol,
        compute_training_score=False,
        verbose=True,
        initialization=initialization,
        linear_residuals=linear_residuals,
    )

    if intercept_hat_guess is None:
        intercept_hat_guess = gs.array(y[0])  # .to(device = device)
    elif intercept_hat_guess.shape != y[0].shape:
        raise ValueError("intercept_hat_guess must be None or have y[0].shape")

    if coef_hat_guess is None:
        coef_hat_guess = gs.array(y[1] - y[0])  # .to(device = device)

    gr.intercept_ = intercept_hat_guess
    gr.coef_ = coef_hat_guess

    print("Intercept guess: ", gr.intercept_.shape)
    print("Coef guess: ", gr.coef_.shape)

    gr.fit(gs.array(X), gs.array(y))

    intercept_hat, coef_hat = gr.intercept_, gr.coef_

    return intercept_hat, coef_hat, gr


def fit_linear_regression(y, X):  # , device = "cuda:0"):
    """Perform linear regression on parameterized meshes.

    Parameters
    ----------
    y:
        for meshes: vertices of mesh sequence to be fit
        for benchmark: points to be fit
    X: list of X corresponding to y

    Returns
    -------
    intercept_hat: intercept of regression fit
    coef_hat: slope of regression fit
    """
    original_y_shape = y[0].shape

    print("y.shape: ", y.shape)
    print("X.shape: ", X.shape)

    y = gs.array(y.reshape((len(X), -1)))
    print("y.shape: ", y.shape)

    X = gs.array(X.reshape(len(X), 1))

    lr = LinearRegression()

    lr.fit(X, y)

    intercept_hat, coef_hat = lr.intercept_, lr.coef_

    intercept_hat = intercept_hat.reshape(original_y_shape)
    coef_hat = coef_hat.reshape(original_y_shape)

    intercept_hat = gs.array(intercept_hat)
    coef_hat = gs.array(coef_hat)

    return intercept_hat, coef_hat, lr
