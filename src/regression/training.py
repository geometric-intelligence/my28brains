"""Functions for parameterized regression."""

import os

import numpy as np

os.environ["GEOMSTATS_BACKEND"] = "pytorch"  # noqa: E402
import inspect

import geomstats.backend as gs
from geomstats.geometry.discrete_surfaces import DiscreteSurfaces
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

import H2_SurfaceMatch.utils.input_output as h2_io  # noqa: E402
import src.import_project_config as pc
from src.regression.geodesic_regression import GeodesicRegression


def save_regression_results(
    dataset_name,
    y,
    X,
    space,
    true_coef,
    regr_intercept,
    regr_coef,
    results_dir,
    model,
    linear_residuals,
    y_hat=None,
    config=None,
    lr_score_array=None,
):
    """Save regression results to files.

    Parameters
    ----------
    dataset_name: string, either "synthetic_mesh" or "menstrual_mesh"
    y: input data given to regression (points on manifold)
    true_intercept: numpy array, the true intercept
    true_coef: numpy array, the true slope
    regr_intercept: numpy array, the intercept calculated via regression
    regr_coef: numpy array, the slope calculated via regression
    model: linear regression or geodesic regression
    linear_residuals: boolean, whether geodesic regression was performed
        with linear residuals
    results_directory: string, the directory in which to save the results
    y_hat: numpy array, the y values predicted by the regression model.
    """
    if config is None:
        calling_script_path = os.path.abspath(inspect.stack()[1].filename)
        config = pc.import_default_config(calling_script_path)
    if model == "linear":
        suffix = f"{dataset_name}_lr"
    if model == "geodesic" and linear_residuals:
        suffix = f"{dataset_name}_gr_linear_residuals"
    else:
        suffix = f"{dataset_name}_gr_geodesic_residuals"
    true_intercept_path = os.path.join(results_dir, f"true_intercept_{suffix}")
    true_coef_path = os.path.join(results_dir, f"true_coef_{suffix}")
    regr_intercept_path = os.path.join(results_dir, f"regr_intercept_{suffix}")
    regr_coef_path = os.path.join(results_dir, f"regr_coef_{suffix}")
    y_path = os.path.join(results_dir, f"y_{suffix}")
    X_path = os.path.join(results_dir, f"X_{suffix}")
    y_hat_path = os.path.join(results_dir, f"y_hat_{suffix}")

    if dataset_name == "synthetic_mesh" or dataset_name == "menstrual_mesh":
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

        if not os.path.exists(y_path):
            os.makedirs(y_path)

        for i, mesh in enumerate(mesh_sequence_vertices):
            mesh_path = os.path.join(y_path, f"mesh_{i}")
            h2_io.save_data(
                mesh_path,
                ".ply",
                gs.array(mesh).numpy(),
                faces,
            )

        # HACK ALERT: uses the plotGeodesic function to plot
        # the original mesh sequence, which is not a geodesic
        # h2_io.plotGeodesic( # plots and saves
        #     geod=gs.array(mesh_sequence_vertices).detach().numpy(),
        #     F=faces,
        #     stepsize=config.stepsize[dataset_name],
        #     file_name=y_path,
        # )

        if y_hat is not None:
            if not os.path.exists(y_hat_path):
                os.makedirs(y_hat_path)

            for i, mesh in enumerate(y_hat):
                mesh_path = os.path.join(y_hat_path, f"mesh_{i}")
                h2_io.save_data(
                    mesh_path,
                    ".ply",
                    gs.array(mesh).numpy(),
                    faces,
                )

            if lr_score_array is not None:
                score_path = os.path.join(y_hat_path, f"R2_score_{suffix}")
                np.savetxt(score_path, lr_score_array)

    np.savetxt(true_coef_path, true_coef)
    np.savetxt(regr_coef_path, regr_coef)
    np.savetxt(X_path, X)


def fit_geodesic_regression(
    y,
    space,
    X,
    tol,
    intercept_hat_guess,
    coef_hat_guess,
    initialization="warm_start",
    linear_residuals=False,
    compute_iterations=False,
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
    # print(f"initialization: {initialization}")
    # print(f"linear_residuals: {linear_residuals}")

    # maxiter was 100
    # method was riemannian
    gr = GeodesicRegression(
        space,
        center_X=False,
        method="extrinsic",
        compute_training_score=False,
        verbose=True,
        tol=tol,
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

    # compute R2 score
    X_pred = X
    X_pred_lr = gs.array(X_pred.reshape(len(X_pred), 1))
    y_pred_for_lr = lr.predict(X_pred_lr)
    y_pred_for_lr = y_pred_for_lr.reshape(y.shape)

    normal_r2_score = r2_score(y, y_pred_for_lr)

    Adj_r2 = 1 - (1 - normal_r2_score) * (len(y) - 1) / (len(y) - X.shape[1] - 1)

    print("Adjusted R2 score: ", Adj_r2)
    print("R2 score: ", normal_r2_score)
    score_array = np.array([Adj_r2, normal_r2_score])

    return intercept_hat, coef_hat, lr, score_array


def fit_polynomial_regression(y, X, degree=2):
    """Perform polynomial regression on parameterized meshes.

    Also used to perform multiple linear regression.

    Parameters
    ----------
    y: vertices of mesh sequence to be fit
    X: list of X corresponding to y

    Returns
    -------
    intercept_hat: intercept of regression fit
    coef_hat: slope of regression fit
    """
    original_y_shape = y[0].shape

    y = gs.array(y.reshape((len(X), -1)))
    X = gs.array(X.reshape(len(X), 1))

    poly = PolynomialFeatures(degree=degree)

    # TODO: is that really what we want?
    X_poly = poly.fit_transform(X)
    lr = LinearRegression()

    lr.fit(X_poly, y)

    intercept_hat, coef_hat = lr.intercept_, lr.coef_

    intercept_hat = intercept_hat.reshape(original_y_shape)
    coef_hat = coef_hat.reshape(original_y_shape)

    intercept_hat = gs.array(intercept_hat)
    coef_hat = gs.array(coef_hat)

    return intercept_hat, coef_hat, lr
