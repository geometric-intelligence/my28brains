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
    regress_dir,
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
    y_hat: numpy array, the y values predicted by the regression model.
    """
    suffix = f"{dataset_name}"
    true_intercept_path = os.path.join(regress_dir, f"true_intercept_{suffix}")
    true_coef_path = os.path.join(regress_dir, f"true_coef_{suffix}")
    regr_intercept_path = os.path.join(
        regress_dir, f"regr_intercept_{suffix}_{duration_time}"
    )
    regr_coef_path = os.path.join(regress_dir, f"regr_coef_{suffix}_{duration_time}")
    y_path = os.path.join(regress_dir, f"y_{suffix}")
    y_hat_path = os.path.join(regress_dir, f"y_hat_{suffix}")

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
    mesh_sequence,
    space,
    X,
    tol,
    intercept_hat_guess,
    coef_hat_guess,
    initialization="warm_start",
    geodesic_residuals=False,
    # device = "cuda:0",
):
    """Perform regression on parameterized meshes.

    Parameters
    ----------
    mesh_sequence: list of vertices of meshes.
    EACH MESH is a numpy array of shape (n, 3)
    space: space on which to perform regression
    X: list of X corresponding to mesh_sequence
    intercept_hat_guess: initial guess for intercept of regression fit
    coef_hat_guess: initial guess for slope of regression fit

    Returns
    -------
    intercept_hat: intercept of regression fit
    coef_hat: slope of regression fit
    """
    print(f"initialization: {initialization}")
    print(f"geodesic_residuals: {geodesic_residuals}")

    # maxiter was 100
    # method was riemannian
    gr = GeodesicRegression(
        space,
        metric=space.metric,
        center_X=False,
        method="extrinsic",
        max_iter=5,
        init_step_size=0.1,
        tol=tol,
        verbose=True,
        initialization=initialization,
        geodesic_residuals=geodesic_residuals,
    )

    if intercept_hat_guess is None:
        intercept_hat_guess = gs.array(mesh_sequence[0])  # .to(device = device)
    elif intercept_hat_guess.shape != mesh_sequence[0].shape:
        raise ValueError(
            "intercept_hat_guess must be None or have mesh_sequence[0].shape"
        )

    if coef_hat_guess is None:
        coef_hat_guess = gs.array(
            mesh_sequence[1] - mesh_sequence[0]
        )  # .to(device = device)

    # NOTE: THIS IS BUGGING on second iteration
    # coeff_hat_guess = METRIC.log(mesh_sequence[1], mesh_sequence[0])

    gr.intercept_ = intercept_hat_guess
    gr.coef_ = coef_hat_guess

    print("Intercept guess: ", gr.intercept_.shape)
    print("Coef guess: ", gr.coef_.shape)

    gr.fit(gs.array(X), gs.array(mesh_sequence), compute_training_score=False)

    intercept_hat, coef_hat = gr.intercept_, gr.coef_

    return intercept_hat, coef_hat, gr


def fit_linear_regression(mesh_sequence_vertices, X):  # , device = "cuda:0"):
    """Perform linear regression on parameterized meshes.

    Parameters
    ----------
    mesh_sequence_vertices: vertices of mesh sequence to be fit
    X: list of X corresponding to mesh_sequence_vertices

    Returns
    -------
    intercept_hat: intercept of regression fit
    coef_hat: slope of regression fit
    """
    original_mesh_shape = mesh_sequence_vertices[0].shape

    print("mesh_sequence_vertices.shape: ", mesh_sequence_vertices.shape)
    print("X.shape: ", X.shape)

    mesh_sequence_vertices = gs.array(mesh_sequence_vertices.reshape((len(X), -1)))
    print("mesh_sequence_vertices.shape: ", mesh_sequence_vertices.shape)

    X = gs.array(X.reshape(len(X), 1))

    lr = LinearRegression()

    lr.fit(X, mesh_sequence_vertices)

    intercept_hat, coef_hat = lr.intercept_, lr.coef_

    intercept_hat = intercept_hat.reshape(original_mesh_shape)
    coef_hat = coef_hat.reshape(original_mesh_shape)

    intercept_hat = gs.array(intercept_hat)
    coef_hat = gs.array(coef_hat)

    return intercept_hat, coef_hat, lr
