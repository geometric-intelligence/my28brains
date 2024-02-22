"""Functions for parameterized regression."""

import os

import numpy as np

os.environ["GEOMSTATS_BACKEND"] = "pytorch"  # noqa: E402
import inspect

import geomstats.backend as gs
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

import H2_SurfaceMatch.utils.input_output as h2_io  # noqa: E402

from geomstats.geometry.discrete_surfaces import DiscreteSurfaces
# from src.regression.discrete_surfaces import DiscreteSurfaces
from src.regression.geodesic_regression import GeodesicRegression


def save_regression_results(
    dataset_name,
    y,
    X,
    space,
    true_coef,
    regr_intercept,
    regr_coef,
    duration_time,
    results_dir,
    config,
    linear_residuals=None,
    model=None,
    y_hat=None,
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
    if model is None:
        suffix = f"{dataset_name}"
    elif model == "linear":
        suffix = f"{dataset_name}_lr"
    elif model == "geodesic" and linear_residuals:
        suffix = f"{dataset_name}_gr_linear_residuals"
    else:
        suffix = f"{dataset_name}_gr_geodesic_residuals"
    true_intercept_path = os.path.join(results_dir, f"true_intercept_{suffix}")
    true_coef_path = os.path.join(results_dir, f"true_coef_{suffix}")
    regr_intercept_path = os.path.join(results_dir, f"regr_intercept_{suffix}")
    y_path = os.path.join(results_dir, f"y_{suffix}")
    X_path = os.path.join(results_dir, f"X_{suffix}")
    y_hat_path = os.path.join(results_dir, f"y_hat_{suffix}")
    duration_time_path = os.path.join(results_dir, f"duration_time_{suffix}")

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
    np.savetxt(X_path, X)
    np.savetxt(duration_time_path, duration_time)

    print("regr_coef.shape: ", regr_coef.shape)
    if len(regr_coef.shape) > 2:
        for i, coef in enumerate(regr_coef):
            regr_coef_path = os.path.join(results_dir, f"regr_coef_{suffix}_degree_{i}")
            np.savetxt(regr_coef_path, coef)


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
    use_cuda=True,
    device_id=1,
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
        use_cuda=use_cuda,
        device_id=device_id,
        embedding_space_dim=3 * len(y[0]),
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
    original_point_shape = y[0].shape

    print("y.shape: ", y.shape)
    print("original_point_shape: ", original_point_shape)
    print("X.shape: ", X.shape)

    y = gs.array(y.reshape((len(X), -1)))
    X = gs.array(X.reshape(len(X), -1))
    print("regression reshaped y.shape: ", y.shape)

    lr = LinearRegression()

    lr.fit(X, y)

    intercept_hat, coef_hat = lr.intercept_, lr.coef_

    if X.shape[1] > 1:
        coef_hat = coef_hat.reshape(
            X.shape[1], original_point_shape[0], original_point_shape[1]
        )

    else:
        coef_hat = coef_hat.reshape(original_point_shape)

    print("coef_hat.shape: ", coef_hat.shape)

    intercept_hat = intercept_hat.reshape(original_point_shape)

    intercept_hat = gs.array(intercept_hat)
    coef_hat = gs.array(coef_hat)

    return intercept_hat, coef_hat, lr


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
    original_point_shape = y[0].shape

    y = gs.array(y.reshape((len(X), -1)))
    X = gs.array(X.reshape(len(X), 1))

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)  # X_poly is a matrix of shape (len(X), degree + 1)
    # The extra row is filled with 1's, which is the "intercept" term.

    print("X_poly.shape: ", X_poly.shape)
    print("X_poly: ", X_poly)

    lr = LinearRegression()
    lr.fit(X_poly, y)

    intercept_hat, coef_hats = lr.intercept_, lr.coef_

    print("coef_hat.shape: ", coef_hats.shape)

    coef_hats = coef_hats.reshape(
        degree, original_point_shape[0], original_point_shape[1]
    )
    print("reshaped coef_hats.shape:", coef_hats.shape)

    # intercept_term = coef_hats[0] # note: this is essentially zero. ignore.
    # coef_hat_linear = coef_hats[0]
    # coef_hat_quadratic = coef_hats[1]

    # coef_hat_linear = coef_hat_linear.reshape(original_point_shape)
    # coef_hat_quadratic = coef_hat_quadratic.reshape(original_point_shape)
    intercept_hat = intercept_hat.reshape(original_point_shape)

    # coef_hat_linear = gs.array(coef_hat_linear)
    # coef_hat_quadratic = gs.array(coef_hat_quadratic)
    coef_hats = gs.array(coef_hats)
    intercept_hat = gs.array(intercept_hat)

    print("original_point_shape: ", original_point_shape)
    print("coef_hats.shape: ", coef_hats.shape)

    return intercept_hat, coef_hats, lr


def compute_R2(y, X, test_indices, train_indices):
    """Compute R2 score for linear regression.

    Parameters
    ----------
    X: list of X corresponding to y
    y: vertices of mesh sequence to be fit
        (flattened s.t. array dimension <= 2)
    lr: linear regression model

    Returns
    -------
    score_array: [adjusted R2 score, normal R2 score]
    """
    X_train = gs.array(X[train_indices])
    X_test = gs.array(X[test_indices])
    y_train = gs.array(y[train_indices])
    y_test = gs.array(y[test_indices])

    print("X_pred: ", X)
    print("X_train: ", X_train)
    print("X_test: ", X_test)

    # X_train = gs.array(X_train.reshape(len(X_train), len(X_train[0])))
    # X_test = gs.array(X_test.reshape(len(X_test), len(X_test[0])))
    X_train = gs.array(X_train.reshape(len(X_train), -1))
    X_test = gs.array(X_test.reshape(len(X_test), -1))
    y_train = gs.array(y_train.reshape((len(X_train), -1)))
    y_test = gs.array(y_test.reshape((len(X_test), -1)))

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred_for_lr = lr.predict(X_test)

    normal_r2_score = r2_score(y_test, y_pred_for_lr)

    train_sample_size = len(y_train)
    n_independent_variables = X_train.shape[1]
    print("train_sample_size (n): ", train_sample_size)
    print("n_independent_variables (p): ", n_independent_variables)

    Adj_r2 = 1 - (1 - normal_r2_score) * (train_sample_size - 1) / (
        train_sample_size - n_independent_variables - 1
    )

    print("Adjusted R2 score: ", Adj_r2)
    print("R2 score: ", normal_r2_score)
    score_array = np.array([Adj_r2, normal_r2_score])

    return score_array
