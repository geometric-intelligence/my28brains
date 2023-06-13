"""Speeds up parameterized geodesic regression with an intentional decimation scheme.

What this script does:
- Fetch a sequence of parameterized meshes, based on data specified in default_config.py
THEN
- Decimates the mesh_sequence to a LOW number of points.
- Performs geodesic regression on the decimated mesh_sequence.
- Uses decimated mesh slope and intercept as starting point for next regression.
REPEATS ABOVE STEPS UNTIL THE INTERCEPT IS CLOSE TO THE TRUE INTERCEPT.
- Compares the regression results to the true slope and intercept of the mesh_sequence.
"""

"""
Performs regression on parameterized meshes.

Mesh sequence chosen in default_config.py
Returns the slope and intercept of the regression fit.
"""
import os
import time

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs

import my28brains.datasets.utils as data_utils
import my28brains.default_config as default_config
import my28brains.parameterized_regression as parameterized_regression

my28brains_dir = default_config.my28brains_dir
synthetic_data_dir = default_config.synthetic_data_dir
parameterized_meshes_dir = default_config.parameterized_meshes_dir
data_dir = default_config.data_dir


mesh_sequence_vertices, mesh_faces, times, true_intercept, true_coef = data_utils.load(
    default_config
)

start_time = time.time()

print("\n--- Linear Regression ---")
linear_intercept_hat, linear_coef_hat = parameterized_regression.linear_regression(
    mesh_sequence_vertices, times
)


linear_duration_time = time.time() - start_time

print(">> Duration: ", linear_duration_time)

linear_intercept_err = gs.linalg.norm(linear_intercept_hat - true_intercept)
linear_coef_err = gs.linalg.norm(linear_coef_hat - true_coef)
print(">> Linear regression errors:")
print(f"On intercept: {linear_intercept_err}, on coef: {linear_coef_err}")

print("Saving results...")
parameterized_regression.save_regression_results(
    default_config.data_type,
    default_config.sped_up,
    mesh_sequence_vertices,
    mesh_faces,
    gs.array(true_coef),
    regression_intercept=linear_intercept_hat,
    regression_coef=linear_coef_hat,
    duration_time=linear_duration_time,
    regression_dir=default_config.linear_regression_dir,
)

# if (residual magnitude is too big... have max residual as a param):
# then do geodesic regression

print("\n--- Geodesic Regression ---")
intercept_hat, coef_hat = parameterized_regression.geodesic_regression(
    mesh_sequence_vertices,
    mesh_faces,
    times,
    tol=10000,
    intercept_hat_guess=linear_intercept_hat,
    coef_hat_guess=linear_coef_hat,
    initialization="warm_start",
)

duration_time = time.time() - start_time

print(">> Duration: ", duration_time)
print(">> Geodesic regression errors:")
print(f"On intercept: {linear_intercept_err}, on coef: {linear_coef_err}")

print("Saving results...")
parameterized_regression.save_regression_results(
    default_config.data_type,
    default_config.sped_up,
    mesh_sequence_vertices,
    mesh_faces,
    gs.array(true_coef),
    regression_intercept=intercept_hat,
    regression_coef=coef_hat,
    duration_time=duration_time,
    regression_dir=default_config.geodesic_regression_dir,
)

print("Evaluating the results")
