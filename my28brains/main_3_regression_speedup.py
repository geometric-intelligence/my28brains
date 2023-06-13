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

(
    lr_intercept_hat,
    lr_coef_hat,
) = parameterized_regression.linear_regression(mesh_sequence_vertices, times)

# if (residual magnitude is too big... have max residual as a param):
# then do geodesic regression

(intercept_hat, coef_hat,) = parameterized_regression.geodesic_regression(
    mesh_sequence_vertices,
    mesh_faces,
    times,
    tolerance=0.0001,
    intercept_hat_guess=lr_intercept_hat,
    coef_hat_guess=lr_coef_hat,
    regression_initialization="warm_start",
)

end_time = time.time()
duration_time = end_time - start_time

print("Duration: ", duration_time)

parameterized_regression.save_regression_results(
    default_config.data_type,
    default_config.sped_up,
    mesh_sequence_vertices,
    mesh_faces,
    gs.array(true_coef),
    regression_intercept=intercept_hat,
    regression_coef=coef_hat,
    duration_time=duration_time,
)
