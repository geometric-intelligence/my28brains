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
##################### Standard Imports #####################

import os
import subprocess
import sys
import time

import numpy as np
import torch
import trimesh
from sklearn.linear_model import LinearRegression

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs

import my28brains.default_config as default_config
import my28brains.parameterized_regression as parameterized_regression

##################### Set up paths and imports #####################

sys_dir = os.path.dirname(default_config.work_dir)
sys.path.append(sys_dir)
sys.path.append(default_config.h2_dir)

import H2_SurfaceMatch.utils.input_output  # noqa: E402
import H2_SurfaceMatch.utils.utils  # noqa: E402

my28brains_dir = default_config.my28brains_dir
synthetic_data_dir = default_config.synthetic_data_dir
parameterized_meshes_dir = default_config.parameterized_meshes_dir
data_dir = default_config.data_dir

##################### Regression Imports #####################

import datasets.synthetic
import geomstats.visualization as visualization
import matplotlib.pyplot as plt
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.learning.frechet_mean import FrechetMean, variance
from geomstats.learning.geodesic_regression import GeodesicRegression

import my28brains.default_config as default_config
import my28brains.discrete_surfaces as discrete_surfaces
from my28brains.discrete_surfaces import DiscreteSurfaces, ElasticMetric

# import geomstats.geometry.discrete_surfaces as discrete_surfaces
# from geomstats.geometry.discrete_surfaces import DiscreteSurfaces, ElasticMetric


##################### Regression Parameters #####################

data_type = default_config.data_type
sped_up = default_config.sped_up

if data_type == "synthetic":
    print("Using synthetic data")
    mesh_dir = synthetic_data_dir
    sphere_mesh = datasets.synthetic.generate_sphere_mesh()
    ellipsoid_mesh = datasets.synthetic.generate_ellipsoid_mesh()
    (
        original_mesh_sequence_vertices,
        original_mesh_faces,
        times,
        true_intercept,
        true_slope,
    ) = datasets.synthetic.generate_synthetic_parameterized_geodesic(
        sphere_mesh, ellipsoid_mesh
    )
    print("Original mesh_sequence vertices: ", original_mesh_sequence_vertices.shape)
    print("Original mesh faces: ", original_mesh_faces.shape)
    print("Times: ", times.shape)

elif data_type == "real":
    print("Using real data")
    mesh_dir = parameterized_meshes_dir
    raise (NotImplementedError)
    # in progress...
    # when i do this, i will most likely change main_2_mesh_parameterization to take in a list of meshes
    # and then call it here.


##################### Perform Regression #####################

# if sped_up:
#     (
#         intercept_hat,
#         coef_hat,
#     ) = parameterized_regression.perform_multi_res_geodesic_regression(
#         decimated_mesh_sequences,
#         decimated_faces,
#         tols,
#         times,
#         regression_initialization="warm_start",
#     )
# else:

n_times = len(original_mesh_sequence_vertices)  # 11
y = original_mesh_sequence_vertices.reshape((n_times, -1))
X = times.reshape((n_times, 1))

lr = LinearRegression().fit(X, y)

intercept_hat = lr.intercept_
coef_hat = lr.coef_

# start_time = time.time()
# (
#     intercept_hat,
#     coef_hat,
# ) = parameterized_regression.perform_single_res_parameterized_regression(
#     original_mesh_sequence_vertices,
#     original_mesh_faces,
#     times,
#     tolerance=0.0001,
#     intercept_hat_guess=None,
#     coef_hat_guess=None,
#     regression_initialization="warm_start",
# )

# end_time = time.time()
# duration_time = end_time - start_time

# print("Duration: ", duration_time)

##################### Calculate True Slope #####################

SURFACE_SPACE = DiscreteSurfaces(faces=original_mesh_faces)

METRIC = ElasticMetric(
    space=SURFACE_SPACE,
    a0=default_config.a0,
    a1=default_config.a1,
    b1=default_config.b1,
    c1=default_config.c1,
    d1=default_config.d1,
    a2=default_config.a2,
)

true_coef = METRIC.log(
    original_mesh_sequence_vertices[1], original_mesh_sequence_vertices[0]
)
true_coef = gs.array(true_slope)

print(f"inferred coef: {coef_hat}")
print(f"true coef: {true_coef}")
print(f"inferred intercept: {intercept_hat}")
print(f"true intercept: {true_intercept}")

print("diff coef: ", coef_hat - true_coef)
print("diff intercept: ", intercept_hat - true_intercept)

##################### Save Results #####################

# parameterized_regression.save_regression_results(
#     data_type,
#     sped_up,
#     original_mesh_sequence_vertices,
#     original_mesh_faces,
#     true_coef,
#     regression_intercept=intercept_hat,
#     regression_coef=coef_hat,
#     duration_time=1, #duration_time,
# )
