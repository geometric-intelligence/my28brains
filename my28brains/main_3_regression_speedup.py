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

import geomstats.visualization as visualization
import matplotlib.pyplot as plt
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.learning.frechet_mean import FrechetMean, variance
from geomstats.learning.geodesic_regression import GeodesicRegression

import data.synthetic_data.generate_syntetic_geodesics as generate_syntetic_geodesics
import my28brains.default_config as default_config
import my28brains.discrete_surfaces as discrete_surfaces
from my28brains.discrete_surfaces import DiscreteSurfaces, ElasticMetric

# import geomstats.geometry.discrete_surfaces as discrete_surfaces
# from geomstats.geometry.discrete_surfaces import DiscreteSurfaces, ElasticMetric


##################### Regression Parameters #####################

data_type = default_config.data_type
sped_up = default_config.sped_up

##################### Load Data (or create synthetic data) #####################

if data_type == "synthetic":
    print("Using synthetic data")
    synthetic_mesh_sequence_dir = default_config.synthetic_mesh_sequence_dir
    start_shape_dir = default_config.start_shape_dir
    end_shape_dir = default_config.end_shape_dir
    start_shape = default_config.start_shape
    end_shape = default_config.end_shape
    n_times = default_config.n_times

    if data_type == "synthetic":

        if not os.path.exists(synthetic_mesh_sequence_dir):
            print(
                f"Creating synthetic geodesic with start_mesh: {start_shape}, end_mesh: {end_shape}, and n_times: {n_times}"
            )

            if not os.path.exists(start_shape_dir):
                print(f"Creating {start_shape} mesh in {start_shape_dir}")
                start_mesh = generate_syntetic_geodesics.generate_synthetic_mesh(
                    start_shape
                )
                # start_mesh = generate_syntetic_geodesics.generate_sphere_mesh()
                start_mesh_vertices = start_mesh.vertices
                start_mesh_faces = start_mesh.faces

                os.makedirs(start_shape_dir)
                np.save(
                    os.path.join(start_shape_dir, "vertices.npy"), start_mesh_vertices
                )
                np.save(os.path.join(start_shape_dir, "faces.npy"), start_mesh_faces)
            else:
                print(
                    f"{start_shape} mesh already exists in {start_shape_dir}. Loading now."
                )
                start_mesh_vertices = np.load(
                    os.path.join(start_shape_dir, "vertices.npy")
                )
                start_mesh_faces = np.load(os.path.join(start_shape_dir, "faces.npy"))
                start_mesh = trimesh.Trimesh(
                    vertices=start_mesh_vertices, faces=start_mesh_faces
                )

            if not os.path.exists(end_shape_dir):
                print(f"Creating {end_shape} mesh in {end_shape_dir}")
                end_mesh = generate_syntetic_geodesics.generate_synthetic_mesh(
                    end_shape
                )
                end_mesh_vertices = end_mesh.vertices
                end_mesh_faces = end_mesh.faces

                os.makedirs(end_shape_dir)
                np.save(os.path.join(end_shape_dir, "vertices.npy"), end_mesh_vertices)
                np.save(os.path.join(end_shape_dir, "faces.npy"), end_mesh_faces)
            else:
                print(
                    f"{end_shape} mesh already exists in {end_shape_dir}. Loading now."
                )
                end_mesh_vertices = np.load(os.path.join(end_shape_dir, "vertices.npy"))
                end_mesh_faces = np.load(os.path.join(end_shape_dir, "faces.npy"))
                end_mesh = trimesh.Trimesh(
                    vertices=end_mesh_vertices, faces=end_mesh_faces
                )

            (
                mesh_sequence_vertices,
                mesh_faces,
                times,
                true_intercept,
                true_slope,
            ) = generate_syntetic_geodesics.generate_synthetic_parameterized_geodesic(
                start_mesh, end_mesh, n_times
            )
            print("Original mesh_sequence vertices: ", mesh_sequence_vertices.shape)
            print("Original mesh faces: ", mesh_faces.shape)
            print("Times: ", times.shape)

            os.makedirs(synthetic_mesh_sequence_dir)
            np.save(
                os.path.join(synthetic_mesh_sequence_dir, "mesh_sequence_vertices.npy"),
                mesh_sequence_vertices,
            )
            np.save(
                os.path.join(synthetic_mesh_sequence_dir, "mesh_faces.npy"), mesh_faces
            )
            np.save(os.path.join(synthetic_mesh_sequence_dir, "times.npy"), times)
            np.save(
                os.path.join(synthetic_mesh_sequence_dir, "true_intercept.npy"),
                true_intercept,
            )
            np.save(
                os.path.join(synthetic_mesh_sequence_dir, "true_slope.npy"), true_slope
            )

        else:
            print(
                f"Synthetic geodesic ALREADY EXISTS with start mesh {start_shape}, end mesh {end_shape}, and n_times {n_times}. Loading now."
            )
            mesh_sequence_vertices = gs.array(
                np.load(
                    os.path.join(
                        synthetic_mesh_sequence_dir, "mesh_sequence_vertices.npy"
                    )
                )
            )
            mesh_faces = gs.array(
                np.load(os.path.join(synthetic_mesh_sequence_dir, "mesh_faces.npy"))
            )
            times = gs.array(
                np.load(os.path.join(synthetic_mesh_sequence_dir, "times.npy"))
            )
            true_intercept = gs.array(
                np.load(os.path.join(synthetic_mesh_sequence_dir, "true_intercept.npy"))
            )
            true_slope = gs.array(
                np.load(os.path.join(synthetic_mesh_sequence_dir, "true_slope.npy"))
            )

elif data_type == "real":
    print("Using real data")
    mesh_dir = parameterized_meshes_dir
    raise (NotImplementedError)
    # in progress...
    # when i do this, i will most likely change main_2_mesh_parameterization to take in a list of meshes
    # and then call it here.

##################### Perform Regression #####################

start_time = time.time()

(
    lr_intercept_hat,
    lr_coef_hat,
) = parameterized_regression.perform_parameterized_linear_regression(
    mesh_sequence_vertices, times
)

# if (residual magnitude is too big... have max residual as a param): then do geodesic regression

(
    intercept_hat,
    coef_hat,
) = parameterized_regression.perform_parameterized_geodesic_regression(
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

##################### Calculate True Slope #####################

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

true_coef = METRIC.log(mesh_sequence_vertices[1], mesh_sequence_vertices[0])
true_coef = gs.array(true_slope)

##################### Save Results #####################

parameterized_regression.save_regression_results(
    data_type,
    sped_up,
    mesh_sequence_vertices,
    mesh_faces,
    true_coef,
    regression_intercept=intercept_hat,
    regression_coef=coef_hat,
    duration_time=duration_time,
)
