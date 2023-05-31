"""Speeds up parameterized geodesic regression with an intentional decimation scheme.

What this script does:
- Fetch a sequence of parameterized meshes, based on data specified in default_config.py
THEN
- Decimates the geodesic to a LOW number of points.
- Performs geodesic regression on the decimated geodesic.
- Uses decimated mesh slope and intercept as starting point for next regression.
REPEATS ABOVE STEPS UNTIL THE INTERCEPT IS CLOSE TO THE TRUE INTERCEPT.
- Compares the regression results to the true slope and intercept of the geodesic.
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

##################### Regression Parameters #####################

data_type = default_config.data_type
sped_up = default_config.sped_up

if data_type == "synthetic":
    print("Using synthetic data")
    mesh_dir = synthetic_data_dir
    sphere_mesh = generate_syntetic_geodesics.generate_sphere_mesh()
    ellipsoid_mesh = generate_syntetic_geodesics.generate_ellipsoid_mesh()
    (
        original_geodesic_vertices,
        original_mesh_faces,
        times,
        true_intercept,
        true_slope,
    ) = generate_syntetic_geodesics.generate_synthetic_parameterized_geodesic(
        sphere_mesh, ellipsoid_mesh
    )
    print("Original geodesic vertices: ", original_geodesic_vertices.shape)
    print("Original mesh faces: ", original_mesh_faces.shape)
    print("Times: ", times.shape)

elif data_type == "real":
    print("Using real data")
    mesh_dir = parameterized_meshes_dir
    raise (NotImplementedError)
    # in progress...
    # when i do this, i will most likely change main_2_mesh_parameterization to take in a list of meshes
    # and then call it here.

##################### Construct output file names #####################

# path_prefix = os.path.join(data_dir, "regression_results")
path_prefix = default_config.regression_dir

true_intercept_file_name = "true_intercept_" + data_type + "_sped_up_" + str(sped_up)
true_slope_file_name = "true_slope_" + data_type + "_sped_up_" + str(sped_up)
regression_intercept_file_name = (
    "regression_intercept_" + data_type + "_sped_up_" + str(sped_up)
)
regression_slope_file_name = (
    "regression_slope_" + data_type + "_sped_up_" + str(sped_up)
)

true_intercept_path = os.path.join(path_prefix, true_intercept_file_name)
true_slope_path = os.path.join(path_prefix, true_slope_file_name)
regression_intercept_path = os.path.join(path_prefix, regression_intercept_file_name)
regression_slope_path = os.path.join(path_prefix, regression_slope_file_name)

##################### Regression Functions #####################


def perform_single_res_parameterized_regression(
    mesh_sequence,
    mesh_faces,
    times,
    tolerance,
    intercept_hat_guess,
    coef_hat_guess,
    regression_initialization="warm_start",
):
    """Performs regression on parameterized meshes.

    inputs:
        mesh_sequence: list of vertices of meshes. Each mesh is a numpy array of shape (n, 3)
        times: list of times corresponding to mesh_sequence
        intercept_hat_guess: initial guess for intercept of regression fit
        coef_hat_guess: initial guess for slope of regression fit


    returns:
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
    gr = GeodesicRegression(
        SURFACE_SPACE,
        metric=METRIC,
        center_X=False,
        method="riemannian",
        max_iter=5,
        init_step_size=0.1,
        tol=tolerance,
        verbose=False,
        initialization=regression_initialization,
    )

    if intercept_hat_guess is None:
        intercept_hat_guess = mesh_sequence[0]
    elif intercept_hat_guess.shape != mesh_sequence[0].shape:
        raise ValueError(
            "intercept_hat_guess must be None or have the same shape as mesh_sequence[0]"
        )

    if coef_hat_guess is None:
        coef_hat_guess = mesh_sequence[1] - mesh_sequence[0]

    # NOTE: THIS IS BUGGING on second iteration
    # coeff_hat_guess = METRIC.log(mesh_sequence[1], mesh_sequence[0])

    gr.intercept_ = intercept_hat_guess
    gr.coef_ = coef_hat_guess

    print("Intercept guess: ", gr.intercept_.shape)
    print("Coef guess: ", gr.coef_.shape)

    gr.fit(times, mesh_sequence, compute_training_score=False)

    intercept_hat, coef_hat = gr.intercept_, gr.coef_

    return intercept_hat, coef_hat


def perform_multi_res_geodesic_regression(
    decimated_geodesics_list,
    mesh_faces_list,
    tols,
    regression_initialization="warm_start",
):
    """Performs regression on parameterized meshes in multiple resolutions.

    inputs:
    -------
    - decimated_geodesics_list: list of geodesics (one for each decimation level).
    - tols: list of tolerances (one for each decimation level).

    returns:
    --------
    - intercept_hat: intercept of regression fit
    - coef_hat: slope of regression fit
    """
    for i_geod in range(
        1, default_config.n_decimations + 1, 1
    ):  # start at 1 because 0 is the original mesh
        print(
            "######### Performing regression on decimation level ", i_geod, " #########"
        )

        geodesic = decimated_geodesics_list[
            -i_geod
        ]  # reverse order so that the original mesh is last
        mesh_faces = mesh_faces_list[-i_geod]
        tolerance = tols[i_geod - 1]  # tols is 0-indexed, but i_geod is 1-indexed
        if i_geod == 1:
            intercept_hat_guess = None
            coef_hat_guess = None
        #     regression_initialization = "frechet"
        #     intercept_hat_guess = None
        #     coeff_hat_guess = None
        # else:
        #     regression_initialization = "warm_start"

        intercept_hat, coef_hat = perform_single_res_parameterized_regression(
            geodesic,
            mesh_faces,
            times,
            tolerance,
            intercept_hat_guess,
            coef_hat_guess,
            regression_initialization=regression_initialization,
        )

        # TODO: must upsample coef_hat to the next resolution level
        print("INTERCEPT HAT SHAPE: ", intercept_hat.shape)
        print("COEFF HAT SHAPE: ", coef_hat.shape)

        intercept_hat_mesh = trimesh.Trimesh(vertices=intercept_hat, faces=mesh_faces)
        coef_hat_mesh = trimesh.Trimesh(vertices=coef_hat, faces=mesh_faces)
        if i_geod < default_config.n_decimations:
            num_upsampled_vertices = len(decimated_geodesics_list[-i_geod - 1][0])
            upsampled_intercept_hat = trimesh.sample.sample_surface(
                intercept_hat_mesh,
                num_upsampled_vertices,
                face_weight=None,
                sample_color=False,
            )
            intercept_hat_guess = gs.array(upsampled_intercept_hat[0])
            assert (
                intercept_hat_guess.shape
                == decimated_geodesics_list[-i_geod - 1][0].shape
            )

            upsampled_coef_hat = trimesh.sample.sample_surface(
                coef_hat_mesh,
                num_upsampled_vertices,
                face_weight=None,
                sample_color=False,
            )
            coef_hat_guess = gs.array(upsampled_coef_hat[0])
            assert (
                coef_hat_guess.shape == decimated_geodesics_list[-i_geod - 1][0].shape
            )

    return intercept_hat, coef_hat


##################### Construct Decimated Mesh Sequences #####################
if sped_up:
    start_time = time.time()

    decimated_geodesics_list = (
        []
    )  # all the decimated geodesics for the geod regr. (0 = original mesh)
    mesh_faces_list = (
        []
    )  # all the decimated mesh faces for the geod regr. (0 = original mesh)

    # TEMP CHANGE
    decimated_geodesics_list.append(original_geodesic_vertices)

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
        for one_mesh in original_geodesic_vertices:
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

    tols = []
    for i_tol in range(default_config.n_decimations):
        tols.append(10 ** (-i_tol))
    tols = gs.array(tols)

##################### Perform Regression #####################

if sped_up:
    intercept_hat, coef_hat = perform_multi_res_geodesic_regression(
        decimated_geodesics_list,
        mesh_faces_list,
        tols,
        regression_initialization="warm_start",
    )
else:
    start_time = time.time()
    intercept_hat, coef_hat = perform_single_res_parameterized_regression(
        original_geodesic_vertices,
        original_mesh_faces,
        times,
        tolerance=0.0001,
        intercept_hat_guess=None,
        coef_hat_guess=None,
        regression_initialization="warm_start",
    )

end_time = time.time()
duration_time = end_time - start_time

print("Duration: ", duration_time)

##################### Save Results #####################

true_intercept_path = os.path.join(path_prefix, true_intercept_file_name)
true_slope_path = os.path.join(path_prefix, true_slope_file_name)
regression_intercept_path = os.path.join(
    path_prefix, regression_intercept_file_name + str(duration_time)
)
regression_slope_path = os.path.join(
    path_prefix, regression_slope_file_name + str(duration_time)
)

# NOTE: is t = 0 the intercept? let's check this if things aren't working.

print("Vertices Data Type: ", type(gs.array(original_geodesic_vertices[0])))
print(" Faces Data Type: ", type(gs.array(original_mesh_faces)))

H2_SurfaceMatch.utils.input_output.save_data(
    true_intercept_path,
    ".ply",
    gs.array(original_geodesic_vertices[0]).numpy(),
    gs.array(original_mesh_faces).numpy(),
)
H2_SurfaceMatch.utils.input_output.save_data(
    regression_intercept_path,
    ".ply",
    gs.array(intercept_hat).numpy(),
    gs.array(original_mesh_faces).numpy(),
)

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

true_slope = METRIC.log(original_geodesic_vertices[1], original_geodesic_vertices[0])
true_slope = gs.array(true_slope)
np.savetxt(true_slope_path, true_slope)
np.savetxt(regression_slope_path, coef_hat)
