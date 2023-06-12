import itertools
import logging
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import torch
import trimesh
import wandb

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


def run_tests():
    """Run wandb with different input parameters and tests.

    uncomment the for loop once we are running multiple tests.
    """
    # for (
    #     data_type,
    #     sped_up,
    #     n_decimations,
    #     regression_decimation_factor_step,
    # ) in itertools.product(
    #     default_config.data_type,
    #     default_config.sped_up,
    #     default_config.n_decimations,
    #     default_config.regression_decimation_factor_step,
    # ):
    #     logging.info(
    #         f"Running tests for {data_type} data with sped_up = {sped_up}, n_decimations = {default_config.n_decimations}:\n"
    #         f"- regression_decimation_factor_step = {regression_decimation_factor_step}\n"
    #     )

    #     run_wandb(
    #         data_type=data_type,
    #         sped_up=sped_up,
    #         n_decimations=default_config.n_decimations,
    #         regression_decimation_factor_step=default_config.regression_decimation_factor_step,
    #     )

    run_wandb(
        data_type=data_type,
        sped_up=sped_up,
        n_decimations=default_config.n_decimations,
        regression_decimation_factor_step=default_config.regression_decimation_factor_step,
    )


def run_wandb(
    data_type=data_type,
    sped_up=sped_up,
    n_decimations=default_config.n_decimations,
    regression_decimation_factor_step=default_config.regression_decimation_factor_step,
):
    """Run wandb script for the following parameters."""
    run_name = f"{data_type}_sped{sped_up}_ndec_{default_config.n_decimations}_factor_{regression_decimation_factor_step}_{default_config.now}"

    wandb.init(
        project="regression_speedup",
        dir=tempfile.gettempdir(),
        config={
            "run_name": run_name,
            "data_type": data_type,
            "sped_up": sped_up,
            "n_decimations": default_config.n_decimations,
            "regression_decimation_factor_step": default_config.regression_decimation_factor_step,
        },
    )

    config = wandb.config
    wandb.run.name = config.run_name

    logging.info(f"Load {data_type} dataset")
    if data_type == "synthetic":
        print("Using synthetic data")
        mesh_dir = synthetic_data_dir
        sphere_mesh = generate_syntetic_geodesics.generate_sphere_mesh()
        ellipsoid_mesh = generate_syntetic_geodesics.generate_ellipsoid_mesh()
        (
            original_mesh_sequence_vertices,
            original_mesh_faces,
            times,
            true_intercept,
            true_slope,
        ) = generate_syntetic_geodesics.generate_synthetic_parameterized_geodesic(
            sphere_mesh, ellipsoid_mesh
        )
        print(
            "Original mesh_sequence vertices: ", original_mesh_sequence_vertices.shape
        )
        print("Original mesh faces: ", original_mesh_faces.shape)
        print("Times: ", times.shape)

    elif data_type == "real":
        print("Using real data")
        mesh_dir = parameterized_meshes_dir
        raise (NotImplementedError)
        # in progress...
        # when i do this, i will most likely change main_2_mesh_parameterization to take in a list of meshes
        # and then call it here.

    else:
        raise ValueError(f"Unknown dataset name {data_type}")

    ##################### Construct Decimated Mesh Sequences #####################

    logging.info("Construct list of decimated mesh sequences and list of tolerances")

    if sped_up:
        start_time = time.time()

        (
            decimated_mesh_sequences,
            decimated_faces,
        ) = parameterized_regression.create_decimated_mesh_sequence_list(
            original_mesh_sequence_vertices, original_mesh_faces
        )

    tols = []
    for i_tol in range(1, default_config.n_decimations + 1):
        tols.append(10 ** (-i_tol))
    tols = gs.array(tols)

    ##################### Perform Regression #####################

    logging.info("Perform Regression")
    if sped_up:
        (
            intercept_hat,
            coef_hat,
        ) = parameterized_regression.perform_multi_res_geodesic_regression(
            decimated_mesh_sequences,
            decimated_faces,
            tols,
            times,
            regression_initialization="warm_start",
        )
    else:
        start_time = time.time()
        (
            intercept_hat,
            coef_hat,
        ) = parameterized_regression.perform_single_res_parameterized_regression(
            original_mesh_sequence_vertices,
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

    ##################### Calculate True Slope #####################
    logging.info("Calculate true slope")
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

    ##################### Save Results #####################

    logging.info("Save results")

    parameterized_regression.save_regression_results(
        data_type,
        sped_up,
        original_mesh_sequence_vertices,
        original_mesh_faces,
        true_coef,
        regression_intercept=intercept_hat,
        regression_coef=coef_hat,
        duration_time=duration_time,
    )

    # TODO: make summary fig to visualize results -- similar to below
    # for i_m, m in enumerate(config.m_grid):
    #     a_steps = iteration_histories_per_i_m[i_m]["a"]
    #     mse_train_steps = iteration_histories_per_i_m[i_m]["mse_train"]
    #     mse_val_steps = iteration_histories_per_i_m[i_m]["mse_val"]

    #     r2_train_steps = iteration_histories_per_i_m[i_m]["r2_train"]
    #     r2_val_steps = iteration_histories_per_i_m[i_m]["r2_val"]

    #     iteration_history_df = pd.DataFrame(
    #         columns=["a", "mse_train", "mse_val", "r2_train", "r2_val"],
    #         data=[
    #             [
    #                 float(a),
    #                 float(mse_train),
    #                 float(mse_val),
    #                 float(r_train),
    #                 float(r_val),
    #             ]
    #             for a, mse_train, mse_val, r_train, r_val in zip(
    #                 a_steps,
    #                 mse_train_steps,
    #                 mse_val_steps,
    #                 r2_train_steps,
    #                 r2_val_steps,
    #             )
    #         ],
    #     )

    #     table_key = f"iteration_history_m_{m}"
    #     iteration_history_df.to_json(
    #         f"saved_figs/optimize_am/{config.run_name}_iteration_history.json"
    #     )
    #     wandb.log({table_key: wandb.Table(dataframe=iteration_history_df)})

    # fig = viz.plot_summary_wandb(
    #     iteration_histories_per_i_m=iteration_histories_per_i_m,
    #     config=config,
    #     noiseless_curve_traj=noiseless_curve_traj,
    #     curve_traj=curve_traj,
    #     noiseless_q_traj=noiseless_q_traj,
    #     q_traj=q_traj,
    #     times_train=times_train,
    #     times_val=times_val,
    #     times_test=times_test,
    #     best_a=best_a,
    #     best_m=best_m,
    #     best_r2_val=best_r2_val,
    #     r2_test_at_best=r2_test_at_best,
    #     baseline_r2_srv_val=baseline_r2_srv_val,
    #     baseline_r2_srv_test=baseline_r2_srv_test,
    # )

    # fig.savefig(f"saved_figs/optimize_am/{config.run_name}_summary.png")
    # fig.savefig(f"saved_figs/optimize_am/{config.run_name}_summary.svg")
    # wandb.log({"summary_fig": wandb.Image(fig)})

    wandb.finish()


run_tests()
