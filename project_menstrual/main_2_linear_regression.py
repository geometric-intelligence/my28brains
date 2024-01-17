"""Parameterized geodesic regression.

Mesh sequence chosen in default_config.py
Returns the slope and intercept of the regression fit.

NOTE: is t = 0 the intercept? let's check this if things aren't working.
"""
import datetime
import itertools
import logging
import os
import time

import numpy as np

os.environ["GEOMSTATS_BACKEND"] = "pytorch"  # noqa: E402
import geomstats.backend as gs

import project_menstrual.default_config as default_config
import src.datasets.utils as data_utils
import src.viz as viz
import wandb
from src.regression import check_euclidean, training

regression_dir = default_config.regression_dir

today = datetime.date.today()


def main_run(config):
    """Run the regression.

    This corresponds to one wandb run (no wandb sweep).

    Notes
    -----
    full_run: bool. If True, at the end of the function, this means the full
        regression was run
    """
    full_run = True
    try:
        print(f"run_start: {today}")
        wandb.init(tags=[f"{today}"])
        wandb_config = wandb.config
        wandb_config.update(config)

        run_name = f"run_{wandb.run.id}"
        wandb.run.name = run_name

        logging.info(f"\n\n---> START run: {run_name}.")

        run_type = default_config.run_type

        linear_regression_dir = os.path.join(regression_dir, f"{run_type}_linear")
        polynomial_regression_dir = os.path.join(
            regression_dir, f"{run_type}_polynomial"
        )
        multiple_regression_dir = os.path.join(regression_dir, f"{run_type}_multiple")
        for one_regress_dir in [
            linear_regression_dir,
            polynomial_regression_dir,
            multiple_regression_dir,
        ]:
            if not os.path.exists(one_regress_dir):
                os.makedirs(one_regress_dir)

        (
            space,
            y,
            y_noiseless,
            X,
            true_intercept,
            true_coef,
        ) = data_utils.load(wandb_config)

        wandb.log(
            {
                "X": np.array(X),
                "true_intercept": np.array(true_intercept),
                "true_coef": np.array(true_coef),
            }
        )

        if wandb_config.dataset_name in [
            "synthetic_mesh",
        ]:
            wandb.log(
                {
                    "noise_factor": wandb_config.noise_factor,
                    "linear_noise": wandb_config.linear_noise,
                }
            )

        if (
            wandb_config.dataset_name == "synthetic_mesh"
            or wandb_config.dataset_name == "menstrual_mesh"
        ):
            mesh_sequence_vertices = y
            mesh_faces = space.faces
            logging.info(
                "\n- Calculating tolerance for geodesic regression."
                "Based on: size of shape, number of time points, number of vertices."
            )
            mesh_diameter = data_utils.mesh_diameter(mesh_sequence_vertices[0])
            tol = (
                wandb_config.tol_factor
                * mesh_diameter
                * len(mesh_sequence_vertices[0])
                * len(mesh_sequence_vertices)
            ) ** 2
            logging.info(
                f"\n- Tolerance calculated for geodesic regression: {tol:.3f}."
            )

            wandb.log(
                {
                    "mesh_diameter": mesh_diameter,
                    "n_faces": len(mesh_faces),
                    "geodesic_tol": tol,
                    # "euclidean_subspace": euclidean_subspace,
                    "mesh_sequence_vertices": wandb.Object3D(
                        mesh_sequence_vertices.numpy().reshape((-1, 3))
                    ),
                    # "test_diff_tolerance": diff_tolerance,
                    "true_intercept_fig": wandb.Object3D(true_intercept.numpy()),
                    "true_coef_fig": wandb.Object3D(true_coef.numpy()),
                }
            )
        else:
            tol = wandb_config.tol_factor
            wandb.log({"geodesic_tol": tol})

        logging.info("\n- Normal Linear Regression")

        (
            linear_intercept_hat,
            linear_coef_hat,
            lr,
            lr_score_array,
        ) = training.fit_linear_regression(y, X)

        wandb.log(
            {
                "linear_intercept_hat": linear_intercept_hat,
                "linear_coef_hat": linear_coef_hat,
                "lr_score_array (adj, normal)": lr_score_array,
            }
        )

        X_pred = gs.linspace(
            0, default_config.n_predicted_points, default_config.n_predicted_points + 1
        )
        X_pred_lr = gs.array(X_pred.reshape(len(X_pred), 1))
        y_pred_for_lr = lr.predict(X_pred_lr)
        y_pred_for_lr = y_pred_for_lr.reshape([len(X_pred_lr), len(y[0]), 3])

        # Save linear_intercept_hat, linear_coef_hat, X_for_lr, y_pred_for_lr in linear_regression_dir
        logging.info("Saving linear regression results...")
        training.save_regression_results(
            dataset_name=wandb_config.dataset_name,
            y=y,
            X=X_pred,
            space=space,
            true_coef=true_coef,
            regr_intercept=linear_intercept_hat,
            regr_coef=linear_coef_hat,
            results_dir=linear_regression_dir,
            model="linear",
            linear_residuals=wandb_config.linear_residuals,
            y_hat=y_pred_for_lr,
            lr_score_array=lr_score_array,
        )

        logging.info("\n- Polynomial Regression")

        wandb_config.update({"full_run": full_run})
        wandb.finish()
    except Exception as e:
        full_run = False
        wandb_config.update({"full_run": full_run})
        logging.exception(e)
        wandb.finish()


def main():
    """Parse the default_config file and launch all experiments.

    This launches experiments with wandb with different config parameters.
    """
    for (
        dataset_name,
        geodesic_initialization,
        linear_residuals,
        tol_factor,
    ) in itertools.product(
        default_config.dataset_name,
        default_config.geodesic_initialization,
        default_config.linear_residuals,
        default_config.tol_factor,
    ):
        main_config = {
            "dataset_name": dataset_name,
            "geodesic_initialization": geodesic_initialization,
            "linear_residuals": linear_residuals,
            "tol_factor": tol_factor,
        }

        if dataset_name == "menstrual_mesh":
            for hemisphere, n_steps in itertools.product(
                default_config.hemisphere, default_config.n_steps
            ):
                config = {
                    "hemisphere": hemisphere,
                    "n_steps": n_steps,
                }
                config.update(main_config)
                main_run(config)
        else:
            print("Please choose valid dataset for this project")


main()
