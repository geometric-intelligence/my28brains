"""Parameterized geodesic regression.

Mesh sequence chosen in default_config.py
Returns the slope and intercept of the regression fit.

NOTE: is t = 0 the intercept? let's check this if things aren't working.
"""
import datetime
import itertools
import logging
import os
import random
import time

import numpy as np

os.environ["GEOMSTATS_BACKEND"] = "pytorch"  # noqa: E402
import geomstats.backend as gs
from sklearn.preprocessing import PolynomialFeatures

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
            regression_dir, f"{run_type}_polynomial_degree{default_config.poly_degree}"
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
            all_hormone_levels,
            true_intercept,
            true_coef,
        ) = data_utils.load_real_data(wandb_config)

        X = all_hormone_levels[default_config.hormone_name].values
        print(X)
        print("X.shape, ", X.shape)

        # X_for_lr = gs.array(X.reshape(len(X), 1))
        # print("regression reshaped X_for_lr.shape: ", X_for_lr.shape)

        wandb.log(
            {
                "X": np.array(X),
                "true_intercept": np.array(true_intercept),
                "true_coef": np.array(true_coef),
                "true_intercept_fig": wandb.Object3D(true_intercept.numpy()),
                "true_coef_fig": wandb.Object3D(true_coef.numpy()),
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

        logging.info("\n- Computing train/test indices")
        n_train = int(default_config.train_test_split * len(X))

        X_indices = np.arange(len(X))
        # Shuffle the array to get random values
        random.shuffle(X_indices)
        train_indices = X_indices[:n_train]
        train_indices = np.sort(train_indices)
        test_indices = X_indices[n_train:]
        test_indices = np.sort(test_indices)

        logging.info("\n- Normal Linear Regression")

        (
            linear_intercept_hat,
            linear_coef_hat,
            lr,
        ) = training.fit_linear_regression(y, X)

        logging.info("\n- Computing R2 scores for linear regression")

        lr_score_array = training.compute_R2(y, X, test_indices, train_indices)

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
            config=wandb_config,
            y_hat=y_pred_for_lr,
            lr_score_array=lr_score_array,
        )

        logging.info("\n- Polynomial Regression")

        (
            poly_intercept_hat,
            # poly_coef_hat_linear,
            # poly_coef_hat_quadratic,
            poly_coef_hats,
            pr,
        ) = training.fit_polynomial_regression(y, X, degree=default_config.poly_degree)

        logging.info("\n- Computing R2 scores for polynomial regression")

        # predictions for polynomial regression
        # TODO: have to make X_poly to do this.
        poly = PolynomialFeatures(degree=default_config.poly_degree, include_bias=False)
        X_poly = poly.fit_transform(X_pred_lr)
        y_pred_for_pr = pr.predict(X_poly)
        y_pred_for_pr = y_pred_for_pr.reshape([len(X_pred_lr), len(y[0]), 3])

        pr_score_array = training.compute_R2(y, X_poly, test_indices, train_indices)

        wandb.log(
            {
                "poly_intercept_hat": poly_intercept_hat,
                "poly_coef_hat_linear": poly_coef_hats[0],
                "poly_coef_hat_quadratic": poly_coef_hats[1],
                "pr_score_array (adj, normal)": pr_score_array,
            }
        )

        logging.info("Saving polynomial regression results...")
        training.save_regression_results(
            dataset_name=wandb_config.dataset_name,
            y=y,
            X=X_pred,
            space=space,
            true_coef=true_coef,
            regr_intercept=poly_intercept_hat,
            regr_coef=poly_coef_hats,
            results_dir=polynomial_regression_dir,
            config=wandb_config,
            y_hat=y_pred_for_pr,
            lr_score_array=pr_score_array,
        )

        logging.info("\n- Multi-variable Regression")

        progesterone_levels = gs.array(all_hormone_levels["Prog"].values)
        estrogen_levels = gs.array(all_hormone_levels["Estro"].values)
        dheas_levels = gs.array(all_hormone_levels["DHEAS"].values)
        lh_levels = gs.array(all_hormone_levels["LH"].values)
        fsh_levels = gs.array(all_hormone_levels["FSH"].values)
        shbg_levels = gs.array(all_hormone_levels["SHBG"].values)

        X_multiple = gs.vstack(
            (
                progesterone_levels,
                estrogen_levels,
                dheas_levels,
                lh_levels,
                fsh_levels,
                shbg_levels,
            )
        ).T  # NOTE: copilot thinks this should be transposed.

        (
            multiple_intercept_hat,
            multiple_coef_hat,
            mr,
        ) = training.fit_linear_regression(y, X_multiple)

        logging.info("\n- Computing R2 scores for multiple regression")

        mr_score_array = training.compute_R2(y, X_multiple, test_indices, train_indices)

        X_multiple_predict = gs.array(X_multiple.reshape(len(X_multiple), -1))
        y_pred_for_mr = mr.predict(X_multiple_predict)
        y_pred_for_mr = y_pred_for_mr.reshape([len(X_multiple), len(y[0]), 3])

        logging.info("Saving multiple regression results...")

        training.save_regression_results(
            dataset_name=wandb_config.dataset_name,
            y=y,
            X=X_pred,
            space=space,
            true_coef=true_coef,
            regr_intercept=multiple_intercept_hat,
            regr_coef=multiple_coef_hat,
            results_dir=multiple_regression_dir,
            config=wandb_config,
            y_hat=y_pred_for_mr,
            lr_score_array=mr_score_array,
        )

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
    main_config = {
        "dataset_name": default_config.dataset_name,
        "geodesic_initialization": default_config.geodesic_initialization,
        "linear_residuals": default_config.linear_residuals,
        "tol_factor": default_config.tol_factor,
        "project_dir": default_config.project_dir,
        "use_cuda": default_config.use_cuda,
    }

    if default_config.dataset_name == "menstrual_mesh":
        config = {
            "hemisphere": default_config.hemisphere,
            "n_steps": default_config.n_steps,
            "structure_id": default_config.structure_id,
            "area_threshold": default_config.area_threshold,
        }
        config.update(main_config)
        main_run(config)
    else:
        print("Please choose valid dataset for this project")


main()
