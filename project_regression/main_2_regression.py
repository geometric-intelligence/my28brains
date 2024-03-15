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
import torch

os.environ["GEOMSTATS_BACKEND"] = "pytorch"  # noqa: E402
import geomstats.backend as gs

import project_regression.default_config as default_config
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
        wandb.init(
            tags=[
                f"{today}",
                f"model_{default_config.model}",
                f"estimator_{config['estimator']}",
                # f"rmsd_y_noiseless",
            ]
        )
        wandb_config = wandb.config
        wandb_config.update(config)

        run_name = f"run_{wandb.run.id}"
        wandb.run.name = run_name

        logging.info(f"\n\n---> START run: {run_name}.")

        linear_regression_dir = os.path.join(regression_dir, f"{run_name}_linear")
        geodesic_regression_dir = os.path.join(regression_dir, f"{run_name}_geodesic")
        for one_regress_dir in [linear_regression_dir, geodesic_regression_dir]:
            if not os.path.exists(one_regress_dir):
                os.makedirs(one_regress_dir)

        (
            space,
            y,
            y_noiseless,
            X,
            true_intercept,
            true_coef,
        ) = data_utils.load_synthetic_data(wandb_config)

        wandb.log(
            {
                "X": np.array(X),
                "true_intercept": np.array(true_intercept),
                "true_coef": np.array(true_coef),
                "estimator": wandb_config.estimator,
                "model": wandb_config.model,
            }
        )

        if wandb_config.dataset_name in [
            "synthetic_mesh",
            "hypersphere",
            "hyperboloid",
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

        logging.info("\n- Performing Linear Regression.")

        start_time = time.time()

        (
            linear_intercept_hat,
            linear_coef_hat,
            lr,
        ) = training.fit_linear_regression(y, X)

        lr_duration_time = time.time() - start_time

        logging.info("Computing points along linear regression prediction...")
        y_pred_for_lr = lr.predict(X.reshape(len(X), 1))
        y_pred_for_lr = y_pred_for_lr.reshape(y.shape)

        lr_intercept_err = linear_intercept_hat - true_intercept
        lr_coef_err = linear_coef_hat - true_coef
        lr_intercept_err_sum = gs.sum(lr_intercept_err)
        lr_coef_err_sum = gs.sum(lr_coef_err)
        lr_intercept_err_norm = gs.linalg.norm(lr_intercept_err)
        lr_coef_err_norm = gs.linalg.norm(lr_coef_err)

        logging.info(f">> Linear Intercept error sum: {lr_intercept_err_sum:.3f}.")
        logging.info(f">> Linear Coef error sum: {lr_coef_err_sum:.3f}.")
        logging.info(f">> Linear Intercept error norm: {lr_intercept_err_norm:.3f}.")
        logging.info(f">> Linear Coef error norm: {lr_coef_err_norm:.3f}.")

        wandb.log(
            {
                "lr_duration_time": lr_duration_time,
                "linear_intercept_hat": np.array(linear_intercept_hat),
                "linear_coef_hat": np.array(linear_coef_hat),
                "lr_intercept_err": lr_intercept_err,
                "lr_coef_err": lr_coef_err,
                "lr_intercept_err_sum": lr_intercept_err_sum,
                "lr_coef_err_sum": lr_coef_err_sum,
                "lr_intercept_err_norm": lr_intercept_err_norm,
                "lr_coef_err_norm": lr_coef_err_norm,
            }
        )

        X_for_lr = gs.array(X.reshape(len(X), 1))
        y_pred_for_lr = lr.predict(X_for_lr)
        y_pred_for_lr = y_pred_for_lr.reshape(y.shape)

        logging.info(f"Computing estimates for {wandb_config.estimator} estimator.")

        if wandb_config.estimator == "LR":
            logging.info("Using Linear Regression estimator.")
            estimator_intercept_hat = gs.array(linear_intercept_hat)
            estimator_coef_hat = gs.array(linear_coef_hat)
            estimator_duration_time = lr_duration_time
            if wandb_config.dataset_name in ["synthetic_mesh", "menstrual_mesh"]:
                estimator_y_prediction = []
                for pred_mesh in y_pred_for_lr:
                    pred_mesh = gs.array(pred_mesh)
                    pred_mesh_mean = gs.mean(pred_mesh)
                    estimator_y_prediction.append(pred_mesh - pred_mesh_mean)
                estimator_y_prediction = gs.array(estimator_y_prediction)
            else:
                estimator_y_prediction = space.projection(gs.array(y_pred_for_lr))

        if wandb_config.estimator == "Lin2015":
            logging.info(
                "Projecting linear regression to the manifold for Lin2015 estimator."
            )
            if wandb_config.dataset_name in ["synthetic_mesh", "menstrual_mesh"]:
                linear_intercept_hat_mean = gs.mean(linear_intercept_hat)
                estimator_intercept_hat = (
                    linear_intercept_hat - linear_intercept_hat_mean
                )
            elif wandb_config.dataset_name in ["hypersphere", "hyperboloid"]:
                estimator_intercept_hat = space.projection(linear_intercept_hat)
            estimator_coef_hat = space.to_tangent(
                linear_coef_hat, base_point=estimator_intercept_hat
            )

            estimator_duration_time = time.time() - start_time
            logging.info(f">> Duration (Lin2015): {estimator_duration_time:.3f} secs.")

            logging.info(
                f"Computing points along {wandb_config.estimator} estimator prediction..."
            )

            estimator_y_prediction = []
            for x_val in X:
                estimator_y_prediction.append(
                    space.metric.exp(
                        tangent_vec=x_val * estimator_coef_hat,
                        base_point=estimator_intercept_hat,
                    )
                )
            estimator_y_prediction = gs.array(estimator_y_prediction)

        elif wandb_config.estimator in ["GLS", "LLS"]:
            logging.info(
                f"\n- Geodesic Regression with {wandb_config.estimator} estimator."
            )

            if wandb_config.estimator == "GLS":
                logging.info("Using GLS estimator.")
                linear_residuals = False
            elif wandb_config.estimator == "LLS":
                logging.info("Using LLS estimator.")
                linear_residuals = True

            (
                estimator_intercept_hat,
                estimator_coef_hat,
                gr,
            ) = training.fit_geodesic_regression(
                y,
                space,
                X,
                tol=tol,
                intercept_hat_guess=linear_intercept_hat,
                coef_hat_guess=linear_coef_hat,
                initialization="warm_start",
                linear_residuals=linear_residuals,
                use_cuda=default_config.use_cuda,
            )

            estimator_duration_time = time.time() - start_time
            logging.info(
                f">> Duration ({wandb_config.estimator}): {estimator_duration_time:.3f} secs."
            )

            n_iterations = gr.n_iterations
            n_function_evaluations = gr.n_fevaluations
            n_jacobian_evaluations = gr.n_jevaluations

            # Compute and evaluate estimator prediction

            logging.info(
                f"Computing points along {wandb_config.estimator} estimator prediction..."
            )
            estimator_y_prediction = gr.predict(X)
            estimator_y_prediction = estimator_y_prediction.reshape(y.shape)

            wandb.log(
                {
                    "n_gr_iterations": n_iterations,
                    "n_gr_function_evaluations": n_function_evaluations,
                    "n_gr_jacobian_evaluations": n_jacobian_evaluations,
                }
            )

        print("estimator_y_prediction: ", estimator_y_prediction.shape)
        print("y: ", y.shape)

        logging.info("Computing errors on intercept and coef...")
        # parallel_transport not implemented for mesh data
        if wandb_config.dataset_name in ["synthetic_mesh", "menstrual_mesh"]:
            estimator_intercept_err = space.metric.log(
                point=true_intercept, base_point=estimator_intercept_hat
            )
            estimator_coef_err = estimator_coef_hat - true_coef

        elif wandb_config.dataset_name in ["hypersphere", "hyperboloid"]:
            estimator_intercept_err = space.metric.log(
                point=true_intercept, base_point=estimator_intercept_hat
            )
            estimator_coef_err = true_coef - space.metric.parallel_transport(
                tangent_vec=estimator_coef_hat,
                base_point=estimator_intercept_hat,
                end_point=true_intercept,
            )

            wandb.log(
                {
                    "synthetic_tan_vec_length": wandb_config.synthetic_tan_vec_length,
                }
            )

        estimator_intercept_err_sum = gs.sum(estimator_intercept_err)
        estimator_coef_err_sum = gs.sum(estimator_coef_err)
        estimator_intercept_err_norm = gs.linalg.norm(estimator_intercept_err)
        estimator_coef_err_norm = gs.linalg.norm(estimator_coef_err)

        logging.info("Computing RMSD...")
        estimator_linear_residuals = gs.array(estimator_y_prediction) - gs.array(y)
        rmsd_linear = gs.linalg.norm(estimator_linear_residuals) / gs.sqrt(len(y))

        estimator_geod_residuals = space.metric.dist(estimator_y_prediction, y)
        rmsd_geodesic = gs.linalg.norm(estimator_geod_residuals) / gs.sqrt(len(y))

        if wandb_config.dataset_name in ["synthetic_mesh", "menstrual_mesh"]:

            rmsd_linear = rmsd_linear / (len(mesh_sequence_vertices[0]) * mesh_diameter)

            if rmsd_geodesic is not None:
                rmsd_geodesic = rmsd_geodesic / (
                    len(mesh_sequence_vertices[0]) * mesh_diameter
                )

            wandb.log(
                {
                    "estimator_intercept_hat_fig": wandb.Object3D(
                        estimator_intercept_hat.numpy()
                    ),
                    "estimator_coef_hat_fig": wandb.Object3D(
                        estimator_coef_hat.numpy()
                    ),
                    "estimator_y_prediction_fig": wandb.Object3D(
                        estimator_y_prediction.detach().numpy().reshape((-1, 3))
                    ),
                    "n_faces": len(mesh_faces),
                    "n_vertices": len(mesh_sequence_vertices[0]),
                }
            )

        wandb.log(
            {
                "estimator_duration_time": estimator_duration_time,
                "estimator_intercept_err": estimator_intercept_err,
                "estimator_coef_err": estimator_coef_err,
                "estimator_intercept_err_sum": estimator_intercept_err_sum,
                "estimator_coef_err_sum": estimator_coef_err_sum,
                "estimator_intercept_err_norm": estimator_intercept_err_norm,
                "estimator_coef_err_norm": estimator_coef_err_norm,
                "rmsd_linear": rmsd_linear,
                "rmsd_geodesic": rmsd_geodesic,
                "estimator_intercept_hat": np.array(estimator_intercept_hat),
                "estimator_coef_hat": np.array(estimator_coef_hat),
            }
        )

        logging.info(">> Regression errors (geodesic):")
        logging.info(
            f"On intercept: {estimator_intercept_err}, on coef: "
            f"{estimator_coef_err}"
        )

        print(f"estimator_y_prediction: " f"{estimator_y_prediction.shape}")

        logging.info("Saving geodesic results...")
        training.save_regression_results(
            dataset_name=wandb_config.dataset_name,
            y=y,
            X=X,
            space=space,
            true_coef=true_coef,
            regr_intercept=estimator_intercept_hat,
            regr_coef=estimator_coef_hat,
            results_dir=geodesic_regression_dir,
            config=wandb_config,
            model=default_config.model,
            estimator=wandb_config.estimator,
            y_hat=estimator_y_prediction,
        )

        if (
            wandb_config.dataset_name in ["hypersphere", "hyperboloid"]
            and space.dim == 2
        ):

            # create high-density sequence of true geodesic
            true_n_X = 100
            true_X = gs.linspace(0, 1, true_n_X)
            true_X -= gs.mean(true_X)
            true_geodesic = space.metric.exp(
                true_X[:, None] * true_coef, base_point=true_intercept
            )

            fig = viz.benchmark_data_sequence(
                space, y, y_pred_for_lr, estimator_y_prediction, true_geodesic
            )
            plt = wandb.Image(fig)

            wandb.log(
                {
                    "line vs geodesic": plt,
                }
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
    for (dataset_name, estimator, tol_factor, n_X, noise_factor) in itertools.product(
        default_config.dataset_name,
        default_config.estimator,
        default_config.tol_factor,
        default_config.n_X,
        default_config.noise_factor,
    ):
        main_config = {
            "dataset_name": dataset_name,
            "estimator": estimator,
            "tol_factor": tol_factor,
            "n_X": n_X,
            "noise_factor": noise_factor,
            "linear_noise": default_config.linear_noise,
            "project_linear_noise": default_config.project_linear_noise,
            "model": default_config.model,
            "use_cuda": default_config.use_cuda,
            "device_id": default_config.device_id,
            "torch_dtype": default_config.torch_dtype,
            "project_dir": default_config.project_dir,
        }
        if dataset_name == "synthetic_mesh":
            for (
                n_subdivisions,
                (start_shape, end_shape),
                n_steps,
            ) in itertools.product(
                default_config.n_subdivisions,
                zip(default_config.start_shape, default_config.end_shape),
                default_config.n_steps,
            ):
                config = {
                    "start_shape": start_shape,
                    "end_shape": end_shape,
                    "n_subdivisions": n_subdivisions,
                    "n_steps": n_steps,
                }
                config.update(main_config)
                main_run(config)

        elif dataset_name == "hypersphere" or dataset_name == "hyperboloid":
            for (space_dimension, synthetic_tan_vec_length) in itertools.product(
                default_config.space_dimension, default_config.synthetic_tan_vec_length
            ):
                config = {
                    "space_dimension": space_dimension,
                    "synthetic_tan_vec_length": synthetic_tan_vec_length,
                }
                config.update(main_config)
                main_run(config)

        # elif dataset_name == "menstrual_mesh":
        #     for hemisphere, n_steps in itertools.product(
        #         default_config.hemisphere, default_config.n_steps
        #     ):
        #         config = {
        #             "hemisphere": hemisphere,
        #             "n_steps": n_steps,
        #         }
        #         config.update(main_config)
        #         main_run(config)


main()
