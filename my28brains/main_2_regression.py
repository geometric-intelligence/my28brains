"""Parameterized geodesic regression.

Mesh sequence chosen in default_config.py
Returns the slope and intercept of the regression fit.

NOTE: is t = 0 the intercept? let's check this if things aren't working.
"""
import itertools
import logging
import os
import time

os.environ["GEOMSTATS_BACKEND"] = "pytorch"  # noqa: E402
import geomstats.backend as gs

import my28brains.datasets.utils as data_utils
import my28brains.default_config as default_config
import my28brains.viz as viz
import wandb
from my28brains.regression import check_euclidean, training

regression_dir = default_config.regression_dir


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
        wandb.init()
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

        start_time = time.time()
        (
            space,
            y,
            y_noiseless,
            X,
            true_intercept,
            true_coef,
        ) = data_utils.load(wandb_config)

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
            or wandb_config.dataset_name == "real_mesh"
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
                }
            )
        else:
            tol = 1e-3
            wandb.log({"geodesic_tol": tol})

        logging.info("\n- Linear Regression")
        (
            linear_intercept_hat,
            linear_coef_hat,
            lr,
        ) = training.fit_linear_regression(y, X)

        linear_duration_time = time.time() - start_time
        linear_intercept_err = gs.linalg.norm(linear_intercept_hat - true_intercept)
        linear_coef_err = gs.linalg.norm(linear_coef_hat - true_coef)

        logging.info("Computing points along linear regression...")
        X_for_lr = gs.array(X.reshape(len(X), 1))
        y_pred_for_lr = lr.predict(X_for_lr)
        y_pred_for_lr = y_pred_for_lr.reshape(y.shape)
        print(f"y_pred_for_lr: {y_pred_for_lr.shape}")

        rmsd_linear = gs.linalg.norm(gs.array(y_pred_for_lr) - gs.array(y)) / gs.sqrt(
            len(y)
        )

        if wandb_config.dataset_name in ["synthetic_mesh", "real_mesh"]:

            rmsd_linear = rmsd_linear / (len(mesh_sequence_vertices[0]) * mesh_diameter)

            offset_mesh_sequence_vertices = gs.array(
                viz.offset_mesh_sequence(mesh_sequence_vertices)
            )[0]
            print(
                f"offset_mesh_sequence_vertices: {offset_mesh_sequence_vertices.shape}"
            )
            wandb.log(  # TODO: implement a general visualization thing instead of this hack
                {
                    "true_intercept": wandb.Object3D(true_intercept.numpy()),
                    "true_coef": wandb.Object3D(true_coef.numpy()),
                    "linear_intercept_hat": wandb.Object3D(
                        linear_intercept_hat.numpy()
                    ),
                    "linear_coef_hat": wandb.Object3D(linear_coef_hat.numpy()),
                    "offset_mesh_sequence_vertices": wandb.Object3D(
                        offset_mesh_sequence_vertices.numpy()
                    ),
                    "y_pred_for_lr": wandb.Object3D(y_pred_for_lr.reshape((-1, 3))),
                }
            )

        nrmsd_linear = rmsd_linear / gs.linalg.norm(y[0] - y[-1])

        wandb.log(
            {
                "linear_duration_time": linear_duration_time,
                "linear_intercept_err": linear_intercept_err,
                "linear_coef_err": linear_coef_err,
                "rmsd_linear": rmsd_linear,
                "nrmsd_linear": nrmsd_linear,
            }
        )

        logging.info(f">> Duration (linear) = {linear_duration_time:.3f} secs.")
        logging.info(">> Regression errors (linear):")
        logging.info(
            f"On intercept: {linear_intercept_err:.6f}, on coef: {linear_coef_err:.6f}"
        )

        logging.info("Saving linear results...")
        training.save_regression_results(
            dataset_name=wandb_config.dataset_name,
            y=gs.array(y),
            space=space,
            true_coef=gs.array(true_coef),
            regr_intercept=linear_intercept_hat,
            regr_coef=linear_coef_hat,
            duration_time=linear_duration_time,
            results_dir=linear_regression_dir,
            model="linear",
            linear_residuals=wandb_config.linear_residuals,
            y_hat=y_pred_for_lr,
        )

        # if (residual magnitude is too big... have max residual as a param):
        # then do geodesic regression

        print(f"linear_intercept_hat: {linear_intercept_hat.shape}")
        print(f"linear_coef_hat: {linear_coef_hat.shape}")
        print(f"y: {y.shape}")
        if wandb_config.dataset_name in ["synthetic_mesh", "real_mesh"]:
            print(f"mesh_faces: {mesh_faces.shape}")

        logging.info("\n- Geodesic Regression")
        (
            geodesic_intercept_hat,
            geodesic_coef_hat,
            gr,
        ) = training.fit_geodesic_regression(
            y,
            space,
            X,
            tol=tol,
            intercept_hat_guess=linear_intercept_hat,
            coef_hat_guess=linear_coef_hat,
            initialization=wandb_config.geodesic_initialization,
            linear_residuals=wandb_config.linear_residuals,
        )

        geodesic_duration_time = time.time() - start_time
        geodesic_intercept_err = gs.linalg.norm(geodesic_intercept_hat - true_intercept)
        geodesic_coef_err = gs.linalg.norm(geodesic_coef_hat - true_coef)

        n_iterations = gr.n_iterations
        n_function_evaluations = gr.n_fevaluations
        n_jacobian_evaluations = gr.n_jevaluations

        logging.info("Computing meshes along geodesic regression...")
        y_pred_for_gr = gr.predict(X)
        y_pred_for_gr = y_pred_for_gr.reshape(y.shape)

        gr_linear_residuals = gs.array(y_pred_for_gr) - gs.array(y)
        rmsd_geod = gs.linalg.norm(gr_linear_residuals) / gs.sqrt(len(y))

        gr_geod_residuals = space.metric.dist(y_pred_for_gr, y)

        if wandb_config.dataset_name in ["synthetic_mesh", "real_mesh"]:

            rmsd_geod = rmsd_geod / (len(mesh_sequence_vertices[0]) * mesh_diameter)

            wandb.log(
                {
                    "geodesic_intercept_hat": wandb.Object3D(
                        geodesic_intercept_hat.numpy()
                    ),
                    "geodesic_coef_hat": wandb.Object3D(geodesic_coef_hat.numpy()),
                    "y_pred_for_gr": wandb.Object3D(
                        y_pred_for_gr.detach().numpy().reshape((-1, 3))
                    ),
                    "n_faces": len(mesh_faces),
                    "n_vertices": len(mesh_sequence_vertices[0]),
                }
            )

        nrmsd_geod = rmsd_geod / gs.linalg.norm(y[0] - y[-1])

        wandb.log(
            {
                "geodesic_duration_time": geodesic_duration_time,
                "geodesic_intercept_err": geodesic_intercept_err,
                "geodesic_coef_err": geodesic_coef_err,
                "geodesic_initialization": wandb_config.geodesic_initialization,
                "n_geod_iterations": n_iterations,
                "n_geod_function_evaluations": n_function_evaluations,
                "n_geod_jacobian_evaluations": n_jacobian_evaluations,
                "rmsd_geod": rmsd_geod,
                "nrmsd_geod": nrmsd_geod,
                "gr_linear_residuals_hist": wandb.Histogram(
                    gr_linear_residuals.numpy()
                ),
                "gr_linear_residuals": gr_linear_residuals.numpy(),
                "gr_geod_residuals_hist": wandb.Histogram(gr_geod_residuals.numpy()),
                "gr_geod_residuals": gr_geod_residuals.numpy(),
            }
        )

        logging.info(f">> Duration (geodesic): {geodesic_duration_time:.3f} secs.")
        logging.info(">> Regression errors (geodesic):")
        logging.info(
            f"On intercept: {geodesic_intercept_err:.6f}, on coef: "
            f"{geodesic_coef_err:.6f}"
        )

        print(f"y_pred_for_gr: " f"{y_pred_for_gr.shape}")

        logging.info("Saving geodesic results...")
        training.save_regression_results(
            dataset_name=wandb_config.dataset_name,
            y=y,
            space=space,
            true_coef=true_coef,
            regr_intercept=geodesic_intercept_hat,
            regr_coef=geodesic_coef_hat,
            duration_time=geodesic_duration_time,
            results_dir=geodesic_regression_dir,
            model="geodesic",
            linear_residuals=wandb_config.linear_residuals,
            y_hat=y_pred_for_gr,
        )

        if (
            wandb_config.dataset_name in ["hypersphere", "hyperboloid"]
            and space.dim == 2
        ):
            fig = viz.benchmark_data_sequence(space, y, y_pred_for_lr, y_pred_for_gr)
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
    for (
        dataset_name,
        geodesic_initialization,
        linear_residuals,
        tol_factor,
        n_steps,
    ) in itertools.product(
        default_config.dataset_name,
        default_config.geodesic_initialization,
        default_config.linear_residuals,
        default_config.tol_factor,
        default_config.n_steps,
    ):
        main_config = {
            "dataset_name": dataset_name,
            "geodesic_initialization": geodesic_initialization,
            "linear_residuals": linear_residuals,
            "tol_factor": tol_factor,
            "n_steps": n_steps,
        }
        if dataset_name == "synthetic_mesh":
            for (
                n_X,
                noise_factor,
                linear_noise,
                n_subdivisions,
                ellipsoid_dims,
                (start_shape, end_shape),
            ) in itertools.product(
                default_config.n_X,
                default_config.noise_factor,
                default_config.linear_noise,
                default_config.n_subdivisions,
                default_config.ellipsoid_dims,
                zip(default_config.start_shape, default_config.end_shape),
            ):
                config = {
                    "n_X": n_X,
                    "start_shape": start_shape,
                    "end_shape": end_shape,
                    "noise_factor": noise_factor,
                    "linear_noise": linear_noise,
                    "n_subdivisions": n_subdivisions,
                    "ellipsoid_dims": ellipsoid_dims,
                    "ellipse_ratio_h_v": ellipsoid_dims[0] / ellipsoid_dims[-1],
                }
                config.update(main_config)
                main_run(config)

        elif dataset_name == "real_mesh":
            for hemisphere in default_config.hemisphere:
                config = {"hemisphere": hemisphere}
                config.update(main_config)
                main_run(config)

        elif dataset_name == "hypersphere" or dataset_name == "hyperboloid":
            for (n_X, noise_factor, linear_noise, space_dimension) in itertools.product(
                default_config.n_X,
                default_config.noise_factor,
                default_config.linear_noise,
                default_config.space_dimension,
            ):
                config = {
                    "n_X": n_X,
                    "noise_factor": noise_factor,
                    "space_dimension": space_dimension,
                    "linear_noise": linear_noise,
                }
                config.update(main_config)
                main_run(config)


main()
