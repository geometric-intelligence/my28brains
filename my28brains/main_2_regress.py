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

regress_dir = default_config.regress_dir


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

        linear_regress_dir = os.path.join(regress_dir, f"{run_name}_linear")
        geodesic_regress_dir = os.path.join(regress_dir, f"{run_name}_geodesic")
        for one_regress_dir in [linear_regress_dir, geodesic_regress_dir]:
            if not os.path.exists(one_regress_dir):
                os.makedirs(one_regress_dir)

        start_time = time.time()
        (
            mesh_sequence_vertices,
            mesh_faces,
            times,
            true_intercept,
            true_coef,
        ) = data_utils.load(wandb_config)

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
        logging.info(f"\n- Tolerance calculated for geodesic regression: {tol:.3f}.")

        logging.info("\n- Testing whether data subspace is euclidean.")
        euclidean_subspace, diff_tolerance = check_euclidean.subspace_test(
            mesh_sequence_vertices,
            times,
            wandb_config.tol_factor,
        )
        logging.info(f"\n- Euclidean subspace: {euclidean_subspace}, ")

        wandb.log(
            {
                "mesh_diameter": mesh_diameter,
                "n_faces": len(mesh_faces),
                "geodesic_tol": tol,
                "euclidean_subspace": euclidean_subspace,
                "mesh_sequence_vertices": wandb.Object3D(
                    mesh_sequence_vertices.numpy().reshape((-1, 3))
                ),
                "test_diff_tolerance": diff_tolerance,
            }
        )

        logging.info("\n- Linear Regression")
        (
            linear_intercept_hat,
            linear_coef_hat,
            lr,
        ) = training.fit_linear_regression(mesh_sequence_vertices, times)

        linear_duration_time = time.time() - start_time
        linear_intercept_err = gs.linalg.norm(linear_intercept_hat - true_intercept)
        linear_coef_err = gs.linalg.norm(linear_coef_hat - true_coef)

        logging.info("Computing meshes along linear regression...")
        times_for_lr = gs.array(times.reshape(len(times), 1))
        meshes_along_linear_regression = lr.predict(times_for_lr)
        meshes_along_linear_regression = meshes_along_linear_regression.reshape(
            mesh_sequence_vertices.shape
        )
        print(f"meshes_along_linear_regression: {meshes_along_linear_regression.shape}")

        offset_mesh_sequence_vertices = viz.offset_mesh_sequence(mesh_sequence_vertices)

        wandb.log(
            {
                "linear_duration_time": linear_duration_time,
                "linear_intercept_err": linear_intercept_err,
                "linear_coef_err": linear_coef_err,
                "true_intercept": wandb.Object3D(true_intercept.numpy()),
                "true_coef": wandb.Object3D(true_coef.numpy()),
                "linear_intercept_hat": wandb.Object3D(linear_intercept_hat.numpy()),
                "linear_coef_hat": wandb.Object3D(linear_coef_hat.numpy()),
                "offset_mesh_sequence_vertices": wandb.Object3D(
                    offset_mesh_sequence_vertices.numpy()
                ),
                "meshes_along_linear_regression": wandb.Object3D(
                    meshes_along_linear_regression.reshape((-1, 3))
                ),
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
            sped_up=wandb_config.sped_up,
            mesh_sequence_vertices=gs.array(mesh_sequence_vertices),
            true_intercept_faces=gs.array(mesh_faces),
            true_coef=gs.array(true_coef),
            regr_intercept=linear_intercept_hat,
            regr_coef=linear_coef_hat,
            duration_time=linear_duration_time,
            regress_dir=linear_regress_dir,
            meshes_along_regression=meshes_along_linear_regression,
        )

        # if (residual magnitude is too big... have max residual as a param):
        # then do geodesic regression

        print(f"linear_intercept_hat: {linear_intercept_hat.shape}")
        print(f"linear_coef_hat: {linear_coef_hat.shape}")
        print(f"mesh_sequence_vertices: {mesh_sequence_vertices.shape}")
        print(f"mesh_faces: {mesh_faces.shape}")

        logging.info("\n- Geodesic Regression")
        (
            geodesic_intercept_hat,
            geodesic_coef_hat,
            gr,
        ) = training.fit_geodesic_regression(
            mesh_sequence_vertices,
            mesh_faces,
            times,
            tol=tol,
            intercept_hat_guess=linear_intercept_hat,
            coef_hat_guess=linear_coef_hat,
            initialization=wandb_config.geodesic_initialization,
            geodesic_residuals=wandb_config.geodesic_residuals,
            n_steps=wandb_config.n_steps,
        )

        geodesic_duration_time = time.time() - start_time
        geodesic_intercept_err = gs.linalg.norm(geodesic_intercept_hat - true_intercept)
        geodesic_coef_err = gs.linalg.norm(geodesic_coef_hat - true_coef)

        logging.info("Computing meshes along geodesic regression...")
        meshes_along_geodesic_regression = gr.predict(times)
        meshes_along_geodesic_regression = meshes_along_geodesic_regression.reshape(
            mesh_sequence_vertices.shape
        )

        wandb.log(
            {
                "geodesic_duration_time": geodesic_duration_time,
                "geodesic_intercept_err": geodesic_intercept_err,
                "geodesic_coef_err": geodesic_coef_err,
                "geodesic_intercept_hat": wandb.Object3D(
                    geodesic_intercept_hat.numpy()
                ),
                "geodesic_coef_hat": wandb.Object3D(geodesic_coef_hat.numpy()),
                "meshes_along_geodesic_regression": wandb.Object3D(
                    meshes_along_geodesic_regression.detach().numpy().reshape((-1, 3))
                ),
                "n_faces": len(mesh_faces),
                "geodesic_initialization": wandb_config.geodesic_initialization,
            }
        )

        logging.info(f">> Duration (geodesic): {geodesic_duration_time:.3f} secs.")
        logging.info(">> Regression errors (geodesic):")
        logging.info(
            f"On intercept: {geodesic_intercept_err:.6f}, on coef: "
            f"{geodesic_coef_err:.6f}"
        )

        print(
            f"meshes_along_geodesic_regression: "
            f"{meshes_along_geodesic_regression.shape}"
        )

        logging.info("Saving geodesic results...")
        training.save_regression_results(
            dataset_name=wandb_config.dataset_name,
            sped_up=wandb_config.sped_up,
            mesh_sequence_vertices=mesh_sequence_vertices,
            true_intercept_faces=mesh_faces,
            true_coef=true_coef,
            regr_intercept=geodesic_intercept_hat,
            regr_coef=geodesic_coef_hat,
            duration_time=geodesic_duration_time,
            regress_dir=geodesic_regress_dir,
            meshes_along_regression=meshes_along_geodesic_regression,
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
        sped_up,
        geodesic_initialization,
        geodesic_residuals,
        tol_factor,
        n_steps,
    ) in itertools.product(
        default_config.dataset_name,
        default_config.sped_up,
        default_config.geodesic_initialization,
        default_config.geodesic_residuals,
        default_config.tol_factor,
        default_config.n_steps,
    ):
        main_config = {
            "dataset_name": dataset_name,
            "sped_up": sped_up,
            "geodesic_initialization": geodesic_initialization,
            "geodesic_residuals": geodesic_residuals,
            "tol_factor": tol_factor,
            "n_steps": n_steps,
        }
        if dataset_name == "synthetic":
            for (
                n_times,
                noise_factor,
                n_subdivisions,
                ellipsoid_dims,
                (start_shape, end_shape),
            ) in itertools.product(
                default_config.n_times,
                default_config.noise_factor,
                default_config.n_subdivisions,
                default_config.ellipsoid_dims,
                zip(default_config.start_shape, default_config.end_shape),
            ):
                config = {
                    "n_times": n_times,
                    "start_shape": start_shape,
                    "end_shape": end_shape,
                    "noise_factor": noise_factor,
                    "n_subdivisions": n_subdivisions,
                    "ellipsoid_dims": ellipsoid_dims,
                    "ellipse_ratio_h_v": ellipsoid_dims[0] / ellipsoid_dims[-1],
                }
                config.update(main_config)
                main_run(config)

        elif dataset_name == "real":
            for hemisphere in default_config.hemisphere:
                config = {"hemisphere": hemisphere}
                config.update(main_config)
                main_run(config)


main()
