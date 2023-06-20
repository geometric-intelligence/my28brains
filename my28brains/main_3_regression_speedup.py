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
import itertools
import logging
import os
import time

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs

import my28brains.datasets.utils as data_utils
import my28brains.default_config as default_config
import my28brains.parameterized_regression as parameterized_regression
import wandb

my28brains_dir = default_config.my28brains_dir
synthetic_data_dir = default_config.synthetic_data_dir
parameterized_meshes_dir = default_config.parameterized_meshes_dir
data_dir = default_config.data_dir


def main_run(config):
    """Run the regression.

    This corresponds to one wandb run (no wandb sweep).
    """
    wandb.init()
    wandb_config = wandb.config
    wandb_config.update(config)

    run_name = f"run_{wandb.run.id}"
    wandb.run.name = run_name
    logging.info(f"\n\n---> START run: {run_name}.")

    linear_regression_dir = os.path.join(
        default_config.regression_dir, f"{run_name}_linear"
    )
    geodesic_regression_dir = os.path.join(
        default_config.regression_dir, f"{run_name}_geodesic"
    )
    for one_regression_dir in [linear_regression_dir, geodesic_regression_dir]:
        if not os.path.exists(one_regression_dir):
            os.makedirs(one_regression_dir)

    start_time = time.time()
    (
        mesh_sequence_vertices,
        mesh_faces,
        times,
        true_intercept,
        true_coef,
    ) = data_utils.load(wandb_config)

    logging.info("\n- Linear Regression")
    linear_intercept_hat, linear_coef_hat = parameterized_regression.linear_regression(
        mesh_sequence_vertices, times
    )

    linear_duration_time = time.time() - start_time
    linear_intercept_err = gs.linalg.norm(linear_intercept_hat - true_intercept)
    linear_coef_err = gs.linalg.norm(linear_coef_hat - true_coef)

    offsets = gs.linspace(0, 200, len(times))
    offset_mesh_sequence_vertices = []
    for i_mesh, mesh in enumerate(mesh_sequence_vertices):
        offset_mesh_sequence_vertices.append(mesh + offsets[i_mesh])
    offset_mesh_sequence_vertices = gs.vstack(offset_mesh_sequence_vertices)

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
        }
    )

    logging.info(f">> Duration (linear) = {linear_duration_time:.3f} secs.")
    logging.info(">> Regression errors (linear):")
    logging.info(
        f"On intercept: {linear_intercept_err:.6f}, on coef: {linear_coef_err:.6f}"
    )

    logging.info("Saving linear results...")
    parameterized_regression.save_regression_results(
        wandb_config.dataset_name,
        wandb_config.sped_up,
        gs.array(mesh_sequence_vertices),
        gs.array(mesh_faces),
        gs.array(true_coef),
        regression_intercept=linear_intercept_hat,
        regression_coef=linear_coef_hat,
        duration_time=linear_duration_time,
        regression_dir=linear_regression_dir,
    )

    # if (residual magnitude is too big... have max residual as a param):
    # then do geodesic regression

    logging.info("\n- Geodesic Regression")
    (
        geodesic_intercept_hat,
        geodesic_coef_hat,
    ) = parameterized_regression.geodesic_regression(
        mesh_sequence_vertices,
        mesh_faces,
        times,
        tol=10000,
        intercept_hat_guess=linear_intercept_hat,
        coef_hat_guess=linear_coef_hat,
        initialization="warm_start",
    )

    geodesic_duration_time = time.time() - start_time
    geodesic_intercept_err = gs.linalg.norm(geodesic_intercept_hat - true_intercept)
    geodesic_coef_err = gs.linalg.norm(geodesic_coef_hat - true_coef)

    wandb.log(
        {
            "geodesic_duration_time": geodesic_duration_time,
            "geodesic_intercept_err": geodesic_intercept_err,
            "geodesic_coef_err": geodesic_coef_err,
            "geodesic_intercept_hat": wandb.Object3D(geodesic_intercept_hat.numpy()),
            "geodesic_coef_hat": wandb.Object3D(geodesic_coef_hat.numpy()),
        }
    )

    logging.info(f">> Duration (geodesic): {geodesic_duration_time:.3f} secs.")
    logging.info(">> Regression errors (geodesic):")
    logging.info(
        f"On intercept: {geodesic_intercept_err:.6f}, on coef: {geodesic_coef_err:.6f}"
    )

    logging.info("Saving geodesic results...")
    parameterized_regression.save_regression_results(
        wandb_config.dataset_name,
        wandb_config.sped_up,
        mesh_sequence_vertices,
        mesh_faces,
        gs.array(true_coef),
        regression_intercept=geodesic_intercept_hat,
        regression_coef=geodesic_coef_hat,
        duration_time=geodesic_duration_time,
        regression_dir=geodesic_regression_dir,
    )

    wandb.finish()


def main():
    """Parse the default_config file and launch all experiments.

    This launches experiments with wandb with different config parameters.
    """
    for (
        dataset_name,
        sped_up,
        gr_with_linear_warm_start,
        gr_with_linear_residuals,
    ) in itertools.product(
        default_config.dataset_name,
        default_config.sped_up,
        default_config.gr_with_linear_warm_start,
        default_config.gr_with_linear_residuals,
    ):
        main_config = {
            "dataset_name": dataset_name,
            "sped_up": sped_up,
            "gr_with_linear_warm_start": gr_with_linear_warm_start,
            "gr_with_linear_residuals": gr_with_linear_residuals,
        }
        if dataset_name == "synthetic":
            for n_times, (start_shape, end_shape) in itertools.product(
                default_config.n_times,
                zip(default_config.start_shape, default_config.end_shape),
            ):
                config = {
                    "n_times": n_times,
                    "start_shape": start_shape,
                    "end_shape": end_shape,
                }
                config.update(main_config)
                main_run(config)

        elif dataset_name == "real":
            for hemisphere in default_config.hemisphere:
                config = {"hemisphere": hemisphere}
                config.update(main_config)
                main_run(config)


main()
