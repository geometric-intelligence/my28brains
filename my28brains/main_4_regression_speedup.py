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

Run:
>>> python main_4_regression_speedup.py
"""
import itertools
import logging
import os
import time

import torch

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

device = torch.device(
    f"cuda:{default_config.use_cuda}" if torch.cuda.is_available() else "cpu"
)


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
    ) = data_utils.load(
        wandb_config
    )  # , device=device)

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
    )
    logging.info(f"\n- Tolerance calculated for geodesic regression: {tol:.3f}.")

    if wandb_config.dataset_name == "synthetic":
        logging.info(
            f"\n- Adding noise to data with factor: {wandb_config.noise_factor}"
        )
        mesh_sequence_vertices = data_utils.add_noise(
            mesh_sequence_vertices, wandb_config.noise_factor
        )

    logging.info("\n- Testing whether data subspace is euclidean.")
    (
        euclidean_subspace_via_ratio,
        euclidean_subspace_via_diffs,
    ) = parameterized_regression.euclidean_subspace_test(
        mesh_sequence_vertices, mesh_faces, wandb_config.tol_factor
    )
    logging.info(
        f"\n- Euclidean subspace via ratio: {euclidean_subspace_via_ratio}"
        f"\n- Euclidean subspace via diffs: {euclidean_subspace_via_diffs}"
    )

    # diffs_log = int(0)
    # if euclidean_subspace_via_diffs:
    #     diffs_log = int(1)

    # ratio_log = int(0)
    # if euclidean_subspace_via_ratio:
    #     diffs_log = int(1)

    wandb.log(
        {
            "mesh_diameter": mesh_diameter,
            "n_faces": len(mesh_faces),
            "ellipse_ratio_h_v": (
                wandb_config.ellipse_dimensions[0] / wandb_config.ellipse_dimensions[-1]
            ),
            "geodesic_tol": tol,
            "euclidean_subspace_via_ratio": euclidean_subspace_via_ratio,
            "euclidean_subspace_via_diffs": euclidean_subspace_via_diffs,
            "mesh_sequence_vertices": wandb.Object3D(
                mesh_sequence_vertices.numpy().reshape((-1, 3))
            ),
        }
    )

    logging.info("\n- Linear Regression")
    (
        linear_intercept_hat,
        linear_coef_hat,
        lr,
    ) = parameterized_regression.linear_regression(mesh_sequence_vertices, times)

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
    parameterized_regression.save_regression_results(
        dataset_name=wandb_config.dataset_name,
        sped_up=wandb_config.sped_up,
        mesh_sequence_vertices=gs.array(mesh_sequence_vertices),
        true_intercept_faces=gs.array(mesh_faces),
        true_coef=gs.array(true_coef),
        regression_intercept=linear_intercept_hat,
        regression_coef=linear_coef_hat,
        duration_time=linear_duration_time,
        regression_dir=linear_regression_dir,
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
    ) = parameterized_regression.geodesic_regression(
        mesh_sequence_vertices,
        mesh_faces,
        times,
        tol=tol,
        intercept_hat_guess=linear_intercept_hat,
        coef_hat_guess=linear_coef_hat,
        initialization=wandb_config.geodesic_initialization,
        geodesic_residuals=wandb_config.geodesic_residuals,
    )

    geodesic_duration_time = time.time() - start_time
    geodesic_intercept_err = gs.linalg.norm(geodesic_intercept_hat - true_intercept)
    geodesic_coef_err = gs.linalg.norm(geodesic_coef_hat - true_coef)

    logging.info("Computing meshes along geodesic regression...")
    meshes_along_geodesic_regression = gr.predict(times)
    meshes_along_geodesic_regression = meshes_along_geodesic_regression.reshape(
        mesh_sequence_vertices.shape
    )

    # geodesic_residuals = 0
    # if wandb_config.geodesic_residuals:
    #     geodesic_residuals = 1

    # geodesic_initialization = 0
    # if wandb_config.geodesic_initialization:
    #     geodesic_initialization = 1

    wandb.log(
        {
            "geodesic_duration_time": geodesic_duration_time,
            "geodesic_intercept_err": geodesic_intercept_err,
            "geodesic_coef_err": geodesic_coef_err,
            "geodesic_intercept_hat": wandb.Object3D(geodesic_intercept_hat.numpy()),
            "geodesic_coef_hat": wandb.Object3D(geodesic_coef_hat.numpy()),
            "exp_solver_n_steps": default_config.n_steps,
            "meshes_along_geodesic_regression": wandb.Object3D(
                meshes_along_geodesic_regression.detach().numpy().reshape((-1, 3))
            ),
        }
    )

    logging.info(f">> Duration (geodesic): {geodesic_duration_time:.3f} secs.")
    logging.info(">> Regression errors (geodesic):")
    logging.info(
        f"On intercept: {geodesic_intercept_err:.6f}, on coef: {geodesic_coef_err:.6f}"
    )

    print(f"meshes_along_geodesic_regression: {meshes_along_geodesic_regression.shape}")

    logging.info("Saving geodesic results...")
    parameterized_regression.save_regression_results(
        dataset_name=wandb_config.dataset_name,
        sped_up=wandb_config.sped_up,
        mesh_sequence_vertices=mesh_sequence_vertices,
        true_intercept_faces=mesh_faces,
        true_coef=gs.array(true_coef),
        regression_intercept=geodesic_intercept_hat,
        regression_coef=geodesic_coef_hat,
        duration_time=geodesic_duration_time,
        regression_dir=geodesic_regression_dir,
        meshes_along_regression=meshes_along_geodesic_regression,
    )

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
    ) in itertools.product(
        default_config.dataset_name,
        default_config.sped_up,
        default_config.geodesic_initialization,
        default_config.geodesic_residuals,
        default_config.tol_factor,
    ):
        main_config = {
            "dataset_name": dataset_name,
            "sped_up": sped_up,
            "geodesic_initialization": geodesic_initialization,
            "geodesic_residuals": geodesic_residuals,
            "tol_factor": tol_factor,
        }
        if dataset_name == "synthetic":
            for (
                n_times,
                noise_factor,
                n_subdivisions,
                ellipse_dimensions,
                (start_shape, end_shape),
            ) in itertools.product(
                default_config.n_times,
                default_config.noise_factor,
                default_config.n_subdivisions,
                default_config.ellipse_dimensions,
                zip(default_config.start_shape, default_config.end_shape),
            ):
                config = {
                    "n_times": n_times,
                    "start_shape": start_shape,
                    "end_shape": end_shape,
                    "noise_factor": noise_factor,
                    "n_subdivisions": n_subdivisions,
                    "ellipse_dimensions": ellipse_dimensions,
                }
                config.update(main_config)
                main_run(config)

        elif dataset_name == "real":
            for hemisphere in default_config.hemisphere:
                config = {"hemisphere": hemisphere}
                config.update(main_config)
                main_run(config)


main()
