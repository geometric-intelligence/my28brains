"""Compare computation of linear vs geodesic square distances.

This computation aims to show that it is not realistic to perform
geodesic regression with the geodesic residuals.

Note: the following paths in are my python path:
For Nina: export PYTHONPATH=/home/nmiolane/code/my28brains/H2_SurfaceMatch:
/home/nmiolane/code/my28brains/

For Adele: export PYTHONPATH=/home/adele/code/my28brains/H2_SurfaceMatch:/home/adele/code/my28brains/
"""

import itertools
import logging
import os
import time

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs
from geomstats.geometry.discrete_surfaces import (
    DiscreteSurfaces,
    ElasticMetric,
    _ExpSolver,
)
from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean, variance

import src.datasets.synthetic as synthetic
import src.datasets.utils as data_utils
import src.viz as viz
import wandb

NOISE_FACTORS = [0, 0.01, 0.1, 0.5, 1.0]  # [0.01, 0.1, 0.5, 1.0]
N_STEPS = [1, 3, 5, 8, 10, 20]  # 3, 5, 8, 10, 20]  #
SUBDIVISIONS = [1, 2, 3]  # [1, 2, 3]
N_TIMES = [5, 10]  # 5, 10
DATASET_NAMES = [
    "hypersphere",
    "hyperboloid",
]  # "synthetic_mesh", "hypersphere", "hyperboloid"
END_MESH = ["different_shape", "noisy"]  # noisy, different_shape

# NOTE: need to do noise_factor 1.0 n_X 10, n_steps 5


def load(config):
    """Load space, start point, end point.

    Parameters
    ----------
    reference mesh: trimesh object. start point.
    """
    if config.dataset_name == "synthetic_mesh":
        reference_mesh = synthetic.generate_ellipsoid_mesh(
            subdivisions=config.subdivisions, ellipsoid_dims=[2, 2, 3]
        )
        reference_vertices = gs.array(reference_mesh.vertices)
        reference_faces = reference_mesh.faces
        start_point = reference_vertices

        if config.end_mesh == "noisy":
            noiseless_vertices = gs.copy(reference_vertices)
            noisy_vertices = synthetic.add_linear_noise(
                mesh_sequence_vertices=[noiseless_vertices],
                dataset_name=config.dataset_name,
                noise_factor=config.noise_factor,
            )
            end_point = noisy_vertices[0]
        elif config.end_mesh == "different_shape":
            end_point = synthetic.generate_sphere_mesh(subdivisions=config.subdivisions)
        space = DiscreteSurfaces(faces=gs.array(reference_faces))
        elastic_metric = ElasticMetric(space=space)
        elastic_metric.exp_solver = _ExpSolver(n_steps=config.n_steps)
        space.metric = elastic_metric
        true_coef = end_point - start_point
    elif config.dataset_name in ["hypersphere", "hyperboloid"]:
        if config.dataset_name == "hypersphere":
            space = Hypersphere(dim=2)
        else:
            space = Hyperbolic(dim=2, default_coords_type="extrinsic")

        _, y, y_noisy, _, true_coef, _ = synthetic.generate_noisy_benchmark_data(
            space, n_samples=2
        )

        if config.end_mesh == "different_shape":
            start_point = y[0]
            end_point = y[1]
        elif config.end_mesh == "noisy":
            start_point = y_noisy[0]
            end_point = y_noisy[1]

    return space, start_point, end_point, true_coef


def main_run(config):
    """Compare computation of line vs geodesic."""
    wandb.init(project="line_vs_geodesic")
    wandb_config = wandb.config
    wandb_config.update(config)
    wandb.run.name = f"run_{wandb.run.id}"
    logging.info(f"\n\n---> MAIN 5 : START run: {wandb.run.name}.")
    logging.info(wandb_config)

    space, start_point, end_point, true_coef = load(wandb_config)

    if wandb_config.dataset_name == "synthetic_mesh":
        n_vertices = len(start_point)
        n_faces = len(start_point)
        diameter = data_utils.mesh_diameter(start_point)
        wandb_config.update(
            {
                "n_faces": n_faces,
                "n_vertices": n_vertices,
                "diameter": diameter,
            }
        )

    logging.info("Computing geodesic distance.")
    start = time.time()
    geodesic_sq_dist = (
        space.metric.squared_dist(start_point, end_point).detach().numpy()
    )  # [0]
    geodesic_dist = gs.sqrt(geodesic_sq_dist)
    geodesic_regression_duration = time.time() - start

    logging.info("Computing geodesic.")
    start = time.time()
    geodesic_fn = space.metric.geodesic(
        initial_point=start_point, initial_tangent_vec=true_coef
    )
    geodesic = geodesic_fn(gs.linspace(0, 1, wandb_config.n_X))
    geodesic_duration = time.time() - start
    logging.info(
        f"--> Done ({geodesic_regression_duration:.1f} sec): "
        f"geodesic_dist = {geodesic_dist}..."
    )

    q_start = geodesic[0]
    q_end = geodesic[-1]

    logging.info("Computing linear distance.")
    start = time.time()
    linear_sq_dist = gs.linalg.norm(q_start - q_end).numpy() ** 2
    linear_dist = gs.sqrt(linear_sq_dist)
    linear_regression_duration = time.time() - start

    logging.info("Computing line.")
    start = time.time()
    line = gs.array(
        [t * q_end + (1 - t) * q_start for t in gs.linspace(0, 1, wandb_config.n_X)]
    )

    line_duration = time.time() - start
    logging.info(
        f"--> Done ({linear_regression_duration:.1f} sec): linear_dist = {linear_dist}"
    )

    diff_dist = linear_dist - geodesic_dist
    relative_diff_dist = diff_dist / linear_dist
    diff_duration = linear_regression_duration - geodesic_regression_duration
    relative_diff_duration = diff_duration / linear_regression_duration

    diff_seq_per_time = gs.linalg.norm(line - geodesic) / (wandb_config.n_X)
    rmsd = gs.linalg.norm(line - geodesic) / gs.sqrt(wandb_config.n_X)
    abs_seq = gs.abs(line - geodesic)

    diff_seq_duration = line_duration - geodesic_duration
    relative_diff_seq_duration = diff_seq_duration / line_duration

    if wandb_config.dataset_name == "synthetic_mesh":
        diff_seq_per_time_and_vertex = diff_seq_per_time / (n_vertices)
        rmsd = rmsd / n_vertices

        diff_seq_per_time_vertex_diameter = diff_seq_per_time_and_vertex / diameter
        rmsd_diameter = rmsd / diameter

        offset_line = gs.array(viz.offset_mesh_sequence(line))
        offset_geodesic = gs.array(viz.offset_mesh_sequence(geodesic))
        print(f"offset_line.shape = {offset_line.shape}")

        wandb.log(
            {
                "linear_dist_per_vertex": linear_dist / n_vertices,
                "geodesic_dist_per_vertex": geodesic_dist / n_vertices,
                "diff_dist_per_vertex": diff_dist / n_vertices,
                "linear_regression_duration_per_vertex": linear_regression_duration
                / n_vertices,
                "geodesic_regression_duration_per_vertex": geodesic_regression_duration
                / n_vertices,
                "start_point_vertices": wandb.Object3D(start_point.numpy()),
                "end_point_vertices": wandb.Object3D(end_point.numpy()),
                "offset_line": wandb.Object3D(
                    offset_line.detach().numpy().reshape((-1, 3))
                ),
                "offset_geodesic": wandb.Object3D(
                    offset_geodesic.detach().numpy().reshape((-1, 3))
                ),
                "diff_seq_per_time_and_vertex": diff_seq_per_time_and_vertex,
                "diff_seq_per_time_vertex_diameter": diff_seq_per_time_vertex_diameter,
                "rmsd_diameter": rmsd_diameter,
                "diff_seq_duration_per_time_and_vertex": diff_seq_duration
                / (wandb_config.n_X * n_vertices),
                "relative_diff_seq_per_time_and_vertex": relative_diff_seq_duration
                / (wandb_config.n_X * n_vertices),
                "diff_duration_per_vertex": diff_duration / n_vertices,
            }
        )

    if wandb_config.dataset_name in ["hypersphere", "hyperboloid"]:
        fig = viz.benchmark_data_sequence(space, line, geodesic)
        plt = wandb.Image(fig)

        # TODO: why is this viz not changing?

        wandb.log(
            {
                "line vs geodesic": plt,
            }
        )

    wandb.log(
        {
            "run_name": wandb.run.name,
            "linear_dist": linear_dist,
            "geodesic_dist": geodesic_dist,
            "diff_dist": diff_dist,
            "relative_diff_dist": relative_diff_dist,
            "line_duration": line_duration,
            "linear_regression_duration": linear_regression_duration,
            "geodesic_regression_duration": geodesic_regression_duration,
            "geodesic_duration": geodesic_duration,
            "diff_duration": diff_duration,
            "relative_diff_duration": relative_diff_duration,
            "rmsd": rmsd,
            "abs_seq": abs_seq,
            "diff_seq_duration": diff_seq_duration,
            "relative_diff_seq_duration": relative_diff_seq_duration,
            # Log actual lines
            "line": line.numpy(),
            "geodesic": geodesic.numpy(),
            "post_error": True,
        }
    )
    wandb.finish()


def main():
    """Parse the default_config file and launch all experiments."""
    for (
        noise_factor,
        n_steps,
        subdivisions,
        n_X,
        dataset_name,
        end_mesh,
    ) in itertools.product(
        NOISE_FACTORS, N_STEPS, SUBDIVISIONS, N_TIMES, DATASET_NAMES, END_MESH
    ):
        config = {
            "dataset_name": dataset_name,
            "n_steps": n_steps,
            "noise_factor": noise_factor,
            "subdivisions": subdivisions,
            "n_X": n_X,
            "end_mesh": end_mesh,
        }

        main_run(config)


main()
