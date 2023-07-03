"""Compare computation of linear vs geodesic square distances.

This computation aims to show that it is not realistic to perform
geodesic regression with the geodesic residuals.

Note: the following paths in are my python path:
export PYTHONPATH=/home/nmiolane/code/my28brains/H2_SurfaceMatch:
/home/nmiolane/code/my28brains/
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

import my28brains.datasets.synthetic as synthetic
import my28brains.datasets.utils as data_utils
import my28brains.viz as viz
import wandb

NOISE_FACTORS = [0.01, 0.1, 0.5, 1.0]
N_STEPS = [3, 5, 8]
SUBDIVISIONS = [1, 2, 3]
N_TIMES = [5, 10]


def main_run(config):
    """Compare computation of line vs geodesic."""
    wandb.init(project="line_vs_geodesic")
    wandb_config = wandb.config
    wandb_config.update(config)
    wandb.run.name = f"run_{wandb.run.id}"
    logging.info(f"\n\n---> MAIN 5 : START run: {wandb.run.name}.")
    logging.info(wandb_config)

    reference_mesh = synthetic.generate_ellipsoid_mesh(
        subdivisions=wandb_config.subdivisions, ellipse_dimensions=[2, 2, 3]
    )
    reference_vertices = gs.array(reference_mesh.vertices)
    reference_faces = reference_mesh.faces
    n_vertices = len(reference_vertices)
    n_faces = len(reference_faces)
    diameter = data_utils.mesh_diameter(reference_vertices)

    noiseless_vertices = gs.copy(reference_vertices)
    noisy_vertices = data_utils.add_noise(
        mesh_sequence_vertices=[reference_vertices],
        noise_factor=wandb_config.noise_factor,
    )
    noisy_vertices = noisy_vertices[0]

    wandb_config.update(
        {
            "n_faces": n_faces,
            "n_vertices": n_vertices,
            "diameter": diameter,
        }
    )

    logging.info("Computing linear distance.")
    start = time.time()
    linear_sq_dist = gs.linalg.norm(noisy_vertices - noiseless_vertices).numpy() ** 2
    linear_dist = gs.sqrt(linear_sq_dist)
    linear_regression_duration = time.time() - start

    logging.info("Computing line.")
    start = time.time()
    line = gs.array(
        [
            t * noiseless_vertices + (1 - t) * noisy_vertices
            for t in gs.linspace(0, 1, wandb_config.n_times)
        ]
    )
    line_duration = time.time() - start
    logging.info(
        f"--> Done ({linear_regression_duration:.1f} sec): linear_dist = {linear_dist}"
    )

    discrete_surfaces = DiscreteSurfaces(faces=gs.array(reference_faces))
    elastic_metric = ElasticMetric(space=discrete_surfaces)
    elastic_metric.exp_solver = _ExpSolver(n_steps=wandb_config.n_steps)

    logging.info("Computing geodesic distance.")
    start = time.time()
    geodesic_sq_dist = (
        discrete_surfaces.metric.squared_dist(noisy_vertices, noiseless_vertices)
        .detach()
        .numpy()
    )[0]
    geodesic_dist = gs.sqrt(geodesic_sq_dist)
    geodesic_regression_duration = time.time() - start

    logging.info("Computing geodesic.")
    start = time.time()
    geodesic_fn = elastic_metric.geodesic(
        initial_point=noiseless_vertices, end_point=noisy_vertices
    )
    geodesic = geodesic_fn(gs.linspace(0, 1, wandb_config.n_times))
    geodesic_duration = time.time() - start
    logging.info(
        f"--> Done ({geodesic_regression_duration:.1f} sec): "
        f"geodesic_dist = {geodesic_dist}..."
    )

    diff_dist = linear_dist - geodesic_dist
    relative_diff_dist = diff_dist / linear_dist
    diff_duration = linear_regression_duration - geodesic_regression_duration
    relative_diff_duration = diff_duration / linear_regression_duration

    diff_seq_per_time_and_vertex = gs.linalg.norm(line - geodesic) / (
        wandb_config.n_times * n_vertices
    )

    diff_seq_per_time_vertex_diameter = diff_seq_per_time_and_vertex / diameter
    diff_seq_duration = line_duration - geodesic_duration
    relative_diff_seq_duration = diff_seq_duration / line_duration

    offset_line = viz.offset_mesh_sequence(line)
    offset_geodesic = viz.offset_mesh_sequence(geodesic)

    wandb.log(
        {
            "run_name": wandb.run.name,
            "linear_dist": linear_dist,
            "linear_dist_per_vertex": linear_dist / n_vertices,
            "geodesic_dist": geodesic_dist,
            "geodesic_dist_per_vertex": geodesic_dist / n_vertices,
            "diff_dist": diff_dist,
            "diff_dist_per_vertex": diff_dist / n_vertices,
            "relative_diff_dist": relative_diff_dist,
            "line_duration": line_duration,
            "linear_regression_duration": linear_regression_duration,
            "linear_regression_duration_per_vertex": linear_regression_duration
            / n_vertices,
            "geodesic_regression_duration": geodesic_regression_duration,
            "geodesic_regression_duration_per_vertex": geodesic_regression_duration
            / n_vertices,
            "geodesic_duration": geodesic_duration,
            "diff_duration": diff_duration,
            "diff_duration_per_vertex": diff_duration / n_vertices,
            "relative_diff_duration": relative_diff_duration,
            "noiseless_vertices": wandb.Object3D(noiseless_vertices.numpy()),
            "noisy_vertices": wandb.Object3D(noisy_vertices.numpy()),
            "offset_line": wandb.Object3D(offset_line.numpy()),
            "offset_geodesic": wandb.Object3D(offset_geodesic.numpy()),
            "diff_seq_per_time_and_vertex": diff_seq_per_time_and_vertex,
            "diff_seq_per_time_vertex_diameter": diff_seq_per_time_vertex_diameter,
            "diff_seq_duration": diff_seq_duration,
            "diff_seq_duration_per_time_and_vertex": diff_seq_duration
            / (wandb_config.n_times * n_vertices),
            "relative_diff_seq_duration": relative_diff_seq_duration,
            "relative_diff_seq_per_time_and_vertex": relative_diff_seq_duration
            / (wandb_config.n_times * n_vertices),
        }
    )
    wandb.finish()


def main():
    """Parse the default_config file and launch all experiments."""
    for (
        noise_factor,
        n_steps,
        subdivisions,
        n_times,
    ) in itertools.product(NOISE_FACTORS, N_STEPS, SUBDIVISIONS, N_TIMES):
        config = {
            "dataset_name": "synthetic",
            "n_steps": n_steps,
            "noise_factor": noise_factor,
            "subdivisions": subdivisions,
            "n_times": n_times,
        }

        main_run(config)


main()
