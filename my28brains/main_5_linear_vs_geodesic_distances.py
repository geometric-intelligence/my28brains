"""Compare computation of linear vs geodesic square distances.

This computation aims to show that it is not realistic to perform
geodesic regression with the geodesic residuals.

Note: the following paths in are my python path:
/Users/ninamiolane/code/my28brains/H2_surfaceMatch:
/Users/ninamiolane/code/my28brains:
/Users/ninamiolane/code/:
/Users/ninamiolane/opt/anaconda3/envs/my28brains/bin:
/Users/ninamiolane/opt/anaconda3/envs/my28brains/lib/python3.10/site-packages
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
import wandb

NOISE_FACTORS = [0.01, 0.1]
N_STEPS = [3, 5]
SUBDIVISIONS = [1, 2, 3]


def main_run(config):
    """Compare computation of linear vs geodesic dist."""
    wandb.init()
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
    noisy_vertices = data_utils.add_noise(
        mesh_sequence_vertices=[reference_vertices],
        noise_factor=wandb_config.noise_factor,
    )
    noisy_vertices = noisy_vertices[0]

    logging.info("Computing linear squared distance.")
    start = time.time()
    linear_sq_dist = (
        gs.linalg.norm(noisy_vertices - reference_vertices, axis=1).numpy() ** 2
    )
    linear_duration = time.time() - start
    logging.info(f"--> Done ({linear_duration} sec): {linear_sq_dist}...")

    discrete_surfaces = DiscreteSurfaces(faces=gs.array(reference_faces))
    elastic_metric = ElasticMetric(space=discrete_surfaces)
    elastic_metric.exp_solver = _ExpSolver(n_steps=wandb_config.n_steps)

    logging.info("Computing geodesic squared distance...")
    start = time.time()
    geodesic_sq_dist = (
        discrete_surfaces.metric.squared_dist(noisy_vertices, reference_vertices)
        .detach()
        .numpy()
    )
    geodesic_duration = time.time() - start
    logging.info(f"--> Done ({geodesic_duration:.1f} sec): {geodesic_sq_dist:.3f}...")

    diff_sq_dist = linear_sq_dist - geodesic_sq_dist
    relative_diff_sq_dist = diff_sq_dist / linear_sq_dist
    diff_duration = linear_duration - geodesic_duration
    relative_diff_duration = diff_duration / linear_duration
    wandb.log(
        {
            "run_name": wandb.run.name,
            "noise_factor": wandb_config.noise_factor,
            "n_steps": wandb_config.n_steps,
            "subdivisions": wandb_config.subdivisions,
            "n_faces": len(reference_faces),
            "n_vertices": len(reference_vertices),
            "linear_sq_dist": linear_sq_dist,
            "geodesic_sq_dist": geodesic_sq_dist,
            "diff_sq_dist": diff_sq_dist,
            "relative_diff_sq_dist": relative_diff_sq_dist,
            "linear_duration": linear_duration,
            "geodesic_duration": geodesic_duration,
            "diff_duration": diff_duration,
            "relative_diff_duration": relative_diff_duration,
            "reference_vertices": wandb.Object3D(reference_vertices.numpy()),
        }
    )
    wandb.finish()


def main():
    """Parse the default_config file and launch all experiments."""
    for (
        noise_factor,
        n_steps,
        subdivisions,
    ) in itertools.product(NOISE_FACTORS, N_STEPS, SUBDIVISIONS):
        config = {
            "dataset_name": "synthetic",
            "n_steps": n_steps,
            "noise_factor": noise_factor,
            "subdivisions": subdivisions,
        }

    main_run(config)


main()
