import itertools
import logging
import os
import time

import torch

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

    reference_mesh = synthetic.generate_ellipsoid_mesh(
        subdivisions=config.subdivisions, ellipse_dimensions=[2, 2, 3]
    )
    reference_faces = reference_mesh.faces
    mesh = data_utils.add_noise(reference_mesh, config.noise_factor)

    start = time.time()
    linear_sq_dist = torch.linalg.norm(mesh - reference_mesh, axis=1) ** 2
    linear_duration = time.time() - start

    discrete_surfaces = DiscreteSurfaces(faces=gs.array(reference_faces))
    elastic_metric = ElasticMetric(space=discrete_surfaces)
    elastic_metric.exp_solver = _ExpSolver(n_steps=config.n_steps)

    start = time.time()
    geodesic_sq_dist = discrete_surfaces.metric.squared_dist(mesh, reference_mesh)
    geodesic_duration = time.time() - start

    diff_sq_dist = linear_sq_dist - geodesic_sq_dist
    relative_diff_sq_dist = diff_sq_dist / linear_sq_dist
    wandb.log(
        {
            "run_name": wandb.run.name,
            "noise_factor": config.noise_factor,
            "n_steps": config.n_steps,
            "subdivisions": config.subdivisions,
            "n_faces": len(reference_faces),
            "n_vertices": len(reference_mesh.vertices),
            "linear_sq_dist": linear_sq_dist,
            "geodesic_sq_dist": geodesic_sq_dist,
            "diff_sq_dist": diff_sq_dist,
            "relative_diff_sq_dist": relative_diff_sq_dist,
            "linear_duration": linear_duration,
            "geodesic_duration": geodesic_duration,
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
