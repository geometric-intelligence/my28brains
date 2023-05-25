"""
Performs regression on parameterized meshes.

Mesh sequence chosen in default_config.py
Returns the slope and intercept of the regression fit.
"""
##################### Standard Imports #####################

import os
import subprocess
import sys

import numpy as np
import torch
import trimesh

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs

##################### Set up paths and imports #####################
gitroot_path = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
)
os.chdir(gitroot_path[:-1])
work_dir = os.getcwd()

my28brains_dir = os.path.join(work_dir, "my28brains")
data_dir = os.path.join(work_dir, "data")

sys_dir = os.path.dirname(work_dir)
sys.path.append(sys_dir)
sys.path.append(my28brains_dir)
sys.path.append(data_dir)

##################### Regression Imports #####################
import geomstats.visualization as visualization
import matplotlib.pyplot as plt
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.learning.frechet_mean import FrechetMean, variance
from geomstats.learning.geodesic_regression import GeodesicRegression

import data.synthetic_data.generate_syntetic_geodesics as generate_syntetic_geodesics
import my28brains.default_config as default_config
import my28brains.discrete_surfaces as discrete_surfaces
from my28brains.discrete_surfaces import DiscreteSurfaces, ElasticMetric

synthetic = default_config.synthetic
real_data = default_config.real_data


def perform_parameterized_regression(mesh_sequence, times):
    """Performs regression on parameterized meshes.

    inputs:
        mesh_sequence: list of trimesh objects
        times: list of times corresponding to mesh_sequence

    returns:
        intercept_hat: intercept of regression fit
        coef_hat: slope of regression fit
    """
    SURFACE_SPACE = DiscreteSurfaces(faces=mesh_sequence[0].faces)

    METRIC = ElasticMetric(
        space=SURFACE_SPACE,
        a0=default_config.a0,
        a1=default_config.a1,
        b1=default_config.b1,
        c1=default_config.c1,
        d1=default_config.d1,
        a2=default_config.a2,
    )

    # maxiter was 100
    gr = GeodesicRegression(
        SURFACE_SPACE,
        metric=METRIC,
        center_X=False,
        method="riemannian",
        max_iter=5,
        init_step_size=0.1,
        verbose=True,
        initialization="frechet",
    )

    gr.fit(times, mesh_sequence, compute_training_score=False)

    intercept_hat, coef_hat = gr.intercept_, gr.coef_

    # Save the regression fit
    true_intercept_file_name = "true_intercept"
    H2_SurfaceMatch.utils.input_output.save_data(file_name, extension, V, F)

    return intercept_hat, coef_hat
