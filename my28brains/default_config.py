"""Default configuration.

Structure_ID names, numbers, and colors:
----------------------------------------
1   255    0    0        1  1  1    "CA1"
2     0  255    0        1  1  1    "CA2+3"
3     0    0  255        1  1  1    "DG"
4   255  255    0        1  1  1    "ERC"
5     0  255  255        1  1  1    "PHC"
6   255    0  255        1  1  1    "PRC"
7    80  179  221        1  1  1    "SUB"
8   255  215    0        1  1  1    "AntHipp"
9   184  115   51        1  1  1    "PostHipp"
2, 6 are expected to grow in volume with progesterone
4, 5 are expected to shrink in volume with progesterone

Information to help elastic metric parameter choosing:
----------------------------------------
The elastic metric is a weighted sum of 3 terms:
    - TODO: add description of the terms

The weights are:
a0 = weighted l2 energy:
    penalizes how much you have to move points (vertices)
    weighted by local area around that vertex.
    Max value 1. If it's too high, it might shrink the mesh down,
    match and then blow up again

a1 = penalizes stretching (was 10)
b1 = penalizes shearing (was 10)
c1 = (was 1) penalizes change in normals:
    for high deformations we want c1 pretty low,
    e.g. when moving an arm.
    in our case try with a1 b1 a bit smaller (10),
    and c1 a bit large (1 or even up to 10)
d1 = (was 0) penalizes how a triangle rotate about normal vector,
    without stretching or shearing. almost never used
    it is everything that a1, b1, c1 don't penalize.

a2 = (was 1) high value = 1. a2 penalizes the laplacian of the mesh.
    it wants to get a smooth mesh.
    if it is too high, we will get bloating. It wants to blow the
    mesh up to get a super smooth mesh.
"""

import datetime
import os
import subprocess
import sys

import torch

gitroot_path = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
)

os.chdir(os.path.join(gitroot_path[:-1], "my28brains"))

now = datetime.datetime.now()
use_wandb = True

# WANDB API KEY
# Find it here: https://wandb.ai/authorize
# Story it in file: api_key.txt (without extra line break)
with open("api_key.txt") as f:
    api_key = f.read()

# Fixed parameters
stepsize = {
    "synthetic": 55,
    "real": 6,
}

# Regression Parameters

dataset_name = ["synthetic", "real"]  # "synthetic" or "real"
sped_up = [True]  # 'True' or 'False' (not currently used)
geodesic_initialization = [
    "warm_start"
]  # "warm_start" or "random" (random on parka server)
geodesic_residuals = [False]  # 'True' or 'False' (alternative is linear residuals)
n_steps = 3  # n steps for the exp solver of geomstats.
tol_factor = [
    0.001,
    0.01,
    0.1,
    0.5,
]  # tolerance for geodesic regression. If none logged, value 0.001.
n_times = [5, 10, 15, 20, 30]  # Only for dataset_name == synthetic
start_shape = ["sphere"]  # "sphere" or "ellipsoid" for synthetic
end_shape = ["ellipsoid"]  # "sphere" or "ellipsoid" for synthetic
noise_factor = [0.0, 0.0001, 0.001, 0.01]  # noise added to the data.
# Will be multiplied by the size of the mesh to calculate the standard
# deviation of added noise distribution.
# only applied to synthetic data.
n_subdivisions = [
    1,
    2,
    3,
    4,
]
# How many times to subdivide the mesh. Note that the number of faces will grow
# as function of 4 ** subdivisions, so you probably want to keep this under ~5.
# if nothing recorded, value 3.
ellipse_dimensions = [
    [2, 2, 3],
    [2, 2, 10],
    [10, 10, 2],
]  # if nothing recorded, [2, 2, 3]

# GPU Parameters

use_cuda = 1
n_gpus = 10
torch_dtype = torch.float64

# Unparameterized Geodesic Mesh Parameters

# number of time points along each interpolating geodesic
resolutions = 0  # don't do several resolutions for our case.
# WORKING
# initial_decimation_fact = 10
# scaling_factor = 2*initial_decimation_fact
# NOT WORKING
initial_decimation_fact = 4
scaling_factor = 10

# Define template structure of the mesh that will be used
# for all mesh in the interpolation
# Every mesh will have the same number of vertices and faces.
i_template = 0

# face area threshold for non-degenerate meshes:
# the less we decimate, the more likely it is to have small faces
# thus the thresholt needs to be higher
area_thresholds = [0.00]  # 0.0001, 0.001, 0.01, 0.1, 1.0]

# Real Data Specific

# specify brain hemispheres to analyze
hemisphere = ["left"]  # , "right"]
structure_ids = [-1]


# range of days to interpolate in between
# Looking at the first 10 days is interesting because:
# - we have 10 gpus, so we can run 10 interpolations at once
# - they contain most of the progesterone peak.
# first menstrual cycle is day 1-30 (pre-pill)
day_range = [2, 11]  # we have parameterized meshes for days 2-11

# determine whether to generate parameterized data or to interpolate between
# meshes with a geodesic.
generate_parameterized_data = True
interpolate_geodesics = not generate_parameterized_data

# Build Paths

# build work path from git root path
gitroot_path = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
)
os.chdir(gitroot_path[:-1])
work_dir = os.getcwd()

my28brains_data_dir = "/home/data/28andMeOC_correct"
my28brains_dir = os.path.join(os.getcwd(), "my28brains")
h2_dir = os.path.join(os.getcwd(), "H2_SurfaceMatch")

# Data
data_dir = os.path.join(os.getcwd(), "data")
synthetic_data_dir = os.path.join(data_dir, "synthetic_data")

# Results
results_dir = os.path.join(os.getcwd(), "my28brains", "results")
meshed_data_dir = os.path.join(results_dir, "meshes")
centered_dir = os.path.join(results_dir, "meshes_centered")
centered_nondegenerate_dir = os.path.join(results_dir, "meshes_centered_nondegenerate")
geodesics_dir = os.path.join(results_dir, "meshes_geodesics")
parameterized_meshes_dir = os.path.join(results_dir, "meshes_parameterized")
sorted_parameterized_meshes_dir = os.path.join(
    results_dir, "meshes_parameterized_sorted_by_hormone"
)
regression_dir = os.path.join(results_dir, "regression")

sys_dir = os.path.dirname(work_dir)
sys.path.append(sys_dir)
sys.path.append(h2_dir)

for mesh_dir in [
    meshed_data_dir,
    centered_dir,
    centered_nondegenerate_dir,
    geodesics_dir,
    regression_dir,
    synthetic_data_dir,
    parameterized_meshes_dir,
    sorted_parameterized_meshes_dir,
]:
    if not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)
    sys.path.append(sys_dir)


# Elastic Metric Parameters

a0 = 0.01  # was 0.1
a1 = 10  # (was 10)
b1 = 10  # (was 10)
c1 = 1  # (was 1)
d1 = 0
a2 = 1  # (was 1)

# Unparameterized Geodesic Parameters

# NOTE: "time_steps" is the number of time points that
#  we will have in the interpolated geodesic
# the "time_steps" value in the last param will set the number
# of time points in the interpolated geodesic

param1 = {
    "weight_coef_dist_T": 10**1,  # target varifold term
    "weight_coef_dist_S": 10**1,  # source varifold term
    "sig_geom": 0.4,
    "max_iter": 1000,  # bfgs gets really close really fast and sometimes
    # worth letting it run for a bunch of iterations + see scipy,
    # esp stopping condition to get decent figures
    "time_steps": 2,
    "tri_unsample": False,  # False
    "index": 0,  # spatial resolution, should increase everytime we to a trip_upsample.
}

param2 = {
    "weight_coef_dist_T": 10
    ** 5,  # increase exponentially bcs in SNRF orignal code they had it and it works
    "weight_coef_dist_S": 10**5,
    "sig_geom": 0.1,
    "max_iter": 1000,
    "time_steps": 3,
    "tri_unsample": False,
    "index": 0,
}
# important to have varifold term with high weight,
# to be sure that source + target are close to data.
# as the match gets better, the varifold terms are decreasing exponetially,
# thus we compensate back with the weights.
# could do 10**1 and 10**5 if we're only doing two parameters.
# e.g. with 3 parameters, emmanuel did: 10**1, 10**5, 10**10
# e.g. corresponding sig_geom: 0.4, 0.1, 0.025


param3 = {
    "weight_coef_dist_T": 10**10,
    "weight_coef_dist_S": 10**10,
    "sig_geom": 0.025,
    "max_iter": 2000,
    "time_steps": 5,
    "tri_unsample": False,
    "index": 0,
}

paramlist = [param1, param2, param3]  # , param4]  # param5, param6]
