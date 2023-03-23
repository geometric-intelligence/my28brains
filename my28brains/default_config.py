"""Default configuration."""

import os
import subprocess

use_cuda = 1
n_gpus = 10

# specify brain hemispheres to analyze
hemispheres = ["left"]  # , "right"]

# specify hippocampus structures to analyze
# 1   255    0    0        1  1  1    "CA1"
# 2     0  255    0        1  1  1    "CA2+3"
# 3     0    0  255        1  1  1    "DG"
# 4   255  255    0        1  1  1    "ERC"
# 5     0  255  255        1  1  1    "PHC"
# 6   255    0  255        1  1  1    "PRC"
# 7    80  179  221        1  1  1    "SUB"
# 8   255  215    0        1  1  1    "AntHipp"
# 9   184  115   51        1  1  1    "PostHipp"
# 2, 6 are expected to grow in volume with progesterone
# 4, 5 are expected to shrink in volume with progesterone

structure_ids = [1]

# number of time points along each interpolating geodesic
n_geodesic_times = [10]  # will not be used
stepsize = 1  # will not be used.
resolutions = 0  # don't do several resolutions for our case.

# range of days to interpolate in between
# Looking at the first 10 days is interesting because:
# - we have 10 gpus, so we can run 10 interpolations at once
# - they contain most of the progesterone peak.
day_range = [0, 7]

# face area threshold for non-degenerate meshes:
# the less we decimate, the more likely it is to have small faces
# thus the thresholt needs to be higher
area_thresholds = [0.0]

# build work path from git root path
gitroot_path = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
)
os.chdir(gitroot_path[:-1])
work_dir = os.getcwd()

data_dir = "/home/data/28andMeOC_correct"
meshes_dir = os.path.join(os.getcwd(), "my28brains", "results", "meshes")
centered_dir = os.path.join(os.getcwd(), "my28brains", "results", "meshes_centered")
centered_nondegenerate_dir = os.path.join(
    os.getcwd(), "my28brains", "results", "meshes_centered_nondegenerate"
)
geodesics_dir = os.path.join(os.getcwd(), "my28brains", "results", "meshes_geodesics")

for mesh_dir in [meshes_dir, centered_dir, centered_nondegenerate_dir, geodesics_dir]:
    if not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)

# parameters for h2_match
h2_dir = os.path.join(work_dir, "H2_SurfaceMatch")

# weighted l2 energy: penalizes how much you have to move points (vertices) weighted by local area around that vertex
a0 = 0.01
# In our case: could be higher (1 max)? if it's too high, it might shrink the mesh down, match and then blow up again
# See paper's figure that highlights links between these parameters.

a1 = 100  # penalizes stretching
b1 = 100  # penalizes shearing
c1 = 0.2  # penalizes change in normals: for high deformations we want c1 pretty low, e.g. when moving an arm.
# in our case try with a1 b1 a bit smaller (10), and c1 a bit large (1 or even up to 10)

# penalizes how a triangle rotate about normal vector,
# without stretching or shearing. almost never uses,
# usually d1 = 0, it's every thing that the others a1, b1, and c1, dont penalize
d1 = 0.01

a2 = 0.01  # high value = 1.
# If a2 is too high, we get bloding : it wants to blow up and get super smooth mesh and then shrink back down to get the matching
# a2 high wants to get a smooth mesh because we're penalizing the mesh laplacian

param1 = {
    "weight_coef_dist_T": 10**1,  # target varifold term
    "weight_coef_dist_S": 10**1,  # source varifold term
    "sig_geom": 0.4,
    "max_iter": 1000,  # bfgs gets really close really fast and sometimes worth letting it run for a bunch of iterations + see scipy, esp stopping condition to get decent figures
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
# important to have varifold term with high weight, to be sure that source + target are close to data.
# as the match gets better, the varifold terms are decreasing exponetially, thus we compensate back with the weights.
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

# param4 = {
#     "weight_coef_dist_T": 10**4,
#     "weight_coef_dist_S": 10**4,
#     "sig_geom": 0.1,
#     "max_iter": 1000,
#     "time_steps": 3,
#     "tri_unsample": False,
#     "index": 1,
# }

# param5 = {
#     "weight_coef_dist_T": 10**5,
#     "weight_coef_dist_S": 10**5,
#     "sig_geom": 0.1,
#     "max_iter": 1000,
#     "time_steps": 4,
#     "tri_unsample": False,
#     "index": 2,
# }

# param6 = {
#     "weight_coef_dist_T": 10**6,
#     "weight_coef_dist_S": 10**6,
#     "sig_geom": 0.05,
#     "max_iter": 1000,
#     "time_steps": 5,
#     "tri_unsample": False,
#     "index": 2,
# }

paramlist = [param1, param2, param3, param4]  # param5, param6]
