"""Default configuration."""

# TODO: implement this in code

import os
import subprocess

# specify brain hemispheres to analyze
hemispheres = ["left", "right"]
# hemispheres = ["right", "left"]

# specify hippocampus structures to analyze
# structure_ids = list(range(1, 10))
structure_ids = list(range(6, 10))
# structure_ids = list(range(5, 6))
structure_ids.append(-1)

# build work path from git root path
gitroot_path = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
)
os.chdir(gitroot_path[:-1])
WORK_DIR = os.getcwd()

# build path to centered meshes (previously generated in jupyter notebooks)
CENTERED_MESHES_DIR = os.path.join(os.getcwd(), "data", "centered_meshes")

# build path to centered_nondegenerate meshes, and create directory if
# it does not exist
CENTERED_NONDEGENERATE_MESHES_DIR = os.path.join(
    os.getcwd(), "data", "centered_nondegenerate_meshes"
)
print("CENTERED_NONDEGENERATE_MESHES_DIR: ", CENTERED_NONDEGENERATE_MESHES_DIR)
if not os.path.exists(CENTERED_NONDEGENERATE_MESHES_DIR):
    os.makedirs(CENTERED_NONDEGENERATE_MESHES_DIR)

# geodesics directory
GEODESICS_DIR = os.path.join(os.getcwd(), "data", "geodesics")
print("GEODESICS_DIR: ", GEODESICS_DIR)
if not os.path.exists(GEODESICS_DIR):
    os.makedirs(GEODESICS_DIR)

# parameters for h2_match
a0 = 0.01
a1 = 100
b1 = 100
c1 = 0.2
d1 = 0.01
a2 = 0.01

param1 = {
    "weight_coef_dist_T": 10**1,
    "weight_coef_dist_S": 10**1,
    "sig_geom": 0.4,
    "max_iter": 2000,
    "time_steps": 2,
    "tri_unsample": True,
    "index": 0,
}

param2 = {
    "weight_coef_dist_T": 10**2,
    "weight_coef_dist_S": 10**2,
    "sig_geom": 0.3,
    "max_iter": 1000,
    "time_steps": 2,
    "tri_unsample": False,
    "index": 1,
}

param3 = {
    "weight_coef_dist_T": 10**3,
    "weight_coef_dist_S": 10**3,
    "sig_geom": 0.2,
    "max_iter": 1000,
    "time_steps": 2,
    "tri_unsample": False,
    "index": 1,
}

param4 = {
    "weight_coef_dist_T": 10**4,
    "weight_coef_dist_S": 10**4,
    "sig_geom": 0.1,
    "max_iter": 1000,
    "time_steps": 3,
    "tri_unsample": True,
    "index": 1,
}

param5 = {
    "weight_coef_dist_T": 10**5,
    "weight_coef_dist_S": 10**5,
    "sig_geom": 0.1,
    "max_iter": 1000,
    "time_steps": 4,
    "tri_unsample": False,
    "index": 2,
}

param6 = {
    "weight_coef_dist_T": 10**6,
    "weight_coef_dist_S": 10**6,
    "sig_geom": 0.05,
    "max_iter": 1000,
    "time_steps": 5,
    "tri_unsample": False,
    "index": 2,
}

paramlist = [param1, param2, param3, param4, param5, param6]
