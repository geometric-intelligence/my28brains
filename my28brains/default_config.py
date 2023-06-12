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
a0 = weighted l2 energy: penalizes how much you have to move points (vertices) weighted by local area around that vertex.
        Max value 1. If it's too high, it might shrink the mesh down, match and then blow up again

a1 = penalizes stretching (was 10)
b1 = penalizes shearing (was 10)
c1 = (was 1) penalizes change in normals: for high deformations we want c1 pretty low, e.g. when moving an arm.
        in our case try with a1 b1 a bit smaller (10), and c1 a bit large (1 or even up to 10)
d1 = (was 0) penalizes how a triangle rotate about normal vector, without stretching or shearing. almost never used
        it is everything that a1, b1, c1 don't penalize.

a2 = (was 1) high value = 1. a2 penalizes the laplacian of the mesh. it wants to get a smooth mesh.
        if it is too high, we will get bloating. It wants to blow the mesh up to get a super smooth mesh.
"""

import os
import subprocess
import time

import numpy as np

##################### Regression Parameters to be adjusted #####################

data_type = "synthetic"  # "synthetic" or "real"
if data_type == "synthetic":
    n_times = 5
    start_shape = "sphere"  # "sphere" or "ellipsoid" or "pill"
    end_shape = "ellipsoid"  # "sphere" or "ellipsoid" or "pill"
sped_up = False  # 'True' or 'False'
geodesic_regression_with_linear_warm_start = True  # 'True' or 'False'
geodesic_regression_with_linear_residual_calculations = False  # 'True' or 'False'

now = time.time()


##################### GPU Parameters #####################

use_cuda = 1
n_gpus = 10

##################### Unparameterized Geodesic Mesh Parameters #####################

# number of time points along each interpolating geodesic
resolutions = 0  # don't do several resolutions for our case.
# WORKING
# initial_decimation_fact = 10
# scaling_factor = 2*initial_decimation_fact
# NOT WORKING
initial_decimation_fact = 4
scaling_factor = 10

# Define template structure of the mesh that will be used for all mesh in the interpolation
# Every mesh will have the same number of vertices and faces.
i_template = 0

# face area threshold for non-degenerate meshes:
# the less we decimate, the more likely it is to have small faces
# thus the thresholt needs to be higher
area_thresholds = [0.00]  # 0.0001, 0.001, 0.01, 0.1, 1.0]

#################### Real Data Specific ####################

# specify brain hemispheres to analyze
hemispheres = ["left"]  # , "right"]
structure_ids = [-1]


# range of days to interpolate in between
# Looking at the first 10 days is interesting because:
# - we have 10 gpus, so we can run 10 interpolations at once
# - they contain most of the progesterone peak.
day_range = [0, 30]  # first menstrual cycle is day 1-30 (pre-pill)

# determine whether to generate parameterized data or to interpolate between
# meshes with a geodesic.
generate_parameterized_data = True
interpolate_geodesics = not generate_parameterized_data

#################### Building Paths ####################

# build work path from git root path
gitroot_path = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
)
os.chdir(gitroot_path[:-1])
work_dir = os.getcwd()

my28brains_data_dir = "/home/data/28andMeOC_correct"
my28brains_dir = os.path.join(os.getcwd(), "my28brains")

## Data ##
data_dir = os.path.join(os.getcwd(), "data")
synthetic_data_dir = os.path.join(data_dir, "synthetic_data")
start_shape_dir = os.path.join(synthetic_data_dir, start_shape)
end_shape_dir = os.path.join(synthetic_data_dir, end_shape)
synthetic_mesh_sequence_dir = os.path.join(
    synthetic_data_dir, f"geodesic_{start_shape}_{end_shape}_{n_times}"
)

## Results ##
meshed_data_dir = os.path.join(os.getcwd(), "my28brains", "results", "meshes")
centered_dir = os.path.join(os.getcwd(), "my28brains", "results", "meshes_centered")
centered_nondegenerate_dir = os.path.join(
    os.getcwd(), "my28brains", "results", "meshes_centered_nondegenerate"
)
geodesics_dir = os.path.join(os.getcwd(), "my28brains", "results", "meshes_geodesics")
parameterized_meshes_dir = os.path.join(
    os.getcwd(), "my28brains", "results", "meshes_parameterized"
)
h2_dir = os.path.join(os.getcwd(), "H2_SurfaceMatch")

regression_dir = os.path.join(os.getcwd(), "my28brains", "results", "regression")


for mesh_dir in [
    meshed_data_dir,
    centered_dir,
    centered_nondegenerate_dir,
    geodesics_dir,
    regression_dir,
    synthetic_data_dir,
]:
    if not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)


# TODO: refactor. have main1 go to /home/adele/code/my28brains/results/main_1
# etc.

##################### Creating synthetic geodesic if it does not already exist #####################

## Note: change this to go to results_main_3.

# if data_type == "synthetic":

#     if not os.path.exists(synthetic_mesh_sequence_dir):
#         os.makedirs(synthetic_mesh_sequence_dir)
#         print(f"Creating synthetic geodesic with start mesh {start_shape}, end mesh {end_shape}, and n_times {n_times}")

#         if not os.path.exists(start_shape_dir):
#             print(f"Creating start shape {start_shape} in {start_shape_dir}")
#             os.makedirs(start_shape_dir)
#             start_mesh = generate_syntetic_geodesics.generate_mesh(start_shape)
#             start_mesh_vertices = start_mesh.vertices
#             start_mesh_faces = start_mesh.faces

#             np.save(os.path.join(start_shape_dir, "vertices.npy"), start_mesh_vertices)
#             np.save(os.path.join(start_shape_dir, "faces.npy"), start_mesh_faces)
#         else:
#             print(f"Start shape {start_shape} already exists in {start_shape_dir}. Loading now.")
#             start_mesh_vertices = np.load(
#                 os.path.join(start_shape_dir, "vertices.npy")
#             )
#             start_mesh_faces = np.load(os.path.join(start_shape_dir, "faces.npy"))
#             start_mesh = trimesh.Trimesh(vertices = start_mesh_vertices, faces = start_mesh_faces)

#         if not os.path.exists(end_shape_dir):
#             print(f"Creating end shape {end_shape} in {end_shape_dir}")
#             os.makedirs(end_shape_dir)
#             end_mesh = generate_syntetic_geodesics.generate_mesh(end_shape)
#             end_mesh_vertices = end_mesh.vertices
#             end_mesh_faces = end_mesh.faces

#             np.save(os.path.join(end_shape_dir, "vertices.npy"), end_mesh_vertices)
#             np.save(os.path.join(end_shape_dir, "faces.npy"), end_mesh_faces)
#         else:
#             print(f"End shape {end_shape} already exists in {end_shape_dir}. Loading now.")
#             end_mesh_vertices = np.load(os.path.join(end_shape_dir, "vertices.npy"))
#             end_mesh_faces = np.load(os.path.join(end_shape_dir, "faces.npy"))
#             end_mesh = trimesh.Trimesh(vertices = end_mesh_vertices, faces = end_mesh_faces)

#         (
#             mesh_sequence_vertices,
#             mesh_faces,
#             times,
#             true_intercept,
#             true_slope,
#         ) = generate_syntetic_geodesics.generate_synthetic_parameterized_geodesic(
#             start_mesh, end_mesh, n_times
#         )
#         print("Original mesh_sequence vertices: ", mesh_sequence_vertices.shape)
#         print("Original mesh faces: ", mesh_faces.shape)
#         print("Times: ", times.shape)

#         np.save(os.path.join(synthetic_mesh_sequence_dir, "mesh_sequence_vertices.npy"), mesh_sequence_vertices)
#         np.save(os.path.join(synthetic_mesh_sequence_dir, "mesh_faces.npy"), mesh_faces)
#         np.save(os.path.join(synthetic_mesh_sequence_dir, "times.npy"), times)
#         np.save(os.path.join(synthetic_mesh_sequence_dir, "true_intercept.npy"), true_intercept)
#         np.save(os.path.join(synthetic_mesh_sequence_dir, "true_slope.npy"), true_slope)

#     else:
#         print(f"Synthetic geodesic ALREADY EXISTS with start mesh {start_shape}, end mesh {end_shape}, and n_times {n_times}. Loading now.")
#         mesh_sequence_vertices = np.load(os.path.join(synthetic_mesh_sequence_dir, "mesh_sequence_vertices.npy"))
#         mesh_faces = np.load(os.path.join(synthetic_mesh_sequence_dir, "mesh_faces.npy"))
#         times = np.load(os.path.join(synthetic_mesh_sequence_dir, "times.npy"))
#         true_intercept = np.load(os.path.join(synthetic_mesh_sequence_dir, "true_intercept.npy"))
#         true_slope = np.load(os.path.join(synthetic_mesh_sequence_dir, "true_slope.npy"))


#################### Elastic Metric Parameters ####################

a0 = 0.01  # was 0.1
a1 = 10  # (was 10)
b1 = 10  # (was 10)
c1 = 1  # (was 1)
d1 = 0
a2 = 1  # (was 1)

#################### Unparameterized Geodesic Parameters ####################

# NOTE: "time_steps" is the number of time points that we will have in the interpolated geodesic
# the "time_steps" value in the last param will set the number of time points in the interpolated geodesic

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
#     "index": 2,  # resolution 2
# }

# param6 = {
#     "weight_coef_dist_T": 10**6,
#     "weight_coef_dist_S": 10**6,
#     "sig_geom": 0.05,
#     "max_iter": 1000,
#     "time_steps": 5,
#     "tri_unsample": False,
#     "index": 2,  # resolution 2
# }

paramlist = [param1, param2, param3]  # , param4]  # param5, param6]
