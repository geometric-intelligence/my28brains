"""Unit tests for discrete_surfaces modules.

Add your code directory to the PYTHON PATH before running.
export PYTHONPATH=/home/nmiolane/code.

Get the .npy files to test this code.
"""

import os

import geomstats.backend as gs
import numpy as np

from my28brains.my28brains.discrete_surfaces import DiscreteSurfaces

# import sphere data vertices and faces
BRAINS_DIR = os.path.join(os.environ["PYTHONPATH"], "my28brains")
DATA_DIR = os.path.join(BRAINS_DIR, "data")
SPHERE_DATA_DIR = os.path.join(DATA_DIR, "sphere_meshes")
test_vertices_path = os.path.join(SPHERE_DATA_DIR, "faces.npy")
test_faces_path = os.path.join(SPHERE_DATA_DIR, "vertices.npy")

test_vertices = np.load(test_vertices_path)
test_faces = np.load(test_faces_path)
test_vertices = gs.cast(test_vertices, gs.int64)
test_faces = gs.cast(test_faces, gs.int64)

# space = DiscreteSurfaces(faces=test_faces)

# get vertices and faces from brain data
# TESTS_DIR = os.path.join(os.getcwd(), "tests")
# test_vertices_path = os.path.join(TESTS_DIR, "test_vertices.npy")
# test_faces_path = os.path.join(TESTS_DIR, "test_faces.npy")
# #test_vertices_source_path = os.path.join(TESTS_DIR, "test_vertices_source.npy")
# #test_faces_source_path = os.path.join(TESTS_DIR, "test_faces_source.npy")
# test_vertices_target_path = os.path.join(TESTS_DIR, "test_vertices_target.npy")
# test_faces_target_path = os.path.join(TESTS_DIR, "test_faces_target.npy")

# test_vertices = np.load(test_vertices_source_path)
# test_faces = np.load(test_faces_source_path)

# print(test_vertices.shape)
# print(test_faces.shape)

# generate test faces
# TODO: set seed and change test_faces to be random instead of all 1's

# random.seed(47)
# TODO: CREATE TEST MESH, AND THEN TEST EVERYTHING ON THIS
# QUESTION: how does this not put all the faces at the same coordinates?
# creating pretty much a point 12 times
# test_faces = gs.ones((12, 3))


# QUESTION doesn't this rely on the fact that random_point() works?
# test_vertices = space.random_point()


def test_belongs():
    """Test that a set of vertices belongs to the manifold of DiscreteSurfaces.

    (Also inheritely checks if the discrete surface has degenerate triangles.)
    """
    vertices = test_vertices
    space = DiscreteSurfaces(faces=test_faces)
    space.belongs(point=vertices)


def test_is_tangent():
    """Test is_tangent.

    TODO.
    """


def test_random_point_1():
    """Test random_point.

    This function tests that:
    - random_point() generates the correct number of samples (3)
    """
    # ambient_dim = 3
    # faces = gs.ones((12, ambient_dim))
    space = DiscreteSurfaces(faces=test_faces)
    # use a random point a test_vertices
    point = space.random_point(n_samples=3)
    assert point.shape[-1] == 3


def test_random_point_2():
    """Test random_point.

    This function tests that:
    - random_point() generates a point on the manifold

    TODO: this is failing. I think its failure has something to do with
    it not being implemented in a way that is compatible with
    the belongs() function???

    error: TypeError: Cannot cast array data from dtype('float64')
    to dtype('int64') according to the rule 'safe'
    """
    space = DiscreteSurfaces(faces=test_faces)
    point = space.random_point(n_samples=1)
    assert space.belongs(point)


def test_vertex_areas():
    """Test vertex_areas.

    TODO.
    """


def test_get_laplacian():
    """Test get_laplacian.

    TODO.
    """


def test_normals():
    """Test normals.

    TODO.
    """


def test_surface_one_forms():
    """Test surface one forms."""
    vertices = test_vertices
    space = DiscreteSurfaces(faces=test_faces)
    one_forms = space.surface_one_forms(point=vertices)
    assert one_forms.shape == (space.n_faces, 2, 3), one_forms.shape


def test_faces_area():
    """Test faces area."""
    vertices = test_vertices
    space = DiscreteSurfaces(faces=test_faces)
    face_areas = space.face_areas(point=vertices)
    assert face_areas.shape == (space.n_faces,), face_areas.shape


def test_surface_metric_matrices():
    """Test surface metric matrices."""
    vertices = test_vertices
    space = DiscreteSurfaces(faces=test_faces)
    surface_metric_matrices = space.surface_metric_matrices(point=vertices)
    assert surface_metric_matrices.shape == (
        space.n_faces,
        2,
        2,
    ), surface_metric_matrices.shape
