"""Unit tests for io (input output) module.

Add your code directory to the PYTHON PATH before running.
export PYTHONPATH=/home/nmiolane/code.

Get the .npy files to test this code.
"""

import os

import numpy as np
from geomstats.geometry.discrete_surfaces import DiscreteSurfaces

import my28brains.src.preprocessing as preprocessing

TESTS_DIR = os.path.join(os.getcwd(), "tests")
test_vertices_path = os.path.join(TESTS_DIR, "test_vertices.npy")
test_faces_path = os.path.join(TESTS_DIR, "test_faces.npy")
test_vertices_source_path = os.path.join(TESTS_DIR, "test_vertices_source.npy")
test_faces_source_path = os.path.join(TESTS_DIR, "test_faces_source.npy")
test_vertices_target_path = os.path.join(TESTS_DIR, "test_vertices_target.npy")
test_faces_target_path = os.path.join(TESTS_DIR, "test_faces_target.npy")

test_vertices = np.load(test_vertices_source_path)
test_faces = np.load(test_faces_source_path)

print(test_vertices.shape)
print(test_faces.shape)


def test_remove_degenerate_faces():
    """Test remove degenerate faces.

    A degenerate face is a face with an area close to 0.

    Test that the new faces do not have an area close to 0.
    """
    vertices = test_vertices
    faces = test_faces
    print("vertices:", vertices.shape)
    print("faces:", faces.shape)

    new_vertices, new_faces = preprocessing.geodesics.remove_degenerate_faces(
        vertices, faces
    )
    print("new vertices:", new_vertices.shape)
    print("new faces:", new_faces.shape)
    new_space = DiscreteSurfaces(faces=new_faces)
    # This also tests that there are no degenerated faces anymore
    res = new_space.belongs(new_vertices)
    assert res
