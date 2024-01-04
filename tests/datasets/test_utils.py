"""Test the utils handling datasets."""

import geomstats.backend as gs

import src.datasets.utils as data_utils


def test_add_noise():
    """Test the add_noise function.

    We test that the add_noise function has modified its input in place.
    We test that if we save an auxiliary copy of the input mesh first, it
    does not get modified.
    """
    mesh = gs.array(
        [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [2.0, 3.0, 4.0], [4.0, -1.0, 3.0]]
    )
    aux_mesh = mesh.copy()
    noisy_mesh = data_utils.add_noise(mesh_sequence_vertices=[mesh], noise_factor=10)[0]

    assert gs.all(gs.isclose(mesh, noisy_mesh))
    assert ~gs.all(gs.isclose(aux_mesh, noisy_mesh))
