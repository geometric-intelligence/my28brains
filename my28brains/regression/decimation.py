"""Functions for regression speed up using decimation strategy.

The strategy is:
- Fetch a sequence of parameterized meshes, based on data specified in default_config.py
THEN
- Decimates the mesh_sequence to a LOW number of points.
- Performs geodesic regression on the decimated mesh_sequence.
- Uses decimated mesh slope and intercept as starting point for next regression.
REPEATS ABOVE STEPS UNTIL THE INTERCEPT IS CLOSE TO THE TRUE INTERCEPT.
- Compares the regression results to the true slope and intercept of the mesh_sequence.
"""
import os

os.environ["GEOMSTATS_BACKEND"] = "pytorch"  # noqa: E402
import geomstats.backend as gs

import H2_SurfaceMatch.utils.utils  # noqa: E402
import my28brains.default_config as default_config


def create_decimated_mesh_sequence_list(
    original_mesh_sequence_vertices, original_mesh_faces
):
    """Create a list of decimated meshes from a list of original meshes.

    The original mesh sequence is first in this list. The second mesh is slightly
    decimated, and the last mesh is very decimated (very few vertices).
    """
    decimated_geodesics_list = (
        []
    )  # all the decimated geodesics for the geod regr. (0 = original mesh)
    mesh_faces_list = (
        []
    )  # all the decimated mesh faces for the geod regr. (0 = original mesh)

    # TEMP CHANGE
    decimated_geodesics_list.append(original_mesh_sequence_vertices)

    # mesh_seq_dict = {
    #     f"/{i_decimation}": my_decimation_function(mesh, i_decimation),
    #     for i_decimation in range(1, default_config.n_decimations+1)
    # }

    # TODO: implement mesh dictionary so that i don't have to keep track of order
    # TODO: change geodesic --> mesh_sequence
    # mesh_seq_dict = {}

    # TODO: print "we are going to decimate the mesh by a factor of 2, 4, 8, ..."

    for i_decimation in range(
        1, default_config.n_decimations + 1
    ):  # reverse(range(n_decimations))
        n_faces_after_decimation = int(
            original_mesh_faces.shape[0]
            / (i_decimation**default_config.regression_decimation_factor_step)
        )

        assert n_faces_after_decimation > 2
        one_decimated_geodesic = []
        for one_mesh in original_mesh_sequence_vertices:
            [
                one_decimated_mesh_vertices,
                decimated_faces,
            ] = H2_SurfaceMatch.utils.utils.decimate_mesh(  # noqa E231
                one_mesh, original_mesh_faces, n_faces_after_decimation
            )
            one_decimated_geodesic.append(one_decimated_mesh_vertices)
        one_decimated_geodesic = gs.array(one_decimated_geodesic)
        decimated_geodesics_list.append(one_decimated_geodesic)
        mesh_faces_list.append(decimated_faces)

        # mesh_seq_dict[f"/{i_decimation}"] = one_decimated_geodesic

    # NOTE: moved mesh_faces_list.append(original_mesh_faces) to after the loop

    # Note: decimated_mesh_sequences must remain a list. It is not a numpy array.

    return decimated_geodesics_list, mesh_faces_list
