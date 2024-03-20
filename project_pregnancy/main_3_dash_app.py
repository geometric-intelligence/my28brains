"""Creates a Dash app where hormone sliders predict hippocampal shape.

Run the app with the following command:
python main_3_dash_app.py

Notes on Dash:
- Dash is a Python framework for building web applications.
- html.H1(children= ...) is an example of a title. You can change H1 to H2, H3, etc.
    to change the size of the title.
"""

import itertools
import os
import random
import subprocess
import sys

import dash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go  # or plotly.express as px
from dash import Dash, Input, Output, callback, dcc, html

# import meshplot as mp
from IPython.display import clear_output, display
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

os.environ["GEOMSTATS_BACKEND"] = "pytorch"  # noqa: E402
import geomstats.backend as gs

import H2_SurfaceMatch.utils.input_output as h2_io
import H2_SurfaceMatch.utils.utils
import project_pregnancy.default_config as default_config
import src.datasets.utils as data_utils
import src.setcwd
from H2_SurfaceMatch.utils.input_output import plotGeodesic
from src.regression import check_euclidean, training

src.setcwd.main()

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

# Multiple Linear Regression

(
    space,
    y,
    all_hormone_levels,
    true_intercept,
    true_coef,
) = data_utils.load_real_data(default_config)

n_vertices = len(y[0])
faces = gs.array(space.faces).numpy()

n_train = int(default_config.train_test_split * len(y))

X_indices = np.arange(len(y))
# Shuffle the array to get random values
random.shuffle(X_indices)
train_indices = X_indices[:n_train]
train_indices = np.sort(train_indices)
test_indices = X_indices[n_train:]
test_indices = np.sort(test_indices)

# TODO: instead, save these values in main_2, and then load them here. or, figure out how to predict the mesh using just the intercept and coef learned here, and then load them.

progesterone_levels = gs.array(all_hormone_levels["prog"].values)
estrogen_levels = gs.array(all_hormone_levels["estro"].values)
lh_levels = gs.array(all_hormone_levels["lh"].values)
# gest_week = gs.array(all_hormone_levels["gestWeek"].values)

progesterone_average = gs.mean(progesterone_levels)
estrogen_average = gs.mean(estrogen_levels)
lh_average = gs.mean(lh_levels)
# gest_week_average = gs.mean(gest_week)

X_multiple = gs.vstack(
    (
        progesterone_levels,
        estrogen_levels,
        lh_levels,
        # gest_week,
    )
).T  # NOTE: copilot thinks this should be transposed.

(
    multiple_intercept_hat,
    multiple_coef_hat,
    mr,
) = training.fit_linear_regression(y, X_multiple)

mr_score_array = training.compute_R2(y, X_multiple, test_indices, train_indices)

# X_multiple_predict = gs.array(X_multiple.reshape(len(X_multiple), -1))
# y_pred_for_mr = mr.predict(X_multiple_predict)
# y_pred_for_mr = y_pred_for_mr.reshape([len(X_multiple), n_vertices, 3])

# Parameters for sliders

hormones_info = {
    "progesterone": {"min_value": 1, "max_value": 103, "step": 10},
    "LH": {"min_value": 1, "max_value": 8, "step": 1},
    "estrogen": {"min_value": 3, "max_value": 10200, "step": 100},
    # "gest_week": {"min_value": -3, "max_value": 162, "step": 10},
}

app = Dash(__name__)  # , external_stylesheets=external_stylesheets)


app.layout = html.Div(
    children=[
        html.Div(
            [
                dcc.Graph(id="mesh-plot"),
            ],
        ),  # style={'padding': '20px 0px 0px 0px'}
        html.Div(
            [
                html.H6("Progesterone ng/ml"),
                dcc.Slider(
                    id="progesterone-slider",
                    min=hormones_info["progesterone"]["min_value"],
                    max=hormones_info["progesterone"]["max_value"],
                    step=hormones_info["progesterone"]["step"],
                    value=progesterone_average,
                    marks={
                        str(i): str(i)
                        for i in range(
                            hormones_info["progesterone"]["min_value"],
                            hormones_info["progesterone"]["max_value"],
                            hormones_info["progesterone"]["step"],
                        )
                    },
                ),
                html.H6("Estrogen pg/ml"),
                dcc.Slider(
                    id="estrogen-slider",
                    min=hormones_info["estrogen"]["min_value"],
                    max=hormones_info["estrogen"]["max_value"],
                    step=hormones_info["estrogen"]["step"],
                    value=estrogen_average,
                    marks={
                        str(i): str(i)
                        for i in range(
                            hormones_info["estrogen"]["min_value"],
                            hormones_info["estrogen"]["max_value"],
                            hormones_info["estrogen"]["step"],
                        )
                    },
                ),
                html.H6("LH ng/ml"),
                dcc.Slider(
                    id="LH-slider",
                    min=hormones_info["LH"]["min_value"],
                    max=hormones_info["LH"]["max_value"],
                    step=hormones_info["LH"]["step"],
                    value=lh_average,
                    marks={
                        str(i): str(i)
                        for i in range(
                            hormones_info["LH"]["min_value"],
                            hormones_info["LH"]["max_value"],
                            hormones_info["LH"]["step"],
                        )
                    },
                ),
                # html.H6("Gestation Week"),
                # dcc.Slider(
                #     id="gest_week-slider",
                #     min=hormones_info["gest_week"]["min_value"],
                #     max=hormones_info["gest_week"]["max_value"],
                #     step=hormones_info["gest_week"]["step"],
                #     value=gest_week_average,
                #     marks={
                #         str(i): str(i)
                #         for i in range(
                #             hormones_info["gest_week"]["min_value"],
                #             hormones_info["gest_week"]["max_value"],
                #             hormones_info["gest_week"]["step"],
                #         )
                #     },
                # ),
            ],
            style={"width": "60%", "display": "inline-block"},
        ),
    ]
)


@callback(
    Output("mesh-plot", "figure"),
    Input("progesterone-slider", "value"),
    Input("LH-slider", "value"),
    Input("estrogen-slider", "value"),
    # Input("gest_week-slider", "value"),
)
def plot_hormone_levels_plotly(progesterone, LH, estrogen):  # , gest_week):
    """Update the mesh plot based on the hormone levels."""
    progesterone = gs.array(progesterone)
    LH = gs.array(LH)
    estrogen = gs.array(estrogen)
    # gest_week = gs.array(gest_week)

    # Predict Mesh
    X_multiple = gs.vstack(
        (
            progesterone,
            estrogen,
            LH,
            # gest_week,
        )
    ).T

    X_multiple_predict = gs.array(X_multiple.reshape(len(X_multiple), -1))

    y_pred_for_mr = mr.predict(X_multiple_predict)
    y_pred_for_mr = y_pred_for_mr.reshape([n_vertices, 3])
    # y_pred_for_mr = gaussian_smoothing(y_pred_for_mr, sigma=0.7)

    faces = gs.array(space.faces).numpy()

    x = y_pred_for_mr[:, 0]
    y = y_pred_for_mr[:, 1]
    z = y_pred_for_mr[:, 2]

    i = faces[:, 0]
    j = faces[:, 1]
    k = faces[:, 2]

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                colorbar_title="z",
                colorscale=[[0, "gold"], [0.5, "mediumturquoise"], [1, "magenta"]],
                # i, j and k give the vertices of triangles
                i=i,
                j=j,
                k=k,
                name="y",
                # showscale=True,
            )
        ]
    )

    fig.update_layout(width=1000)
    fig.update_layout(height=1000)

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                nticks=4,
                range=[-2.5, 2],
            ),
            yaxis=dict(
                nticks=4,
                range=[-2.5, 2],
            ),
            zaxis=dict(
                nticks=4,
                range=[-2.5, 2],
            ),
        ),
    )

    # Default parameters which are used when `layout.scene.camera` is not provided
    # camera1 = dict(
    #     up=dict(x=0, y=0, z=1),
    #     center=dict(x=0, y=0, z=0),
    #     eye=dict(x=2.5, y=-2.5, z=0.0),
    # )

    camera2 = dict(
        up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=0, y=0, z=2.5)
    )

    fig.update_layout(
        scene_camera=camera2, margin=dict(l=0, r=0, b=0, t=0)
    )  # margin=dict(l=0, r=0, b=0, t=0)

    return fig


if __name__ == "__main__":
    # app.run(debug=True)
    app.run_server(
        debug=True, use_reloader=False
    )  # Turn off reloader if inside Jupyter
