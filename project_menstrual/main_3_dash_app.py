""" Creates a Dash app where hormone sliders predict hippocampal shape.

Run the app with the following command:
python main_3_dash_app.py

Notes on Dash:
- Dash is a Python framework for building web applications.
- html.H1(children= ...) is an example of a title. You can change H1 to H2, H3, etc.
    to change the size of the title.
"""

import dash
from dash import Dash, dcc, html, Input, Output, callback
import plotly.graph_objects as go # or plotly.express as px

import os
import subprocess
import sys
import itertools
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# import meshplot as mp
from IPython.display import clear_output, display

os.environ["GEOMSTATS_BACKEND"] = "pytorch"  # noqa: E402
import geomstats.backend as gs

import project_menstrual.default_config as default_config
import H2_SurfaceMatch.utils.input_output as h2_io
import H2_SurfaceMatch.utils.utils
import src.datasets.utils as data_utils
from H2_SurfaceMatch.utils.input_output import plotGeodesic
from src.regression import check_euclidean, training
import src.setcwd

src.setcwd.main()

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

# build work path from git root path
gitroot_path = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
)
os.chdir(gitroot_path[:-1])
work_dir = os.getcwd()  # code/my28brains/
code_dir = os.path.dirname(work_dir)  # code/
raw_dir = "/home/data/28andMeOC_correct"
project_dir = os.path.join(
    os.getcwd(), "project_menstrual"
)  # code/my28brains/project_regression/
src_dir = os.path.join(os.getcwd(), "src")
h2_dir = os.path.join(os.getcwd(), "H2_SurfaceMatch")
sys.path.append(code_dir)
sys.path.append(h2_dir)

# Multiple Linear Regression

# multiple linear regression, as done in project_menstrual/main_2_linear_regression

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

progesterone_levels = gs.array(all_hormone_levels["Prog"].values)
estrogen_levels = gs.array(all_hormone_levels["Estro"].values)
dheas_levels = gs.array(all_hormone_levels["DHEAS"].values)
lh_levels = gs.array(all_hormone_levels["LH"].values)
fsh_levels = gs.array(all_hormone_levels["FSH"].values)
shbg_levels = gs.array(all_hormone_levels["SHBG"].values)

X_multiple = gs.vstack(
    (
        progesterone_levels,
        estrogen_levels,
        dheas_levels,
        lh_levels,
        fsh_levels,
        shbg_levels,
    )
).T  # NOTE: copilot thinks this should be transposed.

(
    multiple_intercept_hat,
    multiple_coef_hat,
    mr,
) = training.fit_linear_regression(y, X_multiple)

mr_score_array = training.compute_R2(y, X_multiple, test_indices, train_indices)

X_multiple_predict = gs.array(X_multiple.reshape(len(X_multiple), -1))
y_pred_for_mr = mr.predict(X_multiple_predict)
y_pred_for_mr = y_pred_for_mr.reshape([len(X_multiple), n_vertices, 3])

# Parameters for sliders

hormones_info = {
    "progesterone": {"min_value": 0, "max_value": 15, "step": 5},
    "FSH": {"min_value": 0, "max_value": 15, "step": 5},
    "LH": {"min_value": 0, "max_value": 50, "step": 10},
    "estrogen": {"min_value": 0, "max_value": 250, "step": 50},
    "DHEAS": {"min_value": 0, "max_value": 300, "step": 50},
    "SHBG": {"min_value": 0, "max_value": 70, "step": 10},
}

def generate_hormone_dataframe(hormones_info):
    data = {
        "min_value": [],
        "max_value": [],
        "step": [],
        "num_points": [],
        "values": [],
    }
    for hormone, info in hormones_info.items():
        min_value = info["min_value"]
        max_value = info["max_value"]
        step = info["step"]
        num_points = int((max_value - min_value) / step) + 1
        values = np.linspace(min_value, max_value, num=num_points)

        data["min_value"].append(min_value)
        data["max_value"].append(max_value)
        data["step"].append(step)
        data["num_points"].append(num_points)
        data["values"].append(values)

    df = pd.DataFrame(data, index=hormones_info.keys())
    return df


# Generate the dataframe
hormone_df = generate_hormone_dataframe(hormones_info)

print(hormone_df)

# Load meshes for all possible hormone combinations

data = {
    "x": [],
    "y": [],
    "z": [],
    "i": [],
    "j": [],
    "k": [],
}

results = []
for progesterone, estrogen, DHEAS, LH, FSH, SHBG in itertools.product(
    hormone_df.loc['progesterone', 'values'], 
    hormone_df.loc['estrogen','values'], 
    hormone_df.loc['DHEAS','values'], 
    hormone_df.loc['LH','values'], 
    hormone_df.loc['FSH','values'], 
    hormone_df.loc['SHBG','values']
):
    
    # Predict Mesh
    X_multiple = gs.vstack(
        (
            gs.array(progesterone),
            gs.array(estrogen),
            gs.array(DHEAS),
            gs.array(LH),
            gs.array(FSH),
            gs.array(SHBG),
        )
    ).T
    X_multiple_predict = gs.array(X_multiple.reshape(len(X_multiple), -1))
    y_pred_for_mr = mr.predict(X_multiple_predict)
    y_pred_for_mr = y_pred_for_mr.reshape([n_vertices, 3])
    faces = gs.array(space.faces).numpy()

    # Extract x, y, z values
    x = y_pred_for_mr[:, 0]
    y = y_pred_for_mr[:, 1]
    z = y_pred_for_mr[:, 2]

    # Extract i, j, k values
    i = faces[:, 0]
    j = faces[:, 1]
    k = faces[:, 2]

    # Append results to list
    results.append([x, y, z, i, j, k])

# Create DataFrame from results
columns = ["x", "y", "z", "i", "j", "k"]
results_df = pd.DataFrame(results, columns=columns)

# Create Dash app

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)


app.layout = html.Div(children=[
       html.Div([
           dcc.Graph(id = 'mesh-plot'),
       ]) ,
       html.Div([
           html.H6('Progesterone'),
              dcc.Slider(
                id='progesterone-slider',
                min=hormones_info['progesterone']['min_value'],
                max=hormones_info['progesterone']['max_value'],
                step=hormones_info['progesterone']['step'],
                value=5,
                marks={str(i): str(i) for i in range(hormones_info['progesterone']['min_value'], hormones_info['progesterone']['max_value'], hormones_info['progesterone']['step'])}
              ),
              html.H6('Estrogen'),
                dcc.Slider(
                    id='estrogen-slider',
                    min=hormones_info['estrogen']['min_value'],
                    max=hormones_info['estrogen']['max_value'],
                    step=hormones_info['estrogen']['step'],
                    value=50,
                    marks={str(i): str(i) for i in range(hormones_info['estrogen']['min_value'], hormones_info['estrogen']['max_value'], hormones_info['estrogen']['step'])}
                ),
                html.H6('DHEAS'),
                dcc.Slider(
                    id='DHEAS-slider',
                    min=hormones_info['DHEAS']['min_value'],
                    max=hormones_info['DHEAS']['max_value'],
                    step=hormones_info['DHEAS']['step'],
                    value=50,
                    marks={str(i): str(i) for i in range(hormones_info['DHEAS']['min_value'], hormones_info['DHEAS']['max_value'], hormones_info['DHEAS']['step'])}
                ),
                html.H6('LH'),
                dcc.Slider(
                    id='LH-slider',
                    min=hormones_info['LH']['min_value'],
                    max=hormones_info['LH']['max_value'],
                    step=hormones_info['LH']['step'],
                    value=50,
                    marks={str(i): str(i) for i in range(hormones_info['LH']['min_value'], hormones_info['LH']['max_value'], hormones_info['LH']['step'])}
                ),
                html.H6('FSH'),
                dcc.Slider(
                    id='FSH-slider',
                    min=hormones_info['FSH']['min_value'],
                    max=hormones_info['FSH']['max_value'],
                    step=hormones_info['FSH']['step'],
                    value=50,
                    marks={str(i): str(i) for i in range(hormones_info['FSH']['min_value'], hormones_info['FSH']['max_value'], hormones_info['FSH']['step'])}
                ),
                html.H6('SHBG'),
                dcc.Slider(
                    id='SHBG-slider',
                    min=hormones_info['SHBG']['min_value'],
                    max=hormones_info['SHBG']['max_value'],
                    step=hormones_info['SHBG']['step'],
                    value=50,
                    marks={str(i): str(i) for i in range(hormones_info['SHBG']['min_value'], hormones_info['SHBG']['max_value'], hormones_info['SHBG']['step'])}
                ),
       ])
])

@callback(
    Output('mesh-plot', 'figure'),
    Input('progesterone-slider', 'value'),
    Input('estrogen-slider', 'value'),
    Input('DHEAS-slider', 'value'),
    Input('LH-slider', 'value'),
    Input('FSH-slider', 'value'),
    Input('SHBG-slider', 'value'),
    )
def plot_hormone_levels_plotly(progesterone, FSH, LH, estrogen, SHBG, DHEAS):
    progesterone = gs.array(progesterone)
    FSH = gs.array(FSH)
    LH = gs.array(LH)
    estrogen = gs.array(estrogen)
    SHBG = gs.array(SHBG)
    DHEAS = gs.array(DHEAS)

    # Predict Mesh
    X_multiple = gs.vstack(
        (
            progesterone,
            estrogen,
            DHEAS,
            LH,
            FSH,
            SHBG,
        )
    ).T

    X_multiple_predict = gs.array(X_multiple.reshape(len(X_multiple), -1))

    y_pred_for_mr = mr.predict(X_multiple_predict)
    print(y_pred_for_mr.shape)
    y_pred_for_mr = y_pred_for_mr.reshape([n_vertices, 3])

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
                # Intensity of each vertex, which will be interpolated and color-coded
                # intensity=[0, 0.33, 0.66, 1],
                # i, j and k give the vertices of triangles
                # here we represent the 4 triangles of the tetrahedron surface
                i=i,
                j=j,
                k=k,
                name="y",
                showscale=True,
            )
        ]
    )

    fig.update_layout(width=1000)
    fig.update_layout(height=800)

    return fig

if __name__ == '__main__':
    # app.run(debug=True)
    app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter
