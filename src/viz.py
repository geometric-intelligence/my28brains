"""Visualization tools."""

import glob
import os
import subprocess

import geomstats.backend as gs
import geomstats.visualization as visualization
import matplotlib
import matplotlib.pyplot as plt
import nibabel
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib import animation

viz_dict = {
    "Hypersphere": visualization.Sphere(n_meridians=30),
    # note: this did not work. points that belonged to poincare ball did not belong to H2
    "PoincareBall": visualization.PoincareDisk(),
    "Hyperboloid": visualization.PoincareDisk(),
}

IMG_DIR = "/home/data/28andme/"
HORMONES = {"Estro": "Estrogen", "Prog": "Progesterone", "LH": "LH", "FSH": "FSH"}

# Colors follow the color scheme from Taylor et al. 2020
COLORS = {
    "Estro": "#AFEEEE",  # pastel blue-turquoise
    "Prog": "#507DBC",  # dark blue
    "LH": "#FF7373",  # red
    "FSH": "#FADA5E",  # pastel yellow
}

COL_TO_TEXT = {
    "diff_seq_per_time_vertex_diameter": "Error per mesh per vertex [% diameter]",
    "diff_seq_duration_per_time_and_vertex": "Time difference [secs] per mesh and vertex",
    "relative_diff_seq_duration": "Time difference per vertex [% line]",
    "n_steps": "Number of steps",
    "n_vertices": "Number of vertices",
    "rmsd": "RMSD",
    "rmsd_diameter": "RMSD, Line vs. Geodesic (per diameter)",
    "speed": "Speed gain",
    "accuracy": "Accuracy",
    "linear_residuals": "Regression",
    "linear_noise": "Noise",
    "geodesic_coef_err": "Geodesic Coef Error",
    "geodesic_duration_time": "Geodesic Duration Time",
    "noise_factor": "Noise Factor",
    "rmsd_geod": "RMSD, Geodesic Regression",
    "nrmsd_geod": "Normalized RMSD, Geodesic Regression",
    "n_X": "Number of Points",
    "n_geod_iterations": "Number of iterations in GR",
}
# build work path from git root path
gitroot_path = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
)
os.chdir(gitroot_path[:-1])
TMP = os.path.join(os.getcwd(), "src", "results", "tmp")
FONTSIZE = 18


def init_matplotlib():
    """Configure style for matplotlib."""
    matplotlib.rc("font", size=FONTSIZE)
    matplotlib.rc("text")
    matplotlib.rc("legend", fontsize=FONTSIZE)
    matplotlib.rc("axes", titlesize=21, labelsize=14)
    matplotlib.rc(
        "font",
        family="sans-serif",
        monospace=["Arial"],
    )


def animate(img_suffix="ashs/right_lfseg_corr_usegray_CT_LQ.nii.gz", slice_z=16):
    """Produce temporal animation of anatomical images.

    This is a time-series of images during 25 days corresponding to 25 sessions.

    Parameters
    ----------
    img_suffix : string
        Suffix of the corresponding images, among:
        - anat/sub-01_ses-**_T1w.nii.gz
        - anat/sub-01_ses-**_T2hipp.nii.gz
        - ashs/left_lfseg_corr_usegray_CT_LQ.nii.gz
        - ashs/right_lfseg_corr_usegray_CT_LQ.nii.gz
    slice_z : int
        Since images are 3D, a slice is chosen to get a 2D images and form a video.

    Returns
    -------
    anima : Animation
        Animation. Display with HTML(anima.to_html5_video())
    """
    string_base = os.path.join(IMG_DIR, f"sub-01/ses-**/{img_suffix}")
    paths = sorted(glob.glob(string_base))

    print(f"Found {len(paths)} image paths. Creating video.")
    print(paths)

    arrays = []
    ses_ids = []
    for path in paths:
        # HACK ALERT: Session ID is at the 30, 31st chars in path:
        ses_ids.append(path[30:32])
        img = nibabel.load(path)
        arrays.append(img.get_fdata())

    array_4d = np.stack(arrays, axis=3)
    if "ashs/left" in img_suffix:
        array_4d = array_4d[120:200, 120:200, :, :]
    elif "ashs/right" in img_suffix:
        array_4d = array_4d[220:320, 120:220, :, :]

    cmap = "viridis"
    if "ashs" in img_suffix:
        array_4d = np.ma.masked_where(array_4d < 0.05, array_4d)
        cmap = matplotlib.cm.get_cmap("tab20b").copy()
        cmaplist = [cmap(2 * i) for i in range(10)]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "Custom cmap", cmaplist, 10
        )
        cmap.set_bad(color="black")

    def quick_play(dT=50):
        fig, ax = plt.subplots()
        im = ax.imshow(array_4d[:, :, slice_z, 0], cmap=cmap)
        ax.set_title(f"{img_suffix}\n Session: {ses_ids[0]}")

        def init():
            im.set_data(array_4d[:, :, slice_z, 0])
            return (im,)

        def animate(i):
            im.set_data(array_4d[:, :, slice_z, i])
            ax.set_title(f"{img_suffix}\n Session: {ses_ids[i]}")
            return (im,)

        anima = animation.FuncAnimation(
            fig,
            animate,
            frames=array_4d.shape[3] - 1,
            init_func=init,
            interval=dT,
            blit=True,
        )
        return anima

    anima = quick_play()
    return anima


def plot_hormones(df, dayID, plot_type="dot", hormones=HORMONES, savefig=False):
    """Plot hormones - original function.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with hormones.
    dayID : int
        Day ID to plot.
    plot_type : string
        Type of plot. Either "dot" or "line".
    hormones : list
        List of hormones to plot.
    savefig : bool
        Whether to save the figure.
    """
    if plot_type == "dot":
        df = df[df["dayID"] < dayID]
    X = df["dayID"]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for h in hormones:
        ax.plot(X, df[h], label=HORMONES[h])

    if dayID > 2:
        if plot_type == "dot":
            for h in hormones:
                ax.scatter(X.values[-1], df[h].values[-1], s=100)

    ax.set_xlim((0, 30))
    ax.set_ylim(0, df["Estro"].max() + 5)
    ax.legend(loc="upper left")
    if savefig:
        fig.savefig(f"{TMP}/plot_hormones_{dayID:02d}.svg")
    return fig


def plotly_hormones(df, by, day, hormones=HORMONES, ymax=None, savefig=False):
    """Plot hormones - plotly version.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with hormones.
    by : string
        Type of day to plot by.
        Either DayID or CycleDay.
    day_id : int
        Day ID to plot.
    hormones : list
        List of hormones to plot.
    ymax : int
        Maximum value for y-axis.
    savefig : bool
        Whether to save the figure.
    """
    day_labels = {
        "dayID": "Day",
        "CycleDay": "Cycle Day",
    }

    if by not in ["dayID", "CycleDay"]:
        raise ValueError("by must be either dayID or CycleDay.")
    if ymax is None:
        ymax = df[hormones].max().max() + 10

    df = df[df[by] <= day]
    df = df.sort_values(by=by)
    X = df[by]

    fig = go.Figure()

    # Add traces
    for h in hormones:
        fig.add_trace(
            go.Scatter(
                x=X,
                y=df[h],
                mode="lines",
                name=HORMONES[h],
                line=dict(color=COLORS[h], width=8),
                showlegend=True,
                line_shape="spline",
            )
        )

    # Update layout
    fig.update_layout(
        template="plotly_white",
        xaxis_title=f"{day_labels[by]}",
        yaxis_title="Hormone",
        xaxis=dict(range=[0, 30], showgrid=False, tickfont=dict(size=20), linewidth=8),
        yaxis=dict(
            range=[-ymax * 0.05, ymax],
            showgrid=False,
            tickfont=dict(size=20),
            linewidth=8,
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=20),
        ),
        margin=dict(l=60, r=30, t=30, b=60),
        width=900,  # Adjust the width as needed
        height=300,  # Adjust the height as needed
    )
    # Set axis titles' font size
    fig.update_xaxes(title_font=dict(size=20))
    fig.update_yaxes(title_font=dict(size=20))

    # Save the figure
    if savefig:
        pio.write_image(fig, f"{TMP}/hormones_day_{day:02d}.png", format="png")
    return fig


def offset_mesh_sequence(mesh_sequence_vertices):
    """Offset a mesh sequence to visualize it better.

    Offset on the x-axis only.

    Parameters
    ----------
    mesh_sequence_vertices : np.array, shape=[n_X, n_vertices, 3]
        Sequence of meshes.

    Returns
    -------
    _ : np.array, shape=[n_X, n_vertices, 3]
        Offset sequence of meshes.
    """
    n_X = len(mesh_sequence_vertices)
    x_max = max([max(mesh[:, 0]) for mesh in mesh_sequence_vertices])
    x_min = min([min(mesh[:, 0]) for mesh in mesh_sequence_vertices])
    x_diameter = np.abs(x_max - x_min)
    max_offset = n_X * x_diameter
    offsets = np.linspace(0, max_offset, n_X)

    offset_mesh_sequence_vertices = []
    for i_mesh, mesh in enumerate(mesh_sequence_vertices):
        offset_mesh = mesh + np.array([offsets[i_mesh], 0, 0])
        offset_mesh_sequence_vertices.append(offset_mesh)
    return offset_mesh_sequence_vertices


def plot_mesh_sequence(mesh_sequence_vertices, savefig=False, label=None):
    """Plot a sequence of meshes.

    NOTE: the plotGeodesic function from H2_SurfaceMatch also works,
    and saves a .ply file with the resulting plot.

    Parameters
    ----------
    mesh_sequence_vertices : np.array, shape=[n_X, n_vertices, 3]
        Sequence of meshes.
    """
    fig = plt.figure(figsize=(10, 20))
    ax = fig.add_subplot(111, projection="3d")
    len_sequence = len(mesh_sequence_vertices)
    plasma_cmap = plt.cm.get_cmap("plasma")

    for i_mesh, mesh in enumerate(mesh_sequence_vertices):
        ax.scatter(
            mesh[:, 0],
            mesh[:, 1],
            mesh[:, 2],
            c=plasma_cmap(i_mesh / len_sequence),
            marker="o",
        )
    ax.view_init(elev=10, azim=-80)
    ax.set_aspect("equal")

    # Save the figure
    if savefig:
        png_filename = f"mesh_sequence_{label}.svg"
        png_path = os.path.join("src", "notebooks", png_filename)
        plt.savefig(png_path)  # Adjust DPI as needed

    plt.show()


def plotly_mesh_sequence(mesh_sequence_vertices):
    """Plot a sequence of meshes with plotly (interactive).

    NOTE: the plotGeodesic function from H2_SurfaceMatch also works,
    and saves a .ply file with the resulting plot.

    Parameters
    ----------
    mesh_sequence_vertices : np.array, shape=[n_X, n_vertices, 3]
        Sequence of meshes.
    """
    plasma_cmap = px.colors.sequential.Plasma
    data = []

    for i_mesh, mesh in enumerate(mesh_sequence_vertices):
        data.append(
            go.Scatter3d(
                x=mesh[:, 0],
                y=mesh[:, 1],
                z=mesh[:, 2],
                mode="markers",
                marker=dict(
                    color=plasma_cmap[i_mesh], size=5, opacity=0.8, symbol="circle"
                ),
            )
        )

    fig = go.Figure(data=data)

    fig.show()


def benchmark_data_sequence(space, sequence_1, sequence_2, sequence_3=None):
    """Compare two benchmark datasets.

    Examples
    --------
    - main_2_regression: compare true sequence vs modeled sequence.
    - main_3_line_vs_geodesic: compare line vs geodesic.

    Parameters
    ----------
    space : space where data points lie.
    sequence_1:
        for regression: true points sequence
        for line vs geodesic: line
    sequence_2:
        for regression: modeled points sequence (lr)
        for line vs geodesic: geodesic
    sequence_3:
        for regression: modeled points sequence (gr)
        for line vs geodesic: None
    """
    sequence_1 = gs.array(sequence_1)
    sequence_2 = gs.array(sequence_2)
    if sequence_3 is not None:
        sequence_3 = gs.array(sequence_3)
    # Plot
    fig = plt.figure(figsize=(8, 8))

    assert space.dim == 2, "space's dimension is not 2 -> can't visualize!"
    manifold_visu = viz_dict[space.__class__.__name__]

    size = 10
    marker = "o"

    if space.__class__.__name__ == "Hypersphere":
        ax = fig.add_subplot(111, projection="3d")
    elif space.__class__.__name__ == "Hyperboloid":
        ax = fig.add_subplot(111)
    ax = manifold_visu.set_ax(ax=ax)
    projected_intercept_hat = space.projection(sequence_2[0])
    projected_sequence_2 = space.projection(sequence_2)
    projected_sequence_1 = space.projection(sequence_1)
    if sequence_3 is not None:
        projected_sequence_3 = space.projection(sequence_3)
    manifold_visu.plot(
        gs.array([projected_intercept_hat]), ax=ax, marker=marker, c="r", s=size
    )
    manifold_visu.plot(
        projected_sequence_1, ax=ax, marker=marker, c="b", s=size, label="True"
    )
    manifold_visu.plot(
        projected_sequence_2, ax=ax, marker=marker, c="g", s=size, label="LR"
    )
    if sequence_3 is not None:
        manifold_visu.plot(
            projected_sequence_3, ax=ax, marker=marker, c="k", s=size, label="GR"
        )

    ax.grid(False)
    plt.axis("off")
    plt.legend()

    return fig


def scatterplot_evaluation(
    df,
    colored_by="noise_factor",
    marked_by="n_steps",
    x_label="n_steps",
    y_label="relative_diff_seq_duration",
):
    """Scatterplot of results.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe made from wandb with config and results.
    colored_by : string
        Column name to color the points by.
    marked_by : string
        Column name to mark the points by, using different symbols.
    x_label : string
        Column name to plot on the x-axis.
    y_label : string
        Column name to plot on the y-axis.
    """
    x = df[x_label]
    y = df[y_label]
    value_to_symbol = dict(
        zip(df[marked_by].unique(), ["square", "x", "cross", "diamond", "star"])
    )

    marked_values = [s for s in df[marked_by].values]
    if marked_by == "linear_noise":
        symbol_value_to_legend_value = {
            s: "Linear Noise" if s else "Manifold Noise" for s in df[marked_by].unique()
        }
        marked_values = [
            symbol_value_to_legend_value[s] if ~np.isnan(s) else s
            for s in df[marked_by].values
        ]

    colored_values = [str(c) for c in df[colored_by].values]
    if colored_by == "linear_residuals":
        color_value_to_legend_value = {
            c: "GRLR" if c else "GR" for c in df[colored_by].unique()
        }
        colored_values = [
            color_value_to_legend_value[c] if ~np.isnan(c) else c
            for c in df[colored_by].values
        ]

    if colored_by == "n_steps":
        color_discrete_sequence = px.colors.sequential.Plasma_r
    else:
        color_discrete_sequence = px.colors.sequential.Viridis_r

    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=colored_values,
        color_discrete_sequence=color_discrete_sequence,
        symbol=marked_values,
        symbol_map=value_to_symbol,
    )

    legend_title = COL_TO_TEXT[colored_by] + ", " + COL_TO_TEXT[marked_by]

    fig.update_layout(
        xaxis_title=dict(
            text=COL_TO_TEXT[x_label],
            font=dict(family="CMU", size=FONTSIZE),
        ),
        yaxis_title=dict(
            text=COL_TO_TEXT[y_label], font=dict(family="CMU", size=FONTSIZE)
        ),
        title_font=dict(family="CMU", size=FONTSIZE),
        xaxis=dict(tickfont=dict(family="CMU", size=FONTSIZE)),
        yaxis=dict(tickfont=dict(family="CMU", size=FONTSIZE)),
        legend=dict(font=dict(family="CMU", size=FONTSIZE), title=legend_title),
        width=650,
        height=370,
    )

    fig.update_traces(marker=dict(size=9, opacity=0.9))
    fig.show()
    return fig
