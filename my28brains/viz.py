"""Visualization tools."""

import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import nibabel
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib import animation

import my28brains.default_config as default_config

IMG_DIR = "/home/data/28andme/"
HORMONES = {"Estro": "Estrogen", "Prog": "Progesterone", "LH": "LH", "FSH": "FSH"}

# Colors follow the color scheme from Taylor et al. 2020
COLORS = {
    "Estro": "#AFEEEE",  # pastel blue-turquoise
    "Prog": "#507DBC",  # dark blue
    "LH": "#FF7373",  # red
    "FSH": "#FADA5E",  # pastel yellow
}

TMP = default_config.tmp_dir


def init_matplotlib():
    """Configure style for matplotlib."""
    fontsize = 18
    matplotlib.rc("font", size=fontsize)
    matplotlib.rc("text")
    matplotlib.rc("legend", fontsize=fontsize)
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
        png_path = os.path.join("my28brains", "notebooks", png_filename)
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
