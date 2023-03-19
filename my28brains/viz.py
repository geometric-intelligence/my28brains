"""Visualization tools."""

import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import nibabel
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib import animation

IMG_DIR = "/home/data/28andme/"
HORMONES = {"Estro": "Estrogen", "Prog": "Progesterone", "LH": "LH", "FSH": "FSH"}
COLORS = {"Estro": "#1f77b4", "Prog": "#ff7f0e", "LH": "#2ca02c", "FSH": "#d62728"}

FIGS = os.path.join(os.getcwd(), "notebooks", "figs")
GIFS = os.path.join(os.getcwd(), "notebooks", "gifs")


def init_matplotlib():
    """Configure style for matplotlib tutorial."""
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
    """Plot hormones - original function."""
    if plot_type == "dot":
        df = df[df["dayID"] < dayID]
    times = df["dayID"]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for h in hormones:
        ax.plot(times, df[h], label=HORMONES[h])

    if dayID > 2:
        if plot_type == "dot":
            for h in hormones:
                ax.scatter(times.values[-1], df[h].values[-1], s=100)

    ax.set_xlim((0, 30))
    ax.set_ylim(0, df["Estro"].max() + 5)
    ax.legend(loc="upper left")
    if savefig:
        fig.savefig(f"{FIGS}/plot_hormones_dot_pic{dayID:02d}.svg")
    return fig


def plotly_hormones(
    df, dayID, plot_type="dot", hormones=HORMONES, ymax=None, savefig=False
):
    """Plot hormones with plotly."""
    if ymax is None:
        ymax = df[hormones].max().max() + 10
    if plot_type == "dot":
        df = df[df["dayID"] < dayID]
    times = df["dayID"]

    fig = go.Figure()

    # Add traces
    for h in hormones:
        fig.add_trace(
            go.Scatter(
                x=times,
                y=df[h],
                mode="lines+markers" if plot_type == "dot" else "lines",
                name=HORMONES[h],
                marker=dict(color=COLORS[h]),
            )
        )

    # Add the last point with a larger marker size if dayID > 2 and plot_type is "dot"
    if dayID > 1:
        if plot_type == "dot":
            for h in hormones:
                fig.add_trace(
                    go.Scatter(
                        x=[times.values[-1]],
                        y=[df[h].values[-1]],
                        mode="markers",
                        marker=dict(size=10, color=COLORS[h]),
                        showlegend=False,
                    )
                )
    if plot_type == "vertical_line":
        for h in hormones:
            fig.add_trace(
                go.Scatter(
                    x=[times.values[dayID], times.values[dayID]],
                    y=[0, ymax],
                    mode="lines",
                    line=dict(color="black"),
                    showlegend=False,
                )
            )

    # Set the axis labels and title
    fig.update_layout(
        xaxis_title="Day",
        yaxis_title="Hormone Level",
        xaxis=dict(range=[0, 30]),
        yaxis=dict(range=[0, ymax]),
        width=900,  # Adjust the width as needed
        height=300,  # Adjust the height as needed
    )

    # Save the figure
    if savefig:
        pio.write_image(fig, f"{FIGS}/plotly_hormones_{dayID:02d}.png", format="png")

    # Show the figure
    fig.show()
