"""Visualization tools."""

import glob
import os

import matplotlib
import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib import animation
import numpy as np

import nibabel

DATA_DIR = "/home/data/28andme/"


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
    string_base = os.path.join(
        DATA_DIR, f"sub-01/ses-**/{img_suffix}")
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
        array_4d  = np.ma.masked_where(array_4d < 0.05, array_4d)
        cmap = matplotlib.cm.get_cmap("tab20b").copy()
        cmaplist = [cmap(2*i) for i in range(10)]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', cmaplist, 10)
        cmap.set_bad(color='black')

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