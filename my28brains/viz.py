"""Visualization tools."""

import copy
import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import nibabel
import numpy as np
import open3d as o3d
from matplotlib import animation

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
    string_base = os.path.join(DATA_DIR, f"sub-01/ses-**/{img_suffix}")
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


def getMeshFromData(mesh, Rho=None, color=None):
    """
    Performs midpoint subdivision. Order determines the number of iterations
    """
    V = mesh[0]
    F = mesh[1]
    # mesh=o3d.geometry.TriangleMesh(o3d.cpu.pybind.utility.Vector3dVector(V),o3d.cpu.pybind.utility.Vector3iVector(F))
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F)
    )

    if Rho is not None:
        Rho = np.squeeze(Rho)
        col = np.stack((Rho, Rho, Rho))
        mesh.vertex_colors = o3d.utility.Vector3dVector(col.T)

    if color is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(color)
    return mesh


def makeGeodMeshes(
    Geod,
    F,
    o_source=None,
    o_target=None,
    Rho=None,
    offset=0,
    offsetstep=2.5,
    stepsize=2,
    axis=[0, 0, 1],
    angle=-1 * np.pi / 2,
):
    """
    Constructs an open3d mesh for the geodesic
    """
    Nt = len(Geod)
    ls = []
    ls_Rho = np.array([], np.float64)
    mesh = getMeshFromData([Geod[0], F]).translate((-stepsize, 0, 0), relative=False)
    R = mesh.get_rotation_matrix_from_axis_angle(np.array(axis) * angle)
    mesh.clear()

    if o_source != None:
        N = o_source[0].shape[0]
        mesh = getMeshFromData(o_source)
        newmesh = copy.deepcopy(mesh)
        newmesh.compute_vertex_normals()
        newmesh.normalize_normals()
        colors_np = np.asarray(newmesh.vertex_normals)
        colors_np = (colors_np + 1) / 2
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors_np)
        if Rho is not None:
            ls_Rho = np.append(ls_Rho, np.ones((N,), np.float64))
        mesh.translate((-stepsize, 0, offset * offsetstep), relative=False)
        mesh.rotate(R, center=(-stepsize, 0, offset * offsetstep))
        ls.append(mesh)
    for i in range(0, Nt):
        V = Geod[i]
        mesh = getMeshFromData([V, F])
        if Rho is not None:
            t = i / float(Nt - 1)
            Rhot = t * Rho + 1 - t
        if i == 0:
            newmesh = copy.deepcopy(mesh)
            newmesh.compute_vertex_normals()
            newmesh.normalize_normals()
            colors_np = np.asarray(newmesh.vertex_normals)
            colors_np = (colors_np + 1) / 2
        if Rho is not None:
            mesh.vertex_colors = o3d.utility.Vector3dVector(
                np.stack((Rhot, Rhot, Rhot), axis=1) * colors_np
            )
            ls_Rho = np.append(ls_Rho, Rhot)
        else:
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors_np)
        mesh.translate((i * stepsize, 0, offset * offsetstep), relative=False)
        mesh.rotate(R, center=(i * stepsize, 0, offset * offsetstep))
        ls.append(mesh)
    if o_target != None:
        N = o_target[0].shape[0]
        mesh = getMeshFromData(o_target)
        newmesh = copy.deepcopy(mesh)
        newmesh.compute_vertex_normals()
        newmesh.normalize_normals()
        colors_np = np.asarray(newmesh.vertex_normals)
        colors_np = (colors_np + 1) / 2
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors_np)
        if Rho is not None:
            ls_Rho = np.append(ls_Rho, np.ones((N,), np.float64))
        mesh.translate((Nt * stepsize, 0, offset * offsetstep), relative=False)
        mesh.rotate(R, center=(Nt * stepsize, 0, offset * offsetstep))
        ls.append(mesh)
    if Rho is not None:
        return ls, ls_Rho
    return ls


def plotGeodesic(
    geod,
    F,
    source=None,
    target=None,
    file_name=None,
    offsetstep=2.5,
    stepsize=2,
    axis=[0, 0, 1],
    angle=-1 * np.pi / 2,
):
    """Plot geodesic evolution after symmetric or asymmetric matching with the H2 metric and varifold relaxation.

    Input:
        - geod: geodesic path [tuple with tuple[k]=vertices of k^th surface in the geodesic stored as an nVx3 ndarray]
        - F: faces for the mesh structure of the surfaces on the geodesic path [nFx3 ndarray]
        - source [tuple with source[0]=vertices, source[1]=faces, default=None]
        - target [tuple with target[0]=vertices, target[1]=faces, default=None]
        - file_name: specified path for saving geodesic mesh [string, default=None]
        - offsetstep: spacing between different geodesics on the plot [default=2.5]
        - stepsize: spacing within a geodesic on the plot [default=2]
        - axis: axis of rotation for each individual surface in the geodesic [default=[0,0,1]]
        - angle: angle of rotation [default=-pi/2]
    Output:
        - Plot of geodesic with source (left), geodesic path (middle) and target (right)
        - file_name.ply file containing geodesic mesh information (optional)
    """

    # Convert data to open3d mesh objects for generating plots of the geodesic path
    ls = makeGeodMeshes(
        geod,
        F,
        source,
        target,
        offsetstep=offsetstep,
        stepsize=stepsize,
        axis=axis,
        angle=angle,
    )
    o3d.visualization.draw_geometries(ls)

    # Save plots if specified by user
    if file_name != None:
        mesh = ls[0]
        for i in range(1, len(ls)):
            mesh += ls[i]
        V, F, Color = getDataFromMesh(mesh)
        if mesh.has_vertex_colors():
            saveData(file_name, "ply", V, F, color=Color)
        else:
            saveData(file_name, "ply", V, F)


def plotPartialGeodesic(
    geod,
    F,
    source=None,
    target=None,
    Rho=None,
    file_name=None,
    offsetstep=2.5,
    stepsize=2,
    axis=[0, 0, 1],
    angle=-1 * np.pi / 2,
):
    """Plot geodesic evolution after partial matching with the H2 metric and weighted varifold relaxation.

    Input:
        - geod: geodesic path [tuple with tuple[k]=vertices of k^th surface in the geodesic stored as an nVx3 ndarray]
        - F: faces for the mesh structure of the surfaces on the geodesic path [nFx3 ndarray]
        - source [tuple with source[0]=vertices, source[1]=faces, default=None]
        - target [tuple with target[0]=vertices, target[1]=faces, default=None]
        - Rho: weights defined on the endpoint of the geodesic [nVx1 numpy ndarray, default=None]
        - file_name: specified path for saving geodesic mesh [string, default=None]
        - offsetstep: spacing between different geodesics on the plot [default=2.5]
        - stepsize: spacing within a geodesic on the plot [default=2]
        - axis: axis of rotation for each individual surface in the geodesic [default=[0,0,1]]
        - angle: angle of rotation [default=-pi/2]
    Output:
        - Plot of geodesic with source (left), geodesic path (middle) and target (right) - with interpolated weights on the path
        - file_name.ply file containing geodesic mesh information (optional)
    """

    # Convert data to open3d mesh objects for generating plots of the geodesic path
    if Rho is not None:
        ls, Rhon = makeGeodMeshes(
            geod,
            F,
            source,
            target,
            Rho=Rho,
            offsetstep=offsetstep,
            stepsize=stepsize,
            axis=axis,
            angle=angle,
        )
        Rhot = np.array(Rhon)
    else:
        ls = makeGeodMeshes(
            geod,
            F,
            source,
            target,
            offsetstep=offsetstep,
            stepsize=stepsize,
            axis=axis,
            angle=angle,
        )
    o3d.visualization.draw_geometries(ls)

    # Save plots if specified by user
    if file_name != None:
        mesh = ls[0]
        for i in range(1, len(ls)):
            mesh += ls[i]
        V, F, Color = getDataFromMesh(mesh)
        if mesh.has_vertex_colors():
            if Rho is not None:
                Rhot = np.asarray(255 * Rhot, dtype=np.int)
                saveData(file_name, "ply", V, F, Rho=Rhot, color=Color)
            else:
                saveData(file_name, "ply", V, F, color=Color)

        else:
            saveData(file_name, "ply", V, F)
