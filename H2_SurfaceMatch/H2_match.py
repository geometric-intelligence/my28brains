# Load Packages
import default_config
import numpy as np
import scipy
import torch
import utils.utils as io
from enr.DDG import computeBoundary
from enr.H2 import *
from H2_param import H2Midpoint
from scipy.optimize import fmin_l_bfgs_b, minimize
from torch.autograd import grad

torch_dtype = torch.float64


def SymmetricH2Matching(
    source, target, geod, F_init, a0, a1, b1, c1, d1, a2, param, device=None
):

    sig_geom = param["sig_geom"]

    if "sig_grass" not in param:
        sig_grass = 1
    else:
        sig_grass = param["sig_grass"]

    if "kernel_geom" not in param:
        kernel_geom = "gaussian"
    else:
        kernel_geom = param["kernel_geom"]

    if "kernel_grass" not in param:
        kernel_grass = "binet"
    else:
        kernel_grass = param["kernel_grass"]

    if "kernel_fun" not in param:
        kernel_fun = "constant"
    else:
        kernel_fun = param["kernel_fun"]

    if "sig_fun" not in param:
        sig_fun = 1
    else:
        sig_fun = param["sig_fun"]

    weight_coef_dist_T = param["weight_coef_dist_T"]
    weight_coef_dist_S = param["weight_coef_dist_S"]
    max_iter = param["max_iter"]

    # Convert Data to Pytorch
    VS = torch.from_numpy(source[0]).to(dtype=torch_dtype, device=device)
    VT = torch.from_numpy(target[0]).to(dtype=torch_dtype, device=device)
    FS = torch.from_numpy(source[1]).to(dtype=torch.long, device=device)
    FT = torch.from_numpy(target[1]).to(dtype=torch.long, device=device)
    geod = torch.from_numpy(geod).to(dtype=torch_dtype, device=device)
    F_sol = torch.from_numpy(F_init).to(dtype=torch.long, device=device)
    B_sol = torch.from_numpy(computeBoundary(F_init)).to(
        dtype=torch.bool, device=device
    )

    N = geod.shape[0]
    n = geod.shape[1]

    FunS = torch.from_numpy(np.zeros((int(np.size(source[0]) / 3),))).to(
        dtype=torch_dtype, device=device
    )
    FunT = torch.from_numpy(np.zeros((int(np.size(target[0]) / 3),))).to(
        dtype=torch_dtype, device=device
    )
    Fun_sol = torch.from_numpy(np.zeros((int(geod.shape[1]),))).to(
        dtype=torch_dtype, device=device
    )

    sig_geom = torch.tensor([sig_geom], dtype=torch_dtype, device=device)
    sig_grass = torch.tensor([sig_grass], dtype=torch_dtype, device=device)
    sig_fun = torch.tensor([sig_fun], dtype=torch_dtype, device=device)

    # Define Energy and set parameters
    energy = enr_match_H2_sym(
        VS,
        FS,
        FunS,
        VT,
        FT,
        FunT,
        F_sol,
        Fun_sol,
        B_sol,
        weight_coef_dist_S=weight_coef_dist_S,
        weight_coef_dist_T=weight_coef_dist_T,
        kernel_geom=kernel_geom,
        kernel_grass=kernel_grass,
        kernel_fun=kernel_fun,
        sig_geom=sig_geom,
        sig_grass=sig_grass,
        sig_fun=sig_fun,
        a0=a0,
        a1=a1,
        b1=b1,
        c1=c1,
        d1=d1,
        a2=a2,
        device=device,
    )

    def gradE(geod):
        qgeod = geod.clone().requires_grad_(True)
        return grad(energy(qgeod), qgeod, create_graph=True)

    def funopt(geod):
        geod = torch.from_numpy(geod.reshape(N, n, 3)).to(
            dtype=torch_dtype, device=device
        )
        return float(energy(geod).detach().cpu().numpy())

    def dfunopt(geod):
        geod = torch.from_numpy(geod.reshape(N, n, 3)).to(
            dtype=torch_dtype, device=device
        )
        [Ggeod] = gradE(geod)
        Ggeod = Ggeod.detach().cpu().numpy().flatten().astype("float64")
        return Ggeod

    xopt, fopt, Dic = fmin_l_bfgs_b(
        funopt,
        geod.cpu().numpy().flatten(),
        fprime=dfunopt,
        pgtol=1e-05,
        epsilon=1e-08,
        maxiter=max_iter,
        iprint=0,
        maxls=20,
        maxfun=150000,
    )
    geod = xopt.reshape(N, n, 3)
    return geod, fopt, Dic


def StandardH2Matching(source, target, geod, F_init, a0, a1, b1, c1, d1, a2, param):
    sig_geom = param["sig_geom"]

    if "sig_grass" not in param:
        sig_grass = 1
    else:
        sig_grass = param["sig_grass"]

    if "kernel_geom" not in param:
        kernel_geom = "gaussian"
    else:
        kernel_geom = param["kernel_geom"]

    if "kernel_grass" not in param:
        kernel_grass = "binet"
    else:
        kernel_grass = param["kernel_grass"]

    if "kernel_fun" not in param:
        kernel_fun = "constant"
    else:
        kernel_fun = param["kernel_fun"]

    if "sig_fun" not in param:
        sig_fun = 1
    else:
        sig_fun = param["sig_fun"]

    weight_coef_dist_T = param["weight_coef_dist_T"]
    max_iter = param["max_iter"]

    # Convert Data to Pytorch
    VS = torch.from_numpy(source[0]).to(dtype=torch_dtype, device=device)
    VT = torch.from_numpy(target[0]).to(dtype=torch_dtype, device=device)
    FS = torch.from_numpy(source[1]).to(dtype=torch.long, device=device)
    FT = torch.from_numpy(target[1]).to(dtype=torch.long, device=device)
    F_sol = torch.from_numpy(F_init).to(dtype=torch.long, device=device)
    B_sol = torch.from_numpy(computeBoundary(F_init)).to(
        dtype=torch.bool, device=device
    )

    N = geod.shape[0]
    n = geod.shape[1]

    FunS = torch.from_numpy(np.zeros((int(np.size(source[0]) / 3),))).to(
        dtype=torch_dtype, device=device
    )
    FunT = torch.from_numpy(np.zeros((int(np.size(target[0]) / 3),))).to(
        dtype=torch_dtype, device=device
    )
    Fun_sol = torch.from_numpy(np.zeros((int(geod.shape[1]),))).to(
        dtype=torch_dtype, device=device
    )

    sig_geom = torch.tensor([sig_geom], dtype=torch_dtype, device=device)
    sig_grass = torch.tensor([sig_grass], dtype=torch_dtype, device=device)
    sig_fun = torch.tensor([sig_fun], dtype=torch_dtype, device=device)

    geod = torch.from_numpy(geod[1:]).to(dtype=torch_dtype, device=device)

    # Define Energy and set parameters
    energy = enr_match_H2(
        VS,
        VT,
        FT,
        FunT,
        F_sol,
        Fun_sol,
        B_sol,
        weight_coef_dist_T=weight_coef_dist_T,
        kernel_geom=kernel_geom,
        kernel_grass=kernel_grass,
        kernel_fun=kernel_fun,
        sig_geom=sig_geom,
        sig_grass=sig_grass,
        sig_fun=sig_fun,
        a0=a0,
        a1=a1,
        b1=b1,
        c1=c1,
        d1=d1,
        a2=a2,
    )

    def gradE(geod):
        qgeod = geod.clone().requires_grad_(True)
        return grad(energy(qgeod), qgeod, create_graph=True)

    def funopt(geod):
        geod = torch.from_numpy(geod.reshape(N - 1, n, 3)).to(
            dtype=torch_dtype, device=device
        )
        return float(energy(geod).detach().cpu().numpy())

    def dfunopt(geod):
        geod = torch.from_numpy(geod.reshape(N - 1, n, 3)).to(
            dtype=torch_dtype, device=device
        )
        [Ggeod] = gradE(geod)
        Ggeod = Ggeod.detach().cpu().numpy().flatten().astype("float64")
        return Ggeod

    xopt, fopt, Dic = fmin_l_bfgs_b(
        funopt,
        geod.cpu().numpy().flatten(),
        fprime=dfunopt,
        pgtol=1e-05,
        epsilon=1e-08,
        maxiter=max_iter,
        iprint=1,
        maxls=20,
        maxfun=150000,
    )
    geod = np.concatenate(
        (np.expand_dims(VS.cpu().numpy(), axis=0), xopt.reshape(N - 1, n, 3)), axis=0
    )
    return geod, fopt, Dic


def H2MultiRes(
    source,
    target,
    a0,
    a1,
    b1,
    c1,
    d1,
    a2,
    resolutions,
    paramlist,
    start=None,
    rotate=False,
    device=None,
):
    N = 2
    [VS, FS] = source
    [VT, FT] = target

    sources = [[VS, FS]]
    targets = [[VT, FT]]

    print("before starting: Vertices then Faces for S then T")
    print(VS.shape, FS.shape)
    print(VT.shape, FT.shape)

    for i in range(0, resolutions):
        decimation_fact = 2
        print(f"FS decimation target: {int(FS.shape[0] / decimation_fact)}")
        print(f"FT decimation target: {int(FT.shape[0] / decimation_fact)}")

        [VS, FS] = io.decimate_mesh(VS, FS, int(FS.shape[0] / decimation_fact))  # was 4
        sources = [[VS, FS]] + sources
        [VT, FT] = io.decimate_mesh(VT, FT, int(FT.shape[0] / decimation_fact))  # was 4
        targets = [[VT, FT]] + targets
        print(f"Resol {i}: Vertices for S then T")
        print(VS.shape, FS.shape)
        print(VT.shape, FT.shape)

    source_init = sources[0]
    target_init = sources[0]

    if start != None:
        geod = np.zeros((N, start[0].shape[0], start[0].shape[1]))
        F0 = start[1]
        for i in range(0, N):
            geod[i, :, :] = start[0]

    else:
        geod = np.zeros((N, source_init[0].shape[0], source_init[0].shape[1]))
        F0 = source_init[1]
        for i in range(0, N):
            geod[i, :, :] = source_init[0]

    iterations = len(paramlist)
    for j in range(0, iterations):
        params = paramlist[j]
        time_steps = params["time_steps"]
        tri_upsample = params["tri_unsample"]
        index = params["index"]
        geod = np.array(geod)
        [N, n, three] = geod.shape
        geod, ener, Dic = SymmetricH2Matching(
            sources[index],
            targets[index],
            geod,
            F0,
            a0,
            a1,
            b1,
            c1,
            d1,
            a2,
            params,
            device=device,
        )
        print(f"\n ############ Iteration {j}:")
        print("F0.shape", F0.shape)
        print("geod.shape:", geod.shape)
        if time_steps > 2:
            # Note: here, geod is still an array.
            print("in timesteps:")
            geod = H2Midpoint(
                geod, time_steps, F0, a0, a1, b1, c1, d1, a2, params, device=device
            )
            print("geod.shape:", geod.shape)
        if tri_upsample:
            # Note: geod is a list at the end of the iteration if tri_upsample is True.
            print("in  triupsample:")
            geod_sub = []
            F_Sub = []
            for i in range(0, N):
                print(f"iteration {i} in range(N) to subdivide mesh")
                geod_subi, F_Subi = io.subdivide_mesh(geod[i], F0, order=1)
                F_Sub.append(F_Subi)
                geod_sub.append(geod_subi)
            geod = geod_sub
            F0 = F_Sub[0]
            print("len(geod):", len(geod))
    print(iterations, F0.shape)
    return geod, F0


def H2StandardIterative(
    source, target, a0, a1, b1, c1, d1, a2, paramlist, rotate=False
):
    N = 2
    [VS, FS] = source
    [VT, FT] = target
    geod = np.zeros((N, source[0].shape[0], source[0].shape[1]))
    F0 = source[1]
    for i in range(0, N):
        geod[i, :, :] = source[0]
    iterations = len(paramlist)
    for j in range(0, iterations):
        params = paramlist[j]
        time_steps = params["time_steps"]
        geod = np.array(geod)
        [N, n, three] = geod.shape
        geod, ener, Dic = StandardH2Matching(
            source, target, geod, F0, a0, a1, b1, c1, d1, a2, params
        )
        print(j, F0.shape)
        if time_steps > 2:
            geod = H2Midpoint(geod, time_steps, F0, a0, a1, b1, c1, d1, a2, params)
    print(iterations, F0.shape)
    return geod, F0
