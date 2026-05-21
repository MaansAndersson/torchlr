import math
import argparse
import numpy as np
import torch
import os
import csv


import matplotlib
from matplotlib import pyplot as plt
from collections import namedtuple
from torchlr.trunc_sum import trunc_sum
from torchlr.crossDEIM import crossDEIM
from torchlr.index_selection import qdeim

import torchlr.torch_config

# ---------------------------------------------------------------------------
# Global device selection
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64

def plotter(it, nt, plotfreq, xx, yy, U, S, V, fig, axes, iter_idx):

    if (it % plotfreq == 0) or (it == nt - 1):
        xx_np = xx.cpu().numpy()
        yy_np = yy.cpu().numpy()
        s = S.flatten()
        W = (U @ torch.diag(s) @ V.T).cpu().numpy()

        axes[0].clear()
        axes[0].plot_surface(xx_np, yy_np, W)
        axes[0].set_aspect('equal')
        axes[0].set_title(f'Timestep {it}')

        axes[1].clear()
        axes[1].contour(xx_np, yy_np, W, levels=np.linspace(0.1, 0.5, 20), linewidths=2)
        J_idx = qdeim(V).cpu().numpy()
        I_idx = qdeim(U).cpu().numpy()
        Xint = xx_np[np.ix_(I_idx, J_idx)]
        Yint = yy_np[np.ix_(I_idx, J_idx)]
        axes[1].scatter(Xint.flatten(), Yint.flatten(), marker='o', c='k')
        axes[1].set_aspect('equal')


        dot1 = axes[2].plot(it, U.shape[1], 'ko', markersize=15, label="rank")
        axes[2].set_xlim([0, int(nt)])
        axes[2].set_ylim([0, 20])

        dot2 = axes[2].plot(it, iter_idx, 'r*', markersize=15, label="iter")

        axes[2].legend(['Rank of the solution','Cross-DEIM iters'])
        axes[2].set_xlim([0, int(nt)])
        axes[2].set_ylim([0, 20])


        
        plt.show()
        plt.pause(0.0001)

# ---------------------------------------------------------------------------
# Core right-hand-side evaluation (vectorized)
# ---------------------------------------------------------------------------

@torch.compile
def grhside(I, J, U, S, V, hi, hj, dt, nu, x, y):
    """
    I, J : 1-D integer tensors of row/col indices
    U    : (n1, r) float tensor
    S    : (r,)  or (r, r) float tensor
    V    : (n2, r) float tensor
    hi, hj, dt, nu : scalars
    x, y : unused (kept for API compatibility)
    """

    # --- singular values as 1-D vector ---
    s = torch.diag(S) if S.ndim == 2 else S.flatten()   # (r,)

    # --- precompute A = U[I,:]*s  and  B = V[J,:]^T ---
    A      = U[I, :] * s          # (ni, r)
    B      = V[J, :].T            # (r,  nj)

    # --- periodic neighbours via torch.roll ---
    U_up    = torch.roll(U, -1, dims=0)   # il -> il+1
    U_down  = torch.roll(U,  1, dims=0)   # il -> il-1
    V_left  = torch.roll(V, -1, dims=0)   # jl -> jl+1
    V_right = torch.roll(V,  1, dims=0)   # jl -> jl-1

    # --- gather neighbours at selected rows/cols ---
    A_up    = U_up  [I, :] * s    # (ni, r)
    A_down  = U_down[I, :] * s    # (ni, r)
    B_left  = V_left [J, :].T     # (r, nj)
    B_right = V_right[J, :].T     # (r, nj)

    # --- stencil weights ---
    r_diff = nu / hi**2
    c_diff = nu / hj**2

    # --- five-point stencil values ---
    w_cc    = A      @ B        # (ni, nj) centre
    w_up    = A_up   @ B        # (ni, nj) il+1, jl
    w_down  = A_down @ B        # (ni, nj) il-1, jl
    w_left  = A      @ B_left   # (ni, nj) il,   jl+1
    w_right = A      @ B_right  # (ni, nj) il,   jl-1

    # --- RHS ---
    rhs = (w_cc
           + dt * (r_diff     * (w_up   + w_down  - 2.0*w_cc)
                 + c_diff     * (w_left + w_right - 2.0*w_cc)
                 + (0.5 / hj) * (w_up   - w_down)
                 + (0.5 / hi) * (w_left - w_right)
                 ))

    return rhs

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(n,
         tend,
         index_selection_var,
         filename,
         plotfreq):
    # ---- parameters ----

    CFL  = 0.5
    nu   = 0.1

    n1 = n
    n2 = n

    h1 = 2.0 / n1
    h2 = 2.0 / n2
    h  = min(h1, h2)
    nu   = h*h
    dt = 0.5*CFL * min(h,h*h/nu)
    nt = math.ceil(tend / dt)
    dt = tend / nt

    print(f"Device : {DEVICE}")
    print(f"dt     : {dt:.6e}   nt={nt}")
    print(f"tend   : {tend:.6e}")

    # ---- grid ----
    x  = torch.linspace(-1.0 + h1, 1.0, n1, dtype=DTYPE, device=DEVICE)
    y  = torch.linspace(-1.0 + h2, 1.0, n2, dtype=DTYPE, device=DEVICE)
    xx, yy = torch.meshgrid(x, y, indexing='ij')   # (n1, n2)

    # ---- initial condition  w0 = exp(-(x/0.1)²) ⊗ exp(-(y/0.1)²) ----
    U0 = torch.exp(-(x / 0.1)**2).reshape(-1, 1)   # (n1, 1)
    V0 = torch.exp(-(y / 0.1)**2).reshape(-1, 1)   # (n2, 1)
    S0 = (torch.linalg.norm(U0) * torch.linalg.norm(V0)).reshape(1, 1)
    U0 = U0 / torch.linalg.norm(U0)
    V0 = V0 / torch.linalg.norm(V0)

    U = U0.clone()
    S = S0.clone()
    V = V0.clone()

    # ---- solver options ----
    hard, soft = 0, 1
    rmax = 40
    tol  = dt**2
    mode = hard

    rin            = min(2 * rmax, min(n1 - 1, n2 - 1))
    max_iter_cross = 20
    Options        = namedtuple('Options', ['tol', 'rmax', 'rin', 'max_iter', 'index_selection'])
    opts           = Options(tol, rmax, rin, max_iter_cross, index_selection_var)

    # ---- matplotlib setup ----

    if filename is not None:
        plt.ion()
        fig = plt.figure(figsize=(15, 4))
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)
        axes = [ax1, ax2, ax3]

    # ---- time integration (Strang-split / RK2 style matching original) ----
    for it in range(nt):
        # Stage 1
        gfunc1 = lambda I, J: grhside(I, J, U, S, V, h1, h2, dt, nu, x, y)
        U1, S1, V1, solver_data1, info1 = crossDEIM(gfunc1, U, S, V, opts)
        # Stage 2
        gfunc2 = lambda I, J: grhside(I, J, U1, S1, V1, h1, h2, dt, nu, x, y)
        U2, S2, V2, solver_data2, info2 = crossDEIM(gfunc2, U1, S1, V1, opts)

        Ui_list = [U,       U2     ]
        Si_list = [0.5 * torch.diag(S), 0.5 * torch.diag(S2)]
        Vi_list = [V,       V2     ]

        U, S, V = trunc_sum(Ui_list, Si_list, Vi_list, tol, rmax, mode, DTYPE)

        if filename is not None:
            plotter(it, nt, plotfreq, xx, yy, U, S, V, fig, axes, info2+info1)

        # Keep S as a 2-D diagonal matrix to match crossDEIM expectations
        #if S.ndim == 1:
        #    S = torch.diag(S)

        
    if filename is not None:
        print('write fig to file')
        plt.savefig(filename)

    return



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                    prog='low-rank advection diffusion solver',
                    description='This program solves a low-rank advection diffusion problem.',
                    epilog='Thank you!')
    parser.add_argument('-n', '--problemsize')
    parser.add_argument('-is','--indexselection')
    parser.add_argument('-t', '--endtime')
    parser.add_argument('-pf', '--plotfreq')
    parser.add_argument('-f', '--filename')
    parser.add_argument('-s','--seed')

    parser.set_defaults(problemsize = 1024,
                        indexselection = 'deim',
                        endtime = 2,
                        plotfreq = 100,
                        filename = "ad_fig",
                        logfile = None,
                        seed = 1)

    args = parser.parse_args()
    
    torch.manual_seed(int(args.seed))

    main(n = int(args.problemsize),
         index_selection_var = args.indexselection,
         tend = float(args.endtime),
         plotfreq = int(args.plotfreq),
         filename = args.filename)

    

