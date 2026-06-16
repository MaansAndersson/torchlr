"""
Cross-DEIM driver and helper steps.

This module contains the high-level crossDEIM interface which iteratively
builds a low-rank approximation of a two-dimensional operator sampled by
`gfun`. The implementation is close to the original algorithm and is
annotated for readability.
"""

from typing import Callable, Optional, Tuple

import torch
from torchlr.scross import scross
from torchlr import index_selection
from torchlr.compute_residual import compute_residual


@torch.compile(dynamic=True)
def crossDEIM(
    gfun: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    U: torch.Tensor,
    S: torch.Tensor,
    V: torch.Tensor,
    opts: Optional[object] = None,
    compute_residual_bool: Optional[bool] = None,
    dtype=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Iterative cross-DEIM algorithm.

    Parameters
    ----------
    gfun : callable
        Function gfun(Iidx, Jidx) -> matrix sampled at (Iidx, Jidx).
    U, S, V
        Initial low-rank factors.
    opts
        Options object with attributes tol, rmax, rin, max_iter,
        index_selection (string).

    Returns
    -------
    U, S, V, solver_data, iter_idx
    """
    if opts is not None:
        tolLR = opts.tol
        tolRES = opts.tol
        tolE = opts.tol
        rmax = opts.rmax
        rin = opts.rin
        max_iter = opts.max_iter
        index_selection_method = opts.index_selection
    else:
        # Provide reasonable defaults when opts is not provided
        tolLR = 1e-10
        tolRES = 1e-10
        tolE = 1e-10
        rmax = min(U.shape[0], V.shape[0])
        rin = max(1, U.shape[1])
        max_iter = 20
        index_selection_method = "deim"

    n1 = U.shape[0]
    n2 = V.shape[0]

    Iall = torch.arange(n1, device=U.device, dtype=torch.long, requires_grad=False)
    Jall = torch.arange(n2, device=V.device, dtype=torch.long, requires_grad=False)

    I0 = torch.tensor([], dtype=torch.long, device=U.device)
    J0 = torch.tensor([], dtype=torch.long, device=V.device)

    solver_data = torch.zeros((max_iter, 8))

    if index_selection_method == "ls":
        pu = torch.empty((n1, max_iter), device=U.device).uniform_()
        pv = torch.empty((n2, max_iter), device=V.device).uniform_()

    Uold = U; Vold = V; Sold = S
    for iter_idx in range(max_iter):

        if index_selection_method == "qdeim":
            I = index_selection.qdeim(U)
            J = index_selection.qdeim(V)
            deim_bool = True
        elif index_selection_method == "deim":
            I = index_selection.deim(U, Iall)
            J = index_selection.deim(V, Jall)
            deim_bool = True
        elif index_selection_method == "ls":
            I = index_selection.leverage_score(U, pu[:, iter_idx])
            J = index_selection.leverage_score(V, pv[:, iter_idx])
            deim_bool = False
        else:
            raise Exception("#### Warning no or inccorect index_selection chosen! #####")

        Ilen = len(I)
        Jlen = len(J)

        U, S, V, I, J, I0, J0 = xdeim_step(
            gfun, U, S, V, Iall, Jall, I, J, I0, J0, rin, iter_idx, dtype, index_selection_method
        )

        resid = compute_residual([U, Uold], [-torch.diag(S), torch.diag(Sold)], [V, Vold])
        Uold = U; Sold = S; Vold = V

        eta1 = 1.0 / torch.linalg.norm(U[I, :], 'fro')
        eta2 = 1.0 / torch.linalg.norm(V[J, :], 'fro')

        solver_data[iter_idx, 0] = iter_idx
        solver_data[iter_idx, 1] = resid
        solver_data[iter_idx, 2] = U.shape[1]
        solver_data[iter_idx, 3] = Ilen
        solver_data[iter_idx, 4] = Jlen
        solver_data[iter_idx, 5] = len(I)
        solver_data[iter_idx, 6] = len(J)
        solver_data[iter_idx, 7] = float('nan')

        if resid < tolRES and min(eta1 * (1 + eta2), eta2 * (1 + eta1)) * S[-1] < tolLR:
            break

    # energy-based rank estimate
    energy = torch.cumsum(S.flip(0) ** 2, dim=0)
    r_st = energy.shape[0] - torch.sum(energy < tolE ** 2)
    r_st = max(min(int(r_st), rmax), 1)

    return U[:, :r_st], S[:r_st], V[:, :r_st], solver_data, iter_idx


@torch.compile(dynamic=True)
def xdeim_step(
    gfun: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    U: torch.Tensor,
    S: torch.Tensor,
    V: torch.Tensor,
    Iall: torch.Tensor,
    Jall: torch.Tensor,
    I: torch.Tensor,
    J: torch.Tensor,
    I0: torch.Tensor,
    J0: torch.Tensor,
    rin: int,
    iter_idx: int,
    dtype,
    index_selection_method: str,
):
    """Single iteration step of cross-DEIM.

    Builds/updates sample sets, calls scross and filters dependent indices.
    """
    II = torch.cat((I, I0))
    I = unique_stable(II)
    JJ = torch.cat((J, J0))
    J = unique_stable(JJ)

    # When using DEIM ensure at least one extra index is sampled to avoid
    # empty sample sets in early iterations.
    if (index_selection_method in ("qdeim", "deim") and len(I0) == len(I)) or iter_idx == 0:
        Iadd = Iall[~torch.isin(Iall, I)]
        if len(Iadd) > 0:
            Iadd_perm = torch.randperm(len(Iadd), device=U.device)
            Iadd_idx = Iadd_perm[:min(1, len(Iadd))]
            I = torch.cat([I, Iadd[Iadd_idx]])

    if (index_selection_method in ("qdeim", "deim") and len(J0) == len(J)) or iter_idx == 0:
        Jadd = Jall[~torch.isin(Jall, J)]
        if len(Jadd) > 0:
            Jadd_perm = torch.randperm(len(Jadd), device=V.device)
            Jadd_idx = Jadd_perm[:min(1, len(Jadd))]
            J = torch.cat([J, Jadd[Jadd_idx]])

    U, S, V, RC, RR = scross(gfun, I, J, Iall, Jall, dtype)
    I0 = I.clone()
    J0 = J.clone()

    if len(I0) > 0:
        diag_RR = torch.diag(RR)
        Idep0 = torch.where(torch.abs(diag_RR) < 1e-14 * torch.max(torch.abs(diag_RR)))[0]
        mask = torch.ones(I0.shape[0], dtype=torch.bool)
        mask[Idep0] = False
        I0 = I0[mask]

    if len(J0) > 0:
        diag_RC = torch.diag(RC)
        Jdep0 = torch.where(torch.abs(diag_RC) < 1e-14 * torch.max(torch.abs(diag_RC)))[0]
        mask = torch.ones(J0.shape[0], dtype=torch.bool)
        mask[Jdep0] = False
        J0 = J0[mask]

    return U, S, V, I, J, I0, J0


@torch.compile(dynamic=True)
def unique_stable(arr: torch.Tensor) -> torch.Tensor:
    """Return unique entries preserving the order of first occurrence.

    This implementation relies on scatter_reduce with amin to find first
    indices of unique values and then selects those entries from the
    original array.
    """
    unique_vals, idx = torch.unique(arr, return_inverse=True)
    perm = torch.arange(arr.size(0), device=arr.device)

    first_indices = torch.full(
        (unique_vals.size(0),),
        arr.size(0),
        device=arr.device
    )

    first_indices = first_indices.scatter_reduce(
        0, idx, perm, reduce='amin'
    )

    return arr[first_indices]
