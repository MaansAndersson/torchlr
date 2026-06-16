"""
Skeleton cross approximation (skeletonization / cross) routines.

scross(gfun, I, J, Iall, Jall) computes a low-rank approximation of the
matrix sampled by gfun using selected rows I and columns J. The implementation
uses QR and small SVDs to build the factors.
"""

from typing import Tuple

import torch
from .linalg import qr_


@torch.compile(dynamic=True)
def qr_local(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return economy QR of A or A.T depending on shape.

    This helper ensures we QR the "tall" matrix for numerical stability.
    """
    m, n = A.shape
    if m >= n:
        return qr_(A, pivoting=False, full_matrices=False)
    else:
        return qr_(A.T, pivoting=False, full_matrices=False)


@torch.compile(dynamic=True)
def scross(gfun, I: torch.Tensor, J: torch.Tensor, Iall: torch.Tensor, Jall: torch.Tensor, dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cross-approximation step.

    Parameters
    ----------
    gfun : callable
        Function gfun(Iidx, Jidx) returning a matrix sampled at the given
        row/column indices. Expected to return a torch.Tensor.
    I, J : 1-D integer tensors
        Chosen row and column indices for the cross approximation.
    Iall, Jall : 1-D integer tensors
        Full index sets (used when building sampled matrices).
    dtype : torch dtype
        Kept for API compatibility.

    Returns
    -------
    U, S, V, RC, RR
        Low-rank factors and R factors from QR of the sampled column and row
        matrices (RC corresponds to columns, RR to rows).
    """
    tol_cond = 1e10
    tol_pinv = 1e-5

    C = gfun(Iall, J)
    R = gfun(I, Jall)

    if C.shape[1] <= R.shape[0]:
        Q, RC, _ = qr_local(C)
        _, RR, _ = qr_local(R.T)

        if torch.linalg.cond(Q[I, :]) > tol_cond:
            UR = torch.linalg.pinv(Q[I, :], rcond=tol_pinv) @ R
        else:
            UR, _, _, _ = torch.linalg.lstsq(Q[I, :], R, driver='gels')

        U, S, Vt = torch.linalg.svd(UR, full_matrices=False)
        V = Vt.T
        U = Q @ U
    else:
        _, RC, _ = qr_local(C)
        Z, RR, _ = qr_local(R.T)

        if torch.linalg.cond(Z[J, :]) > tol_cond:
            CU = torch.linalg.pinv(Z[J, :], rcond=tol_pinv) @ C.T
        else:
            CU, _, _, _ = torch.linalg.lstsq(Z[J, :], C.T, driver='gels')

        U, S, Vt = torch.linalg.svd(CU.T, full_matrices=False)
        V = Z @ Vt.T

    return U, S, V, RC, RR
