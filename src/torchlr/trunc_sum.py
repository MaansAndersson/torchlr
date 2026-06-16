"""
Truncate a sum of low-rank matrices by performing an SVD in the compressed
basis.

Given lists Ui, Si, Vi corresponding to sum_i Ui Si Vi^T the routine
forms concatenated matrices, orthogonalises the bases with QR and computes
an SVD of the compressed core to obtain a truncated low-rank factorisation.
"""

from typing import List, Tuple

import torch
from .linalg import qr_


@torch.compile
def trunc_sum(
    Ui: List[torch.Tensor],
    Si: List[torch.Tensor],
    Vi: List[torch.Tensor],
    tol: float,
    rmax: int,
    soft: int,
    dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Form a truncated low-rank approximation of a sum of factors.

    Parameters
    ----------
    Ui, Si, Vi
        Lists of factor matrices describing the sum of low-rank terms.
    tol
        Tolerance used for energy-based truncation when soft == 0.
    rmax
        Maximum allowed rank for truncation.
    soft
        If soft == 0 perform energy-based hard truncation, otherwise
        behaviour may be extended in the future.
    dtype
        Target dtype (unused currently but kept for API compatibility).

    Returns
    -------
    U, S, V
        Truncated factors such that the approximation is U @ diag(S) @ V.T
    """
    nmat = len(Ui)
    r = torch.zeros(nmat + 1, dtype=torch.int64, device=Ui[0].device)
    n1 = Ui[0].shape[0]
    n2 = Vi[0].shape[0]

    for i in range(nmat):
        _, nn = Vi[i].shape
        r[i + 1] = nn

    rtot = int(torch.sum(r).item())

    # Concatenate into large matrices
    bigU = torch.zeros((n1, rtot), dtype=Ui[0].dtype, device=Ui[0].device)
    bigV = torch.zeros((n2, rtot), dtype=Vi[0].dtype, device=Vi[0].device)
    bigS = torch.zeros((rtot, rtot), dtype=Ui[0].dtype, device=Ui[0].device)

    rc = torch.cumsum(r, dim=0).long()

    for i in range(nmat):
        bigU[:, rc[i]:rc[i + 1]] = Ui[i]
        bigV[:, rc[i]:rc[i + 1]] = Vi[i]
        bigS[rc[i]:rc[i + 1], rc[i]:rc[i + 1]] = Si[i]

    # QR to orthogonalise columns
    QU, RU, _ = qr_(bigU, pivoting=False, full_matrices=False)
    QV, RV, _ = qr_(bigV, pivoting=False, full_matrices=False)

    # Compressed core for SVD
    TEMP = RU @ bigS @ RV.T

    Ust, Sst, Vstt = torch.linalg.svd(TEMP, full_matrices=False)
    sd = Sst
    Vst = Vstt.T

    if soft == 0:
        # Energy-based truncation: keep singular values until residual energy
        # falls below tol^2 (works with squared singular values)
        energy = torch.cumsum(sd.flip(0) ** 2, dim=0)
        r_st = len(energy) - int(torch.sum(energy < tol ** 2).item())
        r_st = max(min(r_st, rmax), 1)

        U = QU @ Ust[:, 0:r_st]
        V = QV @ Vst[:, 0:r_st]
        S = Sst[0:r_st]
    else:
        # Fallback: return full factorisation (could implement soft-thresholding)
        U = QU @ Ust
        V = QV @ Vst
        S = Sst

    return U, S, V
