"""
Compute a Frobenius-norm residual for a sum of low-rank terms.

Given lists Ui, Si, Vi representing matrices U_i S_i V_i^T, this
function forms bigU, bigS, bigV that concatenate the factors along the
rank dimension, orthogonalises the tall factors via QR and computes the
Frobenius norm of RU @ bigS @ RV^T which is equal to the Frobenius norm
of the residual in the orthonormal basis.

The implementation is intentionally minimal and uses PyTorch tensors.
"""

from typing import List

import torch
from .linalg import qr_


@torch.compile
def compute_residual(Ui: List[torch.Tensor], Si: List[torch.Tensor], Vi: List[torch.Tensor]) -> torch.Tensor:
    """Compute a single scalar residual from lists of low-rank factors.

    Parameters
    ----------
    Ui, Si, Vi
        Lists of factors so that the full matrix is sum_i Ui[i] @ Si[i] @ Vi[i].T

    Returns
    -------
    rS : Tensor
        A scalar (0-dim) tensor containing the Frobenius-norm of the
        assembled residual in an orthonormal basis.
    """
    nmat = len(Ui)

    # r will hold cumulative ranks: r[0] = 0, r[i+1] = rank of i-th block
    r = torch.zeros(nmat + 1, dtype=torch.int64, device=Ui[0].device)
    n1 = Ui[0].shape[0]
    n2 = Vi[0].shape[0]

    for i in range(nmat):
        _, nn = Vi[i].shape
        r[i + 1] = nn

    # total rank
    rtot = int(torch.sum(r).item())

    bigU = torch.zeros((n1, rtot), dtype=Ui[0].dtype, device=Ui[0].device)
    bigV = torch.zeros((n2, rtot), dtype=Vi[0].dtype, device=Vi[0].device)
    bigS = torch.zeros((rtot, rtot), dtype=Ui[0].dtype, device=Ui[0].device)

    rc = torch.cumsum(r, dim=0).long()

    # Fill the big block matrices
    for i in range(nmat):
        bigU[:, rc[i]:rc[i + 1]] = Ui[i]
        bigV[:, rc[i]:rc[i + 1]] = Vi[i]
        bigS[rc[i]:rc[i + 1], rc[i]:rc[i + 1]] = Si[i]

    # Orthogonalise tall factors (economy QR) and compute residual norm
    _, RU, _ = qr_(bigU, pivoting=False, full_matrices=False)
    _, RV, _ = qr_(bigV, pivoting=False, full_matrices=False)

    # RU @ bigS @ RV.T has same Frobenius norm as the original sum
    rS = torch.norm(RU @ bigS @ RV.T, 'fro')

    return rS
