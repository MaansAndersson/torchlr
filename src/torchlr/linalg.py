"""
Small wrapper utilities for commonly used linear-algebra operations.
These functions intentionally present a stable signature for the rest of
this package and centralise potential backend changes.
"""

from typing import Tuple

import torch


@torch.compile
def qr_(A: torch.Tensor, pivoting: bool = False, full_matrices: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute an economy QR decomposition.

    Currently this just calls torch.linalg.qr with mode='reduced'. The
    returned permutation array P is a placeholder kept for API
    compatibility with potential pivoted decompositions.
    """
    Q, R = torch.linalg.qr(A, mode="reduced")
    P = torch.arange(A.shape[0], dtype=torch.int32, device=A.device)
    return Q, R, P
