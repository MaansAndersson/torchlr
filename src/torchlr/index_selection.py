"""
Index selection utilities used by cross-DEIM routines.

Provides DEIM-style and leverage-score selection.  A JAX-based QR-DEIM
implementation is present but disabled under torch compilation to avoid
requiring a JAX build at runtime.
"""

from typing import Tuple

import torch
import jax


# Torch-based DEIM: approximate pivot indices using LU pivoting
@torch.compile(dynamic=True)
def deim(U: torch.Tensor, I: torch.Tensor) -> torch.Tensor:
    """Select row indices for DEIM using LU pivoting on U.

    Parameters
    ----------
    U : (n, m) tensor
    I : unused but kept for compatibility with original API
    """
    n, m = U.shape[0], U.shape[1]
    P, _, _ = torch.linalg.lu(U, pivot=True)
    S = P.T @ I
    return S[0:m].long()


# JAX-based Q-DEIM (disabled for torch compilation)
@torch.compiler.disable
def qdeim(U: torch.Tensor) -> torch.Tensor:
    U = U.contiguous()

    n, m = U.shape[0], U.shape[1]
    # Convert to JAX array via DLPack and perform pivoted QR
    U_jax = jax.dlpack.from_dlpack(U.T, copy=False)
    _, _, P_jax = jax.lax.linalg.qr(U_jax, pivoting=True, use_magma=False, full_matrices=False)
    P = torch.from_dlpack(P_jax, copy=False)

    S = P[0:m]
    return S


@torch.compile(dynamic=True)
def leverage_score(U: torch.Tensor, p: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Sample indices by a simple leverage-score heuristic.

    Currently the function computes row-wise squared norms and returns the
    set of indices where the squared norm exceeds random thresholds p. The
    top entry is always included to ensure at least m samples are returned.

    The tolerance differs from the original implementation - it is very lax.
    """
    n, m = U.shape[0], U.shape[1]
    r = m

    probs = (U * U).sum(dim=1)
    _, i = torch.max(probs, dim=0)
    mask = (probs >= p)
    mask[i] = True
    P = torch.where(mask)[0]
    S = P[0:m]
    return S
