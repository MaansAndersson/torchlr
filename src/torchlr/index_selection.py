"""
deim
qdeim (requiers Jax)
leverage_score
"""

import torch
import jax


# Torch
@torch.compile(dynamic=True)
def deim(U, I):
    n, m = U.shape[0], U.shape[1]
    P, _, _ = torch.linalg.lu(U, pivot=True)
    S = P.T@I
    return S[0:m].long()

# Jax
@torch.compiler.disable
def qdeim(U):
    U = U.contiguous()

    n, m = U.shape[0], U.shape[1]
    U_jax = jax.dlpack.from_dlpack(U.T, copy=False)
    _, _, P_jax = jax.lax.linalg.qr(U_jax, pivoting=True, use_magma=False, full_matrices=False)
    P = torch.from_dlpack(P_jax, copy=False)

    S = P[0:m]
    return S

# Torch
@torch.compile(dynamic=True)
def leverage_score(U: torch.Tensor, p: torch.Tensor, eps = 1e-10):
    n, m = U.shape[0], U.shape[1]
    r = m
    #c = 0.5 # * torch.log(torch.tensor(r / eps, device=U.device, dtype=U.dtype))
    probs = (U*U).sum(dim=1) #torch.clamp(c * (U*U).sum(dim=1) / r, max=1.0)
    _, i = torch.max(probs, dim=0)
    mask  = (probs >= p)
    mask[i] = True
    P = torch.where(mask)[0] 
    S = P[0:m]
    return S
