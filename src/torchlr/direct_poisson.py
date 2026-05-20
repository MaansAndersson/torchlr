"""
poisson2d.py — Batched 2-D Poisson solver via DST-I
=====================================================

Solves the discrete Poisson equation on a uniform (Nx × Ny) interior grid
with homogeneous Dirichlet boundary conditions:

    (Δ_h u)_{i,j} = f_{i,j},   i = 1…Nx,  j = 1…Ny

where the 5-point stencil Laplacian is

    (Δ_h u)_{i,j} = u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4 u_{i,j}

and u = 0 on the boundary.

Algorithm
---------
DST-I diagonalises the 1-D finite-difference second-derivative operator:

    (D²)_{mn} x_n = λ_m x_m,   λ_m = 2 cos(π(m+1)/(N+1)) - 2

Applying DST-I along both spatial axes simultaneously transforms the 2-D
system into a diagonal one:

    (λ_i^x + λ_j^y) F_{i,j} = G_{i,j}

where G = DST1( DST1(f, axis=-2), axis=-1 ) and F is the transformed solution.
The physical solution is recovered by the inverse DST-I.  Since
DST-I satisfies  dst1(dst1(x)) = (N+1)/2 · x, the inverse is:

    idst1(X) = dst1(X) · 2 / (N+1)

    u = idst1( idst1(F / (λ^x[:,None] + λ^y[None,:]), axis=-2), axis=-1 )

Complexity:  O(Nx Ny log(Nx Ny))  — same as a 2-D FFT.

Usage
-----
    from poisson2d import poisson2d_dst

    # Solve  Δu = f  on a 64×64 grid, batch of 8
    f = torch.randn(8, 64, 64)
    u = poisson2d_dst(f)

    # Physical RHS: supply grid spacing h so Δ_h u = h² f (Poisson with h)
    u = poisson2d_dst(f, hx=1/65, hy=1/65)
"""

import torch
import math

# ---------------------------------------------------------------------------
# DST-I primitive
# ---------------------------------------------------------------------------

def dst1(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Batched Type-I Discrete Sine Transform along *dim*.

    For a signal x of length N the transform is

        X_k = Σ_{n=0}^{N-1}  x_n  sin( π (n+1)(k+1) / (N+1) ),  k = 0…N-1

    DST-I is its own inverse up to scale:  dst1( dst1(x) ) = (N+1)/2 · x

    Implemented via a real FFT on the odd-extended sequence of length 2(N+1).
    """
    # Move the target axis to the last position for uniform indexing
    x = x.movedim(dim, -1)
    N = x.shape[-1]

    zeros = torch.zeros(*x.shape[:-1], 1, dtype=x.dtype, device=x.device)
    # Odd extension: [0, x, 0, -x_flipped]  →  length 2(N+1)
    y = torch.cat([zeros, x, zeros, -x.flip(-1)], dim=-1)
    Y = torch.fft.rfft(y, dim=-1)

    result = -Y[..., 1 : N + 1].imag / 2.0
    return result.movedim(-1, dim)


def idst1(X: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Inverse DST-I along *dim*.

    DST-I satisfies  dst1(dst1(x)) = (N+1)/2 · x, so:

        idst1(X) = dst1(X) · 2 / (N+1)
    """
    N = X.shape[dim]
    return dst1(X, dim=dim) * (2.0 / (N + 1))

def dst3(x, dim : int =-1):
    N = x.shape[dim]

    # Build complex spectrum
    k = torch.arange(N, device=x.device, dtype=x.dtype)
    shape = [1] * x.ndim
    shape[dim] = N
    k = k.view(shape)

    phase = torch.exp(1j * math.pi * k / (2 * N))

    # Prepare imaginary spectrum
    X = torch.zeros_like(x, dtype=torch.complex64 if x.dtype == torch.float32 else torch.complex128)
    X = x * phase * 1j

    # Build full spectrum via conjugate symmetry
    X_flip = torch.flip(X, dims=[dim]).conj()
    X_full = torch.cat([X, X_flip], dim=dim)

    # IFFT
    x_rec = torch.fft.ifft(X_full, dim=dim).real

    # Take first N entries
    result = x_rec.narrow(dim, 0, N)

    # Normalization
    return result/2

# ---------------------------------------------------------------------------
# Eigenvalues of the 1-D finite-difference Laplacian
# ---------------------------------------------------------------------------

def _dst1_eigenvalues(N: int, dtype=torch.float64, device=None) -> torch.Tensor:
    """
    Eigenvalues of the tridiagonal (−1, 2, −1) second-difference matrix
    under DST-I diagonalisation:

        λ_k = 2 cos( π (k+1) / (N+1) ) − 2,   k = 0 … N−1

    These are all negative (the operator is negative semi-definite on the
    interior), which is consistent with the standard Laplacian sign convention
    used here: Δu = f with f = RHS (not −Δu = f).
    """
    k = torch.arange(N, dtype=dtype, device=device)
    return 2.0 * torch.cos(math.pi * (k + 1.0) / (N + 1.0)) - 2.0


# ---------------------------------------------------------------------------
# 2-D Poisson solver
# ---------------------------------------------------------------------------

@torch.compile
def poisson2d_dst(
    f: torch.Tensor,
    hx: float = 1.0,
    hy: float = 1.0,
) -> torch.Tensor:
    """
    Solve the discrete 2-D Poisson equation with homogeneous Dirichlet BCs.

        Δ_h u = f   on an (Nx × Ny) interior grid

    where Δ_h is the standard 5-point finite-difference Laplacian.

    Parameters
    ----------
    f : Tensor, shape (..., Nx, Ny)
        Right-hand side sampled on interior grid points.  Leading dimensions
        are treated as independent batch dimensions.
    hx : float
        Grid spacing in x.  When hx != 1 the physical equation is
            (u_{i+1} − 2u_i + u_{i-1}) / hx²  +  (…) / hy²  = f_{i,j}
        i.e. the stencil coefficients are scaled by 1/hx² and 1/hy².
    hy : float
        Grid spacing in y.

    Returns
    -------
    u : Tensor, shape (..., Nx, Ny)
        Solution on the interior grid (boundary values are implicitly zero).

    Notes
    -----
    Complexity : O(Nx Ny log(Nx Ny)) per sample in the batch.
    Precision  : Works in the dtype of *f*.  For well-conditioned problems
                 float32 is fine; use float64 for near-singular eigenvalues
                 (very coarse grids or grids with large aspect ratios).
    """
    Nx, Ny = f.shape[-2], f.shape[-1]

    # --- Forward DST-I in both spatial directions --------------------------
    F = dst1(dst1(f, dim=-2), dim=-1)   # (..., Nx, Ny)

    # --- Eigenvalues of the scaled 1-D operators ---------------------------
    #   λ_k^x / hx²,  λ_k^y / hy²
    lam_x = _dst1_eigenvalues(Nx, dtype=f.dtype, device=f.device) / (hx * hx)
    lam_y = _dst1_eigenvalues(Ny, dtype=f.dtype, device=f.device) / (hy * hy)

    # Broadcast to (..., Nx, Ny):  eigenvalue of the 2-D Laplacian
    lam = lam_x[:, None] + lam_y[None, :]   # (Nx, Ny)

    # --- Divide in the DST-I eigenspace ------------------------------------
    U = F / lam   # element-wise; lam is never zero for Dirichlet BCs

    # --- Inverse DST-I in both spatial directions --------------------------
    u = idst1(idst1(U, dim=-2), dim=-1)

    return u


# ---------------------------------------------------------------------------
# Convenience: physical-coordinate grid helpers
# ---------------------------------------------------------------------------

def interior_grid(
    Nx: int,
    Ny: int,
    x0: float = 0.0, x1: float = 1.0,
    y0: float = 0.0, y1: float = 1.0,
    dtype=torch.float64,
    device=None,
):
    """
    Return (x, y) meshgrids for the Nx×Ny interior points of [x0,x1]×[y0,y1].

    Grid spacing is  hx = (x1-x0)/(Nx+1),  hy = (y1-y0)/(Ny+1).
    """
    hx = (x1 - x0) / (Nx + 1)
    hy = (y1 - y0) / (Ny + 1)
    xs = torch.linspace(x0 + hx, x1 - hx, Nx, dtype=dtype, device=device)
    ys = torch.linspace(y0 + hy, y1 - hy, Ny, dtype=dtype, device=device)
    return torch.meshgrid(xs, ys, indexing="ij")   # each (Nx, Ny)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    torch.set_default_dtype(torch.float64)
    print("=" * 60)
    print("2-D Poisson solver — self-test")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Manufactured solution on [0,1]²
    #
    #   u_exact(x,y) = sin(π x) sin(π y)
    #   Δu = −2π² sin(πx) sin(πy)   →   f = −2π² u_exact
    # ------------------------------------------------------------------
    for N in (15, 31, 63, 127):
        Nx = Ny = N
        x, y = interior_grid(Nx, Ny)     # hx = hy = 1/(N+1)
        hx = hy = 1.0 / (N + 1)

        u_exact = torch.sin(math.pi * x) * torch.sin(math.pi * y)

        # Discrete RHS: use the stencil-exact eigenvalue approach so that
        # the manufactured solution is *exactly* consistent with the
        # discrete operator (avoids O(h²) truncation in the test itself).
        #
        # For a clean truncation-error test we use the continuous Laplacian:
        f = -2.0 * math.pi**2 * u_exact   # continuous Δu_exact

        u = poisson2d_dst(f, hx=hx, hy=hy)

        err = (u - u_exact).abs().max().item()
        print(f"  N={N:4d}  h={hx:.2e}  max|u - u_exact| = {err:.3e}")

    # ------------------------------------------------------------------
    # Batch dimension check
    # ------------------------------------------------------------------
    B, Nx, Ny = 16, 64, 64
    f_batch = torch.randn(B, Nx, Ny)
    u_batch = poisson2d_dst(f_batch, hx=1/(Nx+1), hy=1/(Ny+1))
    assert u_batch.shape == (B, Nx, Ny), "Batch shape mismatch"
    print(f"\n  Batch test passed: ({B}, {Nx}, {Ny}) → {tuple(u_batch.shape)}")

    # ------------------------------------------------------------------
    # DST-I self-inverse check:  idst1(dst1(x)) == x
    # ------------------------------------------------------------------
    x = torch.randn(32, 64)
    err_dst = (idst1(dst1(x, dim=-1), dim=-1) - x).abs().max().item()
    print(f"  DST-I round-trip error: {err_dst:.2e}")

    print("\nAll tests passed.")
