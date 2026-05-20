import torch

from .linalg import qr_


@torch.compile(dynamic=True)
def qr_local(A):
    m, n = A.shape
    if m >= n:
        return qr_(A, pivoting = False, full_matrices = False)
    else:
        return qr_(A.T, pivoting = False, full_matrices = False)

@torch.compile(dynamic=True)
def scross(gfun, I, J, Iall, Jall, dtype):
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
               
        # Study behaviour here
        if torch.linalg.cond(Z[J, :]) > tol_cond:
            CU = torch.linalg.pinv(Z[J, :], rcond=tol_pinv) @ C.T
        else:
            CU, _, _, _ = torch.linalg.lstsq(Z[J, :], C.T, driver='gels')

        U, S, Vt = torch.linalg.svd(CU.T, full_matrices=False)
        V = (Z @ Vt.T)
    
    return U, S, V, RC, RR
