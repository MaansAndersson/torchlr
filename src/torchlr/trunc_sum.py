import torch
from .linalg import qr_

@torch.compile
def trunc_sum(Ui, Si, Vi, tol, rmax, soft, dtype):
    nmat = len(Ui)
    r = torch.zeros(nmat + 1, dtype=torch.int64, device=Ui[0].device)
    n1 = Ui[0].shape[0]
    n2 = Vi[0].shape[0]
    
    for i in range(nmat):
        _, nn = Vi[i].shape
        r[i + 1] = nn
    
    rtot = torch.sum(r).item()
    bigU = torch.zeros((n1, rtot), dtype=Ui[0].dtype, device=Ui[0].device)
    bigV = torch.zeros((n2, rtot), dtype=Vi[0].dtype, device=Vi[0].device)
    bigS = torch.zeros((rtot, rtot), dtype=Ui[0].dtype, device=Ui[0].device)
    
    rc = torch.cumsum(r, dim=0).long()
    
    for i in range(nmat):
        bigU[:, rc[i]:rc[i + 1]] = Ui[i]
        bigV[:, rc[i]:rc[i + 1]] = Vi[i]
        bigS[rc[i]:rc[i + 1], rc[i]:rc[i + 1]] = Si[i]
    
        QU, RU, _ = qr_(bigU, pivoting=False, full_matrices=False)
        QV, RV, _ = qr_(bigV, pivoting=False, full_matrices=False)
            
    TEMP = RU  @ bigS @ RV.T

    Ust, Sst, Vstt = torch.linalg.svd(TEMP, full_matrices=False)
    sd = Sst
    Vst = Vstt.T
    
    if soft == 0:
        energy = torch.cumsum(sd.flip(0) ** 2, dim=0)
        r_st = len(energy) - torch.sum(energy < tol ** 2).item()
        r_st = max(min(r_st, rmax), 1)
        U = QU @ Ust[:, 0:r_st]
        V = (QV @ Vst[:, 0:r_st])
        S = Sst[0:r_st]
    
    return U, S, V
