""" 
Implement this based on the trunc_sum function
"""
import torch
from .linalg import qr_

@torch.compile
def compute_residual(Ui, Si, Vi):
    nmat = len(Ui)
    r = torch.zeros(nmat + 1, dtype=torch.int64, device=Ui[0].device)
    n1 = Ui[0].shape[0]
    n2 = Vi[0].shape[0]
    
    for i in range(nmat):
        _, nn = Vi[i].shape
        r[i + 1] = nn
    
    # sum of total rank
    rtot = sum(r);
    
    rtot = torch.sum(r).item()
    bigU = torch.zeros((n1, rtot), dtype=Ui[0].dtype, device=Ui[0].device)
    bigV = torch.zeros((n2, rtot), dtype=Vi[0].dtype, device=Vi[0].device)
    bigS = torch.zeros((rtot, rtot), dtype=Ui[0].dtype, device=Ui[0].device)
    
    rc = torch.cumsum(r, dim=0).long()
    
    for i in range(nmat):
        bigU[:, rc[i]:rc[i + 1]] = Ui[i]
        bigV[:, rc[i]:rc[i + 1]] = Vi[i]
        bigS[rc[i]:rc[i + 1], rc[i]:rc[i + 1]] = Si[i]
    
    _, RU, _ = qr_(bigU, pivoting=False, full_matrices=False);
    _, RV, _ = qr_(bigV, pivoting=False, full_matrices=False);

    rS = torch.norm(RU @ bigS @ RV.T,'fro');

    return rS
