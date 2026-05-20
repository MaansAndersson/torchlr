import torch
import time
from torchlr.scross import scross
from torchlr import index_selection #qdeim, deim, leverage_score
from torchlr.compute_residual import compute_residual

import torch.autograd.profiler as profiler

@torch.compile(dynamic=True)
def crossDEIM(gfun, U, S, V, opts=None, compute_residual_bool=None, dtype=None):
    """
    gfun 
    U 
    S 
    V

    """
    if opts is not None:
        tolLR = opts.tol
        tolRES = opts.tol
        tolE = opts.tol
        rmax = opts.rmax
        rin = opts.rin
        max_iter = opts.max_iter
        index_selection_method = opts.index_selection
    #else:
    #    tolLR = 1e-10
    #    tolRES = 1e-10
    #    tolE = 1e-10
    #    rmax = min(U.shape[0], V.shape[0])
    #    rin = 
    #    max_iter = 20
    #    index_selection_method = 'deim'

    #s = torch.diag(S) if S.ndim == 2 else S.flatten()   # (r,)
    
    n1 = U.shape[0]
    n2 = V.shape[0]

    Iall = torch.arange(n1, device=U.device, dtype=torch.long, requires_grad=False)
    Jall = torch.arange(n2, device=V.device, dtype=torch.long, requires_grad=False)
    
    I0 = torch.tensor([], dtype=torch.long, device=U.device)
    J0 = torch.tensor([], dtype=torch.long, device=V.device)
    
    solver_data = torch.zeros((max_iter, 8))

    if index_selection_method == 'ls':
        pu = torch.empty((n1,max_iter), device=U.device).uniform_()
        pv = torch.empty((n2,max_iter), device=V.device).uniform_()

    Uold = U; Vold = V; Sold = S
    for iter_idx in range(max_iter):
        
        t0 = 0 #time.perf_counter()
        if index_selection_method == 'qdeim': 
            I = index_selection.qdeim(U)
            J = index_selection.qdeim(V)
            deim_bool = True
        elif index_selection_method == 'deim':
            I = index_selection.deim(U, Iall)
            J = index_selection.deim(V, Jall)
            deim_bool = True
        elif index_selection_method == 'ls':
            I = index_selection.leverage_score(U, pu[:,iter_idx])
            J = index_selection.leverage_score(V, pv[:,iter_idx])
            deim_bool = False
        else:
            raise Exception("#### Warning no or inccorect index_selection chosen! #####")
        
        Ilen = len(I)
        Jlen = len(J)

        U, S, V, I, J, I0, J0 = xdeim_step(gfun, U, S, V, Iall, Jall, I, J, I0, J0, rin, iter_idx, dtype, index_selection)

        resid = compute_residual([U, Uold], [-torch.diag(S), torch.diag(Sold)], [V, Vold]) 
        Uold = U; Sold = S; Vold = V

        eta1 = 1.0 / torch.linalg.norm(U[I,:], 'fro')   
        eta2 = 1.0 / torch.linalg.norm(V[J,:], 'fro')   
        
        t1 = float('nan') #time.perf_counter()
        
        solver_data[iter_idx, 0] = iter_idx
        solver_data[iter_idx, 1] = resid
        solver_data[iter_idx, 2] = U.shape[1]
        solver_data[iter_idx, 3] = Ilen
        solver_data[iter_idx, 4] = Jlen 
        solver_data[iter_idx, 5] = len(I)
        solver_data[iter_idx, 6] = len(J) 
        solver_data[iter_idx, 7] = t1-t0
        
        if resid < tolRES and min(eta1 * (1 + eta2), eta2 * (1 + eta1)) * S[-1] < tolLR:
            break
   
    # TODO find native solution
    energy = torch.cumsum(S.flip(0) ** 2, dim=0)
    r_st = energy.shape[0] - torch.sum(energy < tolE ** 2)
    r_st = max(min(r_st, rmax), 1)
    
    #U = U[:, :r_st]
    #S = S[:r_st]
    #V = V[:, :r_st] 

    return U[:, :r_st], S[:r_st], V[:, :r_st], solver_data, iter_idx

@torch.compile(dynamic=True)
def xdeim_step(gfun, U, S, V, Iall, Jall, I, J, I0, J0, rin, iter_idx, dtype, deim_bool):
    
    II = torch.cat((I, I0))
    I = unique_stable(II) 
    JJ = torch.cat((J, J0))
    J = unique_stable(JJ)

    if deim_bool and len(I0) == len(I) or iter_idx == 0:
        Iadd = Iall[~torch.isin(Iall, I)]
        if len(Iadd) > 0:
            Iadd_perm = torch.randperm(len(Iadd), device=U.device)
            Iadd_idx = Iadd_perm[:min(1, len(Iadd))]
            I = torch.cat([I, Iadd[Iadd_idx]])
    
    if deim_bool and len(J0) == len(J) or iter_idx == 0:
        Jadd = Jall[~torch.isin(Jall, J)]
        if len(Jadd) > 0:
            Jadd_perm = torch.randperm(len(Jadd), device=V.device)
            Jadd_idx = Jadd_perm[:min(1, len(Jadd))]
            J = torch.cat([J, Jadd[Jadd_idx]])
    
    #if len(I) > rin:
    #   I = I[:rin]
    #if len(J) > rin:
    #   J = J[:rin]
    
    U, S, V, RC, RR = scross(gfun, I, J, Iall, Jall, dtype)
    I0 = I.clone()
    J0 = J.clone()

    if len(I0) > 0:
        diag_RR = torch.diag(RR)
        Idep0 = torch.where(torch.abs(diag_RR) < 1e-14 * torch.max(torch.abs(diag_RR)))[0]
        mask = torch.ones(I0.shape[0], dtype=torch.bool)
        mask[Idep0] = False
        I0 = I0[mask]
    
    if len(J0) > 0:
        diag_RC = torch.diag(RC)
        Jdep0 = torch.where(torch.abs(diag_RC) < 1e-14 * torch.max(torch.abs(diag_RC)))[0]
        mask = torch.ones(J0.shape[0], dtype=torch.bool)
        mask[Jdep0] = False
        J0 = J0[mask]

    return U, S, V, I, J, I0, J0    

@torch.compile(dynamic=True)
def unique_stable(arr):
    unique_vals, idx = torch.unique(arr, return_inverse=True)
    perm = torch.arange(arr.size(0), device=arr.device)

    first_indices = torch.full(
        (unique_vals.size(0),),
        arr.size(0),
        device=arr.device
    )

    first_indices = first_indices.scatter_reduce(
        0, idx, perm, reduce='amin'
    )

    return arr[first_indices]
