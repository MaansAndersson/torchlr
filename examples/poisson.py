import argparse
import csv

import time 
import torch
import matplotlib.pyplot as plt
from collections import namedtuple
from torchlr.crossDEIM import crossDEIM
from torchlr.torch_dst import dst

import torchlr.torch_config
import numpy as np

# ---------------------------------------------------------------------------
# Global device selection
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64

@torch.compile(fullgraph=True)
def grhside(I, J, U, S, V, hx, hy, lx, ly, device=None):
    """
    Args:
        I, J   : array-like of indices (int)
        U      : (m, k) tensor or numpy array
        S      : (k,) or (k,k) tensor or numpy array
        V      : (n, k) tensor or numpy array
        lx, ly : (m,) and (n,) tensors or numpy arrays
        device : torch.device, e.g. torch.device('cuda') or 'cpu'
    Returns:
        g      : (ni, nj) tensor
    """
    #if device is None:
    #    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Extract diagonal if S is 2D
    s = S.diag() if S.dim() == 2 else S.flatten()  # (k,)

    # Gather rows
    UI = U[I]           # (ni, k)
    VJ = V[J]           # (nj, k)

    VJ_scaled = VJ * s  # (nj, k)  -- broadcasts over k
    g_raw = UI @ VJ_scaled.T   # Matmul instead of dot (ni, nj)
    denom = lx[I].unsqueeze(1) + ly[J].unsqueeze(0)  # broadcast (ni, nj)

    return g_raw / denom


class PoissonProblem():

    def __init__(self, n, indexselection):
        n1 = n
        n2 = n
        self.n1 = n1
        self.n2 = n2
        
        x = torch.linspace(-1, 1, n1 + 2, device = DEVICE)
        y = torch.linspace(-1, 1, n2 + 2, device = DEVICE)
        x = x[1:-1]
        y = y[1:-1]
        
        self.hx = x[1] - x[0]
        self.hy = y[1] - y[0]
        
        j1 = torch.arange(1, n1 + 1, device = DEVICE)
        j2 = torch.arange(1, n2 + 1, device = DEVICE)
        lx = -2 * (1 - torch.cos(torch.pi * j1 / (n1 + 1)))
        ly = -2 * (1 - torch.cos(torch.pi * j2 / (n2 + 1)))
        self.lx = lx/self.hx**2
        self.ly = ly/self.hy**2
        
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        self.xx = xx 
        self.yy = yy

        U0 = torch.exp(-36*(x-0.1)**2).reshape(-1,1)
        V0 = torch.exp(-36*(y-0.55)**2).reshape(-1,1)
        S0 = torch.linalg.norm(U0)*torch.linalg.norm(V0).reshape(-1,1)
        U0 = U0/torch.linalg.norm(U0)
        V0 = V0/torch.linalg.norm(V0)
        
        self.UF = U0.clone()
        self.SF = S0.clone()
        self.VF = V0.clone()
        
        rmax = 100
        tol = 1e-6
        rin = 100
        rmax = 100
        max_iter_cross = 20
        options = namedtuple('opts',['tol','rmax','rin','max_iter','index_selection'])
        self.opts = options(tol,rmax,rin,max_iter_cross, indexselection)
        


    def solve(self):

        #torch.cuda.synchronize()
        t0 = time.perf_counter()
        UF = torch.sqrt(torch.tensor(0.5/(self.n1+1)))*dst(self.UF/2, dim=0, type=1)
        VF = torch.sqrt(torch.tensor(0.5/(self.n2+1)))*dst(self.VF/2, dim=0, type=1)
        
        tdst = time.perf_counter() - t0
        print("dst time: ", tdst)

        t0 = time.perf_counter()
        small_grhside = lambda i, j : grhside(i, j, UF, self.SF, VF, self.hx, self.hy, self.lx, self.ly, DEVICE)
        
        [U, S, V, solver_data, cross_iter] = crossDEIM(small_grhside, UF, self.SF, VF, self.opts)
        
        tcrossdeim = time.perf_counter() - t0
        print("cross time: ", tcrossdeim)
        
        t0 = time.perf_counter()
        U = torch.sqrt(torch.tensor(0.5/(self.n1+1)))*dst(U/2, dim=0, type=1)
        V = torch.sqrt(torch.tensor(0.5/(self.n2+1)))*dst(V/2, dim=0, type=1) 
        #torch.cuda.synchronize()
        t1 = time.perf_counter()

        tdst2 = time.perf_counter() - t0
        print("dst2 time: ", tdst2)
        
        # Rematerialise solution
        B = U@torch.diag(S)@V.T
        
        fig = plt.figure()
        ax1 = fig.add_subplot(projection='3d')
        ax1.plot_surface(self.xx.cpu().numpy(), self.yy.cpu().numpy(), B.cpu().numpy())
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('U')
        #ax1.set_title('')
        
        plt.savefig('Poisson.png')

        return tdst+tcrossdeim+tdst2, solver_data, cross_iter

if __name__ == '__main__':
    
    #torch.set_default_device('cuda')
    parser = argparse.ArgumentParser(
                    prog='low-rank advection diffusion solver',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('-n', '--problemsize')
    parser.add_argument('-is','--indexselection')
    parser.add_argument('-s','--seed')
    parser.add_argument('-log', '--logfile', choices=['timing','solver',None])
    #parser.add_argument('-w','--warmup')

    parser.set_defaults(problemsize = 1000,
                        indexselection = 'deim',
                        seed = 1,
                        logfile = None)
    args = parser.parse_args()

    # ---- warmup comment out if needed ------
    print("warmup: ")
    torch.manual_seed(int(args.seed))
    P = PoissonProblem(n = int(args.problemsize),
                indexselection = args.indexselection)

    _, _, _ = P.solve()

    # ----------- end of warmup --------------

    torch.manual_seed(int(args.seed))
    from torch import profiler # profile, ProfilerActivity, record_function
    
    print("\nrun: ")
    #with profiler.profile(activities=[profiler.ProfilerActivity.CUDA], acc_events=True) as prof: #, with_stack=True, profile_memory=True) as prof:
    total_time, solver_data, cross_iter = P.solve()

    print("total time: ",total_time)
    #print(prof.key_averages(group_by_stack_n=10).table(sort_by=str(DEVICE)+"_time_total", row_limit=10))


    solver_data = solver_data.cpu().numpy()
    solver_data = solver_data[~np.all(solver_data == 0, axis=1)]
    if args.logfile is not None:
        logfile = 'poisson_'+args.logfile+'_log.csv'

        with open(logfile,'a') as file:
            if args.logfile == "timing":
                headers = "problem size,time,index selection,device,corss-iter,final_rank,sresidual,seed"
                writer = csv.writer(file)
                data = [str(int(args.elements)**2),str(total_time),args.indexselection,str(DEVICE),cross_iter,int(solver_data[-1,2]),solver_data[-1,1],args.seed]
                writer.writerow(data)
            elif args.logfile == "solver":
                headers = "iter_idx, Error, Rank, Ilen, Jlen, LenI, LenJ, Time [s]"
                np.savetxt(file, solver_data, delimiter=",",fmt=['%d','%e','%d','%d','%d', '%d', '%d', '%e'], header=headers)

