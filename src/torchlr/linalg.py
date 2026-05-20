"""
Linalg wrappers with the most common usage. 
Simplify interface or remove. 
"""

import torch
import numpy
import time

@torch.compile
def qr_(A, pivoting=False, full_matrices=False):
    Q, R = torch.linalg.qr(A, mode="reduced")
    P = torch.arange(A.shape[0], dtype=torch.int32)
    return Q, R, P
