import torch

@torch.compile
def dst(x, type=2, norm=None, dim=-1):
    if type == 2:
        return _dst_type2(x, norm=norm, dim=dim)
    elif type == 1:
        return _dst_type1(x, norm=norm, dim=dim)
    else:
        raise NotImplementedError("Only DST-I and DST-II are implemented")


def _dst_type1(x, norm=None, dim=-1):
    N = x.shape[dim]
    if N < 2:
        raise ValueError("DST-I requires input length >= 2")

    # Create [0, x, 0, -flip(x)]
    zeros_shape = list(x.shape)
    zeros_shape[dim] = 1
    zeros = torch.zeros(zeros_shape, dtype=x.dtype, device=x.device)

    x_flip = torch.flip(x, dims=[dim])

    x_ext = torch.cat((zeros, x, zeros, -x_flip), dim=dim)

    # rfft again
    X = torch.fft.rfft(x_ext, dim=dim)

    result = -X.imag.narrow(dim, 1, N)

    if norm == "ortho":
        result *= (2 / (N + 1)) ** 0.5
    else:
        result *= 2

    return result

def _dst_type2(x, norm=None, dim=-1, orthogonalize=None):
    N = x.shape[dim]

    # Determine orthogonalize default based on norm (matching scipy behavior)
    if orthogonalize is None:
        orthogonalize = (norm == "ortho")

    # Build odd extension: [x, -flip(x)]
    x_flip = torch.flip(x, dims=[dim])
    x_ext = torch.cat((x, -x_flip), dim=dim)

    # Use rfft (faster than fft)
    X = torch.fft.rfft(x_ext, dim=dim)

    # Take imaginary part of first N entries
    result = -X.imag.narrow(dim, 1, N)

    # Apply normalization
    if norm == "ortho":
        result *= (2 / N) ** 0.5
    elif norm == "forward":
        result *= (2 / N)
    else:  # backward (default)
        result *= 2

    # Apply orthogonalization: divide y[-1] by sqrt(2) for orthonormal matrix
    if orthogonalize:
        result.narrow(dim, N-1, 1).div_(2 ** 0.5)

    return result

if __name__ == '__main__':
    
    print(dst(0.5*torch.eye(8), type=1))
    import numpy as np
    import scipy as sp
    print(sp.fft.dst(np.eye(8), type=1))

