"""
PyTorch config.
"""
import torch
torch.set_default_dtype(torch.float64)

torch._dynamo.config.cache_size_limit = 128
torch._dynamo.config.accumulated_cache_size_limit = 1024 #512
torch._dynamo.config.capture_scalar_outputs = True
torch.set_grad_enabled(False)

#torch.compiler.set_stance = "fail_on_recompile"

#torch.backends.cuda.preferred_linalg_library(backend="cusolver")
#torch.set_default_device('cuda')

