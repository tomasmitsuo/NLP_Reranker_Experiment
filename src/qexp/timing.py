import time
import torch

def qexp_timed(function, *args, **kwargs):
    if hasattr(function, "ensure_loaded"):
        function.ensure_loaded()
    # Sincronização evita medir apenas o despacho assíncrono de operações CUDA.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = function(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, time.perf_counter() - start
