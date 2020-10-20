from functools import partial
import numpy as np

import torch


def benchmark(fn, nrepeats=7):
    res = []
    for _ in range(nrepeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        res.append(start.elapsed_time(end))
    return np.median(res), np.std(res)


def benchmark_fw_bw(fn, inputs, nrepeats=7):
    fw = partial(fn, *inputs)
    out = fw()
    g = torch.randn_like(out).contiguous()
    # bw = lambda: out.backward(g, retain_graph=True)
    bw = lambda: torch.autograd.grad(out, inputs, g, retain_graph=True)
    # fw_bw = lambda: fw().backward(g)
    fw_bw = lambda: torch.autograd.grad(fw(), inputs, g)
    return benchmark(fw, nrepeats), benchmark(bw, nrepeats), benchmark(fw_bw, nrepeats)
