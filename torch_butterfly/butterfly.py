import torch


@torch.jit.script
def butterfly_fw(twiddle: torch.Tensor, input: torch.Tensor,
                 increasing_stride: bool) -> torch.Tensor:
    return torch.ops.torch_butterfly.butterfly_multiply_fw(twiddle, input, increasing_stride)
