
import torch
from torch.autograd import Function

with open("pairwise_distances.metal") as f:
    _library = torch.mps.compile_shader(f.read())

class PairwiseDistances(Function):
    @staticmethod
    def forward(ctx, x):
        # len(x) choose 2
        num_pairs = len(x) * (len(x) - 1) // 2

        out = torch.zeros(num_pairs, device="mps")
        _library.pairwise_distances_forward(out,     # device float* out
                                            x,       # device float2* points
                                            len(x))  # constant uint& num_points

        ctx.save_for_backward(x, out)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, forward_out = ctx.saved_tensors
        grad_output = grad_output.contiguous()

        num_pairs = len(x) * (len(x) - 1) // 2

        out = torch.zeros_like(x)
        _library.pairwise_distances_backward(out,          # device float2* out
                                             len(out),     # constant uint& num_points
                                             num_pairs,    # constant uint& num_pairs
                                             x,            # device float2* points
                                             forward_out,  # device float* distances
                                             grad_output)  # device float* grad_output

        return out

mps_pairwise_distances = PairwiseDistances.apply
