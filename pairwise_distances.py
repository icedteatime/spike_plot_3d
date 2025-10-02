
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
                                            len(x),  # constant uint& num_points
                                            num_pairs)  # constant uint& num_pairs

        ctx.save_for_backward(x, out)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, distances = ctx.saved_tensors

        grad_div_distances = grad_output / distances

        num_pairs = len(x) * (len(x) - 1) // 2

        out = torch.zeros_like(x)
        _library.pairwise_distances_backward(out,          # device float2* out
                                             x,            # device float2* points
                                             len(out),     # constant uint& num_points
                                             num_pairs,    # constant uint& num_pairs
                                             grad_div_distances)  # device float* grad_div_distances

        return out

mps_pairwise_distances = PairwiseDistances.apply
