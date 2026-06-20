**Problem (from baseline).** Compile + static KV-cache reaches 104.9 tok/s at 1397 GB/s — ~70% of the
A100's ~2 TB/s HBM peak. The GPU is now genuinely bandwidth-bound: one token reads all 13.5 GB of bf16
weights once, ~6.75 ms at peak, and even a perfect schedule tops out near 148 tok/s. Scheduling is spent;
the only lever left is fewer bytes per token.

**Key idea.** Store the linear weights in **int8** instead of bf16 — a clean halving of the weight
footprint — quantized **symmetrically, per output channel (per row)**. Each row gets its own fp scale
mapping its abs-max onto [−128, 127]; weights are rounded to int8 and stored with one scale per row. At
decode time the int8 weights are streamed (half the bytes), upcast inside the kernel, multiplied, and the
per-row scale applied in the epilogue.

**Why it works.** Throughput in this regime is ≈ bandwidth / bytes-per-token; halving the weight bytes
roughly doubles the token budget the same HBM can carry. The win is int8 *storage*, not int8 arithmetic —
storage is what crosses the bottleneck, so the matmul stays in bf16 and no tensor-core int8 path is
needed. Per-*channel* granularity is what makes it lossless: a single per-tensor scale would crush
small-magnitude rows, but a per-row scale keeps every row well-resolved, so the surviving rounding noise
(≤ half a step per row) is negligible against the activations and averages out over 4096-wide dot
products. Verified on the EleutherAI harness (hellaswag/winogrande) as **no observable quality loss**.

**Change / code.** Per-channel int8 quantizer + drop-in `WeightOnlyInt8Linear` + handler that swaps every
`nn.Linear`.

```python
def dynamically_quantize_per_channel(x, quant_min, quant_max, target_dtype):
    # symmetric, assumes axis == 0 (per output channel)
    eps = torch.finfo(torch.float32).eps
    min_val, max_val = torch.aminmax(x, dim=1)
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    max_val_pos = torch.max(-min_val_neg, max_val_pos)
    scales = max_val_pos / (float(quant_max - quant_min) / 2)
    scales = torch.clamp(scales, min=eps).to(x.dtype)
    zero_points = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=min_val_neg.device)
    x_div = x / scales.unsqueeze(-1)
    x_round = torch.round(x_div)
    x_zp = x_round + zero_points.unsqueeze(-1)
    quant = torch.clamp(x_zp, quant_min, quant_max).to(target_dtype)
    return quant, scales, zero_points

def replace_linear_weight_only_int8_per_channel(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, WeightOnlyInt8Linear(child.in_features, child.out_features))
        else:
            replace_linear_weight_only_int8_per_channel(child)

class WeightOnlyInt8QuantHandler:
    def __init__(self, mod):
        self.mod = mod
    @torch.no_grad()
    def create_quantized_state_dict(self):
        cur_state_dict = self.mod.state_dict()
        for fqn, mod in self.mod.named_modules():
            if isinstance(mod, torch.nn.Linear):
                int8_weight, scales, _ = dynamically_quantize_per_channel(
                    mod.weight.float(), -128, 127, torch.int8)
                cur_state_dict[f"{fqn}.weight"] = int8_weight
                cur_state_dict[f"{fqn}.scales"] = scales.to(mod.weight.dtype)
        return cur_state_dict
    def convert_for_runtime(self):
        replace_linear_weight_only_int8_per_channel(self.mod)
        return self.mod

class WeightOnlyInt8Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", torch.empty((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scales", torch.ones(out_features, dtype=torch.bfloat16))
    def forward(self, input):
        return F.linear(input, self.weight.to(dtype=input.dtype)) * self.scales

# build offline: python quantize.py --checkpoint_path .../model.pth --mode int8
# run: pass model_int8.pth to generate.py --compile
```
