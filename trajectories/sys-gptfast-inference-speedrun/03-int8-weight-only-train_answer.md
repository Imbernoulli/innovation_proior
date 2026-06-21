Compile plus the static KV-cache reached 104.9 tok/s at 1397 GB/s — about 70% of the A100's ~2 TB/s peak. That second number is the one that matters: the problem has fundamentally changed. The GPU is no longer idle; it is streaming weights from HBM at a serious fraction of peak, and tokens/s now tracks bandwidth almost linearly. There is maybe $1.4\times$ of headroom left to the ceiling, but even a perfect schedule tops out near 148 tok/s and stops, because to generate one token the run reads all 13.5 GB of bf16 weights once, and at 2 TB/s that read is ~6.75 ms no matter how cleverly it is scheduled. Scheduling is spent; the wall *is* the wall.

So the lever has to move from *how fast the bytes arrive* to *how many bytes there are*. In this regime throughput is essentially $\text{bandwidth}/\text{bytes-per-token}$, and bytes-per-token is dominated by the weight read, so halving the weight footprint should roughly double tokens/s — for free on the bandwidth account, since the matmuls were never the limiter and have arithmetic to spare. The weights are stored in bf16, 2 bytes each, and I propose to store them in **int8** instead: a clean halving, and a representation the hardware and compiler both handle well. The constraint is quality — this is inference on frozen weights, so I am not allowed to make the model dumber to make it faster — and the design that makes int8 safe is **symmetric, per-output-channel (per-row) quantization**.

The argument is about granularity. Within a single output channel — one row of the weight matrix, mapping the whole input to one output feature — the entries live in a fairly narrow range. So I find, *per row*, the maximum absolute value, pick a scale that maps that row's abs-max onto the int8 range $[-128, 127]$, round each weight to the nearest integer, and store the int8 weights plus one fp scale per row. At compute time the int8 weights are cast back up to the activation dtype and multiplied, then the result is multiplied by the per-row scale to undo the normalization. The quantization is *symmetric* — zero stays zero, no zero-point — which keeps the dequant a single multiply. The reconstruction error is bounded by half a quantization step per row, and because the scale is chosen per output channel rather than once for the whole matrix, a row with small weights gets a small step and a row with large weights gets a large one, so no row is crushed; the surviving rounding noise is tiny relative to the activations and averages out across the 4096-wide dot products. Per-channel granularity is the load-bearing choice: a single per-tensor scale would have to accommodate the largest-magnitude row across the whole matrix and would quantize the small rows to nearly nothing, which *would* hurt — per-row, it does not.

The crucial subtlety is in the forward: the dequant multiply by `scales` happens *after* `F.linear`, and the int8 weights are upcast to bf16 just before the matmul, so this is not int8×int8 tensor-core arithmetic — the matmul stays in bf16. The win is purely the memory *read*: the weights live in HBM as 1 byte each and get streamed at half the byte count, then are widened inside the kernel for the multiply. That is exactly right for a bandwidth-bound problem — I do not need int8 *arithmetic*, I need int8 *storage*, because storage is what crosses the bottleneck, and the compiler can fuse the upcast and the scale into the matmul epilogue so the extra ops are nearly free. The quantizer runs once, offline, over every `nn.Linear` weight (the QKV projection, the output projection, the three feed-forward matmuls, the LM head — where the 13.5 GB lives), leaving the small, numerically sensitive embeddings and norms alone.

I expect tokens/s to climb meaningfully — around $1.5\times$, into the 150s — but not a clean $2\times$, and the way to read it is instructive: the *achieved* GB/s will actually *drop*, not stay pinned at peak, because reading half-size weights is a less bandwidth-efficient access pattern. Tokens/s up because bytes/token down, even as GB/s achieved goes down, is the signature of a genuinely bandwidth-bound win. The guardrail is the EleutherAI harness on hellaswag/winogrande: per-row symmetric int8 should show no observable degradation, and if it did, the per-channel argument would be wrong and I would back it out — but I expect no observable quality loss.

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
