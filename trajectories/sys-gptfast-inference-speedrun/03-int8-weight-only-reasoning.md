104.9 tokens/s, and the achieved bandwidth is 1397 GB/s — about 70% of the A100's ~2 TB/s. That second
number is the one I keep staring at, because it tells me the problem has fundamentally changed since the
eager baseline. Back then the GPU was idle and the win was deleting host overhead. Now the GPU is busy:
it is streaming weights from HBM at a serious fraction of peak, and the tokens/s is tracking the
bandwidth almost linearly. There is still maybe 1.4× of headroom to the bandwidth ceiling, but even if I
hit the ceiling perfectly I top out around 148 tok/s and stop. Scheduling tricks won't get me past that
wall — I've basically already paid for them. The wall *is* the wall: to generate one token I read all
13.5 GB of bf16 weights once, and at 2 TB/s that read takes ~6.75 ms no matter how cleverly I schedule
it.

So the lever has to move. I can't make the bytes arrive faster than HBM allows; I can only make there be
*fewer bytes*. Tokens/s in this regime is essentially `bandwidth / bytes_per_token`, and bytes_per_token
is dominated by the weight read. Halve the bytes of the weights and, to first order, I should roughly
double the tokens/s — for free on the bandwidth account, because the matmuls were never the limiter and
have arithmetic to spare. The weights are stored in bf16, 2 bytes each. The obvious move is to store them
in fewer bits.

int8 is the natural first stop: 1 byte instead of 2, a clean halving of the weight footprint, and 8-bit
integers are a representation the hardware and the compiler both handle well. The question is whether I
can drop a linear layer's weights to int8 *without observably changing the model's outputs*, because the
whole point of this rung is throughput at no quality cost — this isn't training, the weights are frozen,
and I am not allowed to make the model dumber to make it faster.

Here is why int8 weight-only is safe where a naive cast would not be. A weight matrix's entries, within
a single output channel (a single row, mapping the whole input to one output feature), live in a fairly
narrow numerical range. If I find, *per row*, the maximum absolute value, I can pick a scale that maps
that row's range onto the int8 range [−128, 127], round each weight to the nearest integer, and store
the int8 weights plus one fp scale per row. At compute time I do the matmul in int8 (or just cast the
int8 weights back up and multiply), then multiply the result by the per-row scale to undo the
normalization. The reconstruction error is bounded by half a quantization step *per row*, and because
the scale is chosen per output channel — not one global scale for the whole matrix — a row with small
weights gets a small step and a row with large weights gets a large one, so no row is crushed. The
rounding noise that survives is tiny relative to the activations it feeds into, and it averages out
across the 4096 input dimensions of each dot product. Per-*channel* (per-row) granularity is the design
choice that makes this lossless in practice: a single per-tensor scale would have to accommodate the
largest-magnitude row across the whole matrix and would quantize the small rows to almost nothing, which
*would* hurt. Per-row, it doesn't.

So I quantize symmetrically (zero stays zero — no zero-point, which keeps the dequant a single multiply)
and per output channel. The per-channel quantizer: take each row's absolute max, set the scale so that
max lands at the top of the int8 range, round, clamp:

```python
def dynamically_quantize_per_channel(x, quant_min, quant_max, target_dtype):
    # symmetric, per-row (axis 0)
    eps = torch.finfo(torch.float32).eps
    min_val, max_val = torch.aminmax(x, dim=1)
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    max_val_pos = torch.max(-min_val_neg, max_val_pos)
    scales = max_val_pos / (float(quant_max - quant_min) / 2)   # map |max| -> 127
    scales = torch.clamp(scales, min=eps).to(x.dtype)
    zero_points = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=x.device)
    x_div = x / scales.unsqueeze(-1)
    x_round = torch.round(x_div)
    quant = torch.clamp(x_round, quant_min, quant_max).to(target_dtype)
    return quant, scales, zero_points
```

That runs once, offline, over every `nn.Linear` weight to produce an int8 state dict plus its scales —
the QKV projection, the output projection, the three feed-forward matmuls, the LM head. These are the
big matrices; quantizing them is where the 13.5 GB lives. I leave embeddings and norms alone (small, and
numerically sensitive).

The runtime side is a drop-in linear that stores int8 weights and the per-row scales and applies the
scale after the matmul:

```python
class WeightOnlyInt8Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", torch.empty((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scales", torch.ones(out_features, dtype=torch.bfloat16))

    def forward(self, input):
        return F.linear(input, self.weight.to(dtype=input.dtype)) * self.scales
```

and a handler that swaps every `nn.Linear` for this and builds the quantized weights:

```python
class WeightOnlyInt8QuantHandler:
    def __init__(self, mod): self.mod = mod
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
```

One subtlety the forward reveals: the dequant multiply by `scales` happens *after* `F.linear`, and the
int8 weights are cast up to the activation dtype just before the matmul. So I'm not (in this simple
version) doing int8×int8 tensor-core math — the *win is purely the memory read*. The weights live in HBM
as 1 byte each, get streamed at half the byte count, and are widened to bf16 inside the kernel for the
multiply. That's exactly right for a bandwidth-bound problem: I don't need int8 arithmetic, I need int8
*storage*, because storage is what crosses the bottleneck. The compiler can still fuse the upcast and the
scale into the matmul epilogue so the extra ops are nearly free.

The prediction, against 104.9 tok/s at 1397 GB/s. Halving the weight bytes should let the same bandwidth
carry roughly twice the tokens, so I expect the throughput to climb meaningfully — call it 1.5×, into the
150s — but *not* a clean 2×, and the reason is instructive: the achieved bandwidth will actually *drop*,
not stay pinned at peak. Reading half-size weights is a less bandwidth-efficient access pattern (smaller
contiguous reads, the dequant overhead per byte), so I won't get the full ideal. The right way to read
the result is: tokens/s up because bytes/token down, even though GB/s achieved goes *down* — that's the
signature of a genuinely bandwidth-bound win. And the guardrail is the accuracy harness: per-row
symmetric int8 should show no observable degradation on hellaswag/winogrande, and if it does, the
per-channel argument was wrong and I'd back it out. I expect no observable quality loss. The change is
the per-channel int8 quantizer, the `WeightOnlyInt8Linear`, and the handler that swaps the model's
linears; full scaffold code in the answer.
