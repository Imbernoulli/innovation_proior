The schedule levers — dynamic batch, sequence-length growth — have squeezed the run down a lot, and I've
been leaving two parts of the *network itself* at their textbook defaults: the MLP nonlinearity and the
precision policy. Both are worth re-examining now, because at this point the per-step quality of the
network is what gates how few steps I need, and the per-step *cost* is dominated by the matmuls in those
MLPs.

Start with the MLP. The baseline is the standard one: expand the channels 4×, apply GELU pointwise,
project back. GELU is a smooth gate — it multiplies each pre-activation by a smooth approximation of "is
this value positive." But it's a *fixed* gate: the gating function is the same nonlinearity applied
elementwise, and the only thing the layer learns is the linear map feeding into it. What if the gate itself
were *learned* — if, instead of one linear projection passed through a fixed nonlinearity, the layer
produced two linear projections and used one to gate the other? That's a gated linear unit: split the
widened activation into two halves a and b, and output `a ⊙ σ(b)` — half the channels carry a value, the
other half carry a learned, input-dependent gate on that value. With a SiLU gate (x·sigmoid(x), smooth like
GELU) this is the SiLU-gated linear unit, SiGLU. The win is representational: the gate is now a function of
a *separate learned projection* of the input, so the layer can modulate each value channel by an
input-dependent amount it controls, rather than by a fixed pointwise curve. Gated MLPs of this family
consistently learn more per parameter than a plain activation MLP, which in a step-limited run means each
step makes more progress — fewer steps to the bar.

I have to keep the parameter and FLOP budget honest, though, or I've just made each step more expensive and
given the gain back. A SiGLU needs *two* linear projections into the hidden width (one for the value, one
for the gate) where GELU needed one. If I kept the 4× expansion and doubled it for the two halves, the MLP
would balloon. So I trade expansion factor for the gating: expand to `2 × 3 ×` width — i.e. an effective
hidden expansion of 3× but produced as two stacked halves — then SiGLU gates the two halves down to a 3×
hidden which the projection maps back. The split-and-gate is cheap; the cost is in the linear, and a 3×
gated hidden lands close to the 4× plain MLP in compute while learning more per step:

```python
class SiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.SiLU()
    def forward(self, x, dim=-1):
        x = x.split((x.shape[-1]//2, x.shape[-1]//2), dim=dim)
        return x[0] * self.activation(x[1])      # value half, gated by SiLU of the gate half

class MLPBlock(nn.Module):
    def __init__(self, num_channels, expansion_factor=3):
        super().__init__()
        self.norm    = LayerNorm(num_channels, bias=False)
        self.expand  = nn.Linear(num_channels, num_channels*2*expansion_factor, bias=False)  # 2x for the two halves
        self.project = nn.Linear(expansion_factor*num_channels, num_channels, bias=False)
        self.siglu   = SiGLU()
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.project(self.siglu(self.expand(x)))
        return x + residual
```

Now precision. The baseline runs *mixed* precision: an autocast context where the heavy matmuls happen in
bf16 but the tensors live in fp32 and get cast on the fly. Autocast is the safe default — it keeps a
high-precision master copy and only drops to bf16 inside the cast region — but it costs something. Every
autocast region inserts casts at its boundaries (fp32→bf16 going in, bf16→fp32 coming out), and keeping
fp32 master tensors around means more memory traffic and larger optimizer state. On the A100, bf16 carries
the full fp32 *exponent* range; the only thing it loses relative to fp32 is mantissa bits, i.e. precision,
not range — so the usual reason mixed precision exists (fp16's tiny range needing an fp32 master + loss
scaling) doesn't apply to bf16. That makes me ask: do I need the fp32 master copy at all, or can I just
make the *whole network* bf16 and drop autocast entirely?

If I cast the entire net to bf16 — weights, activations, everything — then there are no cast boundaries to
pay, the memory footprint roughly halves, and every op runs natively at tensor-core speed without autocast
bookkeeping. The risk is the lost mantissa: bf16 has ~8 bits of mantissa, so accumulations and small-update
arithmetic are coarser. For a short run to a flattening-point loss this is exactly the regime where it's
most affordable — I'm not chasing the last fraction of a bit of loss, and bf16's range protects me from the
overflow/underflow that would bite fp16. So: drop the autocast context, cast the network to pure bf16, and
make the few range-sensitive pieces (the positional-bias base matrix, the −∞ fill in the mask) bf16-typed
too so nothing silently upcasts:

```python
net = net.to(hyp['misc']['device'], torch.bfloat16)   # pure-bf16 net, no autocast region
```

The two changes are independent but land together: SiGLU makes each step learn more (fewer steps to the
bar), and pure bf16 makes each step cheaper (no autocast casts, half the memory traffic). The bet against
the previous rung: the MLP was the bulk of the per-step compute and a plain GELU MLP was leaving
per-parameter capacity on the table, while autocast was paying cast overhead for an fp32 safety margin bf16
doesn't need on this hardware — fixing both should knock another chunk off the wall-clock, into the
low-minutes. The risk is bf16's coarser arithmetic destabilizing the late, small-gradient phase; bf16's
fp32-range is the hedge, and the test is the same as always — does it still land at ~3.8.
