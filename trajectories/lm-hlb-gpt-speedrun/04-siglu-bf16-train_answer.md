The sequence-length schedule and the linear positional bias carried the run forward again past the ~3.5-minute mark. The schedule levers — dynamic batch, sequence-length growth — have squeezed the run down a lot, but I've left two parts of the *network itself* at their textbook defaults: the MLP nonlinearity and the precision policy. Both are worth re-examining now, because at this point the per-step *quality* of the network is what gates how few steps I need to reach the bar, and the per-step *cost* is dominated by the matmuls inside those MLPs.

I propose two independent changes that land together: a **SiGLU (SiLU-gated linear unit) MLP** and a **pure-bfloat16 network**. Take the MLP first. The baseline is standard: expand the channels $4\times$, apply GELU pointwise, project back. GELU is a smooth gate — it multiplies each pre-activation by a smooth approximation of "is this value positive" — but it is a *fixed* gate: the gating function is the same nonlinearity applied elementwise, and the only thing the layer learns is the linear map feeding into it. The improvement is to make the gate itself *learned*. Instead of one linear projection passed through a fixed nonlinearity, the layer produces two linear projections and uses one to gate the other — a gated linear unit. Split the widened activation into two halves $a$ and $b$ and output $a \odot \text{SiLU}(b)$, where $\text{SiLU}(x) = x\,\sigma(x)$ is smooth like GELU. Half the channels carry a value, the other half carry a learned, input-dependent gate on that value. The win is representational: the gate is now a function of a *separate learned projection* of the input, so the layer can modulate each value channel by an input-dependent amount it controls, rather than by a fixed pointwise curve. Gated MLPs of this family consistently learn more per parameter than a plain activation MLP, which in a step-limited run means each step makes more progress — fewer steps to the bar.

The budget has to stay honest, or I've just made each step more expensive and given the gain back. A SiGLU needs *two* linear projections into the hidden width — one for the value, one for the gate — where GELU needed one. If I kept the $4\times$ expansion and doubled it for the two halves, the MLP would balloon. So I trade expansion factor for the gating: expand to $2\times 3\times$ width — an effective hidden expansion of $3\times$, but produced as two stacked halves — then SiGLU gates the two halves down to a $3\times$ hidden, which the projection maps back. The split-and-gate is cheap; the real cost lives in the linear, and a $3\times$ gated hidden lands close to the $4\times$ plain MLP in compute while learning more per step.

Now the precision. The baseline runs *mixed* precision: an autocast context in which the heavy matmuls happen in bf16 but the tensors live in fp32 and are cast on the fly. Autocast is the safe default — it keeps a high-precision master copy and only drops to bf16 inside the cast region — but it costs something. Every autocast region inserts casts at its boundaries (fp32→bf16 going in, bf16→fp32 coming out), and keeping fp32 master tensors around means more memory traffic and larger optimizer state. The key observation is that on the A100, bf16 carries the *full fp32 exponent range*; the only thing it loses relative to fp32 is mantissa bits — precision, not range. So the usual reason mixed precision exists — fp16's tiny range needing an fp32 master plus loss scaling — simply doesn't apply to bf16, which raises the question of whether I need the fp32 master copy at all. If I cast the *entire* network to bf16, weights and activations and everything, then there are no cast boundaries to pay, the memory footprint roughly halves, and every op runs natively at tensor-core speed without autocast bookkeeping. The only risk is the lost mantissa — bf16 has ~8 bits, so accumulations and small-update arithmetic are coarser — but a short run to a flattening-point loss is exactly the regime where that is most affordable: I'm not chasing the last fraction of a bit of loss, and bf16's range protects me from the overflow/underflow that would bite fp16. So I drop the autocast context, cast the network to pure bf16, and type the few range-sensitive pieces (the positional-bias base matrix, the $-\infty$ fill in the mask) as bf16 too so nothing silently upcasts.

The two changes compose: SiGLU makes each step learn more (fewer steps to the bar) and pure bf16 makes each step cheaper (no autocast casts, half the memory traffic), and *more-per-step* $\times$ *cheaper-per-step* drives the wall-clock down into the low minutes with the bar held. The MLP was the bulk of the per-step compute and a plain GELU MLP was leaving per-parameter capacity on the table, while autocast was paying cast overhead for an fp32 safety margin bf16 doesn't need on this hardware — fixing both is what knocks off another chunk of time. The remaining risk is bf16's coarser arithmetic destabilizing the late, small-gradient phase; bf16's fp32-range is the hedge, and the test is the same as always — does it still land at ~3.8.

```python
class SiGLU(nn.Module):
    """ SiLU-gated linear unit; assumes the channel tensors are stacked. """
    def __init__(self):
        super().__init__()
        self.activation = nn.SiLU()
    def forward(self, x, dim=-1):
        x = x.split((x.shape[-1]//2, x.shape[-1]//2), dim=dim)
        return x[0] * self.activation(x[1])

class MLPBlock(nn.Module):
    def __init__(self, num_channels, expansion_factor=3):
        super().__init__()
        self.norm    = LayerNorm(num_channels, bias=False)
        self.expand  = nn.Linear(num_channels, num_channels*2*expansion_factor, bias=False)  # 2x for the two gated halves
        self.project = nn.Linear(expansion_factor*num_channels, num_channels, bias=False)
        self.siglu   = SiGLU()
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.expand(x)
        x = self.siglu(x)
        x = self.project(x)
        return x + residual

# pure-bfloat16 network: no autocast context, the whole net is bf16
net = net.to(hyp['misc']['device'], torch.bfloat16)
```
