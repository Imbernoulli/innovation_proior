**Problem (from rung 3).** The network itself is still at textbook defaults in two places: the MLP uses a
4×-expansion GELU (a *fixed* pointwise gate, where the only learned thing is the linear map into it), and
precision is *mixed* via autocast (bf16 matmuls but fp32 master tensors, paying cast overhead at every
region boundary and carrying double the memory). On the A100, bf16 has fp32's exponent range, so the fp32
master copy buys little for a short run.

**Key idea.** (1) **SiGLU MLP**: replace the GELU with a SiLU-gated linear unit — split the widened
activation into two halves and output `value ⊙ SiLU(gate)`, so the gate is a *separate learned projection*
the layer controls, not a fixed curve. Keep the budget honest by trading the 4× plain expansion for a 3×
*gated* hidden produced as two stacked halves (`expand` to `2·3·width`). (2) **Pure bfloat16 net**: drop
the autocast context and cast the entire network to bf16 — no cast boundaries, ~half the memory traffic,
native tensor-core ops — relying on bf16's fp32-range to stay numerically safe where fp16 could not.

**Why it works.** Gated MLPs learn more per parameter, so each step makes more progress and fewer steps
reach the bar; pure bf16 removes autocast's cast/memory overhead so each step is cheaper. The two compose:
more-per-step × cheaper-per-step drives the wall-clock down into the low minutes, bar held. bf16's full
exponent range is what makes dropping the fp32 master safe on this hardware.

**Change / code.** The SiGLU unit, the rebuilt MLP block, and the pure-bf16 cast.

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
