**Problem (from step 3).** With the untied, zero-init head and post-embed norm reaching 10.8 min / 4578
steps, the architecture and optimizers are good; the remaining slack is in how information *flows through*
the residual stream over depth and how stable the logits are at the end. As depth grows, attention
concentrates and later layers lose access to the clean value information the early layers computed; mean-
while nothing bounds a single logit from running away and saturating the softmax late in training.

**Key idea (four bundled shortcuts).** (1) **Value residual** (Zhou et al. 2024, arXiv:2410.17897): the
first block exposes its value v1, and every later attention layer mixes its own value with v1, restoring a
direct line to the pristine early-layer values. (2) **Learnable lambda**: replace the fixed 0.5 mix with a
per-layer scalar `nn.Parameter(0.5)` optimized by Adam(lr=0.02), so each depth decides how much v1 it wants.
(3) **Embed shortcut**: feed the original normalized embedding x0 into every block via two learnable scalars
initialized to (1, 0) (a pure pass-through at init), giving deep blocks a direct path back to the raw input.
(4) **Momentum warmup** for Muon: ramp momentum 0.85→0.95 over the first 500 steps. (5) **Tanh logit
softcap** (Gemma 2, Team et al. 2024, arXiv:2408.00118): `logits = 30*tanh(logits/30)`, bounding logits to
(−30, 30) so the softmax can't saturate.

**Why it works.** Value residual and the embed shortcut both give later blocks a direct path back to clean
early signal (v1 and x0), countering attention concentration and the dilution of the input over twelve
additive updates — better information flow buys real steps. The learnable scalars all start at a pass-through
or the published mix, so each only *adds* capacity the model can ignore. Momentum warmup lets early updates
track the fast-changing gradient before settling into the smoother high-momentum regime; the tanh softcap
keeps the loss curve clean by preventing any logit from blowing up. All five are a couple of lines with
negligible parameters.

**Change / code.** Per-layer `self.lamb` mixes v with first-block v1; per-block `self.lambdas` blends x with
x0; Muon momentum ramps 0.85→0.95 over 500 steps; logits pass through `30*tanh(logits/30)`.

```python
# value residual + learnable lambda, inside the attention block
def __init__(self, ...):
    ...
    self.lamb = nn.Parameter(torch.tensor(0.5))   # value residual lambda, optimized by Adam(lr=0.02)

def forward(self, x, v1=None):
    ...
    v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
    if v1 is None:
        v1 = v                                            # first block defines the shared value
    v = (1 - self.lamb) * v + self.lamb * v1.view_as(v)   # mix with first-block value

# embed shortcut, in the Block
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config.n_embd, config.n_head)
        self.mlp = MLP(config.n_embd)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))
    def forward(self, x, x0, v1):
        x = self.lambdas[0] * x + self.lambdas[1] * x0   # embed shortcut
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)), v1)
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x

# momentum warmup for Muon (optimizer3), in the training loop
frac = min(step/500, 1)
optimizer3.param_groups[0]['momentum'] = (1 - frac) * 0.85 + frac * 0.95

# tanh logit softcap, before computing the loss
logits = 30 * torch.tanh(logits / 30)
```
