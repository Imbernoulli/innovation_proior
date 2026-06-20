Ten point eight minutes now, 4578 steps. The untie experiment told me something concrete: the embedding
and head were the limiting factor, and the wins came from cleaning up how information enters and leaves the
residual stream. The architecture is good, the optimizers are good. So the next place to look is the same
place — the flow of information *through* the residual stream over the twelve blocks, and the stability of
the logits at the very end. None of these is a big new mechanism; they're a handful of small architectural
shortcuts, and I think they compound. Let me derive each from a pain point I can actually name.

Start with depth. A residual transformer is supposed to let every layer read the running sum, but there's a
well-known failure mode: as I stack blocks, attention *concentrates* — later layers pour their attention
mass onto a few positions and effectively stop attending broadly, and in doing so they lose access to the
value information the early layers computed. The value vectors at block 1 carry a clean, lightly-processed
view of each token; by block 8 the values are heavily transformed and the early signal is hard to recover
from the residual stream alone. This is exactly the diagnosis in value residual learning, Zhou et al. 2024
("Value Residual Learning For Alleviating Attention Concentration In Transformers", arXiv:2410.17897). Their
fix is simple and cheap: give every later layer a direct line back to the *first* block's value vector. The
first block computes v1 and exposes it; every subsequent attention layer mixes its own freshly-projected v
with v1 before doing attention. In the original form the mix is a fixed half-and-half, `v = 0.5*v +
0.5*v1`. So each head, no matter how deep, always retains half of the pristine early-layer value, and the
attention output can't drift entirely away from the input's value content. It's the value-stream analogue of
a residual connection, but across depth rather than within a block.

A fixed 0.5 bothers me a little, though. Why should every layer want exactly half of v1? The first decoder
layer is right next to v1 and probably wants very little of it; a deep layer that's badly concentrated
probably wants more. The cheapest possible way to let the model decide is to make the mixing coefficient a
*learnable scalar*, one per attention layer, and let the optimizer find the right value per depth. So
instead of a constant I write `self.lamb = nn.Parameter(torch.tensor(0.5))`, initialized at 0.5 so I start
exactly at the published behavior, and the mix becomes `v = (1 - self.lamb)*v + self.lamb*v1`. It's a single
extra scalar per layer — negligible parameters — and these scalars are tiny 1-D tensors, so I hand them to
Adam rather than Muon, at a fairly aggressive lr=0.02 since they're scalars that should move fast. The model
can now turn the early-value injection up where attention is concentrated and down where it isn't.

That value-residual idea is really an instance of a more general principle: deep blocks benefit from a
*direct path back to something early and clean*. v1 is one such early signal. But there's an even earlier
one — the input embedding itself, the post-lookup, post-norm activation x0 that seeds the whole residual
stream. By block 10 the residual stream is a deep transformation of x0, and the only way a block can still
"see" the raw input is through whatever survived ten rounds of additive updates. Why not give every block a
direct line to x0 as well, exactly as I'm now giving them a direct line to v1? The same shape of fix: before
each block does its attention and MLP, let it form a learnable blend of the current residual x and the
original embedding x0. I add two scalars per block, `self.lambdas = nn.Parameter(torch.tensor([1., 0.]))`,
initialized to (1, 0) so at the start it's a pure pass-through — `x = self.lambdas[0]*x + self.lambdas[1]*x0`
is just `x` when lambdas = (1,0), so I again begin from the known-good behavior and let training dial in how
much raw input each block wants to re-inject. This is the embed shortcut: input → every block, learnable.

Two of these are about *getting clean early signal deep into the net*. Now two more about *not destabilizing
the run*. First, the optimizer side. Muon's momentum is set to 0.95, which is great once the network has
found a sensible region — momentum averages the orthogonalized updates and smooths the trajectory. But at
the very start, when the weights are near their zero-init and the loss is dropping fastest, a heavy momentum
buffer is averaging over updates that are themselves changing direction rapidly; the buffer lags the true
gradient and the early steps are noisier than they need to be. The fix is a momentum *warmup*: start Muon at
a gentler 0.85 and ramp linearly up to 0.95 over the first 500 steps, so early updates track the fresh
gradient more closely while the net is still finding its footing, then settle into the smoother high-momentum
regime once it's in a good region. In the loop: `frac = min(step/500, 1)` and
`optimizer3.param_groups[0]['momentum'] = (1 - frac)*0.85 + frac*0.95`. It costs nothing — it's a schedule on
a hyperparameter I already have.

Second, the logits. I zero-init the head, so the run *starts* from uniform max-entropy logits, which is
clean — but nothing stops a single logit from blowing up *later* in training. As the head sharpens, one
coordinate of the 50304-way output can run away, the softmax saturates, the cross-entropy gradient through
that coordinate goes tiny or spikes, and I get a noisy, slightly unstable loss curve right where I want it
smooth. Gemma 2 (Team et al. 2024, arXiv:2408.00118) handles exactly this with a tanh logit softcap: pass
the logits through a smooth saturating nonlinearity so they're bounded but the function stays differentiable
and monotone. I'll cap at 30: `logits = 30 * torch.tanh(logits / 30)`. For small logits this is essentially
the identity (tanh(z)≈z near zero), so it doesn't distort normal training; but it asymptotes to ±30, so no
single logit can blow past that bound, the softmax never fully saturates, and the loss curve stays smooth.
A free stabilizer with a negligible footprint.

So I'm bundling four shortcuts, each derived from the same instinct — give the network direct paths to clean
signal and don't let any one quantity run away. Value residual with a learnable per-layer lambda (mix in the
first block's v1, decide how much per depth). Embed shortcut (re-inject the original x0 into every block via
two learnable scalars). Momentum warmup for Muon (0.85→0.95 over 500 steps, gentler early updates). And the
tanh logit softcap at 30 (bound the output so the softmax can't saturate). Each is a couple of lines; each
starts from a pass-through or known-good initialization so I'm only ever *adding* capacity the model can
choose to ignore. I expect the value paths to cut the step count meaningfully — better information flow is
worth real steps — and the warmup and softcap to make those steps cleaner and less likely to stall. Adding
them incrementally on top of the 10.8-minute record.

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

The chain: as depth grows attention concentrates and late layers lose the early value signal, so I mix each
layer's value with the first block's v1 (value residual, Zhou et al. 2024) and let a learnable per-layer
lambda decide how much; generalizing "give deep blocks a clean early signal" I also re-inject the original
embedding x0 into every block through two learnable scalars (embed shortcut); I ramp Muon's momentum from
0.85 to 0.95 over 500 steps so early updates track the fast-changing gradient; and I cap the logits with
`30*tanh(logits/30)` (Gemma 2's softcap) so no coordinate blows up and the softmax never saturates. Four
small shortcuts, each a pass-through at init, compounding into fewer and cleaner steps to 3.28.
