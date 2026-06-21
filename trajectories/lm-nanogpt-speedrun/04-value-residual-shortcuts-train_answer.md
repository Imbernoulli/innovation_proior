Untying the embedding and head, zero-initializing the head, and normalizing the embedding after lookup cut the step count to 4578 and the wallclock to ~10.8 minutes. That experiment told me where the slack now lives: the architecture and optimizers are good, and the wins came from cleaning up how information enters and leaves the residual stream. So the next place to look is the same place — how information flows *through* the stream over the twelve blocks, and the stability of the logits at the very end. None of these is a big new mechanism; they are a handful of small shortcuts, each derived from a pain point I can name, and they compound.

I propose **four bundled shortcuts**, each a couple of lines and each a pass-through at initialization so I only ever *add* capacity the model can choose to ignore. The first is the **value residual** (Zhou et al. 2024). As depth grows a transformer hits a known failure mode — attention *concentrates*, later layers pour their mass onto a few positions and lose access to the clean value information the early layers computed. The value vectors at block 1 carry a lightly-processed view of each token; by block 8 the values are heavily transformed and the early signal is hard to recover from the residual stream alone. The fix gives every later layer a direct line back to the first block's value $v_1$: the first block computes and exposes $v_1$, and every subsequent attention layer mixes its own freshly-projected $v$ with $v_1$ before doing attention. In the original form the mix is a fixed half-and-half, $v = 0.5\,v + 0.5\,v_1$ — the value-stream analogue of a residual connection, across depth rather than within a block. A fixed $0.5$ bothers me, though: the first decoder layer sits right next to $v_1$ and probably wants little of it, while a badly-concentrated deep layer probably wants more. So the second shortcut is the **learnable lambda** — replace the constant with a per-layer scalar `self.lamb = nn.Parameter(torch.tensor(0.5))`, initialized at $0.5$ so I start exactly at the published behavior, and let the mix become $v = (1-\lambda)\,v + \lambda\,v_1$. It is one extra scalar per layer, negligible parameters; these 1-D tensors go to Adam at an aggressive `lr=0.02` since scalars should move fast. The model can now turn the early-value injection up where attention concentrates and down where it does not.

That is really an instance of a general principle — *deep blocks benefit from a direct path back to something early and clean* — and $v_1$ is just one such signal. An even earlier one is the input embedding itself, the post-lookup, post-norm activation $x_0$ that seeds the whole stream. By block 10 the residual is a deep transformation of $x_0$, and the only way a block still "sees" the raw input is through whatever survived ten additive updates. So the third shortcut is the **embed shortcut**: give every block a direct line to $x_0$ exactly as I gave it a line to $v_1$. Before each block does its attention and MLP it forms a learnable blend of the current residual $x$ and the original embedding $x_0$ via two scalars `self.lambdas = nn.Parameter(torch.tensor([1., 0.]))`, initialized to $(1, 0)$ so $x = \lambda_0 x + \lambda_1 x_0$ is a pure pass-through at the start, and training dials in how much raw input each block re-injects.

The last two are about not destabilizing the run rather than information flow. The fourth shortcut is a **momentum warmup** for Muon. Its momentum is set to $0.95$, which is great once the network has found a sensible region, but at the very start the weights are near their zero-init and the loss drops fastest, and a heavy buffer averages over updates that are themselves changing direction rapidly — it lags the true gradient and the early steps are noisier than they need to be. So I start Muon at a gentler $0.85$ and ramp linearly to $0.95$ over the first 500 steps: `frac = min(step/500, 1)` and `momentum = (1-frac)*0.85 + frac*0.95`. Early updates track the fresh gradient while the net finds its footing, then settle into the smoother regime; it costs nothing, a schedule on a hyperparameter I already have. The fifth piece is the **tanh logit softcap** (Gemma 2, Team et al. 2024). I zero-init the head so the run *starts* from uniform max-entropy logits, but nothing stops a single logit from blowing up *later* — as the head sharpens, one coordinate of the 50304-way output can run away, the softmax saturates, and the loss curve gets noisy right where I want it smooth. Passing the logits through a smooth saturating nonlinearity bounds them while keeping the function differentiable and monotone: `logits = 30 * torch.tanh(logits / 30)`. For small logits this is essentially the identity ($\tanh(z)\approx z$ near zero), so normal training is undistorted, but it asymptotes to $\pm 30$, so no single logit blows past that bound and the softmax never fully saturates. Each shortcut starts from a pass-through or known-good initialization, so they only add capacity; I expect the value paths to cut the step count meaningfully and the warmup and softcap to make those steps cleaner, added incrementally on top of the 10.8-minute record.

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
