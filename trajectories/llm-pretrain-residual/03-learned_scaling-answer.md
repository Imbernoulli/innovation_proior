**Problem (from step 2).** ProRes (val_loss 2.2707) showed deep layers were under-used and that gentle
control of branch writes helps — but a *fixed schedule* dictates one trajectory per layer, never lets a
layer settle on a residual weight other than 1, and expires back into the rigid unit-weight accumulator.
It also never touched a second weakness: the token embedding is diluted in the additive stream as depth
grows (over-smoothing). The next rung should *learn* the per-layer magnitude (persistently) and add a
direct embedding-to-every-depth route.

**Key idea.** Learnable Residual Scaling + x0 injection (the modded-nanogpt speedrun design). Two
learnable scalars per layer control each residual write:

`x_new = resid_lambda[l]·x + delta + x0_lambda[l]·x0`,

where `delta = block_out − x` is the block's Pre-LN branch and `x0` is the embedding-stage stream
(post-dropout, post-position). `resid_lambda[l]` scales the incoming residual carry; `x0_lambda[l]`
scales a direct re-injection of the token embedding. **Init `resid_lambda = 1.0`, `x0_lambda = 0.0`**, so
at step zero `x_new = x + delta` — exactly vanilla — and the network *refines* from the working floor
rather than rebuilding it from an identity start (the prores numbers say the init-time conditioning win
is small, so do not pay for a zero-init tax).

**Why it works.** `resid_lambda` is the persistent, learnable version of the ProRes idea — a layer finds
its own residual weight and *keeps* it (can leak the carry below 1 to counteract the variance climb, or
amplify above 1), no schedule to expire. `x0_lambda` adds the route the schedule never gave: the
embedding is the least-redundant signal (every later hidden state is already in the running sum; the
embedding is the one thing the accumulator *loses* with depth), so re-injecting it gives every depth a
gradient highway and a forward path to un-diluted token identity. The two knobs are different axes —
attenuate the noisy carry *and* re-anchor on clean token identity — so this is strictly more expressive
than a single scalar (ReZero) or a single scheduled scalar (ProRes), at 2·n_layer = 48 extra params.

**Hyperparameters / optimizer.** The 48 scalars are gains, so they get **no weight decay**: route both
tensors by `id()` into a dedicated no-decay group (decay would pull `resid_lambda`→0 and fight the init).
LR unchanged, base rate; `CONFIG_OVERRIDES = {}`. No step buffer, no `self.training` gate — these are
persistent and apply identically at train and eval.

**What to watch.** Beat *both* prores (2.2707) and vanilla — mid-2.26s would be a clean win.
x0-injection's signature is token identity at depth: WikiText-2 below 44.11, ARC-Easy holding/extending
55.35. LAMBADA may *not* match prores's deep-layer-warm-up win (init at 1, not ramping from 0), a tell
the two methods help through different channels. Risk: with 48 scalars on a 13.5k-step run the
`x0_lambdas` may not learn far from 0, collapsing toward a resid-only model near prores. If the win is
real-but-small, the next rung must drop the rank-one scalar knob entirely and give the network
content-dependent, full-rank mixing over the depth axis — attention over layers.

```python
# EDITABLE regions of custom_pretrain.py — step 3: learnable residual scaling + x0 injection
# (each region shown inside its enclosing method exactly as spliced into the file)

# Block: unchanged — vanilla Pre-LN residual.

class GPT(nn.Module):
    def _init_scaling(self, config):  # GPT.__init__ residual region:
        # ── Learnable residual scaling + x0 injection ──
        # resid_lambdas[i]: scales the incoming residual stream (init 1.0 = vanilla)
        # x0_lambdas[i]:    scales the embedding injection (init 0.0 = no injection)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))

    def _forward_block_loop(self, x):  # GPT.forward block loop:
        # ── Learnable residual scaling + x0 injection ──
        # x0 = embedding output; provides gradient highway to every depth.
        x0 = x
        for i, block in enumerate(self.transformer.h):
            block_out = block(x)
            delta = block_out - x
            x = self.resid_lambdas[i] * x + delta + self.x0_lambdas[i] * x0
        return x

    # GPT.configure_optimizers (route scaling params to no-decay group):
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Route residual scaling params to no-decay group
        scaling_ids = {id(self.resid_lambdas), id(self.x0_lambdas)}
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2 and id(p) not in scaling_ids]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2 and id(p) not in scaling_ids]
        scaling_params = [p for n, p in param_dict.items() if id(p) in scaling_ids]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
            {'params': scaling_params, 'weight_decay': 0.0},
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

# CONFIG_OVERRIDES = {}   (no override).
```
