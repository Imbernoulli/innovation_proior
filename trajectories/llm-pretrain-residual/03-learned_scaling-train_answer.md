ProRes moved the floor exactly where its mechanism predicted: validation loss went $2.2763 \to 2.2707$, WikiText-2 barely budged ($44.28 \to 44.11$), but LAMBADA dropped from 70.09 to 67.21 — the sharpest single move so far, and LAMBADA is the long-range task that leans hardest on the deep layers, so the shallow-to-deep warm-up did wake them up a little. The gain is real but small, and it exposed the schedule's own ceiling: a *fixed schedule* dictates one trajectory, $\alpha(l,t) = \min(t/(T\cdot l), 1)$, for every layer and *ends at vanilla*. It never lets a layer settle on a residual weight other than 1, and gives the network no say in the matter — once the warm-up expires every layer is back to the rigid unit-weight accumulator. I want to keep what worked, gentle conditioned control of how much each branch writes, but hand the magnitude to the network and let it persist; and I want to attack a second weakness the schedule never touched.

I propose Learnable Residual Scaling with $x_0$ injection — two learnable scalars per layer, the modded-nanoGPT speedrun design. The residual write becomes

$$x_{\text{new}} = \texttt{resid\_lambda}[l]\cdot x + \delta + \texttt{x0\_lambda}[l]\cdot x_0,$$

where $\delta = \texttt{block\_out} - x$ is the block's own Pre-LN branch (recovered the same way ProRes recovered it), $x_0$ is the embedding-stage stream (post-dropout, post-position, the thing entering the first block), `resid_lambda[l]` scales the incoming residual carry, and `x0_lambda[l]` scales a direct re-injection of the token embedding. The two knobs are different axes. `resid_lambda` is the persistent, learnable version of the ProRes idea: at 1 it is the plain residual, below 1 it lets a layer *forget* some of the accumulated stream — a learned leak that counteracts the variance climb — and above 1 it amplifies. `x0_lambda` is the new route: at 0 it is off, and as the network grows it the layer pulls token identity straight from the embedding.

The crucial choice is the init, and it departs from ReZero deliberately. The classic move is to zero-init the residual weight for an exact identity start, but I have a measured fact that argues against it *here*: vanilla, the all-$\lambda=1$ model, already trains cleanly to 2.2763, and ProRes — which is $\lambda$ ramping from 0 to 1 — only beat it by a hair, so at 24 layers and 13.5k steps the init-time conditioning problem is mild. If I zero-init I spend a good chunk of a short run growing the weights back toward 1 before the network even has its vanilla capacity online. So I init `resid_lambda = 1.0` and `x0_lambda = 0.0`, which gives $x_{\text{new}} = 1\cdot x + \delta + 0\cdot x_0 = x + \delta$ — *bit-for-bit the vanilla residual at step zero*. Every deviation the network learns from there is a deliberate, gradient-driven refinement: "the floor is good, refine it," rather than "the floor is suspect, re-earn it." That is the honest reading of the prores numbers — the conditioning win was real but tiny, so I should not pay for it twice with an identity-start tax.

Why $x_0$ specifically, and not "blend in some earlier layer's output"? Because $x_0$ is the *least redundant* signal in the stream. In the unrolled stream $x_l = x_1 + \sum_{i<l} F_i(\mathrm{LN}(x_i))$, every later representation $x_i$ for $i>1$ is already reachable through the ordinary additive residual — it is sitting in the running sum. The embedding is the one thing the additive stream is actively *losing* as depth grows: it carries pure token identity (which token is here, before any attention has mixed it across positions), and as the self-attention mixing and MLP writes pile on, that original signal becomes a smaller and smaller fraction of the accumulated total — the over-smoothing deep attention stacks are known for. Re-injecting a generic earlier hidden state would mostly duplicate what the residual already carries; re-injecting the embedding restores the specific thing that gets diluted, and gives every depth a gradient highway and a forward path to un-diluted token identity. It is also the cheapest such route — $x_0$ is a $(B,T,D)$ tensor already in hand at the bottom of the forward, broadcast-added at each layer, no projection, no attention, no new shapes. And the two scalars are not redundant with each other: `resid_lambda` controls the *carry* (how much of the past survives), `x0_lambda` controls the *injection* (how much fresh token identity enters), so a deep layer can attenuate the noisy accumulated stream (`resid_lambda < 1`) *while* pulling in clean token identity (`x0_lambda > 0`) to re-anchor — the over-smoothing repair a single ReZero scalar cannot express. This is strictly more expressive than the step-2 schedule or a one-scalar design, at $2\cdot n_{\text{layer}} = 48$ extra parameters against 355M.

There is one place this rung touches the optimizer, and it matters because these 48 scalars are *gains*, not weights — each sits at a leveraged point multiplying a whole layer's worth of signal. They are one-dimensional, so they must not get weight decay: decay would pull `resid_lambda` toward 0 and `x0_lambda` toward 0, fighting the very init values I chose and degrading the carry. The scaffold's default routing already sends `dim < 2` parameters to a no-decay group, but I want to be explicit and certain — a single mis-decayed gain on the residual carry could quietly hurt the whole stream — so in `configure_optimizers` I pull the two scalar tensors out by `id()` into their own dedicated no-decay group at the base learning rate. That is the entire optimizer change; the LR schedule and `CONFIG_OVERRIDES` stay default. The `Block` stays vanilla and still returns `block_out = x + delta`; I snapshot `x0 = x` after the embedding and position-encoding but before the first block, and there is no step buffer, no schedule, no `self.training` gate — these are persistent learnable scalars that apply identically at train and eval. Against prores's 2.2707 and the 2.2763 floor I expect to beat both — the learnable carry subsumes what the schedule did and the $x_0$ route adds something neither earlier rung had — with the signature of token identity preserved at depth: WikiText-2 finally below 44.11 and ARC-Easy (knowledge recall) holding or extending 55.35. LAMBADA is the interesting one: learned-scaling's gentle init-at-1 carry might *not* reproduce prores's specific deep-layer-conditioning win, so it could land slightly above 67.21 even as val_loss improves — a tell that the two methods help through different channels.

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
