The SiGLU MLP and the pure-bf16 net brought the run down to ~136–140 seconds, and now I want to break a hundred. The time that remains lives in two places. One is the block structure: every layer still runs attention and the MLP as *two separate sublayers*, each with its own LayerNorm, its own input projection, its own output projection, and its own residual add — two norms, two in-projections, two out-projections, two residual adds per layer, a lot of separate small ops and memory round-trips, when the two sublayers are doing closely related things. The other is the microbatch controller from earlier: it works, but it steers off a *local* signal — the per-step grad-norm jitter against a fixed target — which is a bit blind, because it doesn't know where in the run it is or how big the model is. I attack both.

The first piece is the **LatentAttentionBlock**, a fusion of attention and the gated MLP into a single block. Attention and the MLP were two sublayers historically because they do different jobs — attention mixes across positions, the MLP mixes across channels — and two modules is the clean way to write that. But look at what they share: both start by normalizing the residual stream, both project the normed input up into some wider working space, and both end by projecting back down and adding to the residual. The *only* genuinely different middle step is that attention does a softmax mix over positions while the MLP does a pointwise gated nonlinearity over channels; everything around that middle is duplicated. So I do the up-projection *once*, carve the wide working space into the pieces attention needs (queries, keys, values) and the pieces the MLP needs at the same time, run both middles, then project down *once* — paying one norm, one in-projection, one out-projection, one residual add per layer instead of two of each. Concretely: one shared LayerNorm, one `expand` projection from the residual width into a wide space split into four parts — a query slice, a key slice, and two slices for a gated nonlinear path (`linear` and `pre_gelu`). The gated path computes a GeGLU, $\text{linear} \odot \text{GELU}(\text{pre\_gelu})$, the same gated-MLP idea I already adopted, now living inside the fused block.

What makes the fusion tight is that the gated features do double duty. Of the GeGLU output I let *part* of the channels stay local — that is the MLP's contribution to this position — and I use the *other* part as the *values* for attention. So the MLP's nonlinearity and the attention's values come out of the *same* gated computation: the values attention mixes over positions are themselves the nonlinear gated features, not a separate linear projection. Attention then mixes those values across positions, I concatenate the local (MLP) part with the attended part, and one `project` maps the whole thing back to the residual width. Two design choices here are load-bearing. First, *no attention heads* — I run attention as a single head over the full value width, because splitting into heads forces `.contiguous()` reshapes and per-head bookkeeping, and a single head over the fused values skips all of that and works at least as well at this scale. Second, the keys and queries are *linear* and small ($\text{qk\_dim} = \text{dim}/8$) while the values are the *nonlinear* gated features — cheap linear projections decide *where* to attend, the expensive nonlinear computation decides *what* gets mixed. The per-layer `position_bias_mult` carries forward the length-agnostic linear positional bias from before, one learnable slope per block via softplus, and the `project` parameter is init-scaled by $1/\text{num\_blocks}$ to keep the residual stream stable with the fused output adding into it.

The second piece is the **expected-grad-norm power-law microbatch scheduler**, a better target for the batch controller. The earlier version chased a fixed per-step grad-norm *delta* — a local, model-blind target. I can do better by giving it a *model-aware, run-aware* reference. The reasoning is that over a training run the grad norm doesn't just wander, it *decays*, and empirically it decays roughly as a *power law* in the step count. If I know the expected grad-norm trajectory I can schedule the microbatch size to *track* it: when the measured grad-norm-per-parameter runs above the expected curve the model is under-averaging and I grow the microbatch count; when it runs below, I shrink it — a far less blind controller than "keep the per-step jitter near a constant," because it has an explicit expectation for where the grad norm *should* be at this step, for this model size. So I posit
$$\text{grad\_norm\_target} = (\text{scale}\cdot\text{step})^{\text{pow}},$$
and make both the exponent and the scale depend on the parameter count. The exponent should be slightly more negative (faster expected decay) for smaller models and less negative for larger ones; a clean form that does this is $\text{pow} = -0.677\cdot\log(\text{params})^{-0.2}$ — as params grow, $\log(\text{params})^{-0.2}$ shrinks, so $\text{pow}$ moves toward zero, a slower expected decay, exactly the size-dependence I want. The scale folds in the parameter count too, $\text{scale} = \log(\text{params})\cdot\text{params}$, so the curve sits at the right magnitude across model sizes. I normalize the measured grad norm per parameter as $\text{grad\_norm}/\sqrt{\text{params}}$ so the comparison is size-invariant, take the ratio of measured-to-expected, and push the fractional microbatch count multiplicatively toward closing that ratio. It keeps the Bernoulli dithering from before — the fractional `microbatch_steps` is the smooth accumulator, each step uses an integer via `divmod` and `torch.bernoulli` — but now the target is a principled, model-size-aware power-law trajectory instead of a fixed local delta. I only sample the grad norm every few steps because computing it isn't free, and the controller is smooth enough to tolerate that.

This is the closing move. Fusing attention and the MLP into one block — one norm, one up-projection, one down-projection, one residual add per layer, with the gated nonlinear features doing double duty as the attention values — cuts the per-step op count and memory traffic substantially, while the expected-grad-norm power-law controller schedules the effective batch off a principled trajectory instead of a blind local target, spending compute where the run actually needs it. Together they drop the wall-clock *under 100 seconds* to ~3.8 val loss on a single 40 GB A100. The risks are real — the headless single-head attention and the tight `expand`/`project` initializations could cost stability, and the power-law controller could mis-track if the real grad-norm decay departs from the assumed exponent — and the $1/\text{num\_blocks}$ project-init scaling and the $\max(\cdot, 0.1)$ microbatch floor are the hedges. With the fused block holding quality and the controller tracking, this is the fastest known way to train a small GPT to ~3.8 val loss on one A100, and that is where the ladder ends.

```python
class LatentAttentionBlock(nn.Module):
    """ Efficient fused latent-space attention block. Linear keys and queries, nonlinear values. """
    def __init__(self, num_dim):
        super().__init__()
        self.dim        = num_dim
        self.qk_dim     = self.dim // hyp['net']['qk_dim_div']
        self.v_dim      = num_dim
        self.expand_dim = num_dim * hyp['net']['expand_factor']
        self.norm    = nn.LayerNorm(self.dim, bias=False)
        self.expand  = nn.Parameter(.5 * 1./hyp['net']['residual_depth']**.5 * 1./hyp['net']['expand_factor']
                                    * torch.randn(2*self.qk_dim + 2*self.expand_dim, self.dim))
        self.project = nn.Parameter(1. * 1./hyp['net']['residual_depth']**.5 * 1./hyp['net']['expand_factor']
                                    * 1./hyp['net']['num_blocks'] * torch.randn((self.dim, self.expand_dim)))
        self.position_bias_mult = nn.Parameter(torch.tensor(1., device='cuda'))

    def forward(self, x):
        residual = x
        attn_mask = torch.where(causal_mask[:x.shape[1], :x.shape[1]],
                                F.softplus(self.position_bias_mult) * position_bias_base[:x.shape[1], :x.shape[1]],
                                negative_infinity_matrix_base[:x.shape[1], :x.shape[1]])
        x = self.norm(x)
        query, key, linear, pre_gelu = F.linear(x, self.expand).split(
            (self.qk_dim, self.qk_dim, self.expand_dim, self.expand_dim), dim=-1)
        geglu = linear * F.gelu(pre_gelu)
        geglu_local, geglu_attention_value = geglu.split((self.expand_dim - self.v_dim, self.v_dim), -1)
        attention = F.scaled_dot_product_attention(query, key, geglu_attention_value, attn_mask=attn_mask)
        out = F.linear(torch.cat([geglu_local, attention], dim=-1), self.project)
        return residual + out

# expected-grad-norm power-law microbatch scheduler
microbatch_expected_grad_norm_pow = -.677 * math.log(total_trainable_params) ** -.2
microbatch_grad_norm_steps_scale  = math.log(total_trainable_params) * total_trainable_params
# ... every `sample_every` steps:
grad_norm_per_param = grad_norm / (total_trainable_params**.5)
grad_norm_target    = (microbatch_grad_norm_steps_scale * (curr_step + 1e-2)) ** microbatch_expected_grad_norm_pow
ratio_diff          = grad_norm_per_param / grad_norm_target
microbatch_steps   *= 1. + (sample_every * scale_lr * (ratio_diff - 1))
microbatch_steps    = max(microbatch_steps, 1e-1)
base, dither_prob   = divmod(microbatch_steps, 1)
discrete_sampled_microbatch_steps = max(1, int(base + torch.bernoulli(torch.tensor(dither_prob)).item()))
```
