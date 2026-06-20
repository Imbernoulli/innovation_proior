I'm at ~136–140 seconds and I want to break a hundred. Where is the time now? Two places. One is the block
structure: every layer still does attention and MLP as *two separate sublayers*, each with its own
LayerNorm, its own input projection, its own output projection, its own residual add. That's two norms, two
in-projections, two out-projections, two residual adds, per layer — a lot of separate small ops and a lot
of memory round-trips, when the two sublayers are doing closely related things. The other is the batch
controller from earlier: it works, but it steers off a *local* signal (the per-step grad-norm jitter
against a fixed target), which is a bit blind — it doesn't know where in the run it is or how big the model
is. Both are worth attacking to get under 100s.

Take the block first. Why are attention and the MLP two separate sublayers? Historically because they do
different jobs — attention mixes across positions, the MLP mixes across channels — and the clean way to
write that is two modules. But look at what they share. Both start by normalizing the residual stream. Both
project the normed input up into some wider working space. Both end by projecting back down and adding to
the residual. The *only* genuinely different middle step is: attention does a softmax mix over positions;
the MLP does a pointwise gated nonlinearity over channels. Everything around that middle is duplicated. If I
could do the up-projection *once*, carve the wide working space into the pieces attention needs (queries,
keys, values) *and* the pieces the MLP needs (a gated nonlinear path) at the same time, run both middles,
then project down *once* — I'd pay one norm, one in-projection, one out-projection, one residual add per
layer instead of two of each. Fuse the two sublayers into one.

Let me design that fused block. One shared LayerNorm. One `expand` projection from the residual width to a
wide space, which I split into four parts: a query slice, a key slice, and two slices for a gated nonlinear
path (a `linear` part and a `pre_gelu` part). The gated path computes a GeGLU — `linear ⊙ GELU(pre_gelu)` —
the same gated-MLP idea I already adopted, now living inside the fused block. To make the fusion tight, of
that gated nonlinear output I let *part* of the channels stay local (that's the MLP's contribution to this
position) and I use the *other* part as the *values* for attention. So the MLP's
nonlinearity and the attention's values come out of the *same* gated computation — the values attention
mixes over positions are themselves the nonlinear, gated features, not a separate linear projection. Then
attention mixes those values across positions, I concatenate the local (MLP) part with the attended part,
and one `project` maps the whole thing back to the residual width:

```python
class LatentAttentionBlock(nn.Module):
    """ Fused latent-space attention + gated-MLP block. Linear keys/queries, nonlinear gated values. """
    def __init__(self, num_dim):
        super().__init__()
        self.dim        = num_dim
        self.qk_dim     = self.dim // hyp['net']['qk_dim_div']
        self.v_dim      = num_dim
        self.expand_dim = num_dim * hyp['net']['expand_factor']
        self.norm    = nn.LayerNorm(self.dim, bias=False)
        # one fused up-projection producing query, key, and the two gated-MLP slices
        self.expand  = nn.Parameter(.5 * hyp['net']['residual_depth']**-.5 * hyp['net']['expand_factor']**-1
                                    * torch.randn(2*self.qk_dim + 2*self.expand_dim, self.dim))
        # one down-projection, init-scaled by 1/num_blocks for residual-stream stability
        self.project = nn.Parameter(hyp['net']['residual_depth']**-.5 * hyp['net']['expand_factor']**-1
                                    * hyp['net']['num_blocks']**-1 * torch.randn((self.dim, self.expand_dim)))
        # one learnable per-layer position-bias slope (the same length-agnostic linear bias, per block)
        self.position_bias_mult = nn.Parameter(torch.tensor(1., device='cuda'))

    def forward(self, x):
        residual = x
        attn_mask = torch.where(causal_mask[:x.shape[1], :x.shape[1]],
                                F.softplus(self.position_bias_mult) * position_bias_base[:x.shape[1], :x.shape[1]],
                                negative_infinity_matrix_base[:x.shape[1], :x.shape[1]])
        x = self.norm(x)
        # ONE fused projection, split into query / key / gated-MLP-linear / gated-MLP-pregelu
        query, key, linear, pre_gelu = F.linear(x, self.expand).split(
            (self.qk_dim, self.qk_dim, self.expand_dim, self.expand_dim), dim=-1)
        geglu = linear * F.gelu(pre_gelu)                              # gated nonlinear features
        geglu_local, geglu_attention_value = geglu.split((self.expand_dim - self.v_dim, self.v_dim), -1)
        # the gated features ARE the attention values
        attention = F.scaled_dot_product_attention(query, key, geglu_attention_value, attn_mask=attn_mask)
        out = F.linear(torch.cat([geglu_local, attention], dim=-1), self.project)   # ONE down-projection
        return residual + out
```

Two design notes I want to get right. First, no attention heads — I run attention as a single head over the
full value width. Splitting into heads forces `.contiguous()` reshapes and per-head bookkeeping; a single
head over the fused values skips that, and at this scale it works at least as well. Second, the keys and
queries are *linear* (small, `qk_dim = dim/8`) while the values are the *nonlinear* gated features — that's
the asymmetry in the docstring: cheap linear projections decide *where* to attend, the expensive nonlinear
computation decides *what* gets mixed. And the per-layer `position_bias_mult` carries forward the
length-agnostic linear positional bias from before, one learnable slope per block via softplus.

Now the second target: the batch controller. The earlier version chased a fixed per-step grad-norm *delta*
— a local, model-blind target. I can do better by giving it a *model-aware, run-aware* reference. Here's the
reasoning: over a training run the grad norm doesn't just wander, it *decays* — and empirically it decays
roughly as a *power law* in the step count. If I knew the expected grad-norm trajectory, I could schedule
the microbatch size to *track* it: when the measured grad-norm-per-parameter runs above the expected curve,
the model is under-averaging and I grow the microbatch count; when it runs below, I shrink it. That's a far
less blind controller than "keep the per-step jitter near a constant" — it has an explicit expectation for
where the grad norm *should* be at this step, for this model size.

So I posit `grad_norm_target = (scale · step)^pow`, a power-law decay, and I make both the exponent and the
scale depend on the model size. The exponent should be slightly more negative (faster expected decay) for
smaller models and less negative for larger ones; a clean form that does this is `pow = −0.677 ·
log(params)^(−0.2)` — as params grow, `log(params)^(−0.2)` shrinks, so `pow` moves toward zero, a slower
expected decay, exactly the size-dependence I want. The scale folds in the parameter count too, `scale =
log(params) · params`, so the curve sits at the right magnitude across model sizes. I normalize the
measured grad norm per parameter (`grad_norm / √params`) so the comparison is size-invariant, take the ratio
of measured-to-expected, and push the fractional microbatch count multiplicatively toward closing that
ratio:

```python
microbatch_expected_grad_norm_pow = -.677 * math.log(total_trainable_params) ** -.2
microbatch_grad_norm_steps_scale  = math.log(total_trainable_params) * total_trainable_params

# every `sample_every` steps:
grad_norm_per_param = grad_norm / (total_trainable_params**.5)
grad_norm_target    = (microbatch_grad_norm_steps_scale * (curr_step + 1e-2)) ** microbatch_expected_grad_norm_pow
ratio_diff          = grad_norm_per_param / grad_norm_target
microbatch_steps   *= 1. + (sample_every * scale_lr * (ratio_diff - 1))   # grow if above the expected curve
microbatch_steps    = max(microbatch_steps, 1e-1)                          # floor so it can recover
base, dither_prob   = divmod(microbatch_steps, 1)
discrete_sampled_microbatch_steps = max(1, int(base + torch.bernoulli(torch.tensor(dither_prob)).item()))
```

It keeps the Bernoulli dithering from before — the fractional `microbatch_steps` is the smooth accumulator,
each step uses an integer — but now the target is a principled, model-size-aware power-law trajectory for the
grad norm instead of a fixed local delta. I only sample the grad norm every few steps because computing it
isn't free, and the controller is smooth enough to tolerate that.

This is the closing move, so let me set the bar. The standing record going into this rung is ~136–140
seconds on the A100. The bet is that fusing attention and the MLP into one block — one norm, one
up-projection, one down-projection, one residual add per layer, with the gated nonlinear features doing
double duty as the attention values — cuts the per-step op count and memory traffic substantially, while the
expected-grad-norm power-law microbatch controller schedules the effective batch off a principled trajectory
instead of a blind local target, spending compute where the run actually needs it. Together I expect that to
drop the wall-clock *under 100 seconds* to ~3.8 val loss on a single 40 GB A100. The risks are real: the
headless single-head attention and the tight `expand`/`project` initializations could cost stability, and
the power-law controller could mis-track if the real grad-norm decay departs from the assumed exponent — the
`1/num_blocks` project-init scaling and the `max(·, 1e-1)` microbatch floor are the hedges. If the fused
block holds quality and the controller tracks, this is the fastest known way to train a small GPT to ~3.8
val loss on one A100, and that is where the ladder ends.
