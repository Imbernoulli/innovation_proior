**Problem (from rung 4).** At ~136–140 s, time is now lost in two places. (1) Every layer runs attention
and the MLP as two separate sublayers — two norms, two up-projections, two down-projections, two residual
adds per layer — a lot of small ops and memory round-trips for two operations that share almost all their
structure. (2) The earlier microbatch controller steers off a *local, model-blind* signal (per-step
grad-norm jitter vs a fixed target), so it doesn't know where in the run it is or how big the model is.

**Key idea.** (1) **LatentAttentionBlock** — fuse attention and the gated MLP into one block: one shared
LayerNorm, one `expand` up-projection split into query / key / two gated-MLP slices, a GeGLU
(`linear ⊙ GELU(pre_gelu)`) whose features do double duty — part stays local (the MLP path) and part
*becomes the attention values* — single-head `F.scaled_dot_product_attention`, then one `project` back and
one residual add. Linear (cheap) keys/queries decide *where* to attend; nonlinear gated features decide
*what* is mixed. Per-layer learnable position-bias slope carries the length-agnostic linear bias forward.
(2) **Expected-grad-norm power-law microbatch scheduler** — posit `grad_norm_target = (scale·step)^pow`
with model-size-dependent `pow = −0.677·log(params)^(−0.2)` and `scale = log(params)·params`; grow the
(dithered, fractional) microbatch count when the size-normalized measured grad norm runs above the expected
curve, shrink it when below.

**Why it works.** Fusing the two sublayers collapses per-layer ops and memory traffic (the gated features
serve as both MLP output and attention values, and single-head attention skips `.contiguous()` reshapes),
cutting per-step cost; the power-law controller schedules the effective batch off a principled,
size-and-run-aware grad-norm trajectory rather than a blind local delta, spending compute where the run
needs it. Together they drop the wall-clock under 100 seconds at the held ~3.8 val-loss bar.

**Change / code.** The fused block and the power-law microbatch update.

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

**Target.** Reach ~3.8 val loss on a single 40 GB A100 in under 100 seconds — the fastest known training of
a small GPT to this bar on this hardware.
