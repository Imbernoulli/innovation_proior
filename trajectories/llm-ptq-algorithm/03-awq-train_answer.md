GPTQ half-confirmed the bet and half-redirected me. At INT3-g128 it cut RTN's $6.7341$ blow-up to $6.1011$ (degradation $1.1940$) — error compensation gave the deferred residual somewhere to go — and at INT4 it improved to $5.0711$ (g128) and $5.0435$ (g64). But two things bother me. INT3 is still an open wound: $1.1940$ degradation is an order of magnitude worse than INT4's $0.16$, so the 8-level grid is plainly not solved; compensation softened the blow without removing the cause. And GPTQ paid roughly $220$s per setting against RTN's $\sim22$s, a cost that is structural — the dense inverse and the sequential column sweep are intrinsic to error feedback. If compensation does not fully recover INT3, the bottleneck at 8 levels may be the grid *coarseness* itself rather than the rounding *objective*, which says: attack the rounding before it happens — protect the weights that matter — without an inverse Hessian at all.

The method is AWQ, activation-aware weight quantization (Lin et al. 2024). It starts from a question GPTQ never asks: are all weights equally important to the output? The diagnostic is to quantize a layer to INT3 but keep a small fraction of weight channels in FP16. Keeping order $1\%$ recovers most of the lost accuracy, so a small salient set does the heavy lifting. The decisive twist is *which* $1\%$: selecting by weight magnitude barely beats keeping a random $1\%$, but selecting the channels that multiply the largest-magnitude *input* features gives a large recovery. So saliency is set by activation magnitude — the same structure GPTQ's $\mathbf X^\top\mathbf X$ encoded, but now needing only a per-channel vector, not a matrix and its inverse.

Keeping salient channels in FP16 would mean a mixed-precision tensor — a hardware nightmare of irregular layout and custom kernels. The escape is to protect those weights *at fixed bit-width* with an equivalence transform. For a group with output $y=\mathbf w x$ quantized symmetrically, scale one high-activation channel's weights by $s>1$ before rounding and divide the matching activation by $s$ after. Before rounding this is exact, $\mathbf W\mathbf x=(\mathbf W\,\mathrm{diag}(\mathbf s))(\mathrm{diag}(\mathbf s)^{-1}\mathbf x)$. After rounding, that element's error becomes $\Delta'\cdot\mathrm{RoundErr}(ws/\Delta')\cdot|x|/s$ against the original $\Delta\cdot\mathrm{RoundErr}\cdot|x|$; the expected rounding residual is unchanged ($\sim0.25$ of a step either way), so if scaling a small fraction of channels by a modest $s$ leaves the group max — and hence $\Delta'$ — roughly fixed, the error ratio is $\approx 1/s$. The salient channel gets effectively finer resolution, with no FP16 side table and no bit-width change.

The trade-off is the whole game, and it is why $s$ cannot just be cranked up. The $1/s$ shrink applies only to the protected channel; every *non-salient* weight in the same group has error proportional to $\Delta'$. Push $s$ high enough that the scaled salient weight becomes the group maximum and $\Delta'$ grows, and $\Delta'/\Delta>1$ amplifies everyone else — protecting the salient weights quietly damages the bulk RTN was already starving. The honest objective is therefore to pick the whole per-channel scale vector $\mathbf s$ to minimize the layer's *quantized output error*, accounting for both sides. But $Q$ contains a non-differentiable round, and I refuse to reintroduce gradient optimization. I do not need a free search over $\mathbb R^{C_{in}}$: the one thing that sets saliency is the per-channel activation magnitude, so I take a one-parameter family. Let $\mathbf x_{\max}$ be the per-input-channel *average* $|X|$ from calibration — average, not max, to capture typical channel importance robustly to a few outliers — and define

$$\mathbf s=\mathbf x_{\max}^{\alpha},\qquad \alpha\in[0,1).$$

At $\alpha=0$, $\mathbf s=\mathbf 1$: plain RTN. As $\alpha\to1$ the scaling tracks activation magnitude most aggressively: maximum protection, maximum risk of inflating $\Delta$. Because it is one scalar I sweep it over `N_ALPHA = 20` ratios, and for each candidate I scale $\mathbf W$, quantize, undo the scale, and measure the *actual* output error directly — no gradient. To keep scales tame I normalize $\mathbf s$ by the geometric mean of its max and min before applying.

This task's edit surface shapes the rest, and I want the code that runs. The harness hands me one linear layer at a time, with no adjacent operator to absorb $\mathrm{diag}(\mathbf s)^{-1}$ into — so unlike the general form, which migrates the inverse scale onto the previous LayerNorm or linear, I realize the equivalence *in-weight*: scale $\mathbf W$ along input channels, quantize, dequantize, and divide the dequantized weight back by $\mathbf s$ (`W_dq / s`). The layer's effective output is the dequantized-and-unscaled $\mathbf W$, self-contained, same shape and dtype. The loss each $\alpha$ is scored against is therefore per-*linear*: `add_batch` accumulates the per-channel sum of $|X|$ (averaged to $\mathbf x_{\max}$ at quantize-time) and reservoir-samples real input tokens (stride-sampled to a few hundred rows, kept on CPU across layers, moved to device at quantize-time), and I score $\alpha$ by $\lVert X(\mathbf W-\mathbf W_{\text{final}})^\top\rVert^2$ over those samples — the true per-layer output MSE on real activations. The grid is symmetric here ($q_{\min}=-2^{b-1}$, $q_{\max}=2^{b-1}-1$, zero point $0$), the same convention RTN and GPTQ used, so the $1/s$ argument is the symmetric-step version and the whole comparison stays on one grid; this is per-linear symmetric scale-search undone in-weight, not the asymmetric or cross-layer-migration form.

The bulk-damage trade-off motivates a second knob. After the per-channel scale is fixed, the group step is still set by the post-scale group max, which a lone outlier can drag up. So I add a per-group clip search on the scaled weights: shrink each group's max by $1-i/N$ for $i$ up to `CLIP_MAX_SHRINK` $\cdot$ `N_CLIP_GRID` ($0.5\cdot20$, half the grid), and for each candidate clip the weights to $\pm\text{max}$, quantize, and measure the per-(output-channel, group) output error against the unclipped scaled weights on the same sampled $X$, via einsums $\sum_c W_{r,g,c}X_{t,g,c}$ for the original and the quantized-clipped version, choosing per group the clip that minimizes it. Clipping trades a little extra rounding error on the outlier for a tighter $\Delta$ on the bulk — the same outlier-versus-bulk balance, now resolved per group. It is batched over output channels (`OC_BATCH`) for memory; the final quantization then uses the clipped per-group scales and undoes the channel scale.

So the delta from GPTQ is a different philosophy on the same grid: GPTQ rounds then *compensates* the residual through an inverse Hessian; AWQ *protects* salient weights before rounding with a per-channel scale, choosing that scale and a per-group clip by directly minimizing the real per-linear output MSE on sampled activations, and never forms a Hessian or an inverse — a vector and two cheap searches, not a matrix and a solve. The decisive test is INT3-g128: GPTQ left it at $6.1011$ (degradation $1.1940$), and if salience-aware scaling plus clipping works, INT3 should drop clearly below it into the high-5s. At INT4-g128 I expect a real but smaller gain under $5.0711$; at INT4-g64 the mechanisms overlap most (fine grouping already protects the bulk), so AWQ may not beat GPTQ's $5.0435$ there. And with no inverse and no column sweep, quant-time should fall well under GPTQ's $\sim220$s.

```python
# EDITABLE region of gptq/custom_ptq.py (lines 26-157) — step 3: AWQ (activation-aware scaling + clip)

def quantize_tensor(x, scale, zero_point, qmin, qmax):
    """Quantize a float tensor to integers given scale and zero point."""
    x_int = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
    return x_int


def dequantize_tensor(x_int, scale, zero_point):
    """Dequantize integer tensor back to float."""
    return (x_int - zero_point) * scale


def find_scale_zero(weight, num_bits=4, group_size=-1, symmetric=True):
    """Compute per-channel (or per-group) quantization parameters."""
    qmin = -(1 << (num_bits - 1))
    qmax = (1 << (num_bits - 1)) - 1

    if group_size > 0:
        out_features, in_features = weight.shape
        assert in_features % group_size == 0
        w_groups = weight.reshape(out_features, -1, group_size)
        if symmetric:
            w_max = w_groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
            scale = w_max / qmax
            zero_point = torch.zeros_like(scale)
        else:
            w_min = w_groups.amin(dim=-1, keepdim=True)
            w_max = w_groups.amax(dim=-1, keepdim=True)
            w_range = (w_max - w_min).clamp(min=1e-12)
            scale = w_range / (qmax - qmin)
            zero_point = torch.round(qmin - w_min / scale)
        scale = scale.reshape(out_features, -1).repeat_interleave(group_size, dim=1)
        zero_point = zero_point.reshape(out_features, -1).repeat_interleave(group_size, dim=1)
    else:
        if symmetric:
            w_max = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
            scale = w_max / qmax
            zero_point = torch.zeros_like(scale)
        else:
            w_min = weight.amin(dim=1, keepdim=True)
            w_max = weight.amax(dim=1, keepdim=True)
            w_range = (w_max - w_min).clamp(min=1e-12)
            scale = w_range / (qmax - qmin)
            zero_point = torch.round(qmin - w_min / scale)

    return scale, zero_point, qmin, qmax


class LayerQuantizer:
    """AWQ quantizer -- faithful to mit-han-lab/llm-awq.

    Pipeline:
      1. add_batch: accumulate per-channel mean |X|; reservoir-sample raw input
         tokens (up to N_SAMPLE_TOKEN rows) so we can use real activations as
         the loss signal during search.
      2. Per-channel scale alpha-search (auto_scale): for ratio in [0, 1):
             s = x_max^ratio  (clamped, range-normalized: s /= sqrt(max*min))
         loss = mean((X @ (W - W_final).T)^2)  on sampled X.
      3. Per-group max clip-search (auto_clip), on the post-scale weights:
         clip per-group max by 1 - i/N for i in 0..MAX_SHRINK*N, loss is
         per-(out_channel, group) output-error using sampled X:
             org_out[r, t, g] = sum_c W_scaled[r,g,c] * X[t,g,c]
             cur_out[r, t, g] = sum_c Q(clamp(W,±M))[r,g,c] * X[t,g,c]
             err[r, g] = mean_t (cur_out - org_out)^2
      4. Quantize with the clipped per-group scales, undo channel scaling.

    Implemented to fit the LayerQuantizer interface (per-linear, no block ctx),
    so the loss is computed at linear-layer granularity (not full block).
    """

    N_ALPHA = 20             # auto_scale grid size
    N_CLIP_GRID = 20         # auto_clip n_grid
    CLIP_MAX_SHRINK = 0.5    # auto_clip max_shrink (official default)
    N_SAMPLE_TOKEN = 256     # number of input tokens kept for loss computation
    OC_BATCH = 256           # output-channel batching for clip search (memory)

    def __init__(self, layer, num_bits=4, group_size=-1):
        self.layer = layer
        self.num_bits = num_bits
        self.group_size = group_size
        self.out_features, self.in_features = layer.weight.shape
        self.dev = layer.weight.device
        self.nsamples = 0

        # Per-channel sum of |activation| (averaged over tokens at quantize-time)
        self.act_sum = torch.zeros(
            self.in_features, device=self.dev, dtype=torch.float32
        )
        # Reservoir of input tokens (CPU to save GPU memory across layers)
        self._x_buf = []
        self._x_buf_rows = 0
        # Keep H for interface compatibility (unused by AWQ)
        self.H = torch.zeros(
            (self.in_features, self.in_features),
            device=self.dev, dtype=torch.float32
        )

    def add_batch(self, inp):
        """Accumulate per-channel |X| stats and reservoir-sample raw inputs."""
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        inp_f = inp.float()
        n = inp_f.shape[0]
        self.act_sum += inp_f.abs().sum(dim=0)
        self.nsamples += n
        # Keep ~4x N_SAMPLE_TOKEN candidate rows; we'll stride-sample at quantize.
        cap = self.N_SAMPLE_TOKEN * 4
        if self._x_buf_rows < cap:
            take = min(n, cap - self._x_buf_rows)
            # Take an evenly-spaced stride from this batch
            stride = max(1, n // max(take, 1))
            sampled = inp_f[::stride][:take].detach().to('cpu')
            self._x_buf.append(sampled)
            self._x_buf_rows += sampled.shape[0]

    def _get_x_samples(self):
        if not self._x_buf:
            return None
        X = torch.cat(self._x_buf, dim=0)
        if X.shape[0] > self.N_SAMPLE_TOKEN:
            stride = X.shape[0] // self.N_SAMPLE_TOKEN
            X = X[::stride][:self.N_SAMPLE_TOKEN]
        return X.to(self.dev)

    def quantize(self):
        """AWQ: per-channel scale search + per-group clip search + quantize."""
        W = self.layer.weight.data.clone().float()
        num_bits = self.num_bits
        group_size = self.group_size
        qmin = -(1 << (num_bits - 1))
        qmax = (1 << (num_bits - 1)) - 1

        if self.nsamples > 0:
            x_max = (self.act_sum / self.nsamples).clamp(min=1e-5)
        else:
            x_max = torch.ones(self.in_features, device=self.dev)

        X = self._get_x_samples()  # (T, in_features) on dev, may be None

        # ── (1) auto_scale: per-channel scale search ─────────────────────────
        best_err = float('inf')
        best_s = torch.ones(self.in_features, device=self.dev)

        for i in range(self.N_ALPHA):
            ratio = i / self.N_ALPHA
            s = x_max.pow(ratio).clamp(min=1e-4)
            s = s / (s.max() * s.min()).sqrt().clamp(min=1e-5)

            W_scaled = W * s.unsqueeze(0)
            scale_q, zp, _, _ = find_scale_zero(
                W_scaled, num_bits=num_bits, group_size=group_size, symmetric=True
            )
            W_q = quantize_tensor(W_scaled, scale_q, zp, qmin, qmax)
            W_dq = dequantize_tensor(W_q, scale_q, zp)
            W_final = W_dq / s.unsqueeze(0)

            if X is not None:
                # Output-error: ||X @ (W - W_final).T||^2 / (T * out)
                delta = (W - W_final).to(X.dtype)
                err = (X @ delta.T).pow(2).mean().item()
            else:
                err = (W - W_final).pow(2).mul(x_max.unsqueeze(0).pow(2)).sum().item()

            if err < best_err:
                best_err = err
                best_s = s.clone()

        # Apply best per-channel scaling
        W_scaled = W * best_s.unsqueeze(0)

        # ── (2) auto_clip: per-group max clip search ─────────────────────────
        if group_size > 0:
            n_groups = self.in_features // group_size
            gs = group_size
        else:
            n_groups = 1
            gs = self.in_features

        W_groups = W_scaled.reshape(self.out_features, n_groups, gs)  # (O, G, gs)
        base_max = W_groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        best_max = base_max.clone()

        if X is not None:
            X_groups = X.reshape(X.shape[0], n_groups, gs)  # (T, G, gs)

            n_clip_iters = max(1, int(self.CLIP_MAX_SHRINK * self.N_CLIP_GRID))
            oc_batch = self.OC_BATCH
            if self.out_features % oc_batch != 0:
                # fall back to a divisor of out_features
                for cand in (128, 64, 32, 16, 8, 4, 2, 1):
                    if self.out_features % cand == 0:
                        oc_batch = cand
                        break

            for i_b in range(0, self.out_features, oc_batch):
                W_b = W_groups[i_b:i_b + oc_batch]                # (B, G, gs)
                base_max_b = base_max[i_b:i_b + oc_batch]          # (B, G, 1)
                # org_out[r, t, g] = sum_c W_b[r,g,c] * X_groups[t,g,c]
                org_out = torch.einsum('rgc,tgc->rtg', W_b, X_groups.float())
                min_errs = torch.full_like(base_max_b, float('inf'))
                best_max_b = base_max_b.clone()
                for i_s in range(n_clip_iters):
                    cur_max = base_max_b * (1 - i_s / self.N_CLIP_GRID)  # (B, G, 1)
                    cur_w = torch.clamp(W_b, -cur_max, cur_max)
                    scale_b = (cur_max / qmax).clamp(min=1e-12)
                    q_w = (
                        torch.clamp(torch.round(cur_w / scale_b), qmin, qmax) * scale_b
                    )
                    cur_out = torch.einsum('rgc,tgc->rtg', q_w, X_groups.float())
                    err_b = (cur_out - org_out).pow(2).mean(dim=1, keepdim=True)
                    err_b = err_b.permute(0, 2, 1).contiguous()  # (B, G, 1)
                    mask = err_b < min_errs
                    min_errs = torch.where(mask, err_b, min_errs)
                    best_max_b = torch.where(mask, cur_max, best_max_b)
                best_max[i_b:i_b + oc_batch] = best_max_b
                del org_out, cur_out, q_w, cur_w
            del X_groups
        # else: no calibration samples — fall back to base_max (no clipping)

        # ── (3) Final quantization with clipped scales ───────────────────────
        scale_g = (best_max / qmax).clamp(min=1e-12)
        scale_q = scale_g.expand_as(W_groups).reshape(self.out_features, self.in_features)
        zp = torch.zeros_like(scale_q)

        # Clamp scaled weights to the searched per-group range, then quantize
        W_clamped = torch.clamp(
            W_scaled,
            -best_max.expand_as(W_groups).reshape(self.out_features, self.in_features),
            best_max.expand_as(W_groups).reshape(self.out_features, self.in_features),
        )
        W_q = quantize_tensor(W_clamped, scale_q, zp, qmin, qmax)
        W_dq = dequantize_tensor(W_q, scale_q, zp)
        W_final = W_dq / best_s.unsqueeze(0)

        return W_final.to(self.layer.weight.dtype)

    def free(self):
        """Release calibration buffers."""
        del self.H
        del self.act_sum
        del self._x_buf
        self.H = None
        self.act_sum = None
        self._x_buf = None
```
