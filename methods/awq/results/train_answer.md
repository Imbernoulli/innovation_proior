On-device batch-1 LLM inference is memory-bound: the arithmetic intensity is roughly one FLOP per byte, so the only way to speed up generation is to move less weight memory. Weight-only quantization to 4 or even 3 bits (W4A16 / W3A16, with FP16 activations) cuts weight traffic about fourfold and is the natural lever. The obstacle is accuracy. Round-to-nearest at 3-4 bits degrades perplexity sharply because it treats every weight as equally important and lets a single outlier inflate the step size for an entire group. We need a post-training quantizer that recovers near-FP16 accuracy without retraining, without backpropagation, without per-layer reconstruction regression, and without introducing a hardware-unfriendly mixed-precision layout.

The existing baselines each miss one of those requirements. Round-to-nearest is hardware-friendly but accuracy-poor. GPTQ improves accuracy by minimizing layer output error through a Hessian-based column sweep, yet it still performs a calibration-set regression, can overfit, and its dense inverse and column sweep are expensive. A mixed-precision oracle that keeps the most salient 1% of weight channels in FP16 does recover accuracy, but the resulting FP16/INT tensor is irregular and difficult to deploy efficiently. The real question is therefore how to protect the salient channels while keeping every stored weight on the same low-bit grid.

The method is AWQ, Activation-aware Weight Quantization. Its first observation is that saliency is determined by activations, not by weight magnitude. Diagnostic experiments show that keeping the weight channels that multiply the largest-magnitude input features in FP16 recovers accuracy, while keeping channels selected by weight magnitude barely helps. Intuitively, an input feature with large magnitude contributes heavily to the output, so the weights processing it deserve finer quantization resolution. AWQ protects those channels with an equivalence transform rather than a different data type.

For a linear layer WX, apply a per-channel scaling s to the weights and the inverse scaling to the activations: WX = (W diag(s)) (diag(s)^-1 X). This is exact before quantization. After group-wise quantization, a salient channel scaled by s > 1 receives Q(ws)(x/s). Compared with the unscaled error Q(w)x, the compensated error shrinks by roughly 1/s as long as the group step Δ does not grow. The transform therefore gives salient channels finer effective resolution at no extra bits. However, if s is too large the scaled weight becomes the group maximum, Δ grows, and every non-salient weight in that group suffers amplified error. The scale must balance protection of salient channels against harm to ordinary ones.

AWQ captures that balance with a one-parameter search. From a small calibration pass it computes the per-input-channel average activation magnitude s_X. It then considers scales s = s_X^α and searches a scalar α in [0, 1]. For each candidate it scales the weights, applies ordinary group-wise INT3/INT4 quantization, undoes the scale in the stored weight, and scores the real output MSE against the FP16 reference. The best α is chosen with no gradients and no Hessian. An additional per-group clip search trims the worst outliers by shrinking each group's maximum value before rounding. The final stored model remains a regular group-wise low-bit weight tensor, so it packs cleanly into hardware-aligned kernels.

This satisfies all the constraints: no retraining, no backpropagation, no second-order reconstruction, minimal dependence on calibration data, and a uniform hardware-friendly layout. The activation statistic only decides which channels need protection; the equivalence transform supplies that protection without leaving the integer grid.

```python
import torch
import torch.nn as nn

def quantize_tensor(x, scale, zero_point, qmin, qmax):
    return torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)

def dequantize_tensor(x_int, scale, zero_point):
    return (x_int - zero_point) * scale

def find_scale_zero(weight, num_bits=4, group_size=-1, symmetric=True):
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
            w_min = w_groups.amin(dim=1, keepdim=True)
            w_max = w_groups.amax(dim=1, keepdim=True)
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


class AWQLayerQuantizer:
    """Activation-aware weight quantizer: per-channel scale search + per-group clip search."""

    N_ALPHA = 20
    N_CLIP_GRID = 20
    CLIP_MAX_SHRINK = 0.5
    N_SAMPLE_TOKEN = 256

    def __init__(self, layer, num_bits=4, group_size=-1):
        self.layer = layer
        self.num_bits = num_bits
        self.group_size = group_size
        self.out_features, self.in_features = layer.weight.shape
        self.dev = layer.weight.device
        self.nsamples = 0
        self.act_sum = torch.zeros(self.in_features, device=self.dev, dtype=torch.float32)
        self._x_buf = []
        self._x_buf_rows = 0

    def add_batch(self, inp):
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        inp_f = inp.float()
        n = inp_f.shape[0]
        self.act_sum += inp_f.abs().sum(dim=0)
        self.nsamples += n
        cap = self.N_SAMPLE_TOKEN * 4
        if self._x_buf_rows < cap:
            take = min(n, cap - self._x_buf_rows)
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
        W = self.layer.weight.data.clone().float()
        num_bits = self.num_bits
        group_size = self.group_size
        qmin = -(1 << (num_bits - 1))
        qmax = (1 << (num_bits - 1)) - 1

        x_max = (self.act_sum / max(self.nsamples, 1)).clamp(min=1e-5)
        X = self._get_x_samples()

        # Auto-scale search.
        best_err = float('inf')
        best_s = torch.ones(self.in_features, device=self.dev)
        for i in range(self.N_ALPHA):
            ratio = i / self.N_ALPHA
            s = x_max.pow(ratio).clamp(min=1e-4)
            s = s / (s.max() * s.min()).sqrt().clamp(min=1e-5)
            W_scaled = W * s.unsqueeze(0)
            scale_q, zp, _, _ = find_scale_zero(W_scaled, num_bits, group_size, symmetric=True)
            W_q = quantize_tensor(W_scaled, scale_q, zp, qmin, qmax)
            W_dq = dequantize_tensor(W_q, scale_q, zp)
            W_final = W_dq / s.unsqueeze(0)
            if X is not None:
                delta = (W - W_final).to(X.dtype)
                err = (X @ delta.T).pow(2).mean().item()
            else:
                err = (W - W_final).pow(2).mul(x_max.unsqueeze(0).pow(2)).sum().item()
            if err < best_err:
                best_err = err
                best_s = s.clone()

        W_scaled = W * best_s.unsqueeze(0)

        # Auto-clip search.
        if group_size > 0:
            n_groups = self.in_features // group_size
            gs = group_size
        else:
            n_groups = 1
            gs = self.in_features

        W_groups = W_scaled.reshape(self.out_features, n_groups, gs)
        base_max = W_groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        best_max = base_max.clone()

        if X is not None:
            X_groups = X.reshape(X.shape[0], n_groups, gs)
            n_clip_iters = max(1, int(self.CLIP_MAX_SHRINK * self.N_CLIP_GRID))
            for i_b in range(0, self.out_features, 64):
                W_b = W_groups[i_b:i_b + 64]
                base_max_b = base_max[i_b:i_b + 64]
                org_out = torch.einsum('rgc,tgc->rtg', W_b, X_groups.float())
                min_errs = torch.full_like(base_max_b, float('inf'))
                best_max_b = base_max_b.clone()
                for i_s in range(n_clip_iters):
                    cur_max = base_max_b * (1 - i_s / self.N_CLIP_GRID)
                    cur_w = torch.clamp(W_b, -cur_max, cur_max)
                    scale_b = (cur_max / qmax).clamp(min=1e-12)
                    q_w = torch.clamp(torch.round(cur_w / scale_b), qmin, qmax) * scale_b
                    cur_out = torch.einsum('rgc,tgc->rtg', q_w, X_groups.float())
                    err_b = (cur_out - org_out).pow(2).mean(dim=1, keepdim=True).permute(0, 2, 1)
                    mask = err_b < min_errs
                    min_errs = torch.where(mask, err_b, min_errs)
                    best_max_b = torch.where(mask, cur_max, best_max_b)
                best_max[i_b:i_b + 64] = best_max_b

        scale_g = (best_max / qmax).clamp(min=1e-12)
        scale_q = scale_g.expand_as(W_groups).reshape(self.out_features, self.in_features)
        zp = torch.zeros_like(scale_q)
        W_clamped = torch.clamp(
            W_scaled,
            -best_max.expand_as(W_groups).reshape(self.out_features, self.in_features),
            best_max.expand_as(W_groups).reshape(self.out_features, self.in_features),
        )
        W_q = quantize_tensor(W_clamped, scale_q, zp, qmin, qmax)
        W_dq = dequantize_tensor(W_q, scale_q, zp)
        W_final = W_dq / best_s.unsqueeze(0)
        return W_final.to(self.layer.weight.dtype)
```
