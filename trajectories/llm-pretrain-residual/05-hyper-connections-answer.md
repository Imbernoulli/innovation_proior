**Problem (from step 4).** Block AttnRes (val_loss 2.2544) was the largest jump on the ladder, but to
fit the budget it *coarsened* the depth axis — rigid unit-weight residuals inside each block of 4, dynamic
routing only at 6 block boundaries — so the fine scale is back to the accumulator it was fighting. And
every rung shares one root flaw: a *single* residual stream must serve two conflicting jobs at once — be
a clean gradient highway (so shallow layers get gradient) *and* carry each deep layer's output strongly
(so deep features stay distinct). One coefficient cannot do both — the Pre-Norm-vs-Post-Norm seesaw. No
rung gave the network more than one stream.

**Key idea.** Hyper-Connections: widen the residual stream to `n` parallel copies (`H` of shape
`(B,T,n,D)`) and route them with a small `(n+1)×(n+1)` matrix per residual *site* (attention and MLP are
two sites/layer). The matrix has **depth connections** (`β`: write the sublayer output into the copies;
`diag(A_r)`: each copy's carry) and **width connections** (`A_m`: mix copies into the sublayer input;
`A_r`: recombine copies). `n>1` decouples the two demands — one copy a clean highway, another a
strongly-written distinct output — so it *breaks* the seesaw (provably fails at `n=1`). Make it
input-dependent (DHC): static base + `s·tanh(Norm(H)·W)` correction. The `n` rows are summed before
`ln_f`; the head/loss are untouched.

**Why it works.** Different copies hold different depth-connection patterns simultaneously (the
single-stream conflict disappears). The matrix also spans sequential↔parallel layer arrangements, so it
learns a soft/dynamic arrangement. Norm-before-projection keeps the depth-growing stream scale out of the
router; tanh bounds the correction; small init scale makes the net earn its way off the static matrix.

**Initialization (= exactly Pre-Norm at step 0).** Dynamic projections **zero** → `tanh(0)=0` → pure
static at init. Static matrix for site `k`: `β = 1_{1×n}` (full output into every copy), `A_r = I_n`
(each copy a clean identity highway), `A_m = e_{k mod n}` (read from one copy, round-robin). Summing rows
is then bit-for-bit Pre-Norm — so unlike Block AttnRes (which started at a uniform block-average), this
starts at the vanilla operating point and only improves.

**Hyperparameters / optimizer.** Expansion rate `n = 2` for this 24-layer / micro-batch-32 / 2-GPU
budget (where Block AttnRes already had to coarsen) — the point the seesaw breaks; `n = 4` is the reach
if memory allows (`n = 1` is worse than baseline, never use it). Dynamic scales init `0.01`, tanh on.
Overhead: static `O(n²)`, dynamic `O(nd)`/site, width matmul `O(n²d)`/token, memory `O(n·s·b·d)`/layer —
negligible vs attention/FFN. HC params (gains, not weight matrices) go to a **no-decay** group; LR
schedule and `CONFIG_OVERRIDES` unchanged.

**Bar to clear / what to validate (no feedback — this is the endpoint).** Beat 2.2544 val_loss
(WikiText-2 41.82, LAMBADA 64.32, ARC-Easy 55.01, HellaSwag 34.05). The falsifiable test of the
seesaw-break: adjacent-layer feature cosine similarity should *drop* vs Pre-Norm. Deep-layer-sensitive
metrics should move most (LAMBADA < 64.32, WikiText-2 < 41.82). Risks: at `n=2` the gain over Block
AttnRes may be narrow; HC's reported edge grows over long training, and 13.5k steps is short, so the
measured margin may be smaller than large-scale reports — the test is a clean beat of 2.2544 plus a bent
similarity curve, not a 1.8× speedup.

```python
# EDITABLE regions of custom_pretrain.py — finale: Hyper-Connections (DHC, expansion rate n)
# Faithful to the primary source's Algorithm 2/3, re-expressed in the nanoGPT scaffold.
# CausalSelfAttention, MLP, LayerNorm, GPTConfig are FIXED; HC drives the sublayers from
# GPT.forward over an n-stream tensor H of shape (B, T, n, D).

# Block: unchanged container (ln_1, attn, ln_2, mlp); its forward is not called.

class HyperConnection(nn.Module):
    """One residual site. h: (B, T, n, D). Static matrix = Pre-Norm-on-n-copies base;
    dynamic projections zero-init -> exactly Pre-Norm at init."""

    def __init__(self, dim, rate, site_id, dynamic=True):
        super().__init__()
        self.rate = rate
        self.dynamic = dynamic
        # static_beta = B = 1_{1xn} (write full output into every copy)
        self.static_beta = nn.Parameter(torch.ones(rate))
        # static_alpha = [A_m | A_r] = [e_{site_id mod n} | I_n]
        init_alpha0 = torch.zeros(rate, 1)
        init_alpha0[site_id % rate, 0] = 1.0
        self.static_alpha = nn.Parameter(torch.cat([init_alpha0, torch.eye(rate)], dim=1))  # (n, n+1)
        if dynamic:
            self.dynamic_alpha_fn = nn.Parameter(torch.zeros(dim, rate + 1))
            self.dynamic_alpha_scale = nn.Parameter(torch.ones(1) * 0.01)
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(dim))
            self.dynamic_beta_scale = nn.Parameter(torch.ones(1) * 0.01)
            self.layer_norm = LayerNorm(dim, bias=False)

    def width_connection(self, h):                              # h: (B, T, n, D)
        if self.dynamic:
            norm_h = self.layer_norm(h)
            wc = torch.tanh(norm_h @ self.dynamic_alpha_fn) * self.dynamic_alpha_scale
            alpha = wc + self.static_alpha                      # (B, T, n, n+1)
            dc = torch.tanh(norm_h @ self.dynamic_beta_fn) * self.dynamic_beta_scale
            beta = dc + self.static_beta                        # (B, T, n)
        else:
            alpha = self.static_alpha[None, None, ...]          # broadcast over (B, T)
            beta = self.static_beta[None, None, ...]
        mix_h = alpha.transpose(-1, -2) @ h                     # (B, T, n+1, D)
        return mix_h, beta

    def depth_connection(self, mix_h, h_o, beta):              # h_o: (B, T, D)
        return torch.einsum('btd,btn->btnd', h_o, beta) + mix_h[..., 1:, :]


class GPT(nn.Module):
    def _init_hc(self, config):  # GPT.__init__ residual region:
        # ── Hyper-Connections: n parallel residual streams ──
        self.hc_rate = 2  # expansion rate n (2 breaks the seesaw on this budget; 1 is worse than baseline)
        self.hc = nn.ModuleList([
            HyperConnection(config.n_embd, self.hc_rate, site_id=k, dynamic=True)
            for k in range(2 * config.n_layer)  # two sites (attn, mlp) per layer
        ])

    def _forward_block_loop(self, x):  # GPT.forward block loop:
        # ── Hyper-Connections: lift x into n copies, route per site, sum at the top ──
        H = x.unsqueeze(-2).expand(-1, -1, self.hc_rate, -1).contiguous()  # (B, T, n, D)
        s = 0
        for block in self.transformer.h:
            # attention site
            mix_h, beta = self.hc[s].width_connection(H)
            h0 = block.attn(block.ln_1(mix_h[..., 0, :]))
            H = self.hc[s].depth_connection(mix_h, h0, beta)
            s += 1
            # MLP site
            mix_h, beta = self.hc[s].width_connection(H)
            h0 = block.mlp(block.ln_2(mix_h[..., 0, :]))
            H = self.hc[s].depth_connection(mix_h, h0, beta)
            s += 1
        x = H.sum(dim=-2)   # sum the n streams -> single vector for ln_f + head
        return x

    # GPT.configure_optimizers (HC params -> no-decay group):
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        hc_ids = {id(p) for p in self.hc.parameters()}
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2 and id(p) not in hc_ids]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2 and id(p) not in hc_ids]
        hc_params = [p for n, p in param_dict.items() if id(p) in hc_ids]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
            {'params': hc_params, 'weight_decay': 0.0},
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

# CONFIG_OVERRIDES = {}   (no override).
```
