Block AttnRes was the strongest rung — validation loss 2.2544, the first sub-2.26 result, with WikiText-2 down to 41.82 and LAMBADA to 64.32, finally below both prores and learned-scaling — so the jump from a rank-one scalar mix to a content-dependent full-rank mix over depth was the largest single move on the ladder and added genuine capacity. But read what it cost. To fit a fixed 2-GPU, micro-batch-32 run it had to *coarsen* the depth axis: ordinary unweighted residuals inside each block of 4, dynamic routing only at the 6 boundaries — so inside every block I am right back to the rigid unit-weight accumulator the whole ladder has been fighting. And underneath every rung sits a deeper flaw: no matter how I weight or attend over the *single* residual stream, that stream must serve two conflicting jobs at once. Tracing Pre-Norm $h_k = h_{k-1} + T(\mathrm{Norm}(h_{k-1}))$, the stream norm climbs with depth, so a fresh layer's output is a shrinking fraction of the total and adjacent deep features collapse toward each other; the naive counter, Post-Norm, keeps each output a meaningful fraction but puts a normalization Jacobian on the highway and the gradients vanish. One coefficient governs both "how much of the past survives" (the gradient route) and "how much of this layer enters" (the depth influence), and one number cannot satisfy both. This is a seesaw, and it is *structural for a single stream*. Every scalar fix — ProRes, learned-scaling, ReZero, the depth-aware inits — slides along it without tipping it off. The structural escape is to stop having one stream.

I propose Hyper-Connections (the dynamic variant, DHC): widen the residual stream to $n$ parallel copies. Replicate the embedding into a hyper-hidden *matrix* $H \in \mathbb{R}^{n\times d}$ — here a tensor of shape $(B,T,n,D)$ — let every layer operate on the whole matrix, and sum the $n$ copies into one vector only at the very top, right before the final LayerNorm. The point of $n$ copies is decoupling: one copy can act as the clean Pre-Norm gradient highway while another carries a strongly-written, distinct deep output, so a single coefficient no longer has to serve both jobs. With $n=1$ the conflict provably persists; with $n>1$ the network can *reserve multiple patterns* of connecting to preceding layers simultaneously. That is the difference between sliding the seesaw and breaking it.

The routing rule is derived, not guessed, because the overhead must stay negligible at GPT-2 Medium scale. A layer reads $H$, forms a single input for its sublayer, runs the sublayer, and writes the result back into the $n$ copies. There are two axes of connection — *depth* (how the new output is distributed into the copies, and how each copy carries forward, the generalization of the residual skip) and *width* (how the copies mix to form the sublayer input, and how they recombine). I collect every weight into one small $(n+1)\times(n+1)$ matrix where index 0 is the sublayer-output slot and $1..n$ are the copies. Its first row $\beta$ distributes the single output back into the copies (the **depth write**); its first column $A_m$ mixes the copies into the sublayer input $h_0 = A_m^{\top} H$ (the **width read**); its $n\times n$ block $A_r$ carries the copies to the new copies $H' = A_r^{\top} H$ (the stream's own recombination). The whole layer is then: read the input via $A_m$, carry the streams via $A_r$, run the sublayer on $h_0$, and write its output back via $\beta$ plus the carry. Two structural bonuses fall out for free and make this more than a re-parameterization: the matrix *contains* Pre-Norm and Post-Norm as the non-trainable $n=1$ special cases, so I lose nothing; and at $n=2$ specific integer matrices express both the ordinary sequential residual *and* a parallel-block arrangement, so learning the matrix learns a soft, even dynamic, blend of sequential and parallel depth that the fixed residual can never reach.

I want the routing to depend on the input — the right depth/width mix surely differs token to token, the same instinct that made Block AttnRes's per-token attention beat static scalars — so I keep the static matrix as a base and *add* a small dynamic correction predicted from $H$: normalize $H$, take a linear projection, squash with $\tanh$, scale by a small learnable factor, and add to the static base, separately for $\beta$, $A_m$, $A_r$. Each piece earns its place. The norm-before-projection keeps the depth-growing stream scale out of the router — the disease I am curing must not re-enter through it, the same reason Block AttnRes RMS-normed its keys. The $\tanh$ bounds the correction so a runaway logit cannot blow up the connection weights on a 13.5k-step run I cannot afford to lose. The small init scale ($0.01$) makes the dynamic part start negligible so the network has to *earn* its way off the static matrix.

Initialization is the part I cannot get wrong, and it is the cleanest argument for trying this here: I can make the very first forward pass behave *exactly* like the Pre-Norm residual the vanilla floor already trains cleanly, so the model is never worse than baseline at step zero. Two requirements. The dynamic projections start at zero, so $\tanh(0)=0$ and the correction is exactly nothing at init — the layer begins as pure static hyper-connection. And the static matrix encodes Pre-Norm-on-$n$-copies: $\beta = \mathbf{1}_{1\times n}$ writes the full output into every copy (as Pre-Norm adds the full branch), $A_r = I_n$ carries each copy through unchanged (each stream a clean identity highway), and $A_m = e_{k \bmod n}$ is a one-hot that reads the input from a single copy, rotated by site index $k$ so the copies are used round-robin and none is privileged. With this base and zero dynamics, summing the $n$ rows at the top is *bit-for-bit Pre-Norm*. So unlike Block AttnRes — which started at a uniform block-average, a different operating point than vanilla — hyper-connections start at the vanilla operating point and bend away only as the dynamics learn, the safest possible start on this budget.

The budget decides the expansion rate. Static parameters per site are the $(n+1)\times(n+1)$ matrix, $O(n^2)$; dynamic parameters are projections of size $O(nd)$ plus two scalars and a norm, tiny against the $O(d^2)$ of attention and the MLP; the width matmul $A^{\top} H$ is a rounding error for small $n$. Memory is the real cost — $n$ streams cost $O(n\cdot s\cdot b\cdot d)$ activation per layer. The ablations say $n=4$ is where dynamic clearly beats static and $n=1$ is *worse* than baseline (a single dynamic stream has no room to reserve patterns and the dynamic noise just hurts). On this 24-layer, micro-batch-32, 2-GPU budget, where Block AttnRes already had to coarsen, I will not assume $n=4$ is free, but $n=2$ is comfortably affordable and is exactly the point where the seesaw provably breaks, so it is the honest default to land, with $n=4$ the reach if memory allows. The streams sum to one vector before `ln_f`, so the head and loss are untouched.

The fit to the edit surface is exact. Hyper-connections need two residual sites per layer (attention and MLP), each with its own static matrix, dynamic projections, and norm, operating on the $n$-stream tensor rather than the single $x$. So I keep the `Block` as the vanilla container of `ln_1, attn, ln_2, mlp` but do not call its `forward`; instead, as Block AttnRes already did, I drive the sublayers directly from the `GPT.forward` loop. In `GPT.__init__` I build $2\cdot n_{\text{layer}}$ `HyperConnection` modules (one per site, the static one-hot indexed by site id $k \bmod n$) and set the rate $n$. In the forward loop I lift the embedding into $n$ copies, then for each layer run the attention site — the width connection gives the layer input in row 0 and the carried streams in the rest, I norm row 0 with `ln_1`, run `attn`, and the depth connection writes the output back via $\beta$ plus the carry — then the same for the MLP site with `ln_2` and `mlp`. After all layers I sum the $n$ streams and pass the single vector to `ln_f` and the head. In `configure_optimizers` the new parameters (static matrices, dynamic projections, scales, site norms) are gains, not weight matrices, so — following the pattern the ladder established for leveraged routing parameters — I route them into their own no-decay group; weight decay would just pull them back toward the Pre-Norm init I deliberately chose. The LR schedule and `CONFIG_OVERRIDES` stay default.

The bar to clear is 2.2544 (WikiText-2 41.82, LAMBADA 64.32, ARC-Easy 55.01, HellaSwag 34.05). The mechanism predicts hyper-connections clear it: they add the fine-grained dynamic depth routing Block AttnRes gave up to fit the budget, they start at exactly Pre-Norm so they never pay a start-from-a-different-point tax, and they directly target the adjacent-deep-layer feature collapse that the per-block-only routing leaves untouched inside each block. The falsifiable test of the whole seesaw-break story — what distinguishes a real break from a mere reshuffle of the loss — is that the cosine similarity between adjacent layers' features should *drop* relative to Pre-Norm, the direct readout of "deep layers made distinct." On the headline metrics the deep-layer-sensitive ones should move most, LAMBADA below 64.32 and WikiText-2 below 41.82. The honest risks: at $n=2$ the seesaw breaks but the capacity gain is smaller than $n=4$ would give, so the win over Block AttnRes may be narrow; and hyper-connections' reported edge grows over long training, while 13.5k steps on 7B tokens is brief, so the measured margin may be smaller than large-scale reports — the test is a clean beat of 2.2544 plus a bent similarity curve, not a $1.8\times$ convergence speedup at this scale.

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
