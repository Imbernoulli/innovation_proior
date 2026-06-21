Learned-scaling beat both earlier rungs — validation loss 2.2680, under prores's 2.2707 and vanilla's 2.2763, with WikiText-2 finally moving ($44.11 \to 43.91$) and ARC-Easy extending to 55.85, both the "token identity preserved at depth" signature I predicted for the $x_0$ injection — and the *pattern* of the win revealed the ceiling. Every rung so far is a *scalar* knob on the depth flow. Writing any depth-flow rule as $h_l = \sum_{i<l} M_{i\to l}\, v_i$ over the sources (embedding $v_0$ and each earlier output $v_i$), the plain residual makes every causal entry of $M$ equal to 1 — an all-ones kernel, rank one. ProRes multiplied those entries by a scheduled scalar; learned-scaling by a learned scalar plus an $x_0$ term — still scalar coefficients folded through a single running stream, a low-rank semiseparable kernel. The sequence axis escaped this long ago: it stopped using separable mixing and moved to softmax attention over the whole prefix, a dense, full-rank, content-dependent mixing matrix. The thing I have never given the network is the ability to *choose which past layer's output to read*, per token, with content-dependent weighting. That is the largest move left.

I propose Block Attention Residuals: form each destination's input by softmax attention over the depth sources,

$$h_l = \sum_i \alpha_{i\to l}\, v_i, \qquad \alpha_{i\to l} = \operatorname{softmax}_i\!\big(w_l^{\top}\,\mathrm{RMSNorm}(v_i)\big),$$

turning the depth-mixing matrix from rank-one into generically full rank — each destination reading any subset of the past, differently for each token. The design choices are derived, not assumed. The *values* must be the stored representations themselves; transforming them into some new learned state would put me back to summaries of summaries, the compression I am trying to escape. The same vectors serve as *keys*, because the source content is what should make the weight vary across tokens. For the *query* I use a single learned **pseudo-query** $w_l \in \mathbb{R}^d$ per destination rather than projecting from the current hidden state — the keys are already content-dependent so the weights still vary per token, it avoids a $d\times d$ matrix per destination, and it has a structural bonus (the queries are known before their destinations execute, leaving room to batch the scoring). The keys are **RMS-normalized** — not the values — because the whole problem started with depth-dependent norm growth, and a raw $\exp(w_l^{\top} v_i)$ would let a large-norm source win on scale alone; normalizing the keys gives source *direction* at comparable scale while still mixing the *raw* representation once a source is selected. I use **softmax** rather than independent gates because I want a fixed probability budget over sources — emphasizing one source takes mass from the others, which is exactly the retrieval behavior ("which past layer do I read") the depth axis wants. And the query init is exact: random $w_l$ injects an arbitrary depth preference before training knows what the sources mean, whereas $w_l = 0$ makes every logit $\exp(0) = 1$, so the softmax is uniform — the model starts from an equal-weight average over available sources and learns deviations from that neutral prior.

Here I have to be honest about *this task's* budget, because the full "attend over every sublayer output" does not fit. A 24-layer block has two sublayers each, so attending over all sublayer outputs means up to 48 stored source tensors of shape $(B,T,D)$ and an attention at every one of the 48 destinations — $O(L^2 d)$ compute, $O(Ld)$ source memory per token. At GPT-2 Medium under bf16 with `torch.compile` on a fixed 2-GPU, micro-batch-32 run, holding 48 full activation streams and scoring against them at every sublayer does not survive. The method's own structure points at the fix: the dense source list can be *grouped* into summaries and the attention run over those — a scaling variant, not a different idea. So I partition the 24 layers into **6 blocks of 4**. *Within* a block I use ordinary residual connections — the cheap, well-conditioned default — and I run the depth-attention only at **block boundaries**, attending over the $\sim 6$ block outputs instead of the 48 sublayer outputs, roughly an $8\times$ memory reduction while keeping the one thing that matters: dynamic, content-dependent aggregation along depth. The embedding stays the first source (the one representation every later block may want to recover directly, the $x_0$ route now available to the attention itself). At each boundary after the first, a dedicated pseudo-query attends over all preceding block outputs and picks the input to the next block; the first block reads the embedding directly; after the last block a final readout query attends over all 6 block outputs plus the embedding to produce the input to `ln_f`. The parameter cost is one $d$-vector per boundary except the first ($n_{\text{blocks}}-1 = 5$) plus one output query, all zero-initialized — about $6\times 1024 \approx 6\text{k}$ parameters, negligible against 355M and far fewer than the 49 queries the full version would need. The deliberate cost of the coarsening is that I give up *sublayer*-granularity routing — I cannot have block 5 read block 2's attention output but not its MLP output — and the bet is that block-level dynamic aggregation captures most of the depth-routing benefit at a fraction of the cost.

The gradient story changes, and it is worth keeping exact. The plain additive recurrence has a unit-coefficient identity term in the backward product $\prod(I + \partial f/\partial h)$, routing gradient straight through depth. A normalized softmax mixture does not preserve that exact unit $I$; instead it gives direct, differentiable, weighted paths from the loss to every block output with nonzero attention weight, plus the score-gradient path through the keys. At zero init every block source has nonzero (uniform) weight, so at the start of training gradients spread across all earlier block outputs rather than being forced through the immediate predecessor — a different, and for deep routing arguably better, conditioning than the strict identity highway. The within-block residuals keep the clean local identity path on the fine scale; the boundary attention adds global routing on the coarse scale. There is one optimizer wrinkle, the only thing this rung changes about training: the pseudo-queries are zero-initialized and leveraged — they decide the *entire* mixing at each boundary, so a query that moves too fast can swing which block the next block reads and destabilize everything downstream. So I give the query parameters their own group at a *reduced* learning rate, $0.1\times$ the base, with no weight decay (they are not weight matrices and decay would just pull them back toward the uniform init). The main matrices stay decayed, the other 1-D parameters no-decay. That $0.1\times$ query LR is the one departure from the base schedule; the cosine schedule and `CONFIG_OVERRIDES` stay default.

In the edit surface the `Block` stays vanilla — within a block I just call `block(x)`. In `GPT.__init__` I add `attnres_block_size = 4`, compute `n_blocks = n_layer // 4`, and register `attnres_queries` of shape $(n_{\text{blocks}}-1, d)$ and `attnres_query_out` of shape $(d,)$, both zero. In the forward loop I keep a `block_outputs` list seeded with the embedding; for each block after the first I stack the previous outputs, RMS-norm the keys, score with that boundary's query, softmax over the source axis, and mix the *raw* stacked values to form the block input, then run the 4 layers with ordinary residuals and append; after the loop I do the same once more with `attnres_query_out` over all block outputs to feed `ln_f`. Against the 2.2680 learned-scaling number I expect the largest val_loss drop on the ladder — moving from a rank-one scalar mix to a full-rank content-dependent mix is the first rung that adds genuine *capacity* to the depth flow rather than just conditioning it — plausibly the first sub-2.26 number, with LAMBADA finally dropping below both 67.21 and 68.76 and WikiText-2 below 43.91. The honest risks: if sublayer-granularity routing was where the real benefit lived the coarsening captures only part of it; and the leveraged queries could destabilize if $0.1\times$ is too hot or barely move if too cold, collapsing toward a fixed block-averaging model.

```python
# EDITABLE regions of custom_pretrain.py — step 4: Block Attention Residuals
# (each region shown inside its enclosing method exactly as spliced into the file)

# Block: unchanged — vanilla Pre-LN residual (used within each block).

class GPT(nn.Module):
    def _init_attnres(self, config):  # GPT.__init__ residual region:
        # ── Block Attention Residuals: partition layers into blocks ──
        # 24 layers / 4 = 6 blocks; attention at 5 boundaries + 1 output query
        self.attnres_block_size = 4  # layers per block
        n_blocks = config.n_layer // self.attnres_block_size
        # n_blocks-1 boundary queries (first block gets embedding directly)
        self.attnres_queries = nn.Parameter(torch.zeros(n_blocks - 1, config.n_embd))
        self.attnres_query_out = nn.Parameter(torch.zeros(config.n_embd))

    def _forward_block_loop(self, x):  # GPT.forward block loop:
        # ── Block Attention Residuals: standard residual within blocks,
        #    attention aggregation at block boundaries ──
        block_size_layers = self.attnres_block_size
        n_blocks = len(self.transformer.h) // block_size_layers
        block_outputs = [x]  # initial embedding is first source
        for blk_idx in range(n_blocks):
            # At block boundary (except first): attend over previous block outputs
            if blk_idx > 0:
                stacked = torch.stack(block_outputs, dim=0)  # (num_sources, B, T, D)
                keys_normed = F.rms_norm(stacked, (stacked.size(-1),))
                logits = torch.einsum('d, n b t d -> n b t', self.attnres_queries[blk_idx - 1], keys_normed)
                weights = logits.softmax(dim=0)  # (num_sources, B, T)
                x = torch.einsum('n b t, n b t d -> b t d', weights, stacked)
            # Run layers within this block with standard residual connections
            start = blk_idx * block_size_layers
            end = start + block_size_layers
            for layer_idx in range(start, end):
                x = self.transformer.h[layer_idx](x)
            block_outputs.append(x)
        # Final output: attend over all block outputs with dedicated query
        stacked = torch.stack(block_outputs, dim=0)
        keys_normed = F.rms_norm(stacked, (stacked.size(-1),))
        logits = torch.einsum('d, n b t d -> n b t', self.attnres_query_out, keys_normed)
        weights = logits.softmax(dim=0)
        x = torch.einsum('n b t, n b t d -> b t d', weights, stacked)
        return x

    # GPT.configure_optimizers (queries in a 0.1x-LR, no-decay group):
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        attnres_params = [self.attnres_queries, self.attnres_query_out]
        attnres_ids = {id(p) for p in attnres_params}
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2 and id(p) not in attnres_ids]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2 and id(p) not in attnres_ids]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
            {'params': attnres_params, 'lr': learning_rate * 0.1, 'weight_decay': 0.0},
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

# CONFIG_OVERRIDES = {}   (no override).
```
