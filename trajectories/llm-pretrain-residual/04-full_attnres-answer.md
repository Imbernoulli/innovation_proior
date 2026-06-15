**Problem (from step 3).** Learned-scaling (val_loss 2.2680) beat both earlier rungs, but every rung so
far is a *scalar* knob on the depth flow — a rank-one operation. At each layer the stream is still an
unweighted-or-scalar sum of past outputs, mixed the same way for every token. The network has never been
able to *choose which past layer's output to read*, per token, with content-dependent weighting. The
sequence axis escaped fixed mixing via softmax attention; the depth axis can too.

**Key idea.** Block Attention Residuals. In the depth-mixing form `h_l = Σ_i α_{i→l} v_i`, replace the
rank-one all-ones kernel with content-dependent **softmax attention over depth sources**:
`α_{i→l} = softmax_i(w_l^T · RMSNorm(v_i))`. To fit a fixed 2-GPU / micro-batch-32 / `torch.compile`
budget (the full version stores up to 2·n_layer=48 sublayer sources, `O(L²d)` compute, `O(Ld)` memory),
**coarsen the depth axis**: partition 24 layers into 6 blocks of 4, use ordinary residuals *within* each
block, and run depth-attention only at **block boundaries** over the ~6 block outputs (≈8× memory
reduction vs full AttnRes) while keeping the dynamic depth aggregation.

**Why it works.** Values = the stored block representations (transforming them would re-compress);
keys = the same vectors, **RMS-normalized** so a large-norm source can't win on scale alone (the very
depth-norm growth that motivated the ladder); query = one cheap learned **pseudo-query** per boundary
(keys are content-dependent, so weights still vary per token). **Softmax** gives a fixed probability
budget over sources (retrieval: emphasizing one source costs the others). **Zero-init queries** →
`exp(0)=1` → uniform softmax, so the model starts from an equal-weight depth average and learns
deviations from a neutral prior (no random depth bias). The embedding is source 0 — a direct route to
token identity for every block. The mixing matrix becomes dense/full-rank at the block scale: genuine
depth-flow *capacity*, not just conditioning. Cost: ~6k params; trade-off: routing is block-, not
sublayer-, granular.

**Hyperparameters / optimizer.** `block_size_layers = 4` → 6 blocks; `n_blocks−1 = 5` boundary queries
+ 1 output query, all `n_embd`-vectors, **zero-init**. Queries get their own group at **0.1× base LR, no
weight decay** (they are leveraged — they decide the whole boundary mix; decay would pull them to the
uniform init). Main matrices decayed, other 1-D params no-decay. `CONFIG_OVERRIDES = {}`.

**What to watch.** The largest val_loss drop on the ladder (rank-one → full-rank is the first capacity
add) — plausibly the first sub-2.26 number. LAMBADA below *both* 67.21 and 68.76 (per-token deep-block
choice helps completion); WikiText-2 below 43.91; downstream holding, HellaSwag maybe finally moving.
Risks: coarsening may capture only part of the full-version benefit; leveraged queries can destabilize
(0.1× too hot) or barely move (too cold, collapsing toward fixed block-averaging).

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
