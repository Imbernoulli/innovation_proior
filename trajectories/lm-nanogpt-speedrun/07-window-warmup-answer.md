**Problem (from step 6).** FlexAttention at 64K context reached the bar in 1875 steps (≈5.03 min, val_loss
3.2783), but with notable run-to-run variance and the sliding-window size `attn_blocksize` held *fixed* for
the whole run. A fixed window assumes the right amount of context is constant across training, which is wrong
at both ends: early on the representations are still forming and the model can't exploit long-range context
(so a large window is compute spent on interactions it can't use, and slower early steps), while late on the
representations are mature and long-range context genuinely pays off. Any fixed value over-spends early or
under-serves late.

**Key idea (attention window-size warmup).** GROW the sliding window `attn_blocksize` **linearly** over
training, quantized to chunks of 64 (by @fernbear.bsky.social): start at a 64-token window — cheap, fast
early steps where only local context is usable — and ramp to ~1792 tokens by the final step, where long-range
context earns its cost. This is both a compute-allocation win (cheap early steps) and a
short-context→long-context **curriculum**: the model learns local structure first, then widens its horizon,
which also tames the variance the fixed-window record showed.

**Why it works.** Quantizing to multiples of 64 aligns the window to FlexAttention's block granularity, so
the block mask only recompiles a manageable number of times across the run. The window feeds straight into
the unchanged `window_mask = q_idx - kv_idx < attn_blocksize` from step 6 — no mask logic changes, only the
step-dependent value flowing into it. Cheaper early steps trim per-step time over the stretch of training
where most steps live; the curriculum reaches the bar in slightly fewer steps; and the gentler staged early
phase pulls the run-to-run spread back in.

**Change / code.** Replace the constant `attn_blocksize` with the linear, 64-quantized schedule below,
evaluated each step into a device tensor; everything downstream (the `document_causal_mask`,
`create_block_mask`) is untouched.

```python
# Set the attention blocksize for the current step, in chunks of 64. By @fernbear.bsky.social
attn_blocksize = torch.tensor(
    64 * ((step / args.num_iterations * (1792 - 64) + 64) // 64),
    dtype=torch.int, device='cuda'
)
```
