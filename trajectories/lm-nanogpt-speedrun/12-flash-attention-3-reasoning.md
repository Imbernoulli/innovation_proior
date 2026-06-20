The sparse attention gate brought the record to 2.812 minutes — 168,938 ms, 1695 steps, around 99.7 ms a
step. The architecture is now lean and the optimizer is good, so where is the per-step time actually
going? On a packed 64K-token stream with a long-short sliding-window mask, the dominant cost per step is
the attention kernel itself, and right now that kernel is FlexAttention. FlexAttention has been a
workhorse — it let me lengthen the context to 64K with an arbitrary document-plus-window block mask
expressed as a tiny mask function, and `torch.compile` fuses it. But it is a *general* mechanism: it takes
an arbitrary `mask_mod` and synthesizes a kernel, and that generality has a cost. On Hopper specifically,
the hand-written FlashAttention kernels are tuned to the last detail for the H100's memory hierarchy and
its asynchronous tensor-core pipeline. The newest of these, FlashAttention-3, was rewritten to exploit
Hopper's warp-group asynchronous matmul and TMA copies, overlapping the softmax with the matmuls so the
tensor cores stay busy — it reaches substantially higher SM utilization than FlashAttention-2, and far
higher than what a compiled FlexAttention kernel achieves on the same shapes. So the lever is: replace the
FlexAttention call with a FlashAttention-3 call, and pick up the kernel-level speedup on the single fattest
operation in the step.

The catch is the masking. Everything I rely on lives in that block mask: causal, document-boundary
(queries only see keys in the same FineWeb document), and the sliding window whose size warms up over
training. FlashAttention-3 does not take an arbitrary mask function. It takes a *causal* flag, a
*window_size*, and — for variable-length sequences — `cu_seqlens`, the cumulative sequence-length
boundaries. So I have to re-express my three masks in FA3's vocabulary. Causal: the `causal=True` flag.
Sliding window: FA3's `window_size=(left, 0)` argument — attend to `left` tokens behind and 0 ahead, which
is exactly my causal sliding window. The document mask is the interesting one: instead of masking within a
single long sequence, I treat each document as its own variable-length sequence and pack them with the
*varlen* interface. `flash_attn_varlen_func` takes `cu_seqlens_q` and `cu_seqlens_k` — the prefix sums of
the document lengths — and computes attention independently within each document, never across a boundary.
That's the document mask, but now it's free: the kernel simply doesn't look across `cu_seqlens` boundaries,
so I don't pay for masked-out cross-document scores the way a block mask would. @YouJiacheng's suggestion to
use the varlen entry point is what makes the document structure fit FA3 cleanly.

There's a wrinkle that comes from going varlen. With FlexAttention I fed one fixed-length 64K stream and
let the document mask carve it up. With varlen, the *number* of documents in a batch varies — FineWeb
documents have a wide length distribution — but `torch.compile` and the kernel want fixed tensor shapes. So
I keep the number of *tokens* per step fixed (393216) but pack a *variable number of documents* into each
batch, and I pad `cu_seqlens` out to a fixed maximum length larger than any document count I expect in a
batch. And I clip each document to a maximum length — `train_max_seq_len = 2048` — before packing. The
clipping is not just bookkeeping: it lets me pack a greater *diversity* of documents into each fixed
token budget (a few very long documents would otherwise eat the whole budget), which by itself decreases
the loss per step enough to shave training steps. There's also a known finding I can lean on here — that
restricting training to sequences that begin at a true document start (the BoS token) decreases validation
loss — and the varlen packing makes honoring that boundary natural, since I'm already cutting on document
edges. To find those boundaries I scan the token stream for BoS positions and build the `cu_seqlens` from
them.

Now the window-size warmup has to be reconciled with FA3's interface, and this is where I hit friction.
FlashAttention's `window_size` is an *int*, baked into the kernel. With FlexAttention I quantized the
window to multiples of 64 and the compiled kernel tolerated the changing block mask. With FA3, every
distinct `window_size` value forces a *new* compiled graph — if I let the window grow continuously, I
recompile constantly and the compile cost explodes. So I can't keep the smooth per-step window growth. The
fix is to discretize the schedule hard: pick a small number of window sizes and step between them, so the
kernel is compiled exactly that many times. I keep the principle from the warmup rung — windows start small
and grow over training, and I keep the long-short pattern where different layers get different window sizes
— but I express the growth as a short schedule of a few sizes (a `ws_schedule` of a handful of values,
e.g. three steps) with a `get_ws(step)` helper, and I add the per-window block size as an explicit
hyperparameter. Each of those few window values needs its own warmup, so I bump the number of warmup
iterations up to cover them. The compile time is dominated by that first set of iterations, but it's a
fixed one-time cost, not a per-step cost, so it doesn't touch the raced wallclock.

So the change is conceptually a one-for-one kernel swap — FlexAttention → FlashAttention-3 — but it drags
three adaptations with it: document masking moves from an in-sequence block mask to varlen `cu_seqlens`
packing (with documents clipped to a max length and `cu_seqlens` padded to a fixed size); the sliding
window moves from a mask-function predicate to FA3's `window_size=(bm_size, 0)` argument; and the
continuous window warmup becomes a short discrete `ws_schedule` so the kernel only recompiles a handful of
times. The whole point is that on the H100, FA3's hand-tuned asynchronous kernel does the attention
matmuls and softmax far faster than a compiled general kernel, and attention is the fattest op in the step,
so the per-step time should drop. The clipping-driven document diversity should also trim a few steps.

The risk is real and worth holding open. FA3 with `torch.compile` is not natively compatible — the official
module graph-breaks — so this needs the unmerged compile-compatible build of the kernel, and the whole run
is now pinned to a specific torch nightly and a specific FA3 wheel. And re-expressing the masks in FA3's
restricted vocabulary means I lose the freedom the block mask gave me: the window schedule is coarser, and
if the varlen packing or the BoS-clipping interacts badly with the rotary base or the validation sequence
length, the loss could regress and eat the kernel speedup. So I have to validate that it still clears the
bar across repeated runs, not just once.

```python
# FlashAttention-3 varlen call (compile-compatible build), replacing the FlexAttention call.
# - causal sliding window via window_size=(bm_size, 0)
# - document masking via cu_seqlens (varlen): attention is computed independently within each
#   packed document, never across a boundary, so the document mask is free.
flash_attn_interface = get_kernel('kernels-community/flash-attn3', version=1).flash_attn_interface

def forward(self, x, attn_args, qkvo_w):
    B, T = x.size(0), x.size(1)
    assert B == 1, "varlen sequences requires B == 1"
    seqlens, bm_size = attn_args.seqlens, attn_args.bm_size   # cu_seqlens, current window size
    train_max_seq_len, yarn = attn_args.train_max_seq_len, attn_args.yarn
    q, k, v = F.linear(x, qkvo_w[:self.dim * 3].type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
    max_len = train_max_seq_len if self.training else (val_max_len)
    q, k = norm(q), norm(k)                                   # QK norm
    q, k = yarn.rotary(q), yarn.rotary(k)
    # use flash_attn over flex_attn @varunneal; flash_attn_varlen suggested by @YouJiacheng
    y = flash_attn_interface.flash_attn_varlen_func(
        q[0], k[0], v[0],
        cu_seqlens_q=seqlens, cu_seqlens_k=seqlens,          # document boundaries (varlen)
        max_seqlen_q=max_len, max_seqlen_k=max_len,
        causal=True, softmax_scale=yarn.attn_scale,
        window_size=(bm_size, 0),                            # causal sliding window
    )
    y = y.view(B, T, self.num_heads, self.head_dim)
    y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
    return F.linear(y, qkvo_w[self.dim * 3:].type_as(y))

# documents clipped to a max length and packed with variable count per fixed token budget:
#   train_max_seq_len = 2048   (clip each document; packs more document diversity per step)
#   tokens per step fixed at 393216; cu_seqlens padded to a fixed max-document count
# window warmup becomes a short discrete schedule so FA3 recompiles only a few times:
#   ws_schedule = (3, 7, 11)   with get_ws(step); block_size = 128; warmup_iters 10 -> 30
```

The chain: attention is the fattest op per step, and FlexAttention's generality leaves H100 throughput on
the table; FlashAttention-3 is a hand-tuned asynchronous Hopper kernel with far higher SM utilization, so
swapping the call buys the speedup directly — but FA3 takes only causal/window_size/cu_seqlens, so the
document mask becomes free varlen packing (documents clipped to 2048 and `cu_seqlens` padded to fixed
size), the sliding window becomes `window_size=(bm_size, 0)`, and the continuous window warmup becomes a
short discrete `ws_schedule` to bound recompiles. The bar to clear is the sparse-attention-gate record:
168,938 ms (≈2.812 min) at val_loss ≤ 3.28. If FA3's kernel speedup plus the clipping-driven document
diversity hold, this should land a few seconds under that while the validated mean stays below 3.28.
