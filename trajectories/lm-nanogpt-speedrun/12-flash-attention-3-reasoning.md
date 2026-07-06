The sparse attention gate brought the record to 2.812 minutes — 168,938 ms, 1695 steps, around 99.7 ms a
step, and I can read the whole ledger in that product: 1695 × 99.67 ≈ 168,938 ms, so the operating point is
now a lean ~1700-step run at ~100 ms a step. Both factors have been driven hard — the step count down 3.7×
from Muon's 6200, the step time down from 216 ms — and the gate's ~50-step saving plus the intervening
systems work landed exactly on the bar I'm now standing on. The architecture is lean and the optimizer is
good, so where is the per-step time actually going? When I profile, the answer is the same as it's been since
the FlexAttention rung: on a packed 64K-token stream with a long-short sliding-window mask, the dominant cost
per step is the attention kernel itself, and right now that kernel is FlexAttention. FlexAttention has been a
workhorse — it let me lengthen the context to 64K with an arbitrary document-plus-window block mask expressed
as a tiny mask function, and `torch.compile` fuses it. But it is a *general* mechanism: it takes an arbitrary
`mask_mod` and synthesizes a kernel, and that generality has a cost. A kernel that must accept *any* mask
predicate can't be hand-scheduled to the last detail for one specific access pattern; it leaves throughput on
the table.

And on Hopper specifically, there's a lot of throughput to leave on the table. The hand-written FlashAttention
kernels are tuned to the H100's memory hierarchy and its asynchronous tensor-core pipeline — the newest of
them, FlashAttention-3, was rewritten to exploit Hopper's warp-group asynchronous matmul and TMA (tensor
memory accelerator) copies, overlapping the softmax computation with the matmuls so the tensor cores never
stall waiting on data. That overlap is the whole game on Hopper: a naive kernel alternates "load, matmul,
softmax, matmul" and leaves the tensor cores idle during the softmax and the loads, while FA3 keeps them fed
by running the next block's loads and the current block's softmax concurrently with the matmul. It reaches
substantially higher SM utilization than FlashAttention-2, and far higher than what a compiled general
FlexAttention kernel achieves on the same shapes. The reason a general kernel loses is structural, not a lack
of tuning effort: to accept *any* `mask_mod`, FlexAttention has to treat the mask as data consulted inside the
inner loop and can't bake the access pattern into the schedule, so it can't prove ahead of time which blocks
are entirely skippable, can't perfectly overlap the loads with the compute for one specific pattern, and
leaves the tensor cores waiting more often. FA3 knows its pattern is exactly "causal + fixed window +
varlen boundaries" and schedules the whole pipeline around that: the softmax of one tile runs on the special-
function units while the matmul of the next tile runs on the tensor cores and the TMA prefetches the tile
after that, so all three unit types stay busy at once. That concurrency is the entire difference between a
kernel that runs the tensor cores near their roofline and one that stalls them between phases — and attention
being the fattest op, closing that gap is the largest single per-step-time win still available. So the lever is clear: replace the FlexAttention call with a
FlashAttention-3 call and pick up the kernel-level speedup on the single fattest operation in the step. Before
committing let me make sure the swap is the right move against the alternatives. I could keep FlexAttention and
try to optimize the mask further — but I've already done the mask-level work (the window warmup), and the
ceiling now is the *kernel's* generality, not the mask. I could hand-write a custom fused CUDA kernel tuned to
exactly my access pattern — but that's an enormous effort and FA3 is precisely that kernel, already written and
tuned by people who know Hopper's pipeline cold. So swapping to FA3 is the option that buys the hand-tuned
throughput without me writing PTX.

The catch is the masking, and it's the whole difficulty of the swap. Everything I rely on lives in that block
mask: causal, document-boundary (queries only see keys in the same FineWeb document), and the sliding window
whose size warms up over training. FlashAttention-3 does not take an arbitrary mask function — that's exactly
the generality it traded away for speed. It takes a *causal* flag, a *window_size*, and — for variable-length
sequences — `cu_seqlens`, the cumulative sequence-length boundaries. So I have to re-express my three masks in
FA3's restricted vocabulary, and each one maps differently. Causal is the easy one: the `causal=True` flag.
Sliding window: FA3's `window_size=(left, 0)` argument — attend to `left` tokens behind and 0 ahead, which is
exactly my causal sliding window, so the window predicate `q_idx - kv_idx < attn_blocksize` becomes
`window_size=(bm_size, 0)`. The document mask is the interesting one, and it's where FA3's design actually
helps rather than hinders. Instead of masking within a single long sequence, I treat each document as its own
variable-length sequence and pack them with the *varlen* interface. `flash_attn_varlen_func` takes
`cu_seqlens_q` and `cu_seqlens_k` — the prefix sums of the document lengths — and computes attention
independently within each document, never across a boundary.

Let me trace `cu_seqlens` to be sure the varlen semantics reproduce my document mask exactly. Say the packed
stream holds three documents of lengths 3, 5, 2. Then `cu_seqlens = [0, 3, 8, 10]` — the prefix sums. The
kernel reads this as "sequence 0 is tokens [0,3), sequence 1 is [3,8), sequence 2 is [8,10)" and runs
attention independently inside each half-open interval, so a query in [3,8) can attend only to keys in [3,8),
never to [0,3) or [8,10). That is *precisely* the document mask `docs[q_idx] == docs[kv_idx]` I built from the
EoS cumsum in the FlexAttention rung — same partition, same "no attention across boundaries." But now it's
free: the kernel simply doesn't look across `cu_seqlens` boundaries, so I don't pay for masked-out
cross-document scores the way a block mask does (a block straddling a document boundary still gets computed and
then partly masked under FlexAttention; under varlen the boundary *is* the loop bound, so those cross-document
entries are never touched at all). The varlen entry point, suggested by @YouJiacheng, is what makes the
document structure fit FA3 cleanly rather than fighting it.

There's a wrinkle that comes from going varlen, and it forces some packing decisions. With FlexAttention I fed
one fixed-length 64K stream and let the document mask carve it up. With varlen, the *number* of documents in a
batch varies — FineWeb documents have a wide length distribution, from a few tokens to thousands — but
`torch.compile` and the kernel want fixed tensor shapes. So I keep the number of *tokens* per step fixed
(393216) but pack a *variable number of documents* into each batch, and I pad `cu_seqlens` out to a fixed
maximum length larger than any document count I expect in a batch, so the shape is static even though the
document count isn't. And I clip each document to a maximum length — `train_max_seq_len = 2048` — before
packing. The clipping is not just bookkeeping: it lets me pack a greater *diversity* of documents into each
fixed token budget. Think about it as a budget-allocation problem — the step has 393216 tokens to spend, and
if a few very long documents were allowed to run to full length they'd eat most of that budget, so each step
would see only a handful of distinct documents; clipping to 2048 caps how much any one document can consume,
so the same 393216 tokens spread across many more distinct documents. More documents per step is a richer,
less-correlated gradient, which by itself decreases the loss per step enough to shave a few training steps.
The clipping applies at *train* time; the `max_len = train_max_seq_len if self.training else val_max_len` line
keeps the validation path on a larger max sequence length, so I clip for gradient diversity during training
but let the model see full-length context when I measure val_loss — I don't want the clipping to change what
the bar-defining metric sees. And the padding of `cu_seqlens` to a fixed maximum document count is what keeps
the tensor shape static across batches with different document counts: when a batch has fewer documents than
the padded maximum, the surplus `cu_seqlens` entries collapse to zero-length trailing sequences the kernel
skips, so the shape is constant for `torch.compile` while the true document structure varies underneath.
There's also a known finding I can lean on here — that restricting training to sequences that begin at a true
document start (the BoS token) decreases validation loss — and the varlen packing makes honoring that boundary
natural, since I'm already cutting on document edges: I scan the token stream for BoS positions and build the
`cu_seqlens` from them, so every packed sequence starts at a real document start.

Now the window-size warmup has to be reconciled with FA3's interface, and this is where I hit real friction —
the one place the swap costs me something rather than buying something. FlashAttention's `window_size` is an
*int*, baked into the compiled kernel. With FlexAttention I quantized the window to multiples of 64 and the
compiled kernel tolerated the changing block mask — the mask was data flowing into a fixed kernel. With FA3,
every distinct `window_size` value forces a *new* compiled graph, because the window is a kernel parameter,
not data. So if I let the window grow continuously the way the warmup rung did (64 → 1792 in 27 steps of 64),
I'd trigger 27 recompiles — tolerable there, but combined with FA3's heavier per-graph compile and the varlen
shapes, letting it grow smoothly per-step would recompile constantly and the compile cost would explode. So I
can't keep the smooth per-step window growth. The fix is to discretize the schedule hard: pick a small number
of window sizes and step between them, so the kernel is compiled exactly that many times. I keep the principle
from the warmup rung — windows start small and grow over training, and I keep the long-short pattern where
different layers get different window sizes — but I express the growth as a short schedule of a few sizes (a
`ws_schedule` of a handful of values, e.g. `(3, 7, 11)`, in units of the block size 128, so ~384/896/1408
tokens) with a `get_ws(step)` helper, and I add the per-window block size as an explicit hyperparameter. This
is genuinely coarser than the 27-value continuous ramp — three plateaus instead of a smooth line — and I'm
accepting that coarseness as the price of the kernel speed, betting the curriculum benefit survives being
quantized to three levels. Let me check the coarsening doesn't secretly change the compute budget, only its granularity. The old
continuous ramp went 64 → 1792 with average window (64+1792)/2 = 928. The three plateaus 384/896/1408 tokens,
if each covers roughly a third of the run, average (384+896+1408)/3 = 896 — within a few percent of 928. So
the discretized schedule spends about the same total attention budget as the smooth ramp; it just delivers it
in three steps instead of a smooth line. That reassures me the curriculum's *cost* profile is preserved and
only its *smoothness* is sacrificed, which is the coarseness I flagged — the model gets three context regimes
(short, medium, long) instead of a continuum, and the bet is three is enough for the curriculum benefit to
survive. Each of those few window values needs its own warmup, so I bump the number of
warmup iterations up (10 → 30) to cover them. The compile time is dominated by that first set of iterations,
but it's a fixed one-time cost, not a per-step cost, so it doesn't touch the raced wallclock — the compiles
happen during warmup and then every step reuses a compiled kernel.

So the change is conceptually a one-for-one kernel swap — FlexAttention → FlashAttention-3 (the move to flash
over flex is by @varunneal) — but it drags three adaptations with it: document masking moves from an
in-sequence block mask to varlen `cu_seqlens` packing (with documents clipped to a max length and `cu_seqlens`
padded to a fixed size); the sliding window moves from a mask-function predicate to FA3's `window_size=(bm_size,
0)` argument; and the continuous window warmup becomes a short discrete `ws_schedule` so the kernel only
recompiles a handful of times. The whole point is that on the H100, FA3's hand-tuned asynchronous kernel does
the attention matmuls and softmax far faster than a compiled general kernel, and attention is the fattest op in
the step, so the per-step time should drop. The clipping-driven document diversity should also trim a few
steps. So the falsifiable signature is a *step_avg* cut (the FA3 kernel) with the step count flat or slightly
down (the clipping diversity), whose product must land under the bar.

The risk is real and worth holding open honestly, because this is the endpoint and I don't get a free retry.
FA3 with `torch.compile` is not natively compatible — the official module graph-breaks under compile — so this
needs the unmerged compile-compatible build of the kernel, and the whole run is now pinned to a specific torch
nightly and a specific FA3 wheel; that's a fragility I'm taking on for the speed. And re-expressing the masks
in FA3's restricted vocabulary means I lose the freedom the block mask gave me: the window schedule is coarser
(three plateaus, not a smooth ramp), and if the varlen packing or the BoS-clipping interacts badly with the
rotary base or the validation sequence length, the loss could regress and eat the kernel speedup. So I have to
validate that it still clears the bar across repeated runs, not just once — the mean must stay below 3.28, not
merely a lucky single run.

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

The chain: attention is the fattest op per step, and FlexAttention's generality — it synthesizes a kernel from
an arbitrary mask function — leaves H100 throughput on the table; FlashAttention-3 is a hand-tuned asynchronous
Hopper kernel (warp-group-async matmul, TMA copies, softmax overlapped with the matmuls) with far higher SM
utilization, so swapping the call buys the speedup directly — but FA3 takes only causal/window_size/cu_seqlens,
so the document mask becomes free varlen packing (traced: `cu_seqlens = [0,3,8,10]` reproduces the exact
document partition, and boundary entries are never touched rather than computed-then-masked), with documents
clipped to 2048 so a fixed 393216-token step packs more distinct documents (a richer gradient that trims a few
steps), the sliding window becomes `window_size=(bm_size, 0)`, and the continuous window warmup becomes a short
discrete `ws_schedule = (3, 7, 11)` to bound recompiles — accepting a coarser three-plateau curriculum and a
torch-nightly/FA3-wheel pin as the price of the kernel speed. This is the endpoint of the ladder. The bar to
clear is the sparse-attention-gate record: 168,938 ms (≈2.812 min) at val_loss ≤ 3.28. If FA3's kernel speedup
plus the clipping-driven document diversity hold, this should land a few seconds under that while the validated
mean stays below 3.28.
