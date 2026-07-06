Seven minutes and twelve seconds now, 3000 steps, val_loss 3.2753 — the U-net skips plus the doubled LR
got me here, and the record confirms the coupled bet paid off exactly as hoped. The step count fell 3200 →
3000 and, more tellingly, the val_loss *improved* from 3.2791 to 3.2753 despite the shorter, higher-LR run —
the margin went from a thin 0.0009 back out to 0.0047. That's the conditioning offsetting the LR-doubling
risk, the outcome I said I'd need to see; the doubled LR didn't cost margin because the skips paid for it.
The step_avg even dropped a touch, 154.78 → 145.01 ms. So the body is dense and clean, the optimizer is fast,
and now I should stop guessing and *profile* a single step, because the shape of where the milliseconds go
decides the next move. When I do, the cost has migrated almost entirely into one place: attention. Everything
else — the MLPs, the embeddings, the head — is cheap and well-scaled. Attention is what each step is now spent
on, and so attention is what I have to think hard about, because it's the thing standing between me and a
shorter run.

The reason attention is expensive is that it's dense and causal over my context, and dense causal attention
is O(T²). Right now T is modest, about 1024 tokens, and that ceiling on T is not an accident — it's exactly
because the cost grows quadratically that I've kept the context short. But look at what context length buys
me, because it's a different lever than any I've pulled. Every rung so far cut the *step count* by making the
optimization better (Muon, the architecture, the shortcuts, the skips) or cut *step_avg* by making the matmul
cheaper (padded vocab). Context length is a third lever: the total work to reach the bar is roughly fixed by
how many tokens the model needs to see, and if I could pack far more tokens into each forward — a much longer
context — I'd need proportionally fewer *steps* to cover the same token budget, and fewer steps with the same
per-token cost is a straight win on wallclock. Concretely, if each step ingests 64× the tokens, the token
budget is covered in ~64× fewer steps. The obvious lever, then, is to lengthen the context dramatically. The
only thing stopping me is the O(T²) wall: at 1024 tokens the quadratic is tolerable, but push T to tens of
thousands and a dense map is hopeless — 64×1024 = 65536 tokens dense would be a 65536² ≈ 4·10⁹-entry score
matrix per head, thousands of times the work of the 1024² ≈ 10⁶ map I run now. And it's not only compute:
that score matrix is 4·10⁹ entries × 2 bytes ≈ 8.6 GB *per head*, and with 6 heads ≈ 51 GB just to hold the
attention scores for one layer — on an 80 GB H100 that's infeasible before I've stored a single weight or
activation. So a dense 64K context is dead on both axes, memory and compute; the only way long context lives
is a kernel that never materializes the full T² map at all, which is exactly what a fused block-sparse
attention does.

So the question isn't "should I lengthen the context" — it's "can I make attention cheap enough at long T to
afford it." Let me look at the alternatives, because "lengthen context" isn't the only way to feed the model
more tokens. I could just raise the *batch* — more independent 1024-length sequences per step. But that
covers more tokens per step without reducing the *steps needed*, because more sequences at the same length
don't let any single optimization step see a longer-range structure; and it keeps attention dense-per-sequence
anyway. I could switch to a linear-attention approximation (Performer, kernel features) that's O(T) by
construction — but that *changes what the model computes*, replacing exact softmax attention with an
approximation, and I'd be betting the approximation doesn't cost me loss, which is a big risk this late.
The third option keeps exact attention but computes only the part I actually need, and the structure of the
problem says most of the dense map is genuinely wasted. Two facts make that so. First, FineWeb is a packed
stream: many independent documents are concatenated together with an end-of-text delimiter between them, and a
query in document seven has no business attending to a key in document three — that's cross-document leakage
that can only add noise. Second, even within a document, the useful signal for next-token prediction is
overwhelmingly local; a query a long way back from a key rarely needs that key, especially with rotary
already encoding relative position. So the *true* attention pattern I want at long context is sparse: each
query attends only to keys (a) at or before it — causal, as always; (b) in the *same* document as it; and
(c) within some bounded window behind it. That is a tiny fraction of the full T² map, and if I can get a
kernel that only computes the kept entries, the cost of attention stops scaling with T² and starts scaling
with T times the window — linear in context length.

Let me put the arithmetic on why that changes everything. Dense causal attention over a sequence of length T
does ~T²/2 score computations per head. Windowed attention with window W does ~T·W — each query attends to
at most W keys behind it. At my current T ≈ 1024 dense, per-query work is ~T/2 ≈ 512 keys. If I lengthen to
T = 65536 but cap each query to a window of W ≈ 1024, per-query work is ~1024 keys — the *same order* as the
dense 1024-context I run now. So the per-token attention cost barely changes, which means step_avg barely
changes, while each step now ingests 64× the tokens. The step count to cover the token budget collapses ~64×.
That's the whole trade: I'm not making attention cheaper per token, I'm buying 64× more context at the same
per-token price by refusing to compute the far-away pairs I never needed. Linear-in-T attention is what makes
a 64K context affordable, and windowing plus document-masking is how I get linear-in-T without approximating
the softmax.

How do I express that sparsity without hand-writing a fused CUDA kernel? This is exactly what FlexAttention is
for — the `torch.nn.attention.flex_attention` interface that @KoszarskyB has been wiring in. I write a small
boolean predicate over `(q_idx, kv_idx)` that returns whether that query is allowed to see that key, hand it
to `create_block_mask`, and FlexAttention compiles a *block-sparse* fused kernel that skips whole blocks of
the map where the mask is entirely false. I don't pay for the masked-out region at all — it's not computed
and then zeroed, it's never touched, which is the difference between a mask that saves memory and a mask that
saves *compute*. Let me quantify how much gets skipped, since that's the whole payoff. FlexAttention works in
blocks — say 128×128 tiles of the score matrix — and it skips a tile entirely only if *every* entry in it is
masked false. The window mask makes almost all tiles all-false: a query block at position p can only see key
blocks within `attn_blocksize` behind it, so out of the T/128 = 65536/128 = 512 key-block columns, a query
row of blocks touches only ~W/128 + 1 ≈ 1024/128 + 1 ≈ 9 of them. That's ~9 / 512 ≈ 1.8% of the block
columns computed, ~98% of the map skipped outright — and the document mask carves further all-false tiles out
of the surviving band wherever a document boundary falls inside the window. So the block-sparsity isn't a
marginal saving; it turns a would-be 512-column-wide dense band into a ~9-column-wide diagonal stripe, which
is precisely the T·W-not-T² scaling I did the arithmetic for above, now realized at the block granularity the
kernel actually operates on. So the three conditions I want become three predicates AND-ed together. Causal is the
familiar `q_idx >= kv_idx`. The document constraint I get by computing a cumulative-sum document id: every
time I pass the EoS token id 50256 the doc id increments, so `docs = (idx == 50256).cumsum(0)` tags each
position with which document it belongs to, and `docs[q_idx] == docs[kv_idx]` is true exactly when query and
key live in the same document. Let me trace that to be sure I have the boundary right: for a stream
`[a, a, <eos>, b, b]` with EoS at position 2, `(idx==50256)` is `[0,0,1,0,0]` and its cumsum is
`[0,0,1,1,1]`. So positions 0,1 (the a's and could include the eos... position 2 has doc id 1) — position 0
and 1 have doc id 0, position 2 (the EoS itself) has id 1, positions 3,4 (the b's) have id 1. A query at
position 4 (a b) has doc id 1 and can attend to positions 2,3,4 (all doc id 1) but not 0,1 (doc id 0) — so it
sees the b's and the EoS delimiter but not the earlier document's a's. That's exactly the separation I want,
and note the EoS lands with the *following* document, which is fine — it's a delimiter the next document can
attend to as its start marker. And the window is `q_idx - kv_idx < attn_blocksize` — the query only reaches
`attn_blocksize` tokens back. The mask is the AND of the three, and that single predicate is the whole
specification; FlexAttention turns it into the kernel.

One reassuring check on the window predicate: if I set `attn_blocksize` ≥ T, the condition `q_idx - kv_idx <
attn_blocksize` is true for every causal pair, so the window mask disappears and I recover ordinary dense
causal attention *within each document*. The windowed form therefore strictly generalizes the dense
document-causal attention — it's the same computation with a knob that, turned to its maximum, gives back the
full map. So by choosing W ≈ 1024 ≪ T = 65536 I'm not adopting a different, weaker attention; I'm taking the
same attention and declining to compute the long-range pairs, and the only thing the model gives up is the
ability to attend *beyond* 1024 tokens within a single document — which, per the locality argument, is signal
it rarely used. That framing also tells me the safe fallback if loss regresses: widen W toward T and I
continuously approach the dense behavior, trading speed back for range.

The one recurring cost I'm adding is rebuilding the block mask each forward, and I should check it's cheap
enough not to eat the win. `create_block_mask` doesn't evaluate the predicate per token pair — that would be
T² again — it evaluates it per *block*: on the S/128 × S/128 = 512 × 512 block grid, ~2.6·10⁵ predicate
evaluations to decide which tiles are all-false, live-or-skip. That's a rounding error against the actual
attention matmuls the kernel then runs, and `_compile=True` caches the compiled kernel so the compilation
itself is a one-time cost, not a per-step one. The reason it *has* to be rebuilt every step is that the
document boundaries — the EoS positions — differ from one packed stream to the next, so `docs` and hence the
all-false pattern change; but rebuilding a 512×512 block map is negligible, so the per-step accounting holds.

```python
docs = (idx == 50256).cumsum(0)
def document_causal_mask(b, h, q_idx, kv_idx):
    causal_mask = q_idx >= kv_idx
    document_mask = docs[q_idx] == docs[kv_idx]
    window_mask = q_idx - kv_idx < attn_blocksize
    return causal_mask & document_mask & window_mask
```

Now I can be aggressive with context. If attention is windowed and document-masked, its cost no longer
explodes with T, so instead of 1024 tokens I take the sequence length all the way to 64×1024 = 65536 tokens. Why 64×
specifically? It's a round power-of-two multiple that keeps the sequence length a clean multiple of both the
128-token block size and the 1024-ish window, and it's large enough to make the step-count collapse dramatic
(~64× more tokens per step) without pushing the single-stream length so far that the windowed band itself
(T·W FLOPs) starts to dominate the step again. Going to 128× would double the per-step work of even the sparse
band for a further step-count halving that's subject to diminishing returns; 64× is the aggressive-but-clean
choice.
The data that used to be eight separate sequences of length ~1024 becomes, in effect, one very long packed
stream per device — and that forces a structural choice I have to be careful about: with FlexAttention here
the batch dimension B must be 1. There is one long stream, not a batch of short ones, so inside the attention
forward I assert `B == 1` and treat the whole thing as a single sequence of length T, letting the document
mask do the work of keeping the packed documents from talking to each other. Why must B be 1, concretely? The
block mask is built for one specific sequence — the `docs` cumsum and the all-false block pattern are computed
from *these* tokens — so a batch of several streams would each need their own mask, and the document
boundaries differ per stream; FlexAttention's single compiled block mask can't represent B independent
document layouts at once. Collapsing to B=1 sidesteps that: one device holds one long packed stream with one
mask, and the batch dimension I gave up is recovered across the 8 GPUs under data parallelism — each GPU
processes its own 64K stream, so the global batch is 8 streams even though each device sees B=1. So the B=1
constraint isn't a limitation on throughput, it's just a relocation of the batch axis from within-device to
across-device. The block mask itself has to be
rebuilt each forward, because the document boundaries depend on the actual tokens in this step's stream —
`create_block_mask(document_causal_mask, None, None, S, S, device="cuda", _compile=True)` where `S = len(idx)`
— and I let it compile, caching the kernel. The attention call becomes
`flex_attention(q.transpose, k.transpose, v.transpose, block_mask=block_mask)` in place of
`F.scaled_dot_product_attention`, everything else in the block — the QK-norm, the rotary, the value-residual
mix `v = (1-lamb)*v + lamb*vi` — left exactly as it was.

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
flex_attention = torch.compile(flex_attention, dynamic=False)
create_block_mask = torch.compile(create_block_mask, dynamic=False)

def forward(self, x, vi, block_mask):
    B, T = x.size(0), x.size(1)
    assert B == 1, "Must use batch size = 1 for FlexAttention"
    q = self.c_q(x).view(B, T, self.n_head, -1)
    k = self.c_k(x).view(B, T, self.n_head, -1)
    v = self.c_v(x).view(B, T, self.n_head, -1)
    v = (1 - self.lamb) * v + self.lamb * vi.view_as(v)
    q, k = norm(q), norm(k)
    q, k = self.rotary(q), self.rotary(k)
    y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask)
    y = y.transpose(1, 2).contiguous().view_as(x)
    return self.c_proj(y)

S = len(idx)
block_mask = create_block_mask(document_causal_mask, None, None, S, S, device="cuda", _compile=True)
# sequence_length : 64*1024 (was ~1024); batch_size : 8 sequences
```

What I expect: the per-step time barely moves, because each step still does roughly the same amount of
useful attention work per token (the window caps how much any query attends to, at ~1024 keys, the same order
as the old dense 1024-context), but each step now ingests sixty-odd times the tokens, so the number of steps
to cover my token budget collapses. That's the falsifiable signature — a big *step-count* cut at roughly flat
step_avg, the opposite fingerprint from a per-step-time optimization — and it comes from buying context
cheaply rather than from any change to what the model learns. I'd expect the step count down from 3000 toward
~1875. If instead step_avg balloons, that would mean the windowed kernel isn't actually skipping the
masked-out blocks (the block-sparsity isn't kicking in at my block granularity) and I'd check the mask.

There's one wrinkle I want to name honestly before I commit, because it's a real cost and not a free lunch.
Shortening the run this much makes it twitchier. The previous record had already doubled the LR, and a high
LR over a much shorter duration leaves less time for the loss to settle, so run-to-run variance goes up. I
expect something like 0.005 standard deviation across repeats — enough that the *mean* val_loss can sit
safely under the bar, around 3.279, while not literally every single run lands below 3.28; the occasional run
pokes above it. I'll accept that for now as the price of the speed, and treat it as something to manage rather
than a reason to back off — the expected loss is comfortably under the bar and the wallclock saving is large.
If the variance becomes the binding constraint later, the natural place to attack it is the window itself,
which I'm holding fixed at a single size for the whole run here; there may be slack in *how* the window is
scheduled rather than in whether it exists. But that's later. For now: replace dense causal attention with a
windowed, document-masked FlexAttention, lengthen the context to 64K, and let the step count fall.

The chain: the U-net record confirmed both step count and margin moved the right way, and profiling shows each
step is now dominated by dense O(T²) causal attention, which caps the context at ~1024 tokens; but the
attention I actually want at long context is sparse — causal, same-document, within a bounded window — and
the arithmetic shows a T=65536 windowed map with W≈1024 costs the same *per token* as the old 1024 dense map
while ingesting 64× the tokens; FlexAttention compiles exactly that sparse pattern from a three-predicate
block mask (causal `q>=kv`, document `docs[q]==docs[kv]` from the EoS cumsum, window `q-kv<blocksize`), so the
cost goes linear in context and the masked blocks are never touched; that lets me push the sequence length to
64K with B=1 packed streams, which covers the token budget in ~64× fewer steps and cuts the wallclock hard —
at the known price of higher run-to-run variance (~0.005 std), whose mean still clears the bar.
