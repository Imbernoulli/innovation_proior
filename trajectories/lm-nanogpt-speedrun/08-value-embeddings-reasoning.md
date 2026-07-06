Four point six six minutes now, 1750 steps, val_loss 3.2759, about 161 ms a step — the window warmup
squeezed another chunk of wallclock out by letting attention start narrow and grow. Reading the record
against my prediction: the step count fell 1875 → 1750 (7%), the val_loss came back down to 3.2759 with the
margin restored to 0.0041 and the run visibly steadier — that's the curriculum taming variance, which was
half the point. But the step_avg barely moved, 161.84 → 160.89 ms, and I'd predicted the cheaper early
windows would drop it more; the early-window saving was real but small against the fixed MLP/embedding/head
cost, so most of this rung's win came from the step-count cut and the steadiness, not from cheaper steps. Fair
enough — the schedule was nearly free, so a modest gain is still a good trade (1750 × 160.89 ≈ 282k ms against
the prior 303k, a ~7% cut, all of it from the seven-percent-fewer steps). The body is tuned, the
optimizers are tuned, the attention schedule is tuned. So I go back to one of the cheapest wins on the ladder
and ask whether I got everything out of it: the value residual. When attention concentrates with depth, late
layers lose the clean early-layer value signal, and the fix was to give every attention layer a direct line
back to the first block's value vector v1, mixed in with a learnable per-layer lambda — `v = (1 - self.lamb)*v
+ self.lamb * v1`. That helped, and the lambdas confirmed the layers actually wanted the early signal. But let
me look hard at *what* that early signal really is, because I suspect it's a compromised version of the thing
the layers actually want.

v1 is the first decoder block's *computed* value: it's the embedding, RMS-normed, passed through one
attention layer's value projection c_v. So when block 8 reaches back for "clean early value," what it gets
is already one block deep — it's the embedding seen through the lens of the first attention layer's
learned c_v weights, mixed with whatever the first block's residual update did. It's *cleaner* than block
8's own value, yes, but it isn't the rawest possible token signal. It's a derived quantity, and every
layer that wants it has to take the *same* v1, because there's only one first block. The thing I'm
injecting is a single shared vector computed once, and its content is whatever the first attention layer
happened to learn to project. There's a tension hiding here, and it's the same shape as the one I broke when
I untied the embedding from the head: the first block's c_v is being asked to serve two masters — it's
optimized to produce a good value for block 1's *own* attention, but it's also, implicitly, the source of the
early-value signal that all eleven later layers depend on. One projection, two jobs, and where those two jobs
disagree the projection sits at a compromise. I recognize that pattern now; last time it capped both roles
and untying fixed it.

So here's the move. What if every attention layer got a direct line not to a *derived* value but to a
*learned, token-identity-based* value signal — a value that depends only on the token id, looked up from a
table, never carried through the residual stream or filtered through the first block's projection at all?
Think about what the residual stream is: the network's working memory, and everything that flows through it is
a transformation of the input, so the deeper a block sits the more processed and the more lossy its view of
the raw token becomes. Reaching back to v1 is one way to fight that, but v1 is *itself* a residual-stream
quantity — it's subject to the same "everything is a transformation" pressure, just less of it. A learned
table indexed by token id sidesteps the residual stream entirely. It says: for token "the", here is a value
vector layer 5 should mix in, here is a (possibly different) value vector layer 9 should mix in, and these are
parameters the optimizer sets directly, with no constraint that they be reconstructible from the residual
stream. It's the value-stream analogue of what the input embedding wte already does for the *input* — wte is
a learned, token-identity-based seed for the residual stream — except now I'm seeding the *value* of every
attention layer directly from token identity. The symmetry is clean: wte at the input, a value table at every
attention layer. And notice this makes the third token-identity-indexed object in the model: wte seeds the
residual stream from the token id at the input, the untied lm_head reads the token id back out at the output,
and now vte injects the token id's value at every attention layer. All three are the same question — "what
does this token mean" — asked in three different places (entry, exit, and the value stream throughout), and
the arc of the last several rungs has been to stop making one object answer that question for multiple roles:
untying split entry from exit, and vte now splits the value-stream answer out from the residual-derived v1.
The table being *content*-indexed (by token id, not position) is the right split too — rotary already handles
position, so what the value stream is missing is a clean per-token *content* signal, which is exactly what a
token-id table supplies.

Why would that be better than reaching back to v1? Because v1's content is bottlenecked, and I can name the
bottleneck precisely. The first block computes one value per token, and that single vector has to encode
whatever *all* the downstream layers collectively need from the early signal — but they don't all need the
*same* thing. A shallow layer that's only mildly concentrated wants a light touch of early value; a deep,
badly-concentrated layer wants a lot, and possibly a *different aspect* of the token's identity than the
shallow one does. v1 can't give them different things; it's one vector, so the eleven consumers are forced to
share one representation and the per-layer lambda can only scale *how much* of that one vector each takes, not
*which* early value it is. A table can give them different things. If I make the table wide enough to hold a
*separate* value slice per layer, then layer i looks up its own dedicated per-token value, and the optimizer
can shape each layer's slice independently — layer 5's "the"-value and layer 9's "the"-value are different
parameters, free to diverge. The first block's c_v is freed to just be block 1's value projection again, no
longer secretly the universal early-value source. And the injection no longer has to survive the trip through
the residual stream, because it never enters the residual stream — it's looked up fresh from the token ids at
every layer.

Concretely: add a second embedding table, call it vte, the value-token-embedding, alongside wte. wte has
width n_embd = 768. vte I make width n_embd·12 = 9216 — one 768-wide slice for each of the twelve layers. I
look it up once from the token ids, exactly like wte, and chunk the result into twelve per-layer slices. Then
layer i receives its slice vi[i], and inside the attention forward I mix it into the freshly-projected value
with the *same* learnable-lambda mechanism I already built for value residual: `v = (1 - self.lamb)*v +
self.lamb * vi.view_as(v)`. The machinery is identical — the per-layer lambda, the convex mix, starting
behavior the model can dial — but the thing being mixed in has changed meaning. Before, vi was the first
block's computed v1, a derived residual-stream quantity shared by all layers. Now vi is this layer's
dedicated slice of a learned, token-indexed table, independent of the residual stream and independent of the
first block's projection. Same lambda, dedicated learned table instead of one shared computed value.

Now let me count the cost honestly, because this table is not small. vte is vocab_size × (n_embd·12) = 50304
× 9216 ≈ 463.6M parameters — larger than the entire rest of the model (~163M) put together, a genuinely fat
table. Is that admissible? Same accounting as the untie rung, and it comes out fine for the same reason.
First, active parameters per token: a token looks up exactly one row, 9216 values, so the vte cost per token
is 9216 numbers gathered — 12× the 768 that wte gathers, but it's a *gather*, not a matmul, so it adds
memory bandwidth, not FLOPs. There's no 463M-wide matmul anywhere; the table is only ever indexed. Second,
memory: 463.6M parameters plus Adam's two moment buffers is ~3× in fp32 ≈ 5.6 GB per GPU — real, but well
under the 80 GB budget, and it doesn't touch throughput. So the charge is memory-and-bandwidth, not compute,
which in a wallclock race is the cheap kind. It's a lookup table indexed by token id, the same *kind* of
object as wte, so it belongs with wte under Adam, not Muon — Muon orthogonalizes matrix *updates* and that's
meaningless for an embedding table whose rows are updated one-per-token. So I hand vte to the same Adam group
as wte: `optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight,
raw_model.transformer.vte.weight], lr=0.6, betas=(0.8, 0.95), fused=True)`. That lr=0.6 looks alarmingly high
next to the body's learning rate, but it's the right regime for an embedding table and vte belongs there for
the same reason wte does: any given row is touched only on the steps where its token actually appears in the
stream, so a row's updates are *sparse* — most steps it gets no gradient at all. A parameter that is updated
rarely needs a large step when it *is* updated to cover the same ground as a dense parameter nudged every
step, which is why embedding groups run at much higher learning rates than the matmul body. The betas =
(0.8, 0.95) also fit: the lower β₁ = 0.8 (vs the body's 0.9) means the first-moment average decays faster,
appropriate when a row's gradients are intermittent and a stale momentum from many steps ago would be
misleading. So the vte hyperparameters aren't a new tuning problem — they're the wte embedding-group settings,
which vte inherits by being the same kind of object. And there's a per-step compute
cost after all: every forward now does a second, twelve-times-wider embedding lookup and twelve view/mix
operations. That will push the step time up a little — the 12× wider gather is the main contributor, since
it moves 12× the embedding bytes. Let me size that so I know what step-time rise to brace for. Per step the
stream is ~64K tokens; wte gathers 65536 × 768 × 2 bytes ≈ 100 MB, while vte gathers 65536 × 9216 × 2 ≈ 1.2 GB
of rows, and the backward pass writes gradients into those same rows. At an H100's ~3 TB/s HBM bandwidth,
moving 1.2 GB is ~0.4 ms forward, and with the scatter-add on the backward it's a few tenths of a millisecond
more — so I'd expect step_avg to rise by roughly a millisecond or two from the vte traffic, a bandwidth-bound
cost, not a compute-bound one (no matmul touches 463M of anything). Against a ~160 ms step that's ~1%, small,
which is exactly why the bet is favorable: a ~1% step-time rise is easy to beat if a dedicated per-token value
cuts even a few percent of the steps. The bet is that the step *count* falls by more than enough to pay for it.

A quick shape check, since the chunk-and-mix has to line up exactly. vte's row is 9216 wide; `chunk(12,
dim=-1)` splits it into twelve slices of 9216/12 = 768 each, so vi[i] is (B, T, 768). Inside the attention
forward the value v is viewed as (B, T, n_head, head_dim) = (B, T, 6, 128), and 6·128 = 768, so
`vi.view_as(v)` reshapes the 768-wide slice to (B, T, 6, 128) and the convex mix `(1-lamb)·v + lamb·vi` is
elementwise on matching shapes. The widths reconcile: one 9216-wide table row carries exactly one 768-wide
value per layer, which is exactly one full head-partitioned value tensor per layer. No padding, no leftover.
And the routing has to respect the U-net layout from the skip rung: the twelve chunks vi[0..11] are indexed by
*block position*, and since the encoder/decoder relabeling doesn't reorder the block list — blocks 0–5 encode,
6–11 decode, in order — block i simply consumes slice vi[i], so the chunk index and the block index line up
directly. Each block, whether it's acting as encoder or decoder, gets its own dedicated slice; the U-net
skip-connection wiring (which activations get added where) is orthogonal to the vte slicing (which value table
each block mixes in), so the two mechanisms compose without either needing to know about the other.

Before I commit, the alternatives. I could instead widen v1 itself — compute several early values in the
first block and let each later layer pick one — but that's still derived from the residual stream and still
funnels through the first block's projections, so it doesn't escape the "derived, filtered" problem, it just
adds more derived vectors. I could share a *single* vte slice across all layers rather than twelve — cheaper
(38.6M, not 463M) — but then I'm back to one shared vector for all consumers, exactly the v1 bottleneck I'm
trying to break, just relocated into a table; the whole point is per-layer specialization, so the 12× width is
load-bearing, not gratuitous. The dedicated per-layer table is the option that actually removes the
bottleneck, and its cost is memory I have to spare rather than compute I don't.

One thing I want to keep honest with myself: this isn't free structure the model can ignore at init the
way the embed shortcut was. The vte table starts from whatever its initialization is, and lambda controls
how much of it leaks in, so if the table were useless the lambdas could learn to suppress it — the safety
valve is there, and it's the same convex-mix valve as before. But the table is big and it has to *learn* to
be useful, which costs some of the early steps: at step zero those 463M parameters are random, so lambda will
initially want them near zero and the useful per-token values have to emerge. So I expect the loss curve to
look slightly *worse* very early — the model is paying to train a large table that isn't helping yet — and
then pull ahead as the per-token value slices sharpen and the lambdas open up. If the net step count at the
3.28 bar comes down, the extra lookup time is paid for and this is a win. Given how much the lambdas wanted v1
in the first place, I think the layers will want a dedicated, richer version of that signal even more.

If the mechanism is right, the falsifiable signature is a *step-count* drop bought at a *step-time rise* — the
distinctive fingerprint of trading compute/memory for convergence, the opposite of the window warmup. I'd
expect the step count from 1750 down toward ~1530, and the step_avg to rise above 160.89 ms from the wider
lookup, with the wallclock still falling because the step cut outweighs the time rise. If instead the step
count barely moves while step_avg jumps, the table isn't earning its lookup cost — the layers didn't want a
dedicated per-token value after all — and I'd revert. The learned lambdas are again a bonus diagnostic: if
they open up more than they did for v1, that's direct evidence the dedicated table is a richer signal than the
shared computed value was, and if they differ *across layers* more than the v1 lambdas did, that confirms the
per-layer slices are being used for genuinely different things — the whole premise of paying for 12× the
width.

```python
# in GPT.__init__:
self.transformer = nn.ModuleDict(dict(
    wte = nn.Embedding(config.vocab_size, config.n_embd),
    # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual learning
    vte = nn.Embedding(config.vocab_size, config.n_embd*12),
    h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
))

# in GPT.forward, look up and chunk into one slice per layer:
vi = self.transformer.vte(idx[None]).chunk(12, dim=-1)
# ... each block i receives vi[i] (encoder/decoder indexing matches the U-net layout):
# x = self.transformer.h[i](x, vi[i], x0, block_mask)

# in the attention forward, mix the value embedding into v with the learnable lambda:
def forward(self, x, vi, block_mask):
    B, T = x.size(0), x.size(1)
    q = self.c_q(x).view(B, T, self.n_head, -1)
    k = self.c_k(x).view(B, T, self.n_head, -1)
    v = self.c_v(x).view(B, T, self.n_head, -1)
    v = (1 - self.lamb) * v + self.lamb * vi.view_as(v)   # @Grad62304977
    q, k = norm(q), norm(k)
    q, k = self.rotary(q), self.rotary(k)
    y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask)
    y = y.transpose(1, 2).contiguous().view_as(x)
    return self.c_proj(y)

# vte optimized by Adam alongside wte:
optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight, raw_model.transformer.vte.weight], lr=0.6, betas=(0.8, 0.95), fused=True)
```

The chain: the value residual gave every layer a line back to the first block's value v1, but v1 is a
derived, residual-stream quantity, shared across all layers, sourced from one overloaded c_v — the same
two-jobs-one-object bottleneck the untie rung taught me to recognize; replace it with a dedicated learned
table vte of width n_embd·12 = 9216 (≈463.6M params, but a token-indexed *gather* not a matmul, so
memory-and-bandwidth not FLOPs) looked up from the token ids and chunked into one per-layer slice, so each
attention layer mixes in its own direct, per-token value via the same learnable lambda — `v = (1 -
self.lamb)*v + self.lamb * vi.view_as(v)` — with vte optimized by Adam alongside wte. The 12×-wider lookup
raises the step time, but a direct learned value signal that never has to survive the residual stream, and
that can differ per layer where one shared v1 could not, should cut the step count from 1750 toward ~1530 by
more than enough to pay for it.
