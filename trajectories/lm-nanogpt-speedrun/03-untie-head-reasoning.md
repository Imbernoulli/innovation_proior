Fifteen point two minutes now, 5100 steps, val_loss 3.2741, about 179 ms a step. Before I pick the next
target let me confirm the last record did what I predicted, because if the mechanism story was right it tells
me where to look next. Against the Muon record the two factors both moved the way the modern-arch bundle was
supposed to move them: steps 6200 → 5100 (18% fewer, the better-conditioned function fitting faster) and
step_avg 216.33 → 178.89 ms (17% cheaper, the padded vocab and kernel-friendly heads), so wallclock fell
1.34M → 910k ms, a 32% cut, and the val_loss even improved to 3.2741 — more margin above the bar, not less.
That two-pronged confirmation matters: it means the body is now genuinely good on both levers, the optimizer
is condition-blind and cheap, and the block itself is modern. So the one piece I haven't touched since the
baseline is now conspicuous: the input embedding and the output head, and specifically the fact that they are
the *same* matrix. Weight tying — `wte.weight = lm_head.weight` — has been standard since GPT-2 because it
saves ~39M parameters (a 50304×768 matrix is large) and was reported to help small models generalize on
limited data. But those two justifications are about a *different objective* than the one I'm racing. Tying
was defended on parameter economy and generalization; I'm racing wallclock to a fixed loss with the body
already fast. Let me question whether tying still earns its keep in *this* regime.

The two ends of the network do genuinely different jobs. The input embedding is a *lookup table*: it maps
a token id to a vector that seeds the residual stream, and what matters is that semantically related
tokens start nearby and that the vector is at a sensible scale for the first block. The output head is a
*classifier*: it maps the final residual vector to 50304 logits, and what matters is that its rows are
arranged so the dot product with the final hidden state ranks the correct next token highest. Tying forces
one matrix to be good at both, and the two objectives genuinely pull apart. The embedding wants a geometry
where "read into the residual stream at the input" is easy — tokens that behave similarly at the *input*
sit close. The head wants a geometry where "score against the final hidden state after twelve blocks" is
easy — rows arranged so that whatever the deep stack produces for context C lands nearest the true
next-token row. Those are related but not the same arrangement: a token can be easy to inject and hard to
predict, or vice versa, and one shared matrix has to compromise between the two. With Muon now learning
the body fast, the embedding/head is increasingly the limiting factor, and making one matrix serve both
roles caps how well either is learned. Untying gives the optimizer two matrices to specialize — at the cost
of the 39M parameters tying was saving.

There's a sharper way to see the tension, in terms of the gradient the shared matrix actually receives. When
`wte.weight` and `lm_head.weight` are the same tensor W, the backward pass writes into W from *two* places:
the input path contributes ∂L/∂W_embed (how the seed vectors should move so the stream starts better) and
the output path contributes ∂L/∂W_head (how the classifier rows should move so the logits rank better), and
the optimizer sees their *sum*. For a given token's row, those two gradients need not agree — the input path
may want to slide "bank" toward "river" so they seed similarly, while the output path may want to push
"bank" away from "river" so the head can tell them apart in context — and where they disagree the sum is
partial cancellation. The shared row ends up parked at a compromise that fully satisfies neither objective,
and no amount of optimization fixes it because the conflict is structural: one tensor, two opposed gradient
fields. Untie, and each matrix receives only its own gradient field — `wte` follows ∂L/∂W_embed cleanly,
`lm_head` follows ∂L/∂W_head cleanly — so both can move to where their own loss term wants them, no
cancellation. That's the mechanistic content of "specialize each": it's not just more parameters, it's the
removal of a gradient conflict that was capping both roles.

And is the *generalization* argument that justified tying — the reason GPT-2 tied in the first place — still
binding here? Tying acts as a regularizer: fewer free parameters, and the input and output geometries forced
to agree, which helps when you're data-starved relative to model size and would otherwise overfit. But I can
read off this run whether overfitting is the constraint I'm fighting, and it isn't: the val_loss has been
*improving* rung over rung (3.2785 → 3.2741) as the model gets more capable, not degrading, which is the
signature of a model still underfitting its token budget, not overfitting it. In that regime the regularizer
tying provides is not protecting me from anything — it's just a constraint with no upside left, and its
downside (the shared-gradient conflict above) is pure cost. The regime has flipped from the one tying was
designed for: I'm not short of data relative to what the body can absorb, I'm short of *steps*, and
specialization buys steps. So the historical justification for tying doesn't transfer, which is exactly the
license to spend the 39M.

The optimizer assignment also stays consistent with the split I set up when I introduced Muon. Both the
embedding and the now-separate head are lookup/classifier objects, not body projection matrices — Muon's
orthogonalization is the wrong tool for either (an embedding's rows are updated one-per-token with no
cross-row subspace structure; the head's 50304-way output axis shouldn't have its directions equalized) — so
both `wte` and the untied `lm_head` stay in the AdamW group, and only the transformer body remains on Muon.
Untying doesn't perturb that division; it just gives AdamW two matrices to step instead of one, each now free
to move independently.

Is paying 39M parameters legitimate in a speedrun? Let me actually account for it. The tied matrix is
50304 × 768 = 38.6M parameters; untying adds a second matrix of the same size, so the model's parameter
count goes from ~124M to ~163M, a 31% jump. But parameter *count* is not what the race charges for. The
head and the embedding are each used exactly *once* per token regardless of whether they share storage — one
embedding lookup at the input, one head matmul at the output — so untying does not change the number of
active parameters per token, does not add a forward or backward matmul (the head matmul was always there;
it just stops reading its weights from the embedding's storage), and does not touch the body at all. The
39M is therefore a *training-memory* cost — an extra weight matrix plus its Adam optimizer state resident in
HBM — not a compute-per-step or throughput cost. In a race measured by wallclock-to-3.28, if untying lets
the model reach the bar in meaningfully fewer steps at essentially unchanged per-step time, the larger
memory footprint is a fine trade; it's "any%" in the sense that total parameter count goes up while active
params and throughput hold. The only way it backfires is if the extra 39M of fresh parameters *slows*
convergence — more to learn, more to overfit on a small token budget — and that's the real risk to manage,
not the memory. And let me put a number on that memory to confirm it's not secretly the binding constraint:
38.6M extra parameters, plus AdamW's two moment buffers, is roughly 3× that in fp32 optimizer footprint ≈
0.46 GB, spread across 8 GPUs under DDP where each holds the full copy — well under 1 GB per card against an
80 GB H100. So the memory is genuinely free here; the trade is purely "39M more things to learn" against
"fewer steps to the bar," and nothing about capacity or HBM forces my hand. Worth trying, with the
convergence risk named and the memory risk dismissed on arithmetic.

Now, how do I initialize the *untied* head? With tying, the head inherited whatever the embedding was. Now
it's a fresh 50304×768 matrix and I get to choose, and the choice should follow the same principle that made
the modern block train gently: start the head's contribution at *zero*. Zero-initialize `lm_head.weight`. At
step zero, every logit is exactly 0, so the softmax is uniform over the vocabulary and the loss is exactly
log(vocab) — the maximum-entropy starting point, no random head injecting noise into the first gradients.
Let me put the number on it so I know what "starts at max entropy" means concretely: the initial loss should
be ln(50304) = ln(5.0304) + 4·ln(10) ≈ 1.6156 + 9.2103 ≈ 10.83 nats. (A subtlety the padding raises: with a
zero head all 50304 logits are 0, including the 47 pad rows that are never targets, so the softmax spreads
uniformly over 50304 rather than 50257 — but ln(50304) − ln(50257) = ln(1.00094) ≈ 0.0009 nats, utterly
negligible, so the pad rows cost nothing at the start either.) A random-init head, by contrast, would start
above 10.83 with confidently-wrong logits, and the first gradients would spend themselves undoing that random
opinion. The head then has to *learn* to separate tokens from a clean uniform start, which matches the
zero-init residual projections in the body: nothing in the network asserts a confident, random opinion at
initialization. This is the same "earn your output from zero" pattern, now applied to the classifier — and
it's the specific thing that keeps the 39M of fresh parameters from making the early dynamics worse: they
enter the loss as a flat, opinion-free uniform, not as noise.

Let me trace the very first head gradient to check that the max-entropy start is not just tidy but
*productive*, because a start that's flat but generates no useful first step would be a waste. With the head
at zero, all logits are 0 and softmax gives pᵢ = 1/V for every class, V = 50304. Cross-entropy's gradient
with respect to logit i is (pᵢ − yᵢ), which for the true next token t is (1/V − 1) ≈ −1 and for every other
class is 1/V ≈ 2·10⁻⁵. Backpropagated into the head weights (row i's gradient is that scalar times the hidden
state h), the first update pulls the true-token row toward +h with magnitude ~1, while nudging each of the
50303 wrong rows *away* from h by a tiny ~h/V. So the first step is exactly the right shape: a strong, clean
signal that "this hidden state should score token t highly," and an almost-imperceptible push on everything
else — no thrashing, no large wrong corrections to undo. Contrast a random-init head, whose first gradient
also has to cancel out whatever confident-but-wrong logits the random rows produced, wasting the step. The
max-entropy start is both the gentlest and the most informative first step available.

One more thing the untie exposes. With tying, the embedding vector that seeded the residual stream was the
same matrix later RMS-normed all over the body — but the *raw* embedding output, straight out of the
lookup, has whatever scale the embedding table happens to have, and now that the head no longer shares
that matrix, the embedding is free to drift in scale during training in a way that changes the effective
input scale to the first block. Under tying there was an implicit coupling: the head's gradient pressure on
the shared matrix indirectly constrained the embedding's magnitude. Cut the tie and that constraint is gone —
the embedding can grow or shrink its overall scale with nothing pushing back, and a drifting input scale
means the first block sees a moving target. The clean fix is to RMS-norm the embedding *immediately* after
the lookup, before the first block sees it: `x = norm(self.transformer.wte(idx))`. That pins the residual
stream to unit RMS at the input regardless of how the embedding table's scale evolves, so the first block
always receives a well-scaled input and the embedding's learned *direction* is decoupled from its learned
*magnitude* — the table can put its capacity into *where* tokens sit in the 768-space without having to also
manage *how big* they are. It's the same normalization discipline the rest of the network already follows
(pre-attention, pre-MLP, on q/k, before the head), now extended to the very first activation, closing the one
spot where an un-normalized quantity still entered the stream.

It's worth being precise about *why* the pre-attention norm inside the block doesn't already cover this,
because at a glance it looks redundant. The block is x ← x + attn(norm(x)): the norm(x) feeding the sublayer
is renormalized, yes, but the residual *addition* uses the raw x, so the embedding's scale sets the baseline
magnitude of the residual stream that every sublayer's output is added *onto*. If x0's RMS drifts from 1 to
2, the stream is 2× hotter everywhere, and a sublayer output of fixed magnitude m becomes a proportionally
*smaller* correction to a 2×-larger running sum — the blocks' relative influence on the stream silently
shrinks as the embedding inflates. Pinning x0 to unit RMS at the source fixes the baseline the whole residual
stream is measured against, so the sublayers keep a stable proportionate voice regardless of what the
embedding table's overall scale does. The post-embed norm isn't re-normalizing something already normalized;
it's normalizing the one quantity — the residual baseline — that the in-block norms structurally can't reach.

The cheaper alternatives don't actually resolve the tension: a shared matrix with its own head learning rate
or a learned output scale still forces one geometry to be both a good input lookup and a good output
classifier, and a scalar knob can't let the two arrangements differ; untying but random-init'ing the head
pays the 39M *and* forfeits the gentle max-entropy start. Only untie + zero-init head + post-embed norm
spends the parameter cost once and collects both the specialization and the clean start, so that's the
bundle.

Untying adds no forward matmul, so per-step time should be essentially unchanged by the untie itself; what
moves is the step count, downward — two specialized matrices fit the input-and-output problem in fewer
updates than one overloaded matrix — a "fewer steps at flat step_avg" fingerprint, opposite the modern-arch
record's. If the step count *rises* instead, the 39M of fresh parameters slowed convergence more than
specialization helped, and I'd back off — the risk I named up front, now measurable.
The two-matrix `GPT` (separate `wte` and zero-init `lm_head`, `x = norm(wte(idx))`) is in the answer.
