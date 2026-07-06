Five minutes flat now, 1875 steps, val_loss 3.2783, about 162 ms a step. The FlexAttention record did what I
predicted and I want to read it precisely, because it landed two loose ends on the table that I explicitly
said I'd come back to. The step count collapsed 3000 → 1875 — a 37% cut — which is the 64× context buying
fewer steps for the same token budget, the big lever working. The step_avg ticked up, 145.01 → 161.84 ms:
the windowed attention over a 64K stream costs a little more per token than the tiny 1024-dense map did, plus
the per-step block-mask rebuild, so the per-token price wasn't *exactly* flat, it rose ~12%. And the val_loss
rose to 3.2783 — margin down to 0.0017, and I'd flagged that shortening the run this hard would raise
variance to ~0.005 std, so the mean is under the bar but the run is twitchier. On the wallclock product, 1875 × 161.84 ≈ 303k ms
against the U-net's 3000 × 145.01 ≈ 435k — a 30% cut driven entirely by the step-count collapse, partly given
back by the ~12% step-time rise, which is the thing I'd like to claw back next. So the two loose ends are
right there in the numbers: the variance, and a choice I made and never examined — I picked a single window
size, `attn_blocksize`, and held it fixed for the entire run. That fixed window is the obvious next thing to
interrogate, because the assumption behind a fixed window is that the *right* amount of context is the same at
step 0 as at the last step, and when I actually think about what the model is doing across training, that
assumption looks plainly wrong in both directions.

Consider the early steps. The representations are still forming — the residual stream barely encodes
anything yet, the attention heads haven't specialized, the model can't meaningfully exploit a token that's a
thousand positions back because it doesn't yet have the machinery to relate them. So a large window early in
training is compute I'm spending on long-range interactions the model is incapable of using. It's not just
neutral waste, either: it makes every early step slower, since the windowed attention cost scales with the
window size — a query attending over W keys does O(W) work, so a window of 1792 does ~28× the per-token
attention work of a window of 64. Spending 28× on context the early model can't use, precisely where the loss
is dropping fastest and I'd most like the steps cheap, is exactly backwards. Now consider the late steps.
Here it's the opposite — the representations are mature, the heads know what they're doing, and long-range
context genuinely pays off for the harder predictions, so this is exactly where a *large* window earns its
cost. A single fixed window can't be right for both ends: set it large and I over-spend early on context I
can't use; set it small and I under-serve the late training where context matters. Whatever fixed value I
chose, it's a compromise that's wrong everywhere except possibly one moment in the middle.

The moment I phrase it that way the fix follows: don't hold the window fixed — grow it. Start small, when
only local context is usable and I want the steps cheap, and increase it to large by the end, when long-range
context pays off. And this isn't only a compute-allocation trick; it's a *curriculum*. By restricting the
model to a short window early, I'm forcing it to learn local structure first — the easy, high-payoff
short-range statistics, the bigram-and-trigram bones of the language — before it ever has to cope with
long-range dependencies, and only later widening its horizon once the local machinery is in place. Why is local-first the right order and not, say, hard-first? Because the loss is dominated by short-range
predictability. Most of the bits in next-token prediction come from local structure — the immediately
preceding token or two pin down most of the distribution (after "the" comes a noun, after "New" comes "York"),
and these local statistics are both high-frequency (every position exercises them) and high-payoff (they cut
the most entropy per parameter). Long-range dependencies are lower-frequency and each one cuts less entropy on
average. So a model that spends its first, cheapest steps nailing the local statistics is capturing the bulk
of the achievable loss drop with the cheapest possible attention, and only later paying for the wide window to
mop up the residual long-range bits. Learning them in the other order — forcing the model to consider
thousand-token context before it can even predict the next word from the last one — would waste the early
steps on a signal it can't yet isolate from the noise. There's a
clean analogy to the rest of what I've already done: I zero-init the head and the residual projections so the
network starts from a quiet, low-confidence state and *earns* its structure rather than asserting random
opinions; I warm up the LR and the momentum so the optimizer doesn't take violent steps before the loss
landscape is understood. A growing window is the same discipline applied to *what the model is allowed to
look at* — start it with a narrow field of view it can actually make sense of, and only broaden it as it
becomes capable of using the breadth. Short-context-then-long-context is a sensible order to learn language
in, and curricula like that tend to stabilize training. Which connects straight back to the variance I'm
trying to tame, and it's worth being precise about the mechanism rather than just hoping. The variance in the
FlexAttention record — ~0.005 std — comes from a short, high-LR run being sensitive to the luck of
initialization and data order: with few steps, an early wrong turn doesn't have time to wash out, so different
seeds end at measurably different losses. A curriculum reduces the number of consequential early decisions. In
the first phase the model is *only* allowed to learn local structure — there's simply less it can get wrong,
because the wide-context degrees of freedom are masked off and can't be set (well or badly) yet. Fewer early
degrees of freedom to fix by luck means less seed-to-seed divergence in where the run ends up, which is
exactly a variance reduction. So it's not a vague "curricula stabilize training" appeal; it's that masking the
long-range interactions early removes the specific early choices that were the variance source. A gentler,
staged early phase should therefore make the runs less twitchy, not more, so the same change
that makes early steps cheaper should also pull the run-to-run spread back in. That's the kind of change I
want — one lever that improves two things at once (cheaper early steps *and* tighter variance), and a lever I
get essentially for free because the windowing machinery is already in place from the last rung.

So the window becomes a function of the training step instead of a constant. Why linear, and not some
curve? I have no evidence the model's appetite for context follows any particular shape — I could imagine a
cosine, a step-function that jumps at a few milestones, an exponential — but every one of those has knobs
(where the steps are, how fast the exponential) that I'd have to tune, and I have no principled setting for
them. A linear ramp is the maximum-ignorance default: it spreads the widening evenly across training, spends
the cheap small windows over the whole early phase rather than dumping them all at the very start, and has no
knobs to tune beyond its two endpoints. It also has a clean cost consequence I can compute: a window that
ramps linearly from 64 to 1792 has *average* window (64 + 1792)/2 = 928 over the run, versus 1792 if I held
it at the large end the whole way — so on average the windowed attention does roughly half the per-token work,
concentrated as a real saving in the early steps where the window is smallest. The main alternative shape worth considering is warmup-then-hold: ramp the window up to 1792 over the first,
say, third of the run and then hold it there. That would give the late steps their full window sooner. But it
spends the large window over two-thirds of the run instead of just the tail, raising the average window (and
thus the average per-step attention cost) well above the linear ramp's 928 — I'd be paying for wide context
across the whole middle of training, where the model is still consolidating and doesn't yet need the full
reach. The linear-to-the-end schedule reserves the most expensive window for the very end, where the mature
model actually cashes it in, and keeps the average low. And there's a tidy way to see the schedule generalizes
what I had: if I set the start and end of the ramp to the same value, the linear schedule degenerates to a
constant — the fixed window of the last rung is just the zero-slope case of this schedule. So I'm not adopting
a different mechanism, I'm turning on a slope that was implicitly zero. So linear from a small value
at step 0 to a large value at the final step, and I let the measured result tell me whether something fancier
is even worth considering.

There's a practical constraint from FlexAttention I have to respect, though, and it's what pins down the
quantization. Every time the window changes, the block mask changes, and a new mask means recompiling the
block-sparse kernel. If I let the window take an arbitrary value at every single step I'd be recompiling on
every step, and that compilation overhead would swamp the per-step savings I'm chasing — I'd lose more to the
compiler than I'd gain from the smaller window. The fix is to quantize the window to multiples of 64 — the
same block granularity the mask and the kernel already work in — so the window only steps up to a genuinely
new value, and only triggers a recompile, a manageable number of times across the whole run. Let me count
that number: the window goes from 64 to 1792 in steps of 64, which is (1792 − 64)/64 = 27 increments, so ~27
distinct window values and ~27 recompiles across the entire run, not one per step. Twenty-seven one-time
compilations against ~1800 steps is negligible. And every value it takes aligns cleanly to block boundaries
so no partial blocks are wasted. Concretely: at step 0 the window is 64 tokens — one block, as small and
cheap as attention gets — and it ramps linearly to about 1792 tokens by the final step, in 64-token jumps.
The schedule line is a single expression: interpolate `64` up to `1792` over the course of `num_iterations`,
divide by 64, floor to land on a block boundary, multiply back by 64, and materialize it each step into an
int tensor on the device so it can flow straight into the mask predicate.

Let me trace the endpoints of that expression to be sure it does what I intend. At step 0: `64 * ((0 /
num_iterations * (1792 − 64) + 64) // 64) = 64 * ((0 + 64) // 64) = 64 * (64 // 64) = 64 * 1 = 64`. Good —
starts at one block. At `step = num_iterations` (the final step): `64 * ((1 * 1728 + 64) // 64) = 64 * (1792
// 64) = 64 * 28 = 1792`. Good — ends at 1792. And at the midpoint, step = num_iterations/2: `64 * ((864 +
64) // 64) = 64 * (928 // 64) = 64 * 14 = 896` — right at the average I computed, as a linear ramp should be.
So the expression is a clean floor-quantized linear interpolation from 64 to 1792.

```python
# Set the attention blocksize for the current step, in chunks of 64. By @fernbear.bsky.social
attn_blocksize = torch.tensor(
    64 * ((step / args.num_iterations * (1792 - 64) + 64) // 64),
    dtype=torch.int, device='cuda'
)
```

The nice thing is how little has to change downstream. This `attn_blocksize` is the very same name that
already appears inside the `document_causal_mask` from the FlexAttention rung — the `window_mask = q_idx -
kv_idx < attn_blocksize` line — so I don't touch the mask logic at all; I just feed it a step-dependent
window instead of a constant, and `create_block_mask` rebuilds with the current width each forward. The
plumbing is identical; only the number flowing into it now moves. And it composes with the document mask
untouched: a query still can't cross a document boundary, and now additionally can't reach past the current
window, both from the same three-predicate AND. It's also worth stating what this change costs in the ledger:
nothing. No new parameters (the window is a schedule, not a learnable), no new tensors of consequence (one
scalar int per step), no new compute machinery (the windowing kernel already exists from last rung) — it is
purely a schedule on a hyperparameter I already have, which is the cheapest kind of win there is and the same
category as the momentum warmup and the LR schedule. That's why it's worth doing even for a modest expected
gain: the downside is bounded to "the schedule shape was slightly wrong," recoverable by re-sloping, with no
parameters or compute sunk.

Let me put numbers on the "cheaper early steps" to know how much I'm actually saving. The windowed attention
does ~T·W score computations per head, so at fixed T = 65536 the attention work is directly proportional to
the window W. At step 0, W = 64: that's 65536 × 64 ≈ 4.2M score pairs per head. At the end, W = 1792: 65536 ×
1792 ≈ 117M pairs — 28× more. So the earliest steps do roughly one twenty-eighth of the attention arithmetic
of the final steps; the attention matmul is nearly free at the start and grows linearly to full cost by the
end. Averaged over the run, the attention work is proportional to the average window 928 rather than 1792, so
if I were holding at 1792 the whole time the attention portion of every step would be ~1.9× what the ramp
averages. Attention isn't the entire step — the MLPs, embedding, and head are fixed cost — but it's the
dominant piece I profiled last rung, so roughly halving its average should be a visible step_avg cut, largest
in the early steps and shrinking to zero savings by the final step where the ramp reaches 1792. The 27
recompiles don't eat this: each is a one-time compile of the kernel for a given window, reused for every step
until the window next increments, so the compile cost amortizes across the ~65 steps that share each window
value.

What I expect: the early steps get cheaper because they run with a 64-token window instead of the full fixed
one, which trims per-step time over the first stretch of training; the curriculum lets the model nail local
structure before being asked to use long-range context, which should let it reach the bar in slightly fewer
steps; and the staged early phase should tighten the variance that the fixed-window record showed. So the
falsifiable signature is a *step_avg* drop (the distinguishing fingerprint — the last three rungs moved step
count, this one should move per-step time via the cheaper early windows) alongside a small step-count cut and
a steadier, ideally lower val_loss. I'd expect the step count from 1875 toward ~1750 with step_avg falling
below 161.84 ms. If step_avg *doesn't* drop, that would say the early small-window steps aren't actually where
the attention time was, and I'd look at whether the mask rebuild or something else dominates.

There's a tension in the endpoint I should name. Because the window only reaches 1792 at the very last step,
the model gets very little training *at* its full context — the widest window is exercised for only a handful
of final steps, yet the validation loss is what the whole schedule is optimizing toward. If long-range context
genuinely matters for the val set and the model barely trains with the full window, val_loss could regress: I'd
have curriculum-ed the model into a mostly-local regime and never given it enough wide-context steps to exploit
the long dependencies. The reason I think the ramp-to-the-end shape is still right is that the loss is
dominated by local structure (the argument above), so the marginal long-range bits the wide window unlocks are
small, and even a short exposure at the end is enough to pick them up on top of the strong local model the
curriculum built. But it's a genuine knob: if val_loss comes in worse, the fix is to reach the full window
*earlier* (steepen the ramp or cap it before the final step) so the model gets more full-context steps,
trading back some of the early cheapness for late-context training. For now the bet is that local-first with a
brief wide-context tail is the right allocation, because that's where the entropy is.

The chain: the FlexAttention record cut steps 37% but left a fixed window and raised variance, and a fixed
window assumes the right amount of context is constant across training — wrong at both ends, since early on
the model can't use long-range context (and pays ~28× for it in slower steps) while late on it can and
should; so I grow the window linearly from 64 to ~1792 tokens (average 928, ~half the per-token attention
work of holding at 1792), quantized to multiples of 64 so FlexAttention recompiles only ~27 times across the
run, with the schedule expression traced to start at 64 and end at 1792 — which makes early steps cheaper
(step_avg down), acts as a short-to-long-context curriculum that reaches the bar in slightly fewer steps, and
tames the run-to-run variance the fixed window left behind.
