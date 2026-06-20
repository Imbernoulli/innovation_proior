Five minutes flat now, 1875 steps, val_loss 3.2783 — FlexAttention at 64K context did most of the work of
cutting the run, but it left two things on the table that I said I'd come back to. One is the variance: this
record is twitchier than I'd like, around 0.005 std, mean safely under the bar but the odd run pokes above
3.28. The other is that I made a choice I never examined — I picked a single window size, `attn_blocksize`,
and held it fixed for the entire run. That fixed window is the obvious next thing to interrogate, because
the assumption behind a fixed window is that the *right* amount of context is the same at step 0 as at the
last step, and when I actually think about what the model is doing across training, that assumption looks
plainly wrong in both directions.

Consider the early steps. The representations are still forming — the residual stream barely encodes
anything yet, the attention heads haven't specialized, the model can't meaningfully exploit a token that's a
thousand positions back because it doesn't yet have the machinery to relate them. So a large window early in
training is compute I'm spending on long-range interactions the model is incapable of using. It's not just
neutral waste, either: it makes every early step slower, since the windowed attention cost scales with the
window size, and the early steps are precisely where the loss is dropping fastest and I'd most like them
cheap. Now consider the late steps. Here it's the opposite — the representations are mature, the heads know
what they're doing, and long-range context genuinely pays off for the harder predictions, so this is exactly
where a *large* window earns its cost. A single fixed window can't be right for both ends: set it large and
I over-spend early on context I can't use; set it small and I under-serve the late training where context
matters. Whatever fixed value I chose, it's a compromise that's wrong everywhere except possibly one moment
in the middle.

The moment I phrase it that way the fix is obvious: don't hold the window fixed — grow it. Start small, when
only local context is usable and I want the steps cheap, and increase it to large by the end, when long-range
context pays off. And this isn't only a compute-allocation trick; it's a *curriculum*. By restricting the
model to a short window early, I'm forcing it to learn local structure first — the easy, high-payoff
short-range statistics, the bigram-and-trigram bones of the language — before it ever has to cope with
long-range dependencies, and only later widening its horizon once the local machinery is in place. There's a
clean analogy to the rest of what I've already done: I zero-init the head and the residual projections so the
network starts from a quiet, low-confidence state and *earns* its structure rather than asserting random
opinions; I warm up the LR and the momentum so the optimizer doesn't take violent steps before the loss
landscape is understood. A growing window is the same discipline applied to *what the model is allowed to
look at* — start it with a narrow field of view it can actually make sense of, and only broaden it as it
becomes capable of using the breadth. Short-context-then-long-context is a sensible order to learn language
in, and curricula like that tend to stabilize training. Which connects straight back to the variance I'm
trying to tame: a gentler, staged early phase should make the runs less twitchy, not more, so the same change
that makes early steps cheaper should also pull the run-to-run spread back in. That's the kind of change I
want — one lever that improves two things at once, and a lever I get essentially for free because the
windowing machinery is already in place from the last rung.

So the window becomes a function of the training step instead of a constant. Why linear, and not some
curve? I have no evidence the model's appetite for context follows any particular shape, and a linear ramp
is the maximum-ignorance default — it spreads the widening evenly across training, spends the cheap small
windows over the whole early phase rather than dumping them all at the very start, and has no knobs to tune
beyond its two endpoints. So linear from a small value at step 0 to a large value at the final step, and I
let the measured result tell me whether something fancier is even worth considering.

There's a practical constraint from FlexAttention I have to respect, though. Every time the window changes,
the block mask changes, and a new mask means recompiling the block-sparse kernel. If I let the window take an
arbitrary value at every single step I'd be recompiling on every step, and that compilation overhead would
swamp the per-step savings I'm chasing — I'd lose more to the compiler than I'd gain from the smaller window.
The fix is to quantize the window to multiples of 64 — the same block granularity the mask and the kernel
already work in — so the window only steps up to a genuinely new value, and only triggers a recompile, a
manageable number of times across the whole run (roughly once per 64-token increment, not once per step), and
every value it takes aligns cleanly to block boundaries so no partial blocks are wasted. Concretely: at step
0 the window is 64 tokens — one block, as small and cheap as attention gets — and it ramps linearly to about
1792 tokens by the final step, in 64-token jumps. The schedule line is a single expression: interpolate `64`
up to `1792` over the course of `num_iterations`, divide by 64, floor to land on a block boundary, multiply
back by 64, and materialize it each step into an int tensor on the device so it can flow straight into the
mask predicate.

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
plumbing is identical; only the number flowing into it now moves.

What I expect: the early steps get cheaper because they run with a 64-token window instead of the full fixed
one, which trims per-step time over the first stretch of training where most steps live; the curriculum lets
the model nail local structure before being asked to use long-range context, which should let it reach the
bar in slightly fewer steps; and the staged early phase should tighten the variance that the fixed-window
record showed. Cheaper early steps plus fewer steps plus a calmer run — all from making one hyperparameter a
schedule instead of a constant.

The chain: the fixed window I left in place assumes the right amount of context is constant across training,
but early on the model can't use long-range context (and pays for it in slower steps) while late on it can
and should; so I grow the window linearly from 64 to ~1792 tokens, quantized to multiples of 64 so
FlexAttention only recompiles the mask a manageable number of times — which makes early steps cheaper, acts
as a short-to-long-context curriculum that reaches the bar in fewer steps, and tames the run-to-run variance
the fixed window left behind.
