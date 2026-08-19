Both single-constant readings beat PowerSign and AddSign cleanly on every column at both scales —
78.23/84.28/66.13 and 78.19/84.17/65.96 against PowerSign's 77.36/83.39/65.17 on ViT-S/16, and the
same pattern at ViT-B/16. That's real confirmation of the thing I was trying to isolate: freeing
the search to touch how momentum itself gets tracked and signed, not just deepening a fixed
expression tree, is where the gain over the restricted-search-space result actually sits. But
neither Ablation_0.9 nor Ablation_0.99 clears AdamW itself — both fall short on ImageNet at both
scales (78.23/78.19 against 78.89; 79.54/79.90 against 80.12), and mostly on ReaL too. Only one
cell, Ablation_0.99's V2 at ViT-B/16, edges past AdamW, and one cell out of twelve is exactly the
kind of thing I should treat as noise, not signal. So: a single accumulated-then-signed buffer
beats a restricted-tree search, but it doesn't beat the incumbent. The "collapse to one constant,
ship something lighter than AdamW" hope from the last rung is dead. Something about forcing one β
to do everything is costing real accuracy, and I need to understand what before I try anything
else.

Think about what one β is actually being asked to do. It sits inside `interp(g, m, β)`, which
produces the *next* value of the tracked quantity, and that same next value is immediately what
gets signed and stepped with. Those are two different jobs wearing one dial. Job one: remember —
how much of the gradient history should the buffer retain, so that a single noisy step doesn't
whipsaw the direction. That job wants β close to 1, a long memory, because the whole point of
momentum is to average out noise across many steps. Job two: react — how much weight should the
step actually give to *this* step's gradient, right now, before deciding a sign. That job wants
some real weight on the fresh gradient, because a step that's 99% stale history and 1% fresh
information is slow to notice the ground has shifted. A single β sitting at 0.9 undershoots memory
to buy reactivity; a single β sitting at 0.99 buys memory by sacrificing reactivity. Both
ablations, read this way, aren't two independent failures — they're the same tradeoff, sampled at
two different points along the same losing dial. That's a hypothesis about *why* the numbers came
out the way they did, and it makes a sharp, checkable prediction: if it's right, no single β should
be able to do noticeably better than either of these two, because the dial itself is the
bottleneck, not its setting.

Before I test that prediction with more single-β sweeps — which would cost more training runs to
learn something I might already have on hand — let me go back to what the search actually
produced, because I've been provisionally treating its second `interp` call as possibly cosmetic,
and I never actually checked that. The raw discovered program, before I collapsed it down to a
single buffer for the last rung's test, did not use one constant. It used two, chained: one buffer
gets updated from the gradient and the *previous* value of a second buffer at a constant near 0.9;
the second buffer then gets updated from the gradient and the *just-updated* first buffer at a
constant near 1.1. That second constant is worth pausing on: it's *above* one, so that second step
is not a convex interpolation at all, it's an extrapolation past the second buffer's own value —
something a human writing signSGD-momentum by hand would never write, and something I threw away
as noise on the way to the single-β simplification. Given that both single-β readings undershoot
AdamW, and my working hypothesis is that memory and reactivity are two jobs fighting over one dial,
this un-thrown-away detail is exactly the kind of structure that would let two jobs live on two
separate dials. I want to know precisely what it computes before deciding whether it's signal.

Substitute one equation into the other rather than eyeball it. Write the first buffer's update as
`m_t = interp(g_t, v_{t-1}, b1)`, i.e. `m_t = (1−b1)·g_t + b1·v_{t−1}`, and the second as
`v_t = interp(g_t, m_t, b2)`, i.e. `v_t = (1−b2)·g_t + b2·m_t`, with b1 ≈ 0.9 and b2 ≈ 1.1.
Plug the first into the second: `v_t = (1−b2)·g_t + b2·[(1−b1)·g_t + b1·v_{t−1}]`. Multiply out and
collect terms in `g_t` and `v_{t−1}`: the coefficient on `g_t` is `(1−b2) + b2·(1−b1)`, and the
coefficient on `v_{t−1}` is `b2·b1`. Plug in the rounded constants: the `g_t` coefficient is
`(1−1.1) + 1.1·(1−0.9) = −0.1 + 0.11 = 0.01`, and the `v_{t−1}` coefficient is `1.1·0.9 = 0.99`.
Those two coefficients sum to exactly 1 — not approximately, exactly, because that's what
composing two affine interpolations always does (the constant terms of two convex-or-not
combinations that both sum their own coefficients to 1 will themselves sum to 1 after
substitution; it's algebra, not a coincidence of these particular numbers). So
`v_t = 0.01·g_t + 0.99·v_{t−1}`, which is precisely the closed form of a *single* EMA run at
β ≈ 0.99 directly on the raw gradient stream — the same form as `interp(g, v, 0.99)`, full stop.
The second buffer never needed the first buffer's help; it's just a plain, slow momentum EMA in
disguise, and the exact same algebra holds at the raw, unrounded constants the search actually
wrote (0.8999999761581421 and 1.109133005142212), landing at β ≈ 0.99822 instead of the rounded
0.99 — a slightly longer memory than the mental-math version, but the same identity, so this isn't
an artifact of rounding either way I check it.

That changes what I now believe the program is doing, and it directly answers the "is the second
`interp` cosmetic" question I deferred: it is not cosmetic, and it is not the two-independent-jobs
structure I first assumed either. There is really only *one* buffer worth storing — the slow EMA,
which I'll keep calling `m` from here on since it's the one that persists across steps, updated at
β₂ ≈ 0.99. The *other* quantity, the one that actually gets signed and turned into a step, is not
`m` itself — it's the fresh, never-stored blend `interp(g_t, m_{t−1}, β₁)` with β₁ ≈ 0.9, computed
transiently each step and thrown away right after signing. So the two jobs I described above
really do land on two separate dials, exactly as the hypothesis predicted, but not by storing two
buffers — by storing one buffer at the slow rate and *recomputing* the fast blend fresh every step
from that one buffer plus the current gradient. Memory lives in β₂, applied to what persists;
reactivity lives in β₁, applied only to the transient quantity that gets signed. This is a
structure neither Ablation_0.9 nor Ablation_0.99 could express, because both forced the persisted
quantity and the signed quantity to be identical.

Two more pieces of the raw program are worth closing out while I'm here, since they were
provisionally deferred rather than settled. The `clip` and `arcsin` applied to the incoming
gradient: I ablate them the same way — remove, retrain, compare — and there is no quality drop
either way I've checked it, so they're not carrying signal and I drop them. And there's a
`cosh(update)` statement that assigns into the variable I'm calling `m`, but it sits at the very
end of the program body, after everything that reads `m` has already run, and `m` gets completely
overwritten at the top of the *next* iteration before anything reads it again — so this statement's
output is never read by anything, ever. That one isn't an empirical call at all; it's a directly
checkable fact about the program's data flow, so I drop it unconditionally rather than testing it.

Putting the surviving pieces together: one persisted buffer `m`, updated on its own slow constant
β₂ ≈ 0.99; a step formed by signing a fast blend of that buffer with the current gradient at
β₁ ≈ 0.9; decoupled weight decay; no ε, no square root, no second full buffer. As before, since the
signed step is exactly ±1 per coordinate before weight decay, its norm is larger than AdamW's
m/√v step or the single-β ablations' step at the same nominal learning rate, so I carry the same
kind of lr reduction into tuning that the ablations needed, and because the effective decoupled
weight-decay shrinkage is lr·λ, I let λ move opposite to lr in the sweep so I'm not silently
changing two things when I only meant to change the update rule.

So the rung-3 fill I'm committing to is the decoupled two-constant rule the algebra actually
supports, not a guess: `c_t = β₁·m_{t−1} + (1−β₁)·g_t`, step by `sign(c_t)` plus decoupled weight
decay, then update the persisted buffer on its own slower rate, `m_t = β₂·m_{t−1} + (1−β₂)·g_t`,
with β₁ = 0.9, β₂ = 0.99 as the defaults the search itself pointed to. I don't yet know whether
this actually clears AdamW — the memory/reactivity story is a mechanism that explains why the two
single-constant readings failed, not a guarantee that decoupling recovers the lost ground, let
alone exceeds it. What I do have is a sharper, falsifiable expectation than before: if the
mechanism is right, this rule should beat both Ablation_0.9 and Ablation_0.99 by a real, consistent
margin at both scales, not a coin-flip-sized one, since it's the one structure on this ladder that
lets memory and reactivity each get their own dial instead of fighting over one.
