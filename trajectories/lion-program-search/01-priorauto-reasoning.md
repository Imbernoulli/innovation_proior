Let me start from what actually bothers me. Every large model I see being trained runs on AdamW,
or its memory-cheaper cousin Adafactor. Nobody chose that update rule by searching a space of
alternatives — Adam was one person's design, decoupled weight decay was another person's patch on
top of it, and the whole thing stuck because it works well enough and nobody has beaten it badly
enough to dislodge it. That's not evidence it's optimal; it's evidence that hand search is slow
and humans stop once something is "good enough." I want to actually search the space of update
rules, but before I invest in building any search machinery of my own I should ask a cheaper
question first: has anyone already searched, and does what they found already clear the bar?

Two prior attempts at automatic discovery exist, and they fail for opposite reasons, which is
useful — the failure modes tell me what any search I build has to avoid. The "learning to
optimize" line parameterizes the update rule as a small neural network and trains it on a handful
of small tasks. I can rule this branch out on inspection, without spending a single training run:
a network trained on a few thousand steps of a toy task has no mechanism to behave sensibly a
hundred thousand steps into training a model many orders of magnitude larger — it is simply
outside its training distribution, and I have no way to read *why* it does what it does, so I
couldn't even diagnose the failure if I saw it. That's a structural objection, not an empirical
one, and I don't need a number to act on it: I will not build on black-box learned optimizers.

The second branch is different, and it deserves an actual test rather than a dismissal. Neural
Optimizer Search treats the update rule as a small expression tree — combinations of the gradient
g and a bias-corrected momentum m under unary and binary math operators — and searches that tree
with RL or Monte Carlo sampling. This is a *real* automatic search, over a real space of
executable programs, not a black box. It produced two named results, PowerSign and AddSign, both
built around comparing the sign of the current gradient to the sign of the momentum: PowerSign
scales the gradient by e raised to ±1 depending on whether they agree (a large multiplicative
swing, since e^1 ≈ 2.72 and e^−1 ≈ 0.37), and AddSign scales it by 2 when they agree and by 0 —
i.e. skips the step — when they disagree. Both are genuinely novel structures a human wouldn't
have written down by hand; PowerSign and AddSign do not resemble Adam, NAdam, or plain signSGD.
That novelty is exactly why I owe this branch a real check rather than an armchair dismissal: an
automatic search already ran, on a real space, and produced something I don't already know the
answer for.

So the question for this first check is narrow and cheap to answer: plug PowerSign and AddSign,
exactly as published, into my actual target training recipe — ViT-S/16 and ViT-B/16 trained from
scratch on ImageNet with RandAugment and Mixup, each optimizer's own learning rate and weight
decay tuned the same way everyone else's is — and see whether either one clears AdamW's numbers.
If either does, I am close to done: I inherit a validated automatic result, save myself the cost
of building and running a much larger search, and can spend my effort validating and simplifying
rather than discovering. If neither does, I have real evidence, not a hunch, about *why* — and the
"why" matters more here than the bare pass/fail, because it tells me what to fix in my own search
space rather than just "try harder."

Here is my actual worry about the restricted-tree design, stated as a falsifiable prediction
rather than an assumption. The search space that produced PowerSign and AddSign fixes its
operands to {g, m} and its structure to a bounded expression tree. Crucially, m itself — how it
gets tracked, how many buffers exist, what rate it updates at — is not something the search
controls; m is handed to the tree as a fixed ingredient, already computed by a fixed EMA rule
outside the search. So whatever this search space can produce, it can only ever be some function
combining a gradient with an EMA of the gradient computed a fixed, unsearched way. It cannot
discover that the buffer itself should update on a different schedule than it's applied, because
the buffer's update rule was never part of the search in the first place — only the *combination
rule* was searched. If that's the real bottleneck — if the interesting design freedom in an
update rule lives in *how the memory itself is tracked*, not just in how a fixed memory gets
combined with the instantaneous gradient — then a fixed-operand, fixed-buffer search space is
structurally capped below whatever a freer search space could reach, independent of how cleverly
it searches within its box. I don't get to *conclude* that from armchair reasoning alone; I get to
predict it, and then let the actual training numbers either support or contradict the prediction.
The prediction, concretely: PowerSign and AddSign, run under my target protocol, land at or below
AdamW's 78.89 (ViT-S/16) / 80.12 (ViT-B/16) ImageNet numbers, and if they do fall short, the shape
of the shortfall is consistent with an inflexible, off-the-shelf momentum term rather than an
inflexible combination rule — though that second part is a reading of the *result*, not something
this test alone can prove; distinguishing "the operator set is too restrictive" from "the buffer
update is too restrictive" would need a search that frees only one of the two and holds the other
fixed, which is more machinery than a first check like this one buys me.

There is also a purely practical reason to run this check before building anything: cost. Testing
two already-fully-specified update rules costs exactly two more tuned training runs at each of the
two model scales I already have baseline numbers for — nothing new to build, nothing to search,
just plug in the formula and tune lr/λ like every other entrant. Building my own search
machinery — a program representation, mutation operators, a pruning and caching layer to survive
an infinite and sparse program space, a funnel of increasingly expensive meta-validation tasks to
catch overfitting to a cheap proxy — is a large investment I want to make only once I know it's
actually necessary. This check is the cheapest possible experiment that could make that investment
unnecessary.

So, concretely, what I'm committing to for this rung: run PowerSign and AddSign in their default
forms — PowerSign with base α = e and no internal decay (f(t)=1), so update = e^{sign(g)·sign(m)}·g;
AddSign with base α = 1 and no internal decay, so update = (1 + sign(g)·sign(m))·g, with m the
usual bias-corrected gradient EMA — under the exact protocol above, tuning each one's own peak
learning rate and weight decay on a log scale the same way AdamW's were tuned. I'll read the
result off all three columns (ImageNet, ReaL, V2) at both scales, not just the headline ImageNet
number, since a rule that wins on ImageNet but not on the harder-to-game ReaL/V2 splits is a
weaker signal than one that wins everywhere. Whatever comes back settles the question this rung
was built to answer: is a restricted, fixed-operand, fixed-momentum search space enough, or do I
need to widen the search to let the buffer's own tracking rule be part of what gets searched?
