I have to start the ladder somewhere, and the honest place to start is with the simplest thing that
could conceivably work, run it, and let its number tell me what the math-reasoning setting actually
punishes. The cheapest preference loss to stand up — cheapest in every sense, memory, compute, and
conceptual machinery — is the one that needs no reference model at all, so that is where I begin. Let
me derive it from scratch so I know exactly what I am putting in the `compute_preference_loss` slot and
exactly what I expect it to do, because the whole point of a first rung is to be a clean, interpretable
floor: not the best number I can get, but the number whose *failure mode* I can read cleanly enough to
tell the second rung what to fix.

The thing that bothers me about the obvious baseline, DPO, before I have even run it, is a cost and a
mismatch. Take the cost literally, because on this box it is not rhetorical. The base is a 1.5B model
trained full-parameter with AdamW in mixed precision. Per parameter that is, roughly, an fp32 master
weight (4 bytes), a bf16 working copy (2 bytes), a gradient, and two Adam moments in fp32 (8 bytes) —
call it ~18 bytes/param, so ~27 GB of optimizer-side state before activations. ZeRO-2 shards the
gradients and the Adam moments across the 4 GPUs, which is exactly why this fits. Now bolt a frozen
reference on top. The reference has no gradient and no optimizer state, but its *weights* are resident
and, being frozen, are not sharded the way the trainable state is — a full bf16 copy, ~3 GB, replicated
on every GPU, plus the activation memory and FLOPs of a second forward pass. Concretely the DPO step is
two full forwards over the concatenated chosen+rejected batch — one through the policy (with grad), one
through the reference (under `no_grad`) — and one policy backward; the reference-free step is one policy
forward and one backward. So DPO's reference is, per GPU, ~3 GB I do not get back and a whole extra
forward over a doubled batch, and all of it exists only to compute a baseline I immediately subtract
off. On a memory-tight 4×GPU box that 3 GB is often the difference between fitting `cutoff_len=2048` and
not.

The mismatch is the one that actually itches, and it is not about money. At generation — which is
exactly how this task is scored, greedy decoding graded by sympy — there is no reference model anywhere.
The model produces a chain of tokens and is judged on whether that chain is correct, i.e. on the
per-token likelihood the policy itself assigns. But the reward DPO optimizes is
`r(x,y) = β log[π_θ(y)/π_ref(y)]`, a log-*ratio* against the reference. Those are different functions of
the same response. Satisfying `r(x,y_w) > r(x,y_l)` rearranges to
`log π_θ(y_w) − log π_θ(y_l) > log π_ref(y_w) − log π_ref(y_l)` — a condition on summed log-probs
*offset by whatever the reference assigns* — and there is no reason that implies the thing I am graded
on, that the policy's per-token score ranks the correct chain above the wrong one. Two policies can win
the DPO inequality identically while assigning wildly different *absolute* likelihoods to the correct
chain, because the reference offset absorbs the difference. The reward I optimize and the metric I am
judged by are not the same object, and nothing in the DPO loss forces them to converge.

So the cleaner thing to want is to make the reward I optimize *be* the metric the model is generated and
ranked by. Don't optimize one quantity and hope it transfers to another; optimize the transfer target
directly. The generation-ranking metric is the average per-token log-likelihood, `(1/|y|) log π_θ(y|x)`
— that is the quantity greedy decoding implicitly maximizes step by step, and the quantity that decides,
token by token, whether the correct chain stays on the beam. Let me try to build the preference reward
directly out of that. The naive first instinct — just use the policy's own log-probability,
`r = β log π_θ(y)`, dropping the reference — is reference-free for free, and the memory problem
evaporates. But it is the *summed* log-prob, and summed log-prob has a structural length problem: every
extra token contributes another `log π ≤ 0`, so longer sequences score systematically lower. Put a
number on it. Suppose the policy is fluent and assigns, on average, −0.5 nats/token to both a correct
and an incorrect chain. A 200-token correct derivation then scores −100; a 100-token truncated wrong
answer scores −50. The wrong chain "wins" by 50 nats on nothing but length. When `y_w` happens to be
longer than `y_l` — which in math, where correct full derivations run longer than the wrong chains that
give up early, is common — the model has to overcome that handicap, and the only lever it has is to
crank token probabilities on the long winning sequence, baking in a "long = good" artifact that is
verbosity, not quality. The summed-log-prob reward would teach the model to pad.

The fix is sitting in what I already said I want. The generation metric is not the summed log-prob, it
is the *average*: `(1/|y|) log π_θ(y|x)`. The `1/|y|` is exactly the length normalization that cancels
the 50-nat handicap I just computed — every response is scored per token, so the 200-token and
100-token chains both come back to −0.5 and sit on the same footing, and there is no incentive to
inflate probabilities just to beat a length penalty. And it *is* the quantity the model is ranked by at
decode. So the two problems — the train/generation mismatch and the length bias — collapse into one
fix: use the average log-likelihood as the reward,
`r_SimPO(x,y) = (β/|y|) log π_θ(y|x)`. Reference-free (it is the policy's own per-token score),
generation-aligned (it *is* the ranking metric, scaled by β), and length-debiased (the `1/|y|` puts
winner and loser per token).

There is a third option in the design space I should knock out explicitly rather than leave implicit,
because it is the tempting compromise: keep the reference but length-normalize it too — a
length-normalized DPO. That would fix the length bias without dropping the anchor, and it is a real,
sensible loss. But it does not solve the problem I opened this rung to solve, which is the *cost* and the
train/generation *mismatch*. It still pays the 3 GB and the extra forward every batch, so I have not
bought the cheap floor; and its reward is still a log-*ratio* against the reference, so the reward I
optimize is still a different function than the per-token score I am graded on. Normalizing DPO trades
one of SimPO's two virtues away to keep an anchor I have not yet earned the right to say I need. The
disciplined thing at rung one is to drop the reference entirely, take the pure policy-only per-token
reward, and let the number tell me whether the anchor was load-bearing. If it was, the next rung buys it
back knowing exactly what it is paying for; if it was not, I have the cheap floor for free. So the
reference-free averaged reward is not just the simplest option, it is the most *informative* first
experiment.

Do I actually need `π_ref` for anything once I have done this? The reference was there for the KL leash,
to keep the policy from running off to degenerate high-reward strings. Dropping it loses the explicit
regularizer, so I should be honest that I am trading a theorem for a hope. But the training regime is a
practical leash here, and I can enumerate the pieces of it: I start from a strong math-SFT model rather
than a blank slate, so the policy is already sitting in a good basin; the learning rate is `5e-7`, two
to three orders of magnitude below typical SFT, so each step barely perturbs the weights; and there are
only four passes over 10K diverse problems, so there is little time to drift far. None of that is a
bound on the KL, but together they are enough reason to try the simpler policy-only reward before paying
for a frozen reference every batch. I will watch for the drift the KL term would have prevented — the
signature would be the training reward margin looking healthy while generation quality falls — rather
than assume it away.

Now plug `r_SimPO` straight into Bradley-Terry, exactly as DPO does with its reward:
`L = −E[ log σ( (β/|y_w|) log π(y_w) − (β/|y_l|) log π(y_l) ) ]`. Before I trust it, check the gradient,
because a loss whose gradient does the wrong thing is worse than no loss. With
`u = (β/|y_w|) log π(y_w) − (β/|y_l|) log π(y_l)` and loss `−log σ(u)`, `d/du[−log σ(u)] = −(1 − σ(u)) =
−σ(−u)`, so the per-example weight is `σ(−u)`, large exactly when `u` is negative — when the policy
*wrongly* gives the loser a higher average log-likelihood than the winner. That is the right thing to
up-weight, and there is no reference model anywhere in it. And each log-prob gradient is divided by its
own length, so `∇r_SimPO(y_w) = (β/|y_w|) ∇log π(y_w)`: a long response and a short one push with
comparable magnitude, because the `1/|y|` that debiased the reward also debiases the gradient. Contrast
DPO, whose gradient moves `∇log π(y_w) − ∇log π(y_l)` un-normalized, letting a response with twice the
tokens contribute roughly twice the gradient and dominate the batch. The reward choice and the gradient
agree, and they agree on the same length-normalization principle — that is the internal consistency I
wanted before running anything.

Two quick sanity checks on the reward before I complicate it with a margin. First, a degenerate limit: a
completely untrained, uniform policy over a vocabulary of size `V` assigns every token log-prob `−log V`,
so both winner and loser have average per-token log-prob `−log V`, their difference is exactly 0
regardless of the two lengths, and `L = −log σ(−γ) > 0` still hands back a nonzero gradient. Good — the
loss is not accidentally satisfied by a policy with no opinion; it pushes until a real gap opens, and the
length normalization means it does so identically for a 10-token and a 500-token pair. Second, the reason
the *average* is the right ranking metric and not an arbitrary choice: greedy decoding, which is how this
task scores, builds one chain token by token, and the sympy checker grades that single chain. Among the
candidate continuations the policy could have produced, the one that survives to be graded is the one the
policy scores highest, and under any length-aware comparison of unequal-length chains that score is the
per-token average, not the sum — the sum would always prefer the shortest chain. So training the average
log-likelihood to rank the correct chain above the wrong one is training the exact quantity that decides,
at decode, which chain gets graded. The reward and the metric are, by construction, the same object —
which is the mismatch I opened this rung complaining DPO does not have.

Is the Bradley-Terry skeleton enough? It only asks `r(y_w) > r(y_l)` — it is satisfied the instant the
winner outscores the loser by an infinitesimal amount. Getting the sign right is a weak requirement; the
lesson of margins (the max-margin idea behind SVMs, the "home advantage" offset in Bradley-Terry ranking
models) is that a comfortable gap generalizes better than a barely-separating one. So put a margin into
the preference model itself: demand the reward gap exceed `γ > 0` before the loss is satisfied,
`p(y_w ≻ y_l|x) = σ(r(x,y_w) − r(x,y_l) − γ)`. The `−γ` shifts the sigmoid so the loss is not near-minimal
until `r(y_w) − r(y_l)` has crossed `γ`, keeping the model pulling the winner above the loser until there
is a real cushion. Plug in `r_SimPO`:
`L_SimPO = −E[ log σ( (β/|y_w|) log π(y_w) − (β/|y_l|) log π(y_l) − γ ) ]`.

Let me watch the margin actually work on a couple of concrete pairs, because "it demands a gap" is only
convincing if I can see the gradient refuse to switch off before the gap arrives. Take `β = 2.0`,
`γ = 1.0`, so the margin in per-token units is `γ/β = 0.5` nats — the winner must be, on average,
`e^{0.5} ≈ 1.65×` more likely per token than the loser before the loss relaxes. Suppose a pair where the
winner already leads: `avg_w = −0.40`, `avg_l = −0.60`, so `Δ = 0.20` nats/token. The argument to the
sigmoid is `u = βΔ − γ = 2(0.20) − 1 = −0.60`, `σ(u) = 0.354`, loss `−ln 0.354 = 1.04`, and the weight
`σ(−u) = σ(0.60) = 0.646`. So even though the winner is *already* per-token more likely, because the
0.20 gap has not cleared the 0.50 margin the loss is still large and the gradient is still pushing hard.
Widen the pair to `Δ = 0.60`: `u = 0.20`, weight `σ(−0.20) = 0.450`, loss `0.60`. At `Δ = 1.0`:
`u = 1.0`, weight `0.269`, loss `0.31`. The gradient does not shut off at `Δ = 0` (the bare BT resting
point); it keeps pouring in until `Δ` is comfortably past `γ/β = 0.5` and only then tapers — which is
exactly the "make the gap comfortable, not just correct" behavior I put the margin in for. Setting
`γ = 0` recovers the bare Bradley-Terry loss on average log-probs, so the margin is a clean additive
generalization, not a different object.

How big should `γ` be, and why does `β` want to be 2.0 here when DPO runs at 0.1? Both fall out of the
per-token scale. The reward feeding the sigmoid is an average log-prob difference, and those differences
live around 0.1–0.3 nats/token on this kind of data — small. If I used DPO's `β = 0.1`, the logit would
be `0.1 × 0.2 = 0.02`, essentially the flat middle of the sigmoid, no learning signal at all. `β = 2.0`
turns that same 0.2-nat gap into a logit of 0.4, in the responsive region — `β` has to scale *inversely*
with the magnitude of the reward it multiplies, and the per-token average is ~20× smaller than DPO's
summed log-ratio, so `β` is ~20× larger. That is not a coincidence to memorize, it is the calibration.
Given `β = 2.0`, the margin `γ` trades off two failures: too small and I am back to merely asking for the
right sign; too large and I demand an unrealistic per-token gap and over-suppress fluent losing responses
whose tokens the model still needs. `γ = 1.0` (a 0.5-nat/token cushion) is the standard middle for a 1.5B
math base, and I will treat it as a knob to tune rather than derive.

Now I have to land this in the actual edit surface, and here the trajectory's substrate does most of the
work for me. The frozen loop already exposes exactly the two hooks SimPO needs. First, because the
`pref_loss=simpo` flag puts `simpo` in the `["ipo","orpo","simpo"]` set, `concatenated_forward` divides
each per-response log-prob by `valid_length` *before* my loss ever sees it — so the
`policy_chosen_logps`/`policy_rejected_logps` handed to me are already the **average** per-token
log-probs, the length normalization done for me. Second, because `simpo` is in the reference-free set in
`finetuning_args.py`, `use_ref_model` is False, the reference model is never loaded, and my loss lands in
the top, reference-free branch of `compute_preference_loss`. So my entire contribution is the
`simpo_loss` helper: form the average-log-prob difference `pi_logratios = chosen − rejected`, subtract the
code-space margin `γ/β`, and return `−logsigmoid(β·logits)`.

I should verify the code-space margin actually reproduces the objective, because the harness applies the
single `β` multiply outside the subtraction and it is easy to double-count. What I want the sigmoid to
see is `βΔ − γ`. What the code computes is `β·(Δ − γ/β)`. Expand: `β·Δ − β·(γ/β) = βΔ − γ`. With the
concrete numbers, `γ/β = 1.0/2.0 = 0.5`, so `logits = Δ − 0.5` and the loss is `−logsigmoid(2.0·(Δ −
0.5)) = −logsigmoid(2Δ − 1)`, and `βΔ − γ = 2Δ − 1` — identical. So subtracting `γ/β` in code-space and
multiplying by `β` once is exactly the margin objective, not a scaled version of it. The implicit rewards
for logging are `β·policy_chosen_logps` and `β·policy_rejected_logps`, the length-normalized policy-only
reward. (The full scaffold fill is in the answer.)

I should flag, before I run it, the one regime where I expect this to be fragile — and it is precisely
this task's regime. Preferences over *math* solutions, where the winning and losing chains can be nearly
identical, differing in one wrong step. A contrastive objective there can do a perverse thing: it widens
the *reward margin* by pushing the loser's probability down, but because the chosen sequence shares
almost every token with the rejected one, dragging the rejected down drags the chosen down too — the
absolute likelihood of the *correct* answer can fall even as the margin grows. And I can see in the
gradient why SimPO has no defense against this. The weight `σ(−u)` shuts off only when the *margin* `u`
is large; it says nothing about whether that margin was won by raising `avg_w` or by lowering `avg_l`.
Nothing in `L_SimPO` mentions `avg_w` in isolation — the loss is a function of the difference alone —
so there is no term that notices, let alone objects, when the correct chain's absolute per-token
likelihood is sliding. SimPO's margin, if anything, asks for a *bigger* gap, which on near-duplicate
pairs means a *harder* shove on the shared tokens. And SimPO has no anchor on the chosen sequence's
absolute likelihood at all (it is reference-free and purely relative). The quantity the benchmark
actually rewards is greedy correctness, which lives on the absolute likelihood of a correct chain, not
the margin.

It is worth being precise about *where* the erosion comes from, because the naive story — "the shared
tokens get dragged down" — is not quite right and the correct version is sharper. On a near-duplicate
pair, the winner and loser share a long prefix and branch at one step. A shared token `t_i` in that
prefix receives `+β/|y_w|` from the winner term of the gradient and `−β/|y_l|` from the loser term. If
the two chains were exactly equal length these would cancel and the prefix would be untouched; but they
are not — the correct chain runs longer, so `β/|y_w| < β/|y_l|`, the loser term wins, and every shared
prefix token gets a small *net negative* push. That residual, times a long shared prefix, is real
downward pressure on tokens the correct chain depends on. Add the softmax coupling at the branch point,
where suppressing the wrong-step logit reshapes the whole distribution the correct step also lives in,
and the mechanism is concrete: it is the length *mismatch* in the normalizer, not the sharing per se,
that leaks the loser's suppression onto the winner. And SimPO cannot see it happening, because — I said
this above and the gradient confirms it — the loss is a function of the margin alone and has no term that
reads `avg_w` by itself.

So my falsifiable expectation for the floor is concrete. SimPO will train stably — the loss is
well-behaved and the tiny learning rate keeps it in the SFT basin — and the reward margin and reward
accuracy logged during training will look *healthy*, because the loss optimizes exactly them. GSM8K,
near-saturated for this base model, will barely move: there is no headroom there for either a gain or the
erosion to show. The erosion, if it is real, surfaces where the correct chain is long and the wrong chain
is a near-duplicate — MATH-500 and, most sharply, AIME, the hardest and highest-variance benchmark, where
a correct solution is a long competition-grade derivation and the reward-margin push has the most shared
tokens to drag down. AIME moves in ~3.3-point quanta (one problem out of thirty), so I am watching for it
to sit at the bottom of that quantized ladder — a single correct problem, or none — while GSM8K and
MATH-500 hold. If that is the shape of the number, it is the signature of exactly this failure: a healthy
margin bought by an eroding absolute likelihood. And it tells the next rung what to fix — stop letting
the relative objective drag the correct chain's likelihood down, either by re-anchoring the correct
chain to a reference so its fall is measured and penalized, or by trading the saturating sigmoid, which
keeps paying out as the margin grows, for a target the model can reach and sit at instead of one it
forever climbs. If instead AIME comes in respectably, my diagnosis is wrong and the unanchored relative
objective is fine on this data — but I do not expect that, and the AIME number is the single reading I am
setting up the ladder to take.
