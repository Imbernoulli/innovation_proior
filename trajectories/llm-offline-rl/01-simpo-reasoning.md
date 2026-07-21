I have to start somewhere, and the honest place to start is the simplest thing that could conceivably
work, run it, and let its number tell me what the math-reasoning setting actually punishes. The cheapest
preference loss to stand up — cheapest in memory, compute, and conceptual machinery — is the one that
needs no reference model at all. The point of a first experiment is not the best number I can get, but the
one whose failure mode I can read cleanly enough to tell the next attempt what to fix.

Two things bother me about the obvious baseline, DPO, before I have even run it: a cost and a mismatch.
The cost is not rhetorical on this box. The base is a 1.5B model trained full-parameter with AdamW; ZeRO-2
shards the gradients and Adam moments across the 4 GPUs, which is exactly why it fits. Bolt a frozen
reference on top and its *weights* are resident and, being frozen, not sharded the way the trainable state
is — a full bf16 copy, ~3 GB, replicated on every GPU, plus the activations and FLOPs of a second forward.
The DPO step is two full forwards over the concatenated chosen+rejected batch (policy with grad, reference
under `no_grad`) plus one policy backward; the reference-free step is one forward and one backward. So
DPO's reference is ~3 GB per GPU I do not get back and a whole extra forward, all to compute a baseline I
immediately subtract off — and on a memory-tight box that 3 GB is often the difference between fitting
`cutoff_len=2048` and not.

The mismatch is the one that actually itches. At generation — which is exactly how this task is scored,
greedy decoding graded by sympy — there is no reference model anywhere. The model produces a chain of
tokens and is judged on the per-token likelihood the policy itself assigns. But the reward DPO optimizes
is `r(x,y) = β log[π_θ(y)/π_ref(y)]`, a log-*ratio* against the reference. Satisfying
`r(x,y_w) > r(x,y_l)` rearranges to `log π_θ(y_w) − log π_θ(y_l) > log π_ref(y_w) − log π_ref(y_l)` — a
condition on log-probs *offset by whatever the reference assigns* — and nothing in it implies the thing I
am graded on, that the policy's own score ranks the correct chain above the wrong one. Two policies can win
the DPO inequality identically while assigning wildly different *absolute* likelihoods to the correct
chain, because the reference offset absorbs the difference.

So the cleaner thing to want is to make the reward I optimize *be* the metric the model is ranked by at
generation: the per-token log-likelihood, `(1/|y|) log π_θ(y|x)`, the quantity greedy decoding implicitly
maximizes step by step. The naive first instinct — just use `r = β log π_θ(y)`, dropping the reference —
is reference-free for free, but it is the *summed* log-prob, and summed log-prob has a structural length
problem: every extra token contributes another `log π ≤ 0`, so longer sequences score systematically
lower. Put a number on it. If the policy assigns, on average, −0.5 nats/token to both a correct and an
incorrect chain, a 200-token correct derivation scores −100 while a 100-token wrong answer scores −50 —
the wrong chain wins by 50 nats on nothing but length. In math, where correct derivations run longer than
the wrong chains that give up early, `y_w` is often the longer one, so the model would have to overcome
that handicap by cranking token probabilities on long winning sequences, baking in a "long = good"
artifact. Summed log-prob would teach the model to pad.

The fix is in what I already said I want: not the sum but the *average*, `(1/|y|) log π_θ(y|x)`. The
`1/|y|` cancels the 50-nat handicap — both chains come back to −0.5 nats/token — and it *is* the quantity
the model is ranked by at decode. So the train/generation mismatch and the length bias collapse into one
move: use the average log-likelihood as the reward, `r_SimPO(x,y) = (β/|y|) log π_θ(y|x)`. Reference-free,
generation-aligned, length-debiased.

There is a tempting compromise I should knock out explicitly: keep the reference but length-normalize it
too — a length-normalized DPO. It fixes the length bias without dropping the anchor, and it is a sensible
loss. But it does not solve the two problems I opened on: it still pays the 3 GB and the extra forward, so
I have not bought the cheap floor, and its reward is still a log-ratio against the reference, so I am still
optimizing a different function than the per-token score I am graded on. Normalizing DPO trades one of
SimPO's virtues away to keep an anchor I have not yet earned the right to say I need. The disciplined move
is to drop the reference entirely and let the number tell me whether it was load-bearing. If it was, the
next attempt buys it back knowing exactly what it is paying for.

Dropping `π_ref` loses the explicit KL leash, so I should be honest that I am trading a theorem for a
hope. But the training regime is a practical leash: I start from a strong math-SFT model already sitting
in a good basin, the learning rate is `5e-7` (two to three orders below typical SFT, so each step barely
perturbs the weights), and there are only four passes over 10K problems, so there is little time to drift
far. None of that bounds the KL, but together it is enough reason to try the policy-only reward first. I
will watch for the drift the KL term would have prevented — the signature would be the training reward
margin looking healthy while generation quality falls.

Now plug `r_SimPO` into Bradley-Terry exactly as DPO does with its reward:
`L = −E[ log σ( (β/|y_w|) log π(y_w) − (β/|y_l|) log π(y_l) ) ]`. With
`u = (β/|y_w|) log π(y_w) − (β/|y_l|) log π(y_l)`, `d/du[−log σ(u)] = −σ(−u)`, so the per-example weight is
`σ(−u)`, large exactly when `u` is negative — when the policy *wrongly* gives the loser the higher average
log-likelihood. And each log-prob gradient is divided by its own length,
`∇r_SimPO(y_w) = (β/|y_w|) ∇log π(y_w)`, so a long response and a short one push with comparable magnitude.
Contrast DPO, whose gradient moves `∇log π(y_w) − ∇log π(y_l)` un-normalized, letting a response with twice
the tokens contribute roughly twice the gradient. The reward choice and the gradient agree on the same
length-normalization principle.

Is the Bradley-Terry skeleton enough? It only asks `r(y_w) > r(y_l)` — satisfied the instant the winner
outscores the loser by an infinitesimal amount. Getting the sign right is a weak requirement; the lesson
of margins (max-margin SVMs, the "home advantage" offset in Bradley-Terry ranking) is that a comfortable
gap generalizes better than a barely-separating one. So put a margin into the preference model itself:
demand the reward gap exceed `γ > 0`, `p(y_w ≻ y_l|x) = σ(r(x,y_w) − r(x,y_l) − γ)`, keeping the model
pulling the winner above the loser until there is a real cushion. Plug in `r_SimPO`:
`L_SimPO = −E[ log σ( (β/|y_w|) log π(y_w) − (β/|y_l|) log π(y_l) − γ ) ]`. Setting `γ = 0` recovers the
bare Bradley-Terry loss, so the margin is a clean additive generalization.

How big should `γ` be, and why does `β` want to be 2.0 here when DPO runs at 0.1? Both fall out of the
per-token scale. The reward feeding the sigmoid is an average log-prob difference, and those differences
live around 0.1–0.3 nats/token — small. At DPO's `β = 0.1` the logit would be `0.1 × 0.2 = 0.02`,
essentially the flat middle of the sigmoid, no signal. `β = 2.0` turns that same 0.2-nat gap into a logit
of 0.4, in the responsive region: `β` scales *inversely* with the magnitude of the reward it multiplies,
and the per-token average is ~20× smaller than DPO's summed log-ratio, so `β` is ~20× larger. That is the
calibration, not a coincidence. Given `β = 2.0`, `γ = 1.0` is a 0.5-nat/token cushion — small enough not
to over-suppress fluent losing responses whose tokens the model still needs — and I treat it as a knob to
tune rather than derive.

Now land it in the edit surface, where the substrate does most of the work. Because `pref_loss=simpo`
puts `simpo` in the `["ipo","orpo","simpo"]` set, `concatenated_forward` divides each per-response log-prob
by `valid_length` before my loss sees it — the `policy_chosen_logps`/`policy_rejected_logps` handed to me
are already the **average** per-token log-probs. And because `simpo` is in the reference-free set in
`finetuning_args.py`, `use_ref_model` is False, no reference is loaded, and my loss lands in the top branch
of `compute_preference_loss`. So my entire contribution is the `simpo_loss` helper: form
`pi_logratios = chosen − rejected`, subtract the code-space margin `γ/β`, and return `−logsigmoid(β·logits)`.
The harness applies the single `β` multiply outside the subtraction, so the code computes `β·(Δ − γ/β) =
βΔ − γ` — exactly the margin objective. The logged implicit rewards are `β·policy_chosen_logps` and
`β·policy_rejected_logps`. (The full fill is in the answer.)

Before I run it, I should flag the one regime where I expect this to be fragile — and it is precisely this
task's regime: preferences over *math* solutions, where the winning and losing chains can be near-identical,
differing at one wrong step. A contrastive objective there can do a perverse thing: it widens the reward
*margin* by pushing the loser's probability down, but because the chosen sequence shares almost every token
with the rejected one, dragging the rejected down drags the chosen down too. The gradient shows SimPO has
no defense: the weight `σ(−u)` shuts off only when the margin `u` is large; it says nothing about whether
that margin was won by raising `avg_w` or lowering `avg_l`. Nothing in `L_SimPO` mentions `avg_w` in
isolation — the loss is a function of the difference alone — so no term notices when the correct chain's
absolute likelihood is sliding, and SimPO has no reference anchor on it either. The benchmark rewards
greedy correctness, which lives on the absolute likelihood of a correct chain, not the margin.

The mechanism is sharper than "shared tokens get dragged down." On a near-duplicate pair the winner and
loser share a long prefix; a shared token receives `+β/|y_w|` from the winner term and `−β/|y_l|` from the
loser term. If the chains were equal length these would cancel — but the correct chain runs longer, so
`β/|y_w| < β/|y_l|`, the loser term wins, and every shared prefix token gets a small *net negative* push.
It is the length *mismatch* in the normalizer, not the sharing per se, that leaks the loser's suppression
onto the winner, and SimPO cannot see it because the loss reads only the margin.

So my expectation for the floor is concrete. SimPO will train stably — the loss is well-behaved and the
tiny learning rate keeps it in the SFT basin — and the logged reward margin and accuracy will look
healthy, because the loss optimizes exactly them. GSM8K, near-saturated for this base model, has no
headroom and will barely move. The erosion, if it is real, surfaces where the correct chain is long and
the wrong chain is a near-duplicate — MATH-500 and, most sharply, AIME, the hardest and highest-variance
benchmark, where a correct solution is a long derivation with the most shared tokens to drag down. AIME
moves in ~3.3-point quanta, so I am watching for it to sit near the bottom of that quantized ladder while
GSM8K and MATH-500 hold — a healthy margin bought by an eroding absolute likelihood. That reading tells the
next attempt what to fix: stop letting the relative objective drag the correct chain down, either by
re-anchoring it to a reference so its fall is measured and penalized, or by trading the saturating sigmoid
— which keeps paying out as the margin grows — for a target the model can reach and sit at. If AIME instead
comes in respectably, my diagnosis is wrong and the unanchored relative objective is fine on this data; but
I do not expect that, and the AIME number is the single reading I am setting up to take.
