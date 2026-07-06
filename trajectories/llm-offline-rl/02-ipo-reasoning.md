SimPO landed exactly where I feared it might, and the shape of the number tells me what to fix. Seed 42:
GSM8K 86.05, MATH-500 74.0, AIME 3.33 — average 54.46. Read the three columns the way I set them up to be
read. GSM8K at 86.05 is essentially the base model's ceiling; it is near-saturated, it was never going to
move, and it did not. MATH-500 at 74.0 is respectable and roughly where a competent preference stage
should sit. But AIME at 3.33 is one correct problem out of thirty — the benchmark quantizes in 3.33-point
steps, so 3.33 is the *floor above zero*, a single problem, and that is the tell. This is precisely the
failure I flagged for the floor: a purely relative, reference-free objective that has *no anchor on the
absolute likelihood of the correct chain*. On AIME the correct solution is a long competition-grade
derivation and the rejected chain is a near-duplicate that branches at one wrong step; SimPO widens the
reward margin by pushing the rejected down, and because the two chains share almost every token — and,
as I worked out, the correct chain runs longer so the length-normalizer leaks the loser's suppression
onto it — the correct chain's likelihood is dragged down with it. The benchmark is greedy correctness,
which lives on that absolute likelihood, so the place where correct chains are longest and most fragile —
AIME — is exactly where it cratered.

Let me quantify how lopsided this is, because the average hides it. SimPO's 54.46 is carried by the two
easy benchmarks; the hard one is on the floor. If I imagine a fix that only repairs AIME and leaves the
other two alone, one recovered AIME problem is +3.33 on that column and +1.11 on the average — a single
problem out of thirty is worth a full point of headline number. That is the leverage point, and it is the
one SimPO gave away. So the diagnosis is sharp: the relative margin is not the problem, the *unanchored
absoluteness* is. I need to bring back a reference so "the correct chain stays likely" is measured against
something, and — separately — I want to ask whether the saturating sigmoid is even the right loss shape,
because a sigmoid that keeps paying out as the margin grows is exactly what licenses the unbounded push
that drags the chosen down. Let me derive the next rung from those two suspicions rather than bolt on a
patch.

Start from what actually goes wrong with the Bradley-Terry *logit* objective, because once I see it I will
know what to change. I have preference pairs and I want a policy near a reference `π_ref`, with one knob —
the regularization coefficient β — that genuinely controls *how near*. The reward-modeling route fits an
Elo reward with Bradley-Terry, `p(y_w ≻ y_l) = σ(r(y_w) − r(y_l))`, then maximizes
`E_π[r] − β·KL(π‖π_ref)`, whose optimum is the exponential tilt `π* ∝ π_ref exp(r/β)`. Now take the
simplest deterministic preference, `p*(y_w ≻ y_l) = 1`, and follow it all the way through, because the
disease shows up cleanest at the extreme. To represent `p* = 1`, Bradley-Terry must send
`r(y_w) − r(y_l) → +∞`. Feed that into the tilt: `π*(y_l)/π*(y_w) = (π_ref(y_l)/π_ref(y_w))·exp((r_l −
r_w)/β) → 0`, so `π*(y_l) = 0` — and read the `β` in that exponent carefully: the limit is 0 *for every
finite β*. I can crank β to a million, demand the policy barely move from `π_ref`, and the optimum still
annihilates the loser. The KL term, the one thing supposed to keep me near `π_ref`, has silently stopped
binding. The more deterministic the preference, the weaker the regularization, and at `p* = 1` it does not
regularize at all.

And it gets worse with finite data, which is my actual situation. Even a true `p* = 0.8` can come out
empirically `1` (two heads in two tosses), and for a language model almost every pair is observed exactly
once, so the empirical preference lands in `{0,1}` constantly — every recorded `(y_w, y_l)` is a
`p̂ = 1`. This is not a corner case, it is the *typical* case, and it means the "for every β" annihilation
above is not a pathological limit I can avoid, it is what the objective is pointed at on essentially every
example. It is the offline version of the same disease I just watched eat SimPO's AIME: an unbounded
objective overfitting the preference and ignoring the anchor.

Why does this matter for the loss *shape*, mechanically? Both RLHF and DPO are optimizing the
Bradley-Terry logit `Ψ(q) = log(q/(1−q))` of the preference probability, and that logit is **unbounded**
as `q → 1`. A single deterministic comparison can therefore contribute an unbounded amount to the
objective and overwhelm the fixed `β·KL` term — there is no `β` large enough to cap an infinite pull.
DPO, which folded the reward away and put the implicit log-ratio inside the `log σ`, inherits exactly this:
where the data says the winner always wins, the loss `−log σ(β·Δ)` keeps decreasing as `Δ` grows without
bound, with no finite resting point, so gradient descent never stops pushing the log-ratio apart. SimPO
inherits the same saturating `log σ` and the same "keep pushing" behavior, only reference-free, which is
*worse* for my AIME problem because there is not even a reference holding the correct chain in place while
the push happens. So the cure I want is a **bounded** objective with a **finite resting point** — a loss
the model can sit at instead of one it forever climbs.

Before I reach for a whole new family, I should check the cheaper repair, because if it worked I would
take it. The minimal edit to SimPO that addresses my diagnosis is: keep SimPO's `−log σ` shape but paste
the reference back in, subtracting the reference's winner-over-loser log-ratio from the policy's — a
"reference-anchored SimPO." That fixes the *anchor* half of the diagnosis: the correct chain is now
measured against `π_ref`. But it leaves the *shape* half untouched. The loss is still `−log σ(β·(anchored
gap − γ))`, still saturating, still paying out more the larger the gap grows, so the optimizer is still
licensed to win the gap by any means including pushing the loser (and the near-duplicate winner) down
without bound. The anchor would slow that but not stop it, because `−log σ` has no finite resting point —
it always prefers a larger gap. My two suspicions off SimPO's AIME were *two* mechanisms, anchor and
shape, and the reference-anchored-SimPO patch only touches one. So the cheap repair is genuinely
insufficient, and that is the argument for changing the loss *shape*, not just re-inserting the reference.

Put the whole family in view so the fix is a choice within it, not an ad-hoc swap. For a nondecreasing
`Ψ: [0,1] → ℝ`, maximize `J(π) = E_{y∼π,y'∼μ}[Ψ(p*(y ≻ y'))] − β·KL(π‖π_ref)`. With `Ψ = logit` and
Bradley-Terry this is exactly RLHF/DPO, and the disease is the unboundedness of `Ψ` at the endpoints. The
simplest bounded nondecreasing choice is the identity, `Ψ(q) = q`, mapping `[0,1] → [0,1]`. Then the score
in the exponent of the optimum becomes the *total* preference `p*(y ≻ μ) ∈ [0,1]`, bounded — so no matter
how deterministic any individual preference, the exponent cannot run to infinity and β keeps biting. That
is the fix in principle: cap `Ψ` and the KL regularizer is restored. But I want it *offline*, no RL, no
reward model — the thing DPO bought me and I refuse to give back. So I follow the same
analytic-optimum-to-equations route DPO used. From `π* ∝ π_ref exp(g/β)` with `g(y) = p*(y ≻ μ)`, take the
log-ratio of two actions to kill the normalizer and define the reference-corrected log-ratio
`h_π(y,y') = log[π(y)π_ref(y') / (π(y')π_ref(y))]`. The optimum satisfies `h*(y,y') = (g(y) − g(y'))/β`,
one scalar equation per pair. There is no Bradley-Terry likelihood to plug into here — `g` is a preference
probability, not a reward — so instead of a likelihood I fold the root-finding into one squared residual:
`L(π) = E[(h_π(y,y') − (p*(y≻μ) − p*(y'≻μ))/β)²]`. Minimizing the squared residual drives `h_π` to satisfy
the optimum's equation; that is the whole trick.

I need to know the squared landscape has no spurious optima before I trust "minimize the residual" to mean
anything. Parametrize policies by logits `s`; then `h_{π_s}(y,y') = (s(y) − s(y')) + log(π_ref(y')/π_ref(y))`,
and `L` is quadratic in `s`, with pure-quadratic part `Σ μ(y)μ(y')(s(y) − s(y'))²`. That form is a graph
Laplacian quadratic — positive-semidefinite — so `L` is convex and every local min is global. The only
flat direction is the all-ones shift `s → s + c`, which leaves every difference `s(y) − s(y')` fixed; but
a constant logit shift does not change the *policy* (the softmax quotients it out), so the minimizing
policy is unique given `Supp(μ) = Supp(π_ref)`. Good — no spurious optima, the residual has one answer.

Now make it usable, because I never observe `p*` — only the Bernoulli labels `I(y,y')` in my dataset. Swap
the unknown gap for the sampled label: `E[(h_π(y,y') − I(y,y')/β)²]`. The naive term-by-term expectation
does not obviously match, because the inner expectation of `I` is a *single* pairwise preference, not the
*total*-preference gap I derived. The equality holds up to a π-independent constant by a symmetry over the
random draw of the pair: `h_π` is additive and antisymmetric — write `h_π(y,y') = a_y − a_{y'}` with
`a_y = log π(y) − log π_ref(y)` — and `y, y'` are iid from μ, so partner-averaging the label recovers the
total preference, `E_{y'}[I(y,y')|y] = p*(y ≻ μ)`. Each recorded comparison `(y_w, y_l)` then furnishes two
oriented terms, `(y_w, y_l, 1)` and `(y_l, y_w, 0)`; average them, use antisymmetry `h_π(y_l,y_w)² =
h_π(y_w,y_l)²`, and the bracket is `(H − 1/β)² + H²` with `H = h_π(y_w,y_l)`.

Let me actually complete that square rather than assert where it lands, because the target `1/(2β)` is the
whole payoff and I want to see it fall out. Expand: `(H − 1/β)² + H² = H² − 2H/β + 1/β² + H² = 2H² − 2H/β +
1/β²`. Factor the 2 out of the `H`-dependent part: `2(H² − H/β) + 1/β² = 2(H − 1/(2β))² − 2·(1/(4β²)) +
1/β² = 2(H − 1/(2β))² + 1/(2β²)`. The `1/(2β²)` is a π-independent constant; drop it and the overall factor
of 2, and the whole thing collapses to one strikingly simple regression:
`L_IPO(π) = E_D[(h_π(y_w,y_l) − 1/(2β))²]`. (I checked the algebra numerically at `H = 0.7, β = 0.1`: both
sides equal 86.98, so the constant I dropped is genuinely constant.)

Look at the two gradients side by side, because the shape difference is the whole point of the swap.
SimPO's is `−β·σ(−u)·(∇r_w − ∇r_l)` — the weight `σ(−u)` shrinks toward zero as the gap `u` grows, so
the *loss* keeps decreasing but only ever asymptotically; there is no gap at which the gradient is exactly
zero, so descent keeps nudging the gap wider forever, which is the "keep pushing" behavior. IPO's is
`2(h_π − 1/(2β))·(∇h_π)` — the weight `2(h_π − 1/(2β))` is *linear in the error* and crosses exactly zero
when `h_π = 1/(2β)`, and past that point it flips sign and *pulls the gap back down*. That sign flip is
the thing `−log σ` structurally cannot do: an overshooting pair is corrected, not merely under-rewarded.
So the squared loss is not a cosmetic change of the sigmoid — it converts a monotone "always want more
gap" pressure into a restoring force around a fixed set-point, which is exactly the finite resting point I
argued the objective needed. Read what this tells the policy to do, because it is exactly the two things I
wanted off the back of SimPO's AIME collapse. First, it brings the **reference back**: `h_π(y_w,y_l) = [log π(y_w) − log π_ref(y_w)]
− [log π(y_l) − log π_ref(y_l)]` regresses the gap between the policy's winner-over-loser log-ratio and the
*reference's* onto a target. The correct chain's likelihood is now measured against `π_ref`, so dragging it
down below the reference is penalized — the anchor SimPO never had. Second, it replaces the **saturating
sigmoid with a finite target** `1/(2β)`, the same target for every pair. There is no `log σ` that keeps
paying out as the gap grows; if the policy already separates winner from loser by `1/(2β)` more than the
reference does, the loss is zero and the gradient vanishes — it *stops pushing*. That is precisely the
brake the unbounded objective lacked: a deterministic preference just means `I = 1` always, which still
only ever asks the gap to hit `1/(2β)`, never `+∞`. The unboundedness that let SimPO march the chosen
chain's probability down to win a margin it would never stop wanting — gone by construction.

Let me confirm the knob now works on the minimal deterministic instance, since that is the disease I am
trying to cure. Two actions, `p*(y_1 ≻ y_2) = 1`, uniform `π_ref`, uniform `μ` over `{y_1, y_2}`. The total
preferences: `p*(y_1 ≻ μ) = ½·p*(y_1 ≻ y_1) + ½·p*(y_1 ≻ y_2) = ½·½ + ½·1 = ¾`, and by symmetry
`p*(y_2 ≻ μ) = ¼`, so the gap is `½`. The optimum `π* ∝ π_ref exp(p*(·≻μ)/β)` with uniform `π_ref` gives
`π*(y_1)/π*(y_2) = exp((¾ − ¼)/β) = exp(1/(2β))`, hence `π*(y_1) = σ(1/(2β))`. Now watch β do its job: as
`β → ∞` (maximal regularization) `π*(y_1) = σ(0) = ½`, the policy stays *exactly* at the uniform reference
— which the logit objective could never do for any finite β, it sat at `π(y_2) = 0` always. As `β → 0`
(no regularization) `σ(∞) = 1`, the deterministic optimum. At `β = 0.1`, my setting, `σ(1/0.2) = σ(5) =
0.993` — close to but not at the deterministic corner, a slightly-regularized winner. The whole continuum
between reference and deterministic is reachable and *governed by β*; contrast the logit objective pinned
at `π(y_2) = 0` for every β. The bounded `Ψ` is the entire difference, and the minimal instance shows it.

The target `1/(2β)` passes the same β limit-check the minimal instance did, which reassures me the
regression is regularized the way I claim. As `β → ∞`, the target `1/(2β) → 0`, so the loss becomes
`(h_π − 0)² = h_π²`, which is minimized at `h_π = 0` — the policy's winner-over-loser gap is regressed to
*exactly the reference's* gap, i.e. the policy is pinned at `π_ref`. Maximal regularization holds the
policy at the reference, as it should. As `β → 0`, the target `1/(2β) → ∞`, demanding infinite separation
beyond the reference — no regularization, the deterministic corner. So the single scalar `1/(2β)` carries
the entire regularization dial, continuously from "be the reference" to "separate at all costs," which is
the property the unbounded logit objective lost. My `β = 0.1` sits far toward the low-regularization end
(target 5.0), which is consistent with the "far target, anchor-dominated" reading I just reached.

One decision the bandit derivation did not force but the sequence setting does, and it dovetails with the
substrate. For a token sequence, `log π(y)` is the *sum* of per-token log-probs, so `h_π` — a difference of
summed log-ratios — scales with completion length, and the single fixed target `1/(2β)` would mean
different things at different lengths. The fix is the *average* per-token log-prob, dividing each sequence
log-prob by its length, so `h_π` is in per-token units and the target is a per-token gap. The harness
already does this for me: `ipo` is in the `["ipo","orpo","simpo"]` set, so `concatenated_forward` hands
`compute_preference_loss` the length-averaged log-probs. Unlike SimPO, though, `ipo` is *not* in the
reference-free set in `finetuning_args.py`, so `use_ref_model` is True, the frozen reference is loaded, and
my loss lands in `self.dpo_loss` — which is exactly where the reference-corrected `h_π` is available,
because `dpo_loss` receives all four log-probs. In TRL's `dpo_loss`, selecting `loss_type="ipo"` forms
`logits = (chosen_logps − rejected_logps) − (reference_chosen_logps − reference_rejected_logps)` — that is
`h_π` — and returns `losses = (logits − 1/(2β))²`. So for this named baseline my "edit" is just selecting
the IPO branch the harness already routes to; the average log-probs and the reference are supplied by the
frozen loop. (The full scaffold dispatch is in the answer.)

Before I write down predictions I should look hard at one number, because it changes how much I expect the
finite target to actually contribute. My β is 0.1, so the target is `1/(2β) = 5.0` — but `h_π` is now in
*per-token averaged* units, where the raw quantities are log-probs of order −0.3 to −0.7 and their
reference-corrected differences are fractions of a nat. A target of 5.0 nats/token is enormous on that
scale; a pair would have to separate the policy's winner-over-loser gap from the reference's by five full
nats *per token* to zero the loss, which on near-duplicate math pairs is almost certainly never reached.
So I should be honest about what IPO is really doing here: with an out-of-reach target, `(h_π − 5)²` has
gradient `2(h_π − 5)`, which for realistic `h_π ≈ 0` is a roughly *constant* gentle pull upward on the
gap — the squared loss behaves, in the regime the data actually occupies, like a mild non-saturating
push rather than a brake that clicks off. The finite-target brake is real but it sits so high that on this
data it mostly serves as insurance against runaway rather than as an active resting point. Which means the
mechanism I should expect to carry the improvement over SimPO is not the target so much as the **reference
anchor** — the `−(reference_chosen_logps − reference_rejected_logps)` term that penalizes the correct
chain for falling below `π_ref`. I will hold that distinction in mind when I read the number, because if
the gain is small it is consistent with "the anchor helps a little and the far target barely engages,"
whereas a large gain would say the far target is engaging more than I think.

The cost, so I am not pretending this is free: IPO reintroduces exactly the frozen reference SimPO shed —
the ~3 GB resident copy per GPU and the second forward every batch, four forwards per step (chosen and
rejected, each through policy and reference) versus SimPO's two. SimPO was reference-free and cheap but
had no anchor; IPO pays for the anchor with the reference. That is a deliberate trade at this rung: I am
buying back the one thing whose absence I diagnosed as the cause of the AIME collapse, and the number will
tell me whether it was worth the memory.

Now the falsifiable expectations against SimPO's 86.05 / 74.0 / 3.33. The two changes — reference anchor
and finite target — both attack the absolute-likelihood erosion that I diagnosed as the cause of SimPO's
AIME 3.33. So my prediction is specific: IPO should *recover AIME* relative to SimPO, because the reference
now penalizes letting the correct chain fall and there is no unbounded push to drive the collapse — one or
two more correct problems, moving AIME off its 3.33 floor. But given the anchor-not-target reading above, I
expect the recovery to be *modest*, not a jump — the far target is not adding aggressive growth, it is just
removing the license to erode. GSM8K is near-saturated and should sit around SimPO's 86 — no headroom for
the fix to show. MATH-500 I expect roughly flat or slightly up: it was not collapsing under SimPO (74.0),
so there is less to repair, and IPO's squared loss is conservative — once the (far) target is nominally
approached it stops improving, which on the middle benchmark may leave it near where SimPO already was. So
the signature I am betting on is "AIME up a little, GSM8K flat, MATH-500 flat-to-slightly-up, average up
modestly." If instead AIME stays on the floor, my diagnosis was wrong — either the erosion was not the
cause, or the reference anchor at β = 0.1 is too weak to hold the long chains — and the next rung would
have to attack the absolute likelihood more directly, and more aggressively, rather than through a
conservative reference-relative regression. That is the test IPO is running.
