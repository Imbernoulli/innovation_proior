SimPO landed exactly where I feared. Seed 42: GSM8K 86.05, MATH-500 74.0, AIME 3.33, average 54.46. Read
the three columns the way I set them up. GSM8K at 86.05 is essentially the base model's ceiling; it was
never going to move, and it did not. MATH-500 at 74.0 is respectable, roughly where a competent preference
stage should sit. But AIME at 3.33 is one correct problem out of thirty — the benchmark quantizes in
3.33-point steps, so that is the *floor above zero*, and it is the tell. This is the failure I flagged: a
purely relative, reference-free objective with no anchor on the absolute likelihood of the correct chain.
On AIME the correct solution is a long derivation and the rejected chain is a near-duplicate branching at
one wrong step; SimPO widens the margin by pushing the rejected down, the longer correct chain's
length-normalizer leaks that suppression onto the shared prefix, and greedy correctness — which lives on
the absolute likelihood — craters exactly where the correct chains are longest and most fragile.

The average hides how lopsided this is: 54.46 is carried by the two easy benchmarks while the hard one sits
on the floor. A fix that only repairs AIME is worth +3.33 on that column and +1.11 on the average — one
problem out of thirty is a full point of headline number, and it is the leverage SimPO gave away. So the
diagnosis is sharp: the relative margin is not the problem, the *unanchored absoluteness* is. I need a
reference so "the correct chain stays likely" is measured against something, and — separately — I want to
ask whether the saturating sigmoid is even the right shape, because a sigmoid that keeps paying out as the
margin grows is exactly what licenses the unbounded push that drags the chosen down.

Start from what goes wrong with the Bradley-Terry *logit* objective. I have preference pairs, want a policy
near `π_ref`, with one knob β controlling *how near*: fit an Elo reward with Bradley-Terry
`p(y_w ≻ y_l) = σ(r_w − r_l)`, then maximize `E_π[r] − β·KL(π‖π_ref)`, whose optimum is the exponential
tilt `π* ∝ π_ref exp(r/β)`. Now take the simplest deterministic preference, `p* = 1`. To represent it,
Bradley-Terry must send `r_w − r_l → +∞`; feed that into the tilt and
`π*(y_l)/π*(y_w) = (π_ref(y_l)/π_ref(y_w))·exp((r_l − r_w)/β) → 0` *for every finite β*. I can crank β to a
million, demand the policy barely move, and the optimum still annihilates the loser. The KL term, the one
thing supposed to keep me near `π_ref`, has silently stopped binding.

And it gets worse with finite data, which is my situation. A true `p* = 0.8` can come out empirically 1
(two heads in two tosses), and for a language model almost every pair is observed exactly once, so every
recorded `(y_w, y_l)` is a `p̂ = 1`. The annihilation is not a pathological limit I can avoid; it is what
the objective is pointed at on essentially every example — the offline version of the same disease that ate
SimPO's AIME. Mechanically: both RLHF and DPO optimize the logit `Ψ(q) = log(q/(1−q))`, which is
*unbounded* as `q → 1`, so a single deterministic comparison contributes an unbounded pull that no finite
`β·KL` can cap. DPO folded this inside `log σ`: where the data says the winner always wins, `−log σ(β·Δ)`
keeps decreasing as `Δ` grows with no finite resting point, so descent never stops pushing the log-ratio
apart. SimPO inherits the same saturating shape reference-free, which is worse for my AIME problem — not
even a reference holding the correct chain in place. So the cure I want is a **bounded** objective with a
**finite resting point** — a loss the model can sit at instead of one it forever climbs.

Before a whole new family, check the cheaper repair. Keep SimPO's `−log σ` shape but paste the reference
back in, subtracting the reference's winner-over-loser log-ratio — a "reference-anchored SimPO." That fixes
the *anchor* half: the correct chain is measured against `π_ref`. But the loss is still `−log σ(β·(anchored
gap − γ))`, still saturating, still paying out more the larger the gap grows, so the optimizer is still
licensed to win the gap by pushing the loser (and the near-duplicate winner) down. The anchor would slow
that but not stop it, because `−log σ` has no finite resting point. My two suspicions were two mechanisms —
anchor *and* shape — and the patch touches only one. That is the argument for changing the loss shape, not
just re-inserting the reference.

Put the whole family in view. For a nondecreasing `Ψ: [0,1] → ℝ`, maximize
`J(π) = E[Ψ(p*(y ≻ y'))] − β·KL(π‖π_ref)`. With `Ψ = logit` this is RLHF/DPO, and the disease is `Ψ`'s
unboundedness at the endpoints. The simplest bounded nondecreasing choice is the identity, `Ψ(q) = q`; then
the score in the optimum's exponent becomes the *total* preference `p*(y ≻ μ) ∈ [0,1]`, bounded — so no
matter how deterministic any individual preference, the exponent cannot run to infinity and β keeps biting.
That is the fix in principle. But I want it *offline*, no RL, no reward model, so I follow DPO's
analytic-optimum-to-equations route. From `π* ∝ π_ref exp(g/β)` with `g(y) = p*(y ≻ μ)`, take the log-ratio
of two actions to kill the normalizer and define the reference-corrected log-ratio
`h_π(y,y') = log[π(y)π_ref(y') / (π(y')π_ref(y))]`. The optimum satisfies `h*(y,y') = (g(y) − g(y'))/β`,
one scalar equation per pair. There is no Bradley-Terry likelihood to plug in — `g` is a preference
probability, not a reward — so I fold the root-finding into a squared residual,
`L(π) = E[(h_π(y,y') − (g(y) − g(y'))/β)²]`; minimizing it drives `h_π` to satisfy the optimum's equation.

The squared landscape has no spurious optima: parametrize policies by logits `s`, and `L` is quadratic in
`s` with pure-quadratic part `Σ μ(y)μ(y')(s(y) − s(y'))²` — a graph Laplacian quadratic, positive-
semidefinite, so `L` is convex and every local min is global. The only flat direction is the all-ones
shift, which the softmax quotients out, so the minimizing policy is unique. Good — "minimize the residual"
has one answer.

Now make it usable, because I never observe `p*`, only the Bernoulli labels `I(y,y')`. Swap the gap for the
label: `E[(h_π(y,y') − I(y,y')/β)²]`. The equality holds up to a π-independent constant by a symmetry over
the random draw: `h_π` is additive and antisymmetric, `h_π(y,y') = a_y − a_{y'}` with
`a_y = log π(y) − log π_ref(y)`, and `y, y'` are iid from μ, so partner-averaging the label recovers the
total preference, `E_{y'}[I(y,y')|y] = p*(y ≻ μ)`. Each recorded comparison furnishes two oriented terms,
`(y_w, y_l, 1)` and `(y_l, y_w, 0)`; averaging with antisymmetry gives `(H − 1/β)² + H²` with
`H = h_π(y_w,y_l)`. Complete the square: `(H − 1/β)² + H² = 2H² − 2H/β + 1/β² = 2(H − 1/(2β))² + 1/(2β²)`.
Drop the constant `1/(2β²)` and the factor of 2, and it collapses to one strikingly simple regression,
`L_IPO(π) = E_D[(h_π(y_w,y_l) − 1/(2β))²]`.

The gradient side by side is the whole point of the swap. SimPO's is `−β·σ(−u)·(∇r_w − ∇r_l)` — the weight
`σ(−u)` shrinks toward zero as the gap grows but is never exactly zero, so descent nudges the gap wider
forever. IPO's is `2(h_π − 1/(2β))·∇h_π` — the weight is *linear in the error*, crosses exactly zero at
`h_π = 1/(2β)`, and past that flips sign and *pulls the gap back down*. That sign flip is what `−log σ`
structurally cannot do: the squared loss converts a monotone "always want more gap" pressure into a
restoring force around a set-point, the finite resting point the objective needed. And it delivers the two
things I wanted off SimPO's collapse. It brings the **reference back**:
`h_π = [log π(y_w) − log π_ref(y_w)] − [log π(y_l) − log π_ref(y_l)]`, so dragging the correct chain below
`π_ref` is penalized. And it replaces the **saturating sigmoid with a finite target** `1/(2β)`: once the
policy separates winner from loser by `1/(2β)` more than the reference does, the loss is zero and the
gradient vanishes — it stops pushing. A deterministic preference just means `I = 1` always, which still
only asks the gap to hit `1/(2β)`, never `+∞`.

Confirm β now binds on the minimal deterministic instance: two actions, `p*(y_1 ≻ y_2) = 1`, uniform
`π_ref` and `μ`. Total preferences `p*(y_1 ≻ μ) = ¾`, `p*(y_2 ≻ μ) = ¼`, gap `½`, so
`π*(y_1) = σ((¾ − ¼)/β) = σ(1/(2β))`. As `β → ∞` this is `σ(0) = ½`, the policy staying *exactly* at the
uniform reference — which the logit objective could never do for any finite β (it sat at `π(y_2) = 0`
always). As `β → 0`, `σ(∞) = 1`, the deterministic corner. The single scalar `1/(2β)` carries the entire
regularization dial, continuously from "be the reference" to "separate at all costs" — the property the
unbounded logit lost. At my `β = 0.1` the target is 5.0, far toward the low-regularization end.

One decision the bandit derivation did not force but the sequence setting does, and it dovetails with the
substrate. For a token sequence `log π(y)` is the *sum* of per-token log-probs, so `h_π` scales with
length and the single fixed target `1/(2β)` would mean different things at different lengths. The fix is
the *average* per-token log-prob, putting `h_π` and the target in per-token units. The harness does this:
`ipo` is in the `["ipo","orpo","simpo"]` set, so `concatenated_forward` hands the length-averaged
log-probs. Unlike SimPO, `ipo` is *not* in the reference-free set, so `use_ref_model` is True, the frozen
reference is loaded, and my loss lands in `self.dpo_loss` — exactly where the reference-corrected `h_π` is
available, since `dpo_loss` receives all four log-probs. Its `ipo` branch forms
`logits = (chosen − rejected) − (ref_chosen − ref_rejected) = h_π` and returns `(logits − 1/(2β))²`. So for
this named baseline my edit is selecting the branch the harness already routes to. (The full dispatch is
in the answer.)

I should look hard at one number, because it changes what I expect. With `β = 0.1` the target is 5.0 — but
`h_π` is now per-token averaged, where the reference-corrected differences are fractions of a nat. A target
of 5.0 nats/token is enormous on that scale; a pair would have to separate the policy's gap from the
reference's by five full nats *per token* to zero the loss, which on near-duplicate math pairs is almost
never reached. So `(h_π − 5)²` has gradient `2(h_π − 5)`, which for realistic `h_π ≈ 0` is a roughly
*constant* gentle pull upward — the squared loss behaves, in the regime the data occupies, like a mild
non-saturating push, not a brake that clicks off. The finite-target brake is real but sits so high it
mostly serves as insurance against runaway. Which means the mechanism I expect to carry the improvement
over SimPO is not the target but the **reference anchor** — the `−(ref_chosen − ref_rejected)` term
penalizing the correct chain for falling below `π_ref`. I hold that distinction: a small gain is consistent
with "the anchor helps and the far target barely engages," a large one would say the target is engaging
more than I think.

The cost, so I am not pretending it is free: IPO reintroduces the frozen reference SimPO shed — ~3 GB
resident per GPU and four forwards per step versus SimPO's two. SimPO was cheap but unanchored; IPO buys
back the one thing whose absence I diagnosed as the AIME collapse, and the number will tell me whether the
memory was worth it.

Against SimPO's 86.05 / 74.0 / 3.33: both changes attack the absolute-likelihood erosion, so I expect IPO
to *recover AIME* — the reference now penalizes letting the correct chain fall and there is no unbounded
push — by one or two problems off the 3.33 floor. But given the anchor-not-target reading, I expect the
recovery *modest*: the far target adds no aggressive growth, it just removes the license to erode. GSM8K
sits near 86, no headroom. MATH-500 was not collapsing (74.0), so I expect it roughly flat. If AIME instead
stays on the floor, my diagnosis was wrong — either the erosion was not the cause or the anchor at β = 0.1
is too weak to hold the long chains — and the next attempt would have to grow the absolute likelihood more
directly and aggressively than a conservative reference-relative regression.
