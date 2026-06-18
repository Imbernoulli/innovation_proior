SimPO landed exactly where I feared it might, and the shape of the number tells me what to fix. Seed 42:
GSM8K 86.05, MATH-500 74.0, AIME 13.33 — wait, no: AIME came in at **3.33**, a single problem out of
thirty. That is the tell. GSM8K (86.05) is essentially the base model's ceiling — near-saturated, it was
never going to move — and MATH-500 (74.0) is respectable, but AIME collapsing to one correct problem is
the signature of precisely the failure I flagged for the floor: a purely relative, reference-free
objective that has *no anchor on the absolute likelihood of the correct chain*. On AIME the correct
solution is a long competition-grade derivation and the rejected chain is a near-duplicate that branches
at one wrong step; SimPO widens the margin by pushing the rejected down, and because the two chains share
almost every token, the correct chain's likelihood is dragged down with it. The benchmark is greedy
correctness, which lives on that absolute likelihood, so the place where correct chains are longest and
most fragile — AIME — is exactly where it cratered. SimPO's average, 54.46, is being held up by the two
easy benchmarks while the hard one bleeds. So the diagnosis is sharp: the relative margin is not the
problem, the *unanchored absoluteness* is. I need to bring back a reference so "the correct chain stays
likely" is measured against something, and — separately — I want to ask whether the saturating sigmoid is
even the right loss shape, because a sigmoid that keeps paying out as the margin grows is exactly what
licenses the unbounded push that drags the chosen down. Let me derive the next rung from those two
suspicions.

Start from what actually goes wrong with the Bradley-Terry *logit* objective, because once I see it I will
know what to change. I have preference pairs and I want a policy near a reference `π_ref`, with one knob —
the regularization coefficient β — that genuinely controls *how near*. The reward-modeling route fits an
Elo reward with Bradley-Terry, `p(y_w ≻ y_l) = σ(r(y_w) − r(y_l))`, then maximizes
`E_π[r] − β·KL(π‖π_ref)`, whose optimum is the exponential tilt `π* ∝ π_ref exp(r/β)`. Now take the
simplest deterministic preference, `p*(y_w ≻ y_l) = 1`. To represent it Bradley-Terry must send
`r(y_w) − r(y_l) → +∞`. Feed that into the tilt: `π*(y_l)/π*(y_w) = (π_ref(y_l)/π_ref(y_w))·exp((r_l −
r_w)/β) → 0`, so `π*(y_l) = 0` — and that happened *for every β*. I can crank β to a million, demand the
policy barely move, and the optimum still annihilates the loser. The KL term, the one thing supposed to
keep me near `π_ref`, has silently stopped binding. The more deterministic the preference, the weaker the
regularization. And it gets worse with finite data: even a true `p* = 0.8` can come out empirically `1`
(two-of-two), and for a language model almost every pair is observed once, so the empirical preference
lands in `{0,1}` constantly. This is not a corner case, it is the typical case — and it is the offline
version of the same disease I just watched eat SimPO's AIME: an unbounded objective overfitting the
preference and ignoring the anchor.

Why does this matter for the loss *shape*? Both RLHF and DPO are optimizing the Bradley-Terry logit
`Ψ(q) = log(q/(1−q))` of the preference probability — and that logit is **unbounded** as `q → 1`. A
single deterministic comparison can contribute an unbounded amount and overwhelm the fixed `β·KL` term.
DPO, which folded the reward away and put the implicit log-ratio inside the `log σ`, has an unbounded
logit inside: where the data says the winner always wins, the loss keeps decreasing as the log-ratio
grows without bound, with no finite resting point. SimPO inherits the same saturating `log σ` and the
same "keep pushing" behavior, only reference-free, which is *worse* for my AIME problem because there is
not even a reference holding the correct chain in place. So the cure I want is a **bounded** objective
with a **finite resting point** — a loss the model can sit at instead of one it forever climbs.

Put the whole family in view: for a nondecreasing `Ψ: [0,1] → ℝ`, maximize
`J(π) = E_{y∼π,y'∼μ}[Ψ(p*(y ≻ y'))] − β·KL(π‖π_ref)`. With `Ψ = logit` and Bradley-Terry, this is exactly
RLHF/DPO. The disease is unboundedness of `Ψ`. The simplest bounded nondecreasing choice is the identity,
`Ψ(q) = q`, mapping `[0,1] → [0,1]`. Then the score in the exponent of the optimum becomes the *total*
preference `p*(y ≻ μ) ∈ [0,1]`, bounded — so no matter how deterministic any individual preference, the
exponent cannot run to infinity and β keeps biting. That is the fix in principle. But I want it offline,
no RL, no reward model — what DPO got me. So follow the analytic-optimum-to-equations route. From
`π* ∝ π_ref exp(g/β)` with `g(y) = p*(y ≻ μ)`, take the log-ratio of two actions to kill the normalizer and
define the reference-corrected log-ratio `h_π(y,y') = log[π(y)π_ref(y') / (π(y')π_ref(y))]`. The optimum
satisfies `h*(y,y') = (g(y) − g(y'))/β`, one scalar equation per pair. Rather than plug into a
Bradley-Terry likelihood (there is no Bradley-Terry here — `g` is a preference probability, not a reward),
fold the root-finding into one squared residual:
`L(π) = E[(h_π(y,y') − (p*(y≻μ) − p*(y'≻μ))/β)²]`.

I need to know the squared landscape has no spurious optima. Parametrize policies by logits `s`; then
`h_{π_s}(y,y') = (s(y) − s(y')) + log(π_ref(y')/π_ref(y))`, and `L` is quadratic in `s`, with pure-quadratic
part `Σ μ(y)μ(y')(s(y) − s(y'))²` — positive-semidefinite, so `L` is convex, every local min global. The
only flat direction is the all-ones shift `s → s + c`, which leaves every difference fixed; but a constant
logit shift does not change the *policy* (softmax quotients it out), so the minimizing policy is unique
(given `Supp(μ) = Supp(π_ref)`). Good — no spurious optima.

Now make it usable. I never observe `p*`, only Bernoulli labels `I(y,y')`. Swap the unknown gap for the
sampled label: `E[(h_π(y,y') − I(y,y')/β)²]`. The naive term-by-term expectation does not match — the
inner expectation of `I` is a *single* pairwise preference, not the total-preference gap — but the equality
holds up to a π-independent constant by a symmetry over the random draw of the pair, exploiting that `h_π`
is additive and antisymmetric (`h_π(y,y') = (a_y − a_{y'})` with `a_y = log π(y) − log π_ref(y)`) and that
`y,y'` are iid from μ. Partner-averaging the label recovers the total preference:
`E_{y'}[I(y,y')|y] = p*(y ≻ μ)`. Then each recorded comparison `(y_w, y_l)` furnishes two oriented terms,
`(y_w,y_l,1)` and `(y_l,y_w,0)`; averaging them and using antisymmetry `h_π(y_l,y_w)² = h_π(y_w,y_l)²`, the
bracket `(H − 1/β)² + H²` with `H = h_π(y_w,y_l)` completes the square to `(H − 1/(2β))² + const`. Drop the
constant and the whole thing collapses to one strikingly simple regression:
`L_IPO(π) = E_D[(h_π(y_w,y_l) − 1/(2β))²]`.

Read what this tells the policy to do, because it is exactly the two things I wanted off the back of
SimPO's AIME collapse. First, it brings the **reference back**: `h_π(y_w,y_l) = [log π(y_w) − log π_ref(y_w)]
− [log π(y_l) − log π_ref(y_l)]` regresses the gap between the policy's winner-over-loser log-ratio and the
*reference's* onto a target. The correct chain's likelihood is now measured against `π_ref`, so dragging it
down below the reference is penalized — the anchor SimPO never had. Second, it replaces the **saturating
sigmoid with a finite target** `1/(2β)`, the same target for every pair. There is no `log σ` that keeps
paying out as the gap grows; if the policy already separates winner from loser by `1/(2β)` more than the
reference does, the loss is zero and the gradient vanishes — it *stops pushing*. That is precisely the
brake the unbounded objective lacked: a deterministic preference just means `I = 1` always, which still
only ever asks the gap to hit `1/(2β)`, never `+∞`. The unboundedness that let SimPO march the chosen
chain's probability down to win a margin it would never stop wanting — gone.

Let me confirm the knob now works on the minimal deterministic instance, since that is the disease. Two
actions, `p*(y_1 ≻ y_2) = 1`, uniform `π_ref`, uniform μ. Total preferences: `p*(y_1 ≻ μ) = 3/4`,
`p*(y_2 ≻ μ) = 1/4`. The optimum `π* ∝ π_ref exp(p*(·≻μ)/β)` gives `π*(y_1) = σ(1/(2β))`. As β → ∞,
σ(0) = 1/2 — strong regularization actually keeps me at `π_ref`, which the logit objective could never do
for any β. As β → 0, σ → 1, the deterministic optimum. The whole continuum is reachable, governed by β —
contrast the logit objective sitting at `π(y_2) = 0` for *all* β. The bounded `Ψ` is the entire difference.

One decision the bandit derivation did not force but the sequence setting does, and it dovetails with the
substrate. For a token sequence, `log π(y)` is the *sum* of per-token log-probs, so `h_π` — a difference of
summed log-ratios — scales with completion length, and the single fixed target `1/(2β)` would mean
different things at different lengths. The fix is the *average* per-token log-prob, dividing each sequence
log-prob by its length. And the harness already does this for me: `ipo` is in the `["ipo","orpo","simpo"]`
set, so `concatenated_forward` hands `compute_preference_loss` the length-averaged log-probs. Unlike SimPO,
though, `ipo` is *not* in the reference-free set in `finetuning_args.py`, so `use_ref_model` is True, the
frozen reference is loaded, and my loss lands in `self.dpo_loss` — which is exactly where the
reference-corrected `h_π` is available, because `dpo_loss` receives all four log-probs. In TRL's
`dpo_loss`, selecting `loss_type="ipo"` forms `logits = (chosen_logps − rejected_logps) −
(reference_chosen_logps − reference_rejected_logps)` — that is `h_π` — and returns
`losses = (logits − 1/(2β))²`. So for this named baseline my "edit" is just selecting the IPO branch the
harness already routes to; the average log-probs and the reference are supplied by the frozen loop. (The
full scaffold dispatch is in the answer.)

Now the falsifiable expectations against SimPO's numbers. The two changes — reference anchor and finite
target — both attack the absolute-likelihood erosion that I diagnosed as the cause of SimPO's AIME 3.33.
So my prediction is specific: IPO should *recover AIME* relative to SimPO, because the reference now
penalizes letting the correct chain fall and the finite target stops the unbounded push that drove the
collapse. GSM8K is near-saturated and should sit around SimPO's 86 — no headroom for the fix to show.
MATH-500 I expect to be roughly flat or slightly up: it was not collapsing under SimPO (74.0), so there is
less to repair, and IPO's squared loss can be conservative — once the target gap is met it stops
improving, which on the middle benchmark may leave it near where SimPO already was. So the signature I am
betting on is "AIME up, GSM8K flat, MATH-500 flat-to-slightly-up, average up." If instead AIME stays low,
my diagnosis was wrong — the erosion was not the cause, or the reference anchor at β = 0.1 is too weak to
hold the long chains — and the next rung would have to attack the absolute likelihood more directly rather
than through a reference-relative regression. That is the test IPO is running.
