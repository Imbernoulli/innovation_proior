I have to start the ladder somewhere, and the honest place to start is with the simplest thing that
could conceivably work, run it, and let its number tell me what the math-reasoning setting actually
punishes. The cheapest preference loss to stand up — cheapest in every sense, memory, compute, and
conceptual machinery — is the one that needs no reference model at all, so that is where I begin. Let
me derive it from scratch so I know exactly what I am putting in the `compute_preference_loss` slot and
exactly what I expect it to do, because the whole point of a first rung is to be a clean, interpretable
floor.

The thing that bothers me about the obvious baseline, DPO, before I have even run it, is a cost and a
mismatch. The cost: DPO keeps a frozen copy of the SFT model resident and runs a second forward pass on
every batch to get the reference log-probs `log π_ref(y|x)`. On a memory-tight 4×GPU box doing
full-parameter training of a 1.5B model, that is a whole extra model's footprint and a doubled forward,
purely to compute a baseline I subtract off. The mismatch is the one that actually itches. At
generation — which is exactly how this task is scored, greedy decoding graded by sympy — there is no
reference model anywhere. The model produces a chain of tokens and is judged on whether that chain is
correct, i.e. on the per-token likelihood the policy itself assigns. But the reward DPO optimizes is
`r(x,y) = β log[π_θ(y)/π_ref(y)]`, a log-*ratio* against the reference. Those are different functions of
the same response. Satisfying `r(x,y_w) > r(x,y_l)` rearranges to
`log π_θ(y_w) − log π_θ(y_l) > log π_ref(y_w) − log π_ref(y_l)` — a condition on summed log-probs
*offset by whatever the reference assigns* — and there is no reason that implies the thing I am graded
on, that the policy's per-token score ranks the correct chain above the wrong one. The reward I optimize
and the metric I am judged by are not the same object.

So the cleaner thing to want is to make the reward I optimize *be* the metric the model is generated and
ranked by. Don't optimize one quantity and hope it transfers to another. The generation-ranking metric
is the average per-token log-likelihood, `(1/|y|) log π_θ(y|x)`. Let me try to build the preference
reward directly out of that. The naive first instinct — just use the policy's own log-probability,
`r = β log π_θ(y)`, dropping the reference — is reference-free for free, and the memory problem
evaporates. But it is the *summed* log-prob, and summed log-prob has a structural length problem: every
extra token contributes another `log π ≤ 0`, so longer sequences score systematically lower. When `y_w`
happens to be longer than `y_l` — which in math, where correct full derivations are often longer than
truncated wrong ones, is common — the model has to overcome a length handicap, and the only lever it has
is to crank token probabilities on the long winning sequence, baking in a "long = good" artifact that is
verbosity, not quality.

The fix is sitting in what I already said I want. The generation metric is not the summed log-prob, it is
the *average*: `(1/|y|) log π_θ(y|x)`. The `1/|y|` is exactly the length normalization that cancels the
structural handicap — every response is scored per token, so winner and loser sit on the same footing,
and there is no incentive to inflate probabilities just to beat a length penalty. And it *is* the
quantity the model is ranked by at decode. So the two problems — the train/generation mismatch and the
length bias — collapse into one fix: use the average log-likelihood as the reward,
`r_SimPO(x,y) = (β/|y|) log π_θ(y|x)`. Reference-free (it is the policy's own per-token score),
generation-aligned (it *is* the ranking metric, scaled by β), and length-debiased (the `1/|y|` puts
winner and loser per-token).

Do I actually need `π_ref` for anything once I have done this? The reference was there for the KL leash,
to keep the policy from running off to degenerate high-reward strings. Dropping it loses the explicit
regularizer, but the training regime is a practical leash here: start from a strong math-SFT model, use
a tiny learning rate (`5e-7`), make only four passes over 10K diverse problems. That is not a theorem, so
I will watch for drift, but it is enough reason to try the simpler policy-only reward before paying for a
frozen reference every batch.

Now plug `r_SimPO` straight into Bradley-Terry, exactly as DPO does with its reward:
`L = −E[ log σ( (β/|y_w|) log π(y_w) − (β/|y_l|) log π(y_l) ) ]`. Before I trust it, check the gradient.
With `u = (β/|y_w|) log π(y_w) − (β/|y_l|) log π(y_l)` and loss `−log σ(u)`, the per-example weight is
`σ(−u)`, large exactly when the policy *wrongly* gives the loser a higher average log-likelihood than the
winner — the right thing to up-weight, with no reference model in it. And each log-prob gradient is
divided by its own length, so a long response and a short one push with comparable magnitude — DPO's
gradient moves `∇log π(y_w) − ∇log π(y_l)` un-normalized, letting a response with twice the tokens get
roughly twice the gradient. The reward choice and the gradient agree.

Is the Bradley-Terry skeleton enough? It only asks `r(y_w) > r(y_l)` — it is satisfied the instant the
winner outscores the loser by an infinitesimal amount. Getting the sign right is a weak requirement; the
lesson of margins (the max-margin idea behind SVMs, the "home advantage" offset in Bradley-Terry ranking
models) is that a comfortable gap generalizes better than a barely-separating one. So put a margin into
the preference model itself: demand the reward gap exceed `γ > 0` before the loss is satisfied,
`p(y_w ≻ y_l|x) = σ(r(x,y_w) − r(x,y_l) − γ)`. The `−γ` shifts the sigmoid so the loss is not near-minimal
until `r(y_w) − r(y_l)` has crossed `γ`, keeping the model pulling the winner above the loser until there
is a real cushion. Plug in `r_SimPO`:
`L_SimPO = −E[ log σ( (β/|y_w|) log π(y_w) − (β/|y_l|) log π(y_l) − γ ) ]`. How big should `γ` be? Too
small and I am back to merely asking for the right sign; too large and I demand an unrealistic per-token
gap and over-suppress fluent losing responses. So `γ` is a knob to tune, not derive — for this 1.5B math
base the standard setting is `β = 2.0`, `γ = 1.0`.

Now I have to land this in the actual edit surface, and here the trajectory's substrate does most of the
work for me. The frozen loop already exposes exactly the two hooks SimPO needs. First, because the
`pref_loss=simpo` flag puts `simpo` in the `["ipo","orpo","simpo"]` set, `concatenated_forward` divides
each per-response log-prob by `valid_length` *before* my loss ever sees it — so the
`policy_chosen_logps`/`policy_rejected_logps` handed to me are already the **average** per-token
log-probs, the length normalization done for me. Second, because `simpo` is in the reference-free set in
`finetuning_args.py`, `use_ref_model` is False, the reference model is never loaded, and my loss lands in
the top, reference-free branch of `compute_preference_loss`. So my entire contribution is the
`simpo_loss` helper: form the average-log-prob difference `pi_logratios = chosen − rejected`, subtract the
code-space margin `γ/β` (so that the single `β`-multiply reproduces `β·(Δ) − γ` exactly,
`β·((Δ) − γ/β) = β·Δ − γ`), and return `−logsigmoid(β·logits)`. The implicit rewards for logging are
`β·policy_chosen_logps` and `β·policy_rejected_logps`, the length-normalized policy-only reward. (The full
scaffold fill is in the answer.)

I should flag, before I run it, the one regime where I expect this to be fragile — and it is precisely
this task's regime. Preferences over *math* solutions, where the winning and losing chains can be nearly
identical, differing in one wrong step. A contrastive objective there can do a perverse thing: it widens
the *reward margin* by pushing the loser's probability down, but because the chosen sequence shares
almost every token with the rejected one, dragging the rejected down drags the chosen down too — the
absolute likelihood of the *correct* answer can fall even as the margin grows. SimPO's margin term, if
anything, asks for a bigger gap, so it will not rescue this; and SimPO has no anchor on the chosen
sequence's absolute likelihood at all (it is reference-free and purely relative). The quantity the
benchmark actually rewards is greedy correctness, which lives on the absolute likelihood of a correct
chain, not the margin. So my falsifiable expectation for the floor is concrete: SimPO will train
stably, the reward margin and reward accuracy will look healthy, GSM8K (near-saturated for this base)
will barely move, and MATH-500/AIME — where the correct chain is long and the wrong chain is a
near-duplicate — are where I expect the absolute-likelihood erosion to bite, most visibly on AIME, the
hardest and highest-variance benchmark. If the AIME number comes in low while the easier benchmarks hold,
that is the signature of exactly this failure, and it tells the next rung what to fix: stop letting the
relative objective drag the correct chain's likelihood down, either by re-anchoring to a reference or by
trading the saturating sigmoid for a target the model can sit at. That is the diagnosis I am setting up
the ladder to read.
