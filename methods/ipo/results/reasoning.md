Let me start from what actually goes wrong, because something is wrong and I keep seeing it. I have a fixed dataset of pairwise human preferences, `(x, y_w, y_l)`: for a context, a rater said the chosen generation beats the rejected one. I want a policy `π(y|x)` that produces preferred generations while staying near a reference `π_ref`, and I want a single knob — a regularization coefficient `τ` — that genuinely controls *how near*. The clean abstraction is a contextual bandit, and to think clearly I'll drop the context `x` for now and reason per-context; everything carries back action-by-action. So: actions `y`, a behavior policy `μ` that generated the candidates, a reference `π_ref`, a true preference `p*(y ≻ y')`, and I only observe samples — Bernoulli draws `I(y, y')` that are 1 when `y` was preferred.

The reward-modeling route I know goes: fit a scalar Elo reward `r(y)` with Bradley–Terry, `p(y ≻ y') = σ(r(y) − r(y'))`, then maximize `E_π[r] − τ KL(π ‖ π_ref)`. And there's a closed form I lean on constantly: for any per-action score `f(y)`, the maximizer of `E_π[f] − τ KL(π ‖ π_ref)` is `π*(y) ∝ π_ref(y) exp(τ⁻¹ f(y))`. Let me re-derive that so I trust it, since the whole construction will hang off it. Write the objective over `δ ∈ Δ_S` against a fixed `η`: `L_τ(δ)/τ = Σ_s δ(s) f(s)/τ − KL(δ ‖ η) = Σ_s δ(s)[ f(s)/τ − log(δ(s)/η(s)) ]`. Define the softmax target `δ*(s) = η(s) exp(τ⁻¹ f(s)) / Z`, `Z = Σ_{s'} η(s') exp(τ⁻¹ f(s'))`. Then `f(s)/τ = log(exp(τ⁻¹ f(s))) = log(δ*(s) Z / η(s))`, so

  `L_τ(δ)/τ = Σ_s δ(s)[ log(δ*(s)Z/η(s)) − log(δ(s)/η(s)) ] = Σ_s δ(s) log(δ*(s)/δ(s)) + Σ_s δ(s) log Z = −KL(δ ‖ δ*) + log Z`.

`log Z` doesn't depend on `δ`, and `−KL(δ ‖ δ*)` is maximized uniquely at `δ = δ*` (KL is nonnegative, zero only when the arguments coincide). So `δ* = argmax L_τ`, unique when `η` has full support. Apply it with `η = π_ref` and `f = r`: the maximizer of the RLHF objective is `π*(y) ∝ π_ref(y) exp(τ⁻¹ r(y))`. And Rafailov's reparameterization (invert this for `r`; the normalizer cancels in Bradley–Terry differences) turns the whole thing into a reward-free supervised loss, DPO. Fine. So why am I unhappy.

Here's what's been nagging me. Take the simplest deterministic preference: `p*(y ≻ y') = 1`, `y` always beats `y'`. What does Bradley–Terry have to do to represent that? `σ(r(y) − r(y')) = 1` forces `r(y) − r(y') → +∞`. Now feed that into the optimal policy: `π*(y')/π*(y) = (π_ref(y')/π_ref(y)) exp(τ⁻¹(r(y') − r(y))) → 0`. So `π*(y') = 0`. And — stare at this — that happened *for every value of `τ`*. I can crank `τ` to a million, demand the policy barely move from `π_ref`, and the optimum still annihilates `y'`. The KL term, the one thing that's supposed to keep me near `π_ref`, has silently stopped binding. The more deterministic the preference, the weaker the regularization, until at full determinism the coefficient I'm tuning does literally nothing.

And it gets worse the moment I admit I never see `p*`, only finite samples. Suppose the *true* preference is a perfectly reasonable `p*(y ≻ y') = 0.8`. With a handful of comparisons the empirical estimate `p̂(y ≻ y')` can easily come out exactly `1` — two out of two, say. The empirical optimum then sets `π(y') = 0` for any `τ`, on the strength of a fluke. For a language model, where the action space is the space of sequences and the context space is the space of prompts, almost every pair is observed once or never; the empirical preference for a pair lands in `{0, 1}` constantly. So this isn't a corner case, it's the typical case. The objective overfits the empirical preferences and ignores `π_ref`, and the knob I thought controlled that doesn't.

Let me make sure I understand *why RLHF-with-a-reward-model seems to dodge this in practice*, because if I understand the accidental shield, I'll understand what DPO threw away. In RLHF the infinite optimal rewards for `{0,1}` preferences are never actually reached — you can't train a network to output `+∞`, and people deliberately regularize the reward model on top of that. So the reward ends up *underfit*: the gap `r(y) − r(y')` stays finite even where the data says it should be infinite. A finite reward gap, plugged into `π* ∝ π_ref exp(τ⁻¹ r)`, gives a finite, regularized policy. The underfitting of the reward is doing the regularizing. DPO, by folding the reward away and fitting only the policy through `log σ(τ log(π(y_w)/π_ref(y_w)) − τ log(π(y_l)/π_ref(y_l)))`, has an *unbounded* logit inside the `log σ`: where the data says `y_w` always wins, the loss keeps decreasing as the log-ratio `log(π(y_w)/π(y_l))` grows without bound, there's no finite resting point, and there's no underfit reward model standing in the way anymore. DPO removed the reward model — a genuine win, no RL, no reward fitting — but in doing so it removed the accidental regularizer too.

So both RLHF and DPO are, at bottom, doing the same thing: optimizing some function of the preference probability, and that function is the Bradley–Terry logit `log(q/(1−q))`. Let me name that and see how general it is. Suppose I consider a whole family: pick a nondecreasing `Ψ: [0,1] → R` and maximize

  `J(π) = E_{y~π, y'~μ}[ Ψ(p*(y ≻ y')) ] − τ KL(π ‖ π_ref)`.

When does this reproduce RLHF/DPO? Take `Ψ(q) = log(q/(1−q))`, and suppose Bradley–Terry holds, `p*(y ≻ y') = σ(r(y) − r(y')) = e^{r(y)}/(e^{r(y)} + e^{r(y')})`. Then

  `E_{y'~μ}[Ψ(p*(y ≻ y'))] = E_{y'~μ}[ log( e^{r(y)} / e^{r(y')} ) ] = E_{y'~μ}[ r(y) − r(y') ] = r(y) − E_{y'~μ}[r(y')]`.

That last term is a constant in `y`, so `E_{y'}[Ψ(p*)]` equals the reward `r(y)` up to an additive constant — and the closed-form tilt only cares about scores up to additive constants (the constant gets absorbed into `Z`). So under Bradley–Terry, the logit `Ψ` makes my family's optimum coincide with RLHF's, which coincides with DPO's. The logit choice, then, just *is* RLHF and DPO sitting inside this family. That's reassuring for two reasons: the family is wide enough to contain the methods I'm trying to improve on, and the failure I've been circling isn't a property of "RLHF" or "DPO" as pipelines — it must be a property of this one function `Ψ`. So let me locate it there.

Now what *is* the disease, in terms of `Ψ`? The logit `Ψ(q) = log(q/(1−q))` is **unbounded**: as `q → 1`, `Ψ → +∞`. That unboundedness is exactly what let the deterministic preference send the score to infinity and overwhelm the fixed `τ KL` term. Turn that around: if `Ψ` were *bounded*, say `|Ψ| ≤ M`, then the score `E_{y'}[Ψ(p*)]` in the objective is at most `M` in magnitude, so the most any preference — deterministic or not — can contribute to `J` is `M`, and I can always pick `τ` large enough that `τ · KL` dominates a swing of size `M`. The regularizer would have to keep biting. That's the property I want from `Ψ`: bounded, so it can't run away from the `τ KL` term. It still has to be nondecreasing (more-preferred should score higher). What's the simplest function that's both? The identity, `Ψ(q) = q` — it maps `[0,1] → [0,1]`, monotone, bounded by construction, and adds no new parameters. I don't yet know whether it gives a *workable* offline loss, only that it can't suffer the unboundedness pathology. So let me follow it through and see what objective it produces.

With `Ψ = I`,

  `g(y) := E_{y'~μ}[Ψ(p*(y ≻ y'))] = E_{y'~μ}[p*(y ≻ y')] =: p*(y ≻ μ)`,

the *total preference* of `y` against the behavior distribution. And the objective becomes

  `max_π E_{y~π}[ p*(y ≻ μ) ] − τ KL(π ‖ π_ref) = max_π p*(π ≻ μ) − τ KL(π ‖ π_ref)`,

direct regularized maximization of the total preference. No logit, no Elo, no Bradley–Terry assumption that pairwise preferences reduce to pointwise rewards — I'm maximizing the preference probability itself. By the tilt formula, the optimum is

  `π*(y) ∝ π_ref(y) exp(τ⁻¹ p*(y ≻ μ))`.

And the score in the exponent, `p*(y ≻ μ) ∈ [0,1]`, is bounded. So no matter how deterministic any individual preference is, the exponent can't run off to infinity; `τ` keeps biting. That's the fix in principle. The standard way to *optimize* this would be RLHF again with reward `r(y) = p*(y ≻ μ)` — but that's back to RL and estimating a reward. I want what DPO got me: an offline loss straight on the preference pairs, no RL, no reward model. Can I get there for this objective?

Let me follow Rafailov's trick — turn the analytic optimum into a system of equations on `π`. From `π*(y) ∝ π_ref(y) exp(τ⁻¹ g(y))`, take a ratio for two actions to kill the normalizer:

  `π*(y)/π*(y') = (π_ref(y)/π_ref(y')) exp( τ⁻¹ (g(y) − g(y')) )`.

Take logs and group the policy terms on one side. Define

  `h*(y, y') := log( π*(y) π_ref(y') / (π*(y') π_ref(y)) )`,

the *reference-corrected* log-ratio. Then rearranging the displayed equation,

  `h*(y, y') = τ⁻¹ ( g(y) − g(y') )`.

This is one scalar equation per ordered pair `(y, y')`, and `π*` is exactly the policy whose reference-corrected log-ratios match `τ⁻¹` times the score gaps. So define, for a candidate `π`,

  `h_π(y, y') = log( π(y) π_ref(y') / (π(y') π_ref(y)) )`,

and I want to *solve* `h_π(y, y') = τ⁻¹(g(y) − g(y'))` for all pairs. That's a root-finding problem. Here is where I peel away from Rafailov, who plugged this kind of relation into a Bradley–Terry likelihood; with `Ψ = I` there's no Bradley–Terry to plug into — `g(y) = p*(y ≻ μ)` is a preference probability, not a reward. So I'll fold the root-finding into one optimization directly. The natural way to make "this equation holds for all pairs" into a single scalar objective is to penalize the squared residual of each equation and average:

  `L(π) = E_{y, y' ~ μ}[ ( h_π(y, y') − (p*(y ≻ μ) − p*(y' ≻ μ)) / τ )² ]`.

It's an expectation of squares, so `L(π) ≥ 0`, and from `h*(y,y') = τ⁻¹(g(y) − g(y'))` the residual at `π*` is identically zero, so `L(π*) = 0`: `π*` is a global minimizer. The only worry: are there *other* minimizers — could the squared-residual landscape have spurious optima that aren't `π*`? I need to actually check, because if it does, minimizing `L` doesn't recover `π*`.

So let me parametrize. Restrict to policies supported on `J = Supp(μ)`, via logits `s ∈ R^J`, `π_s(y) = e^{s(y)} / Σ_{y'∈J} e^{s(y')}`. Then `log(π_s(y)/π_s(y')) = s(y) − s(y')`, and

  `h_{π_s}(y,y') = (s(y) − s(y')) + log(π_ref(y')/π_ref(y))`.

Substitute into `L` and call it `𝓛(s) = L(π_s)`:

  `𝓛(s) = E_{y,y'~μ}[ ( (p*(y≻μ) − p*(y'≻μ))/τ − (s(y) − s(y')) − log(π_ref(y')/π_ref(y)) )² ]`.

This is *quadratic in `s`*. Expand the square: the only place `s` enters quadratically is through `(s(y) − s(y'))²`, with the cross terms and the rest linear or constant. So the Hessian (the pure-quadratic part) is, up to the positive weights `μ(y)μ(y')`,

  `Σ_{y,y'∈J} μ(y) μ(y') (s(y) − s(y'))²`.

Every term is a nonnegative weight times a square, so this is a positive-*semidefinite* quadratic form, hence `𝓛(s)` is convex. Convex means every local minimizer is global — already this kills the "spurious local optimum" fear at the level of the logits. But "semidefinite, not definite" means there could be a flat direction, a whole subspace of minimizing logits. Which direction is flat? `(s(y) − s(y'))²` is unchanged exactly when I shift all logits by the same constant: `s → s + λ·(1,…,1)` leaves every difference `s(y) − s(y')` fixed, so it leaves the entire quadratic form fixed, and it's the *only* such direction (any non-constant shift changes some difference and strictly increases the form). So `𝓛` is strictly convex in every direction *except* the all-ones direction.

And now the lovely part: that one flat direction doesn't matter, because shifting logits by a constant doesn't change the *policy*. For `y ∈ J`,

  `π_{s+λe}(y) = e^{s(y)+λ} / Σ_{y'} e^{s(y')+λ} = e^λ e^{s(y)} / (e^λ Σ_{y'} e^{s(y')}) = π_s(y)`.

The softmax quotients out the all-ones direction. So even though `𝓛` has a flat valley in logit space, every point in that valley maps to the *same* `π`. Strict convexity modulo the only degenerate direction, plus that direction being policy-invariant, plus `π*` being a global min — together these say `π*` is the *unique* local-and-global minimizing policy. I should pin down the support condition that makes this clean, though: I assumed `Supp(μ) = Supp(π_ref)` and restricted `π` to that support. What if I let `π` range over a *larger* support than `μ`? Then there are pairs the loss never constrains. Concretely: one state, three actions `y_1, y_2, y_3`; `π_ref` uniform; `μ` puts `1/2` on each of `y_1, y_2` and `0` on `y_3`. The loss only ever samples pairs from `{y_1, y_2}`, so it reduces to `L(π) = 2( τ⁻¹(p*(y_1≻μ) − p*(y_2≻μ)) − log(π(y_1)/π(y_2)) )²`, which pins only the *ratio* `π(y_1)/π(y_2)` and says nothing about `π(y_3)`. Any `π = (p, q, 1−p−q)` with `p/q = e^{τ⁻¹(p*(y_1≻μ)−p*(y_2≻μ))}` is a global minimum. To be sure I'm not fooling myself, I put numbers on it: `τ = 1`, the loss pins only `s(y_1) − s(y_2)` and leaves `s(y_3)` free, so `s = (0.5, 0, 0)` and `s = (0.5, 0, 5)` both achieve loss exactly `0` — but as policies they are `(0.452, 0.274, 0.274)` and `(0.011, 0.007, 0.982)`, wildly different. So the non-uniqueness is concrete, not a worry I can hand-wave away: when `μ` misses part of `π_ref`'s support, the minimizing policy is genuinely not pinned down. The conclusion I'll carry forward is the condition that *does* make it unique: `Supp(μ) = Supp(π_ref)`, with `π` ranging over that support. Under that condition the convexity argument above gives a single minimizing policy `π*`; without it, all bets on uniqueness are off, and I now have an explicit witness to why.

Now the practical problem: `L(π)` is written in terms of `p*(y ≻ μ)`, which I do not observe. I observe Bernoulli samples `I(y, y')` with mean `p*(y ≻ y')`. So I'd love to swap the unknown gap `p*(y ≻ μ) − p*(y' ≻ μ)` for something I can sample. The obvious candidate is the per-comparison label `I(y, y')`. Consider the **sampled** loss

  `E_{y,y'~μ}[ ( h_π(y, y') − τ⁻¹ I(y, y') )² ]`.

Is this the same objective? My first instinct says no, and I should be honest about that, because the naive expectation-swap fails. Condition on a fixed pair `(y, y')`: the inner conditional expectation `E[ h_π − τ⁻¹ I | y, y' ]` involves `E[I | y,y'] = p*(y ≻ y')`, a single pairwise preference — not the *total* preference difference `p*(y ≻ μ) − p*(y' ≻ μ)` that appears in `L`. So term by term they're not equal. The equality has to come from a *symmetry* over the random draw of the pair, not from matching residuals pointwise. Let me check whether the two losses agree in expectation up to a `π`-independent constant — that's all I need, since constants don't move the argmin.

Expand both squares. The squared `h_π²` term and the constant `τ⁻²(...)` term: `I² = I` for a Bernoulli, and the gaps are bounded — the only `π`-dependent piece that could differ between the two losses is the *cross term* `−2 τ⁻¹ E[ h_π · (·) ]`. So it's enough to show

  `E_{y,y'~μ}[ h_π(y,y') I(y,y') ] = E_{y,y'~μ}[ h_π(y,y') (p*(y≻μ) − p*(y'≻μ)) ]`.

The key structural fact about `h_π` is that it's **additive and antisymmetric**: writing `π_y = log π(y)`, `π^R_y = log π_ref(y)`,

  `h_π(y, y') = (π_y − π^R_y) − (π_{y'} − π^R_{y'})`.

Let me abbreviate `p_y = p*(y ≻ μ)` and use that `y, y'` are drawn iid from `μ`, plus the identity `E_{y~μ}[p_y] = E_{y~μ} E_{y'~μ}[p*(y≻y')] = 1/2` (since `p*(y≻y') + p*(y'≻y) = 1` and `y, y'` are exchangeable). Start with the right-hand side:

  `E[ h_π(y,y') (p_y − p_{y'}) ] = E[ ((π_y − π^R_y) − (π_{y'} − π^R_{y'}))(p_y − p_{y'}) ]`.

Set `a_y = π_y − π^R_y`, so the expression is `E[(a_y − a_{y'})(p_y − p_{y'})]`. Expanding gives `E[a_y p_y] − E[a_y p_{y'}] − E[a_{y'} p_y] + E[a_{y'} p_{y'}]`. The first and last terms are the same by iid relabeling. In the two cross terms, the variables separate, so `E[a_y p_{y'}] = E[a_y]E[p_{y'}] = (1/2)E[a_y]`, and similarly `E[a_{y'} p_y] = (1/2)E[a_y]`. Therefore the whole right-hand side is

  `E[ h_π(y,y')(p_y − p_{y'}) ] = E_{y~μ}[ (2 p_y − 1)(π_y − π^R_y) ]`.

Now the left-hand side, the one with the sampled label. Use additivity of `h_π` to split it across `y` and `y'`, then take the conditional expectation of `I` against the *other* variable:

  `E[ h_π(y,y') I(y,y') ] = E_y[ (π_y − π^R_y) · E_{y'}[I(y,y') | y] ] + E_{y'}[ (−π_{y'} + π^R_{y'}) · E_y[I(y,y') | y'] ]`.

Here `E_{y'~μ}[I(y,y') | y] = E_{y'~μ}[p*(y ≻ y')] = p*(y ≻ μ) = p_y` — that's exactly the total preference, recovered from the *partner-averaged* label. And `E_{y~μ}[I(y,y') | y'] = E_{y~μ}[p*(y ≻ y')] = 1 − E_{y~μ}[p*(y' ≻ y)] = 1 − p_{y'}`. Substitute:

  `E[ h_π I ] = E_y[ (π_y − π^R_y) p_y ] + E_{y'}[ (−π_{y'} + π^R_{y'})(1 − p_{y'}) ] = E_{y~μ}[ (2 p_y − 1)(π_y − π^R_y) ]`,

after relabeling `y' → y` in the second term and combining `p_y − (1 − p_y) = 2 p_y − 1`. Both sides reduce to the same expression `E_{y~μ}[(2 p_y − 1)(π_y − π^R_y)]`.

That's a long enough chain that I don't trust it from the algebra alone — the whole argument hinges on the two cross-terms collapsing to the same thing, and it would be easy to drop a factor. So I check it on a small instance. Take four actions, a random `μ`, and a true preference matrix built as `p*(y≻y') = σ(c_y − c_{y'})` for random latent `c` (this automatically satisfies `p* + p*ᵀ = 1` and `p*(y≻y) = 1/2`, the only structure I used), and a random policy via random `a_y = π_y − π^R_y`. Computing all three expectations directly over the `4×4` pairs: `E_μ[p_y] = 0.500000` (the exchangeability fact I leaned on), and the true-gap form `E[(a_y−a_{y'})(p_y−p_{y'})] = −0.0239126`, the sampled-label form `E[(a_y−a_{y'}) p*(y≻y')] = −0.0239126`, and the claimed closed form `E[(2p_y−1)a_y] = −0.0239126` all coincide to machine precision. So the sampled loss and the true loss really do agree up to a `π`-independent constant. The partner-averaging is what turns a single pairwise label into the total-preference gap I needed: conditioning `I` on one action and averaging over its random partner recovers `p_y`, and that step is only legitimate because of the additive/antisymmetric structure of `h_π` and the iid draw. My first instinct that the term-by-term mismatch sinks the swap was wrong; it's the averaging over the partner, not a per-pair match, that makes the two losses agree.

Now make it empirical. My dataset is `(y_{w,i}, y_{l,i})`: an observed comparison where the first beat the second. The sampled loss is over an ordered iid draw of a pair together with the label `I` saying whether the first item beats the second. A single recorded comparison `(y_w, y_l)` actually furnishes *two* ordered terms: `(y, y', I) = (y_w, y_l, 1)` and the swapped orientation `(y_l, y_w, 0)`. Using both halves of every datapoint cuts the variance of the loss estimate, and it costs nothing. So the empirical loss is the average of the two:

  `(1/2) E_D[ (h_π(y_w, y_l) − τ⁻¹·1)² + (h_π(y_l, y_w) − τ⁻¹·0)² ] = (1/2) E_D[ (h_π(y_w, y_l) − τ⁻¹)² + h_π(y_l, y_w)² ]`.

And `h_π` is antisymmetric, `h_π(y_l, y_w) = −h_π(y_w, y_l)`, so `h_π(y_l, y_w)² = h_π(y_w, y_l)²`. Let `H := h_π(y_w, y_l)`. The bracket is `(H − τ⁻¹)² + H²`. Expand and complete the square:

  `(H − τ⁻¹)² + H² = H² − 2τ⁻¹H + τ⁻² + H² = 2H² − 2τ⁻¹H + τ⁻² = 2(H − τ⁻¹/2)² + τ⁻²/2`.

Halve it: `(1/2)[(H − τ⁻¹)² + H²] = (H − τ⁻¹/2)² + τ⁻²/4`. The `τ⁻²/4` is a constant in `π`. Drop it. The empirical loss collapses to a single, strikingly simple squared term:

  `L_IPO(π) = E_{(y_w, y_l) ~ D}[ ( h_π(y_w, y_l) − τ⁻¹/2 )² ]`.

Let me read what this is actually telling the policy to do. Unfold `h_π(y_w, y_l) = log(π(y_w)/π(y_l)) − log(π_ref(y_w)/π_ref(y_l))`. So the loss regresses the *gap between the policy's log-likelihood ratio of winner-over-loser and the reference's* onto a single fixed target, `τ⁻¹/2`, the **same target for every pair in the dataset**. That's the whole behavior. It's nothing like DPO's `log σ`: there's no saturating sigmoid that keeps paying out as the log-ratio grows; there's a finite target it wants to *sit at*. If the policy already separates winner from loser by `τ⁻¹/2` more than the reference does, the loss is zero and the gradient vanishes — it stops pushing. Weaken the regularization (smaller `τ`, larger `τ⁻¹/2`) and the target gap grows, so the policy is allowed to separate winner and loser more; strengthen it (large `τ`, target → 0) and the policy is pulled to match `π_ref`'s own separation. The coefficient is doing exactly what it advertises — controlling the distance from `π_ref` — even when the preferences are deterministic, because the target is a *finite number*, not a logit running to infinity. The unboundedness that broke the logit objective is gone; a deterministic preference just means `I = 1` always, which still only ever asks the gap to hit `τ⁻¹/2`, never `+∞`.

I should check whether this really cures the deterministic case, on the minimal instance, and I want to check it two ways — through the analytic optimum and by actually minimizing the loss — because the analytic optimum is what I *designed* the loss to hit, and I haven't yet confirmed the loss hits it. Two actions, `p*(y_1 ≻ y_2) = 1`, uniform `π_ref` and `μ`. The total preferences: `p*(y_1 ≻ μ) = (1/2)p*(y_1 ≻ y_1) + (1/2)p*(y_1 ≻ y_2) = (1/2)(1/2) + (1/2)(1) = 3/4`, and by symmetry `p*(y_2 ≻ μ) = 1/4`. Plug into the optimal tilt `π*(y) ∝ π_ref(y) exp(τ⁻¹ p*(y ≻ μ))`: `π*(y_1) = exp(τ⁻¹·3/4)/(exp(τ⁻¹·3/4) + exp(τ⁻¹·1/4)) = 1/(1 + exp(−τ⁻¹·1/2)) = σ(τ⁻¹/2)`, and `π*(y_2) = σ(−τ⁻¹/2)`.

Now the second way. The dataset here only ever records `(y_w, y_l) = (y_1, y_2)`, so with uniform `π_ref` the reference term in `h_π` vanishes and `L_IPO = (s(y_1) − s(y_2) − τ⁻¹/2)²` over logits `s`. That's minimized at `s(y_1) − s(y_2) = τ⁻¹/2`, i.e. `π(y_1) = σ(τ⁻¹/2)` — the same as the analytic optimum, as it must be if the derivation is consistent. To be sure the *optimizer* gets there and not just my pen, I gradient-descend `L_IPO` from `s = 0`: at `τ = 0.5` it converges to `s(y_1) − s(y_2) = 1.000000`, giving `π(y_1) = 0.731059`, matching `σ(1/(2·0.5)) = σ(1) = 0.731059`; at `τ = 1`, `s` gap `0.500000`, `π(y_1) = 0.622459 = σ(0.5)`; at `τ = 5`, gap `0.100000`, `π(y_1) = 0.524979 = σ(0.1)`. The loss and its closed form agree, and the optimizer lands on it.

Now watch the knob work, using `π(y_1) = σ(τ⁻¹/2)` and plugging in actual numbers. `τ = 100`: `σ(1/200) = 0.501250`, essentially the uniform `π_ref`. `τ = 10⁶`: `σ(5·10⁻⁷) = 0.500000` — strong regularization holds the policy *at* the reference even though the preference is perfectly deterministic, which is exactly what the logit objective could never do (it sat at `π(y_2) = 0` for that same instance, for every `τ`). Going the other way, `τ = 0.1`: `σ(5) = 0.993307`, nearly the deterministic policy; `τ → 0` drives it to `1`. So the whole continuum from `π_ref` to deterministic is reachable, governed by `τ`, on the very instance where the logit objective collapsed. The difference is entirely that the target is the finite number `τ⁻¹/2` rather than a logit free to run to `+∞`. (The same story holds for a never-observed action: where the logit objective would push an action that never wins to probability 0 regardless of `τ`, the bounded score lets the policy keep it near `π_ref` and only shade it down as `τ` shrinks — the safe behavior when I simply lack data on an action.)

So the algorithm, in one line: starting from `π = π_ref`, minimize `E_{(x, y_w, y_l) ~ D}[ ( h_π(y_w, y_l, x) − τ⁻¹/2 )² ]`, where `h_π(y_w, y_l, x) = log( π(y_w|x) π_ref(y_l|x) / (π(y_l|x) π_ref(y_w|x)) )`. No reward model, no RL, no Bradley–Terry assumption, and a regularizer that stays alive on deterministic data. Now to the code, filling the one empty slot — the `preference_objective` map from a preference pair to a scalar loss.

There's one more decision the abstraction didn't force but the sequence setting does. For a bandit, `log π(y)` is a single number. For a language model, the generation `y` is a token sequence, and the "log-probability of `y`" is the *sum* of per-token log-probs over the completion. If I use the raw sum, then `h_π` — a difference of summed log-ratios — scales with completion length: a long winner-versus-loser comparison contributes a numerically larger residual than a short one, purely because there are more tokens, and the squared loss then has the target `τ⁻¹/2` mean different things at different lengths. The clean fix is to use the *average* per-token log-probability — divide each sequence log-prob by its completion length — so `h_π` is on a per-token scale and the single fixed target `τ⁻¹/2` is comparable across pairs of any length. This isn't something the bandit derivation needed; it's a sequence-length normalization that keeps the one-target-for-all-pairs logic intact when actions are variable-length strings, and it's the form used in practice.

In code I have to translate the coefficient name without changing its meaning. In the derivation I've called it `τ`, the multiplier on `KL` in `J(π) = E[Ψ(p*)] − τ KL`, and the target came out `τ⁻¹/2`. The implementation harness exposes that same coefficient under the name `beta`, so `beta` plays the role of `τ` and the target is written `1/(2*beta)`. The per-sequence loss is a squared term, so I average it over the batch to get the scalar to backprop.

```python
import torch


def preference_objective(policy_chosen_lp, policy_rejected_lp,
                         ref_chosen_lp, ref_rejected_lp,
                         chosen_len, rejected_len, beta):
    """IPO loss for one batch of preference pairs.

    Args are sequence-level summed log-probs of policy and reference on the
    chosen / rejected completions, the two completion lengths, and beta (= the
    KL-regularization coefficient tau; the regression target is 1/(2*beta)).
    """
    chosen_len = chosen_len.clamp_min(1).to(policy_chosen_lp.dtype)
    rejected_len = rejected_len.clamp_min(1).to(policy_rejected_lp.dtype)

    # per-token average log-probs: keeps h_pi on a per-token scale so the single
    # fixed target is comparable across completions of different lengths.
    pol_chosen   = policy_chosen_lp   / chosen_len
    pol_rejected = policy_rejected_lp / rejected_len
    ref_chosen   = ref_chosen_lp      / chosen_len
    ref_rejected = ref_rejected_lp    / rejected_len

    # reference-corrected log-ratio gap  h_pi(y_w, y_l) =
    #   [log pi(y_w) - log pi_ref(y_w)] - [log pi(y_l) - log pi_ref(y_l)]
    chosen_logratio   = pol_chosen   - ref_chosen
    rejected_logratio = pol_rejected - ref_rejected
    h_pi = chosen_logratio - rejected_logratio

    # regress the gap onto the single fixed target 1/(2*beta) = tau^{-1}/2;
    # bounded target keeps KL alive even for deterministic preferences.
    losses = (h_pi - 1.0 / (2.0 * beta)) ** 2
    return losses.mean()
```

That is exactly the squared term I derived, with the per-token averaging the sequence setting demands. To see it in the shape it actually ships: one trainer pattern length-averages the policy and reference sequence log-probs before the preference-loss branch; another first forms the chosen and rejected reference-corrected log-ratio sums, then divides those scores by their completion lengths inside the preference-loss branch. Either way, the branch forms the same averaged `h_pi` (called the delta) and squares the distance to `1/(2*beta)`:

```python
import torch
import torch.nn.functional as F


class IPOTrainer:
    """Offline preference training with the IPO (Identity-PO) loss.

    beta is the KL-regularization coefficient (tau in the derivation); the IPO
    loss regresses the reference-corrected log-ratio gap onto the fixed target
    1/(2*beta). Sequence log-probs are averaged per completion token.
    """

    def __init__(self, policy_model, reference_model, beta=0.1, lr=5e-7):
        self.policy = policy_model
        self.ref = reference_model.eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)
        self.beta = beta
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr)

    @staticmethod
    def avg_seq_logp(model, input_ids, completion_mask):
        # per-token log p over the completion, then average by completion length
        logits = model(input_ids).logits[:, :-1, :].log_softmax(-1)
        labels = input_ids[:, 1:]
        per_token = torch.gather(logits, 2, labels.unsqueeze(2)).squeeze(2)
        mask = completion_mask[:, 1:]
        length = mask.sum(-1).clamp_min(1).to(per_token.dtype)
        return (per_token * mask).sum(-1) / length          # average log-prob, shape (B,)

    def ipo_loss(self, pol_chosen, pol_rejected, ref_chosen, ref_rejected):
        # h_pi = [log pi(y_w) - log pi_ref(y_w)] - [log pi(y_l) - log pi_ref(y_l)]
        chosen_logratio = pol_chosen - ref_chosen
        rejected_logratio = pol_rejected - ref_rejected
        h_pi = chosen_logratio - rejected_logratio
        # IPO: squared distance to the single fixed target tau^{-1}/2 = 1/(2*beta)
        losses = (h_pi - 1.0 / (2.0 * self.beta)) ** 2
        # implicit-reward signals (for logging only): beta * (log pi - log pi_ref)
        chosen_rewards = self.beta * (pol_chosen - ref_chosen).detach()
        rejected_rewards = self.beta * (pol_rejected - ref_rejected).detach()
        return losses.mean(), chosen_rewards, rejected_rewards

    def train_step(self, batch):
        pol_c = self.avg_seq_logp(self.policy, batch["chosen_ids"],   batch["chosen_mask"])
        pol_r = self.avg_seq_logp(self.policy, batch["rejected_ids"], batch["rejected_mask"])
        with torch.no_grad():                                # frozen reference
            ref_c = self.avg_seq_logp(self.ref, batch["chosen_ids"],   batch["chosen_mask"])
            ref_r = self.avg_seq_logp(self.ref, batch["rejected_ids"], batch["rejected_mask"])
        loss, _, _ = self.ipo_loss(pol_c, pol_r, ref_c, ref_r)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return loss
```
