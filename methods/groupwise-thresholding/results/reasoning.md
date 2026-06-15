Let me start from what actually goes wrong, because the setup looks innocent. I have a base classifier I can't touch and a fixed train/calibration/test pipeline. The only thing I get to design is the acceptance rule: at a target coverage — say I'm allowed to predict on 80% of test points and must defer the other 20% to a human reviewer — which points do I accept and which do I defer? Errors on accepted points are expensive; deferrals are tolerable but each one costs a reviewer. The base classifier never consumes the subgroup id `g`, but the post-hoc policy harness can hand me subgroup ids at calibration and decision time. So the question is really: out of one fixed stream of softmax outputs, manufacture an accept/defer decision that is good not just on average but on every subgroup, and that doesn't dump all the deferrals on one subgroup.

The obvious thing, and the thing I'd reach for first, is Chow's rule. If I had the true posterior and a fixed cost for a wrong prediction versus a deferral, the Bayes-optimal policy is to abstain exactly when the maximum class posterior `max_y P(y|x)` is below a threshold and predict otherwise. So I take the score `s(x) = max_y P̂(y|x)`, the maximum softmax probability of the predicted class `ŷ(x) = argmax_y P̂(y|x)`, and I threshold it. To hit a target coverage I don't even have to think about the cost: I just set the global threshold to the `(1 − target_coverage)` quantile of `s` over the calibration set, so that exactly an 80% fraction clears the bar. Accept iff `s(x) ≥ threshold`. Done — this is the canonical softmax-response selective classifier, it's training-free, and across a huge range of applications more confident predictions really are more accurate, so the ranking by `s` is meaningful.

Before I trust it I want to be careful about the score itself, because people report confidence in a logit-transformed form rather than the bare probability and I want to know whether that matters. The logit confidence is `ĉ(x) = ½ log( P̂(ŷ|x) / (1 − P̂(ŷ|x)) )` for binary, and `½ log( P̂(ŷ|x)/(1−P̂(ŷ|x)) ) + ½ log(k−1)` for `k` classes. It's non-negative because `P̂(ŷ|x) ≥ 1/k`, it's zero at the maximally-unsure prediction, and at `k=2` the general form collapses to the binary one. Now, `ĉ` is a strictly increasing function of `s = P̂(ŷ|x)` — log of `p/(1−p)` is monotone in `p` on `(0,1)`. The accept set `{ĉ ≥ τ}` is exactly `{s ≥ σ}` for the matching threshold `σ`, so the two scores produce identical accept sets and therefore identical accuracy-coverage curves. Let me actually prove this generally because I'll want it later as license to use the cheap raw probability. Suppose I summarize a selective classifier by the distribution of its *margin* `m(x) = ĉ(x)` on correct predictions and `−ĉ(x)` on incorrect ones, with CDF `F`. Apply any strictly increasing, *odd* transform `T` to the margins, `Y = T(X)`. Odd means `T(−t) = −T(t)`, which is the right structure because the margin is signed by correctness. By the change-of-variables for a monotone map, `F_X(τ) = F_Y(T(τ))` and `F_X(−τ) = F_Y(T(−τ)) = F_Y(−T(τ))`. The selective accuracy is `A_F(τ) = (1 − F(τ)) / (1 − F(τ) + F(−τ))`, so

  A_{F_X}(τ) = (1 − F_Y(T(τ))) / (1 − F_Y(T(τ)) + F_Y(−T(τ))) = A_{F_Y}(T(τ)).

The curve is reparameterized but pointwise identical. So the accuracy-coverage curve depends only on the *ranking* of the confidence, not on the parameterization — I can use the bare `max(probs)` and lose nothing. Good, that simplifies the score to one line.

Now let me convince myself the global threshold is actually fine, and this is where I hit the wall. I'll set up the margin distribution properly because the whole accept/defer behavior is a functional of it. At threshold `τ` on confidence: a predicted-and-correct point has `m ≥ τ`; the classifier abstains on `−τ < m < τ`; a predicted-and-wrong point has `m ≤ −τ`. So `coverage(τ) = 1 − F(τ) + F(−τ)` and selective accuracy `A_F(τ) = (1 − F(τ)) / (F(−τ) + 1 − F(τ))`. I want to know how `A_F` moves as I raise `τ` — does deferring the least-confident points always help? Differentiate. Let me grind it out:

  dA_F/dτ = d/dτ [ (1 − F(τ)) / (1 − F(τ) + F(−τ)) ].

Quotient rule. Numerator of the whole thing, call the denominator `D(τ) = 1 − F(τ) + F(−τ)` with `D'(τ) = −f(τ) − f(−τ)`. Top is `N(τ) = 1 − F(τ)`, `N'(τ) = −f(τ)`. Then

  dA_F/dτ = [ N' D − N D' ] / D²
          = [ (−f(τ))(1 − F(τ) + F(−τ)) − (1 − F(τ))(−f(τ) − f(−τ)) ] / D².

Expand the bracket: `−f(τ) + f(τ)F(τ) − f(τ)F(−τ) + f(τ) + f(−τ) − f(τ)F(τ) − f(−τ)F(τ)`. The `−f(τ)` and `+f(τ)` cancel; the `+f(τ)F(τ)` and `−f(τ)F(τ)` cancel. Left with `f(−τ) − f(τ)F(−τ) − f(−τ)F(τ) = f(−τ)(1 − F(τ)) − f(τ)F(−τ)`. So

  dA_F/dτ = [ f(−τ)(1 − F(τ)) − f(τ)F(−τ) ] / D².

The denominator is positive, so accuracy increases at `τ` iff `f(−τ)(1 − F(τ)) ≥ f(τ)F(−τ)`, i.e.

  f(−τ)/F(−τ) ≥ f(τ)/(1 − F(τ)),    for all τ ≥ 0.

That's a clean monotonicity test. It's increasing everywhere iff that holds for all `τ ≥ 0`, and decreasing iff the inequality flips. Now I want to know when it holds. Suppose the margin distribution is *symmetric* about its mean `μ` — a decent first model since real margin distributions look roughly Gaussian. Symmetry gives `f(τ)/(1 − F(τ)) = f(2μ − τ)/F(2μ − τ)` (reflect `τ` about `μ`). So the condition becomes `f(−τ)/F(−τ) ≥ f(2μ − τ)/F(2μ − τ)`.

The natural property that controls `f/F` is log-concavity: if `F` is log-concave then `f/F` is decreasing, so the smaller argument has the larger `f/F`. But I don't need `F` log-concave everywhere — and that's important, because real margin distributions are bimodal (a hump of correct and a hump of wrong). I only need it on the lower tail. So let me define what I actually need: call a distribution *left-log-concave* if its CDF `F` is log-concave on `(−∞, μ]` where `μ` is the mean. This is strictly weaker than log-concavity. In fact a symmetric mixture of two Gaussians `0.5 𝒩(μ,σ²) + 0.5 𝒩(−μ,σ²)` is left-log-concave for *all* `μ, σ`, but only log-concave when `|μ| ≤ σ`. Let me verify that, because the bimodal case is exactly the worst-group regime I care about. Take `σ = 1`, `μ > 0`. The log-derivative of the density is

  f'(x)/f(x) = −x + μ · (1 − e^{−2xμ})/(1 + e^{−2xμ}).

Differentiate: `d/dx[f'/f] = −1 + 4μ² e^{−2xμ}/(1 + e^{−2xμ})²`. Set to zero: `(1 + e^{−2xμ})² = 4μ² e^{−2xμ}`, a quadratic in `u = e^{−2xμ}`: `u² + (2 − 4μ²)u + 1 = 0`, roots `u = 2μ² − 1 ± 2μ√(μ²−1)`. For `μ > 1` there are two distinct positive roots, so `f'/f` has two critical points symmetric about 0. Let the larger root be `v(μ) = 2μ² − 1 + 2μ√(μ²−1)`; it's increasing for `μ ≥ 1` with `v(1) = 1`, so `v(μ) > 1` for `μ > 1`. The smaller critical point is at `x = −a = −log v(μ)/(2μ) < 0`. The second derivative `d²/dx²[f'/f] = 8μ³ e^{−2xμ}(e^{−2xμ} − 1)/(e^{−2xμ} + 1)³` is positive at `x = −a` (since `v(μ) > 1`), so `−a` is a local minimum of `f'/f`, and it's the only critical point below 0. Therefore `f'/f` is decreasing on `(−∞, −a]`, which makes `f` and hence `F` log-concave there. On `[−a, 0]` I use `f(x)/F(x) ≥ 0` and `f'(x)/f(x) ≤ 0` (since `−a` is the minimum and `f'(0)/f(0) = 0`): then `d/dx[f/F] = (f/F)(f'/f − f/F) ≤ 0`, so `f/F` is decreasing on `[−a, 0]` too, i.e. `F` is log-concave on `(−∞, 0]`. Since `μ > 1 > 0`, that's log-concavity up to the mean — left-log-concave. So even a clearly bimodal margin distribution is left-log-concave.

Now use it. For symmetric, left-log-concave `F`: if the full-coverage accuracy `A_F(0) ≥ ½` (equivalently `F(0) ≤ ½`, equivalently `μ ≥ 0`), then for `τ ≥ 0` we have `−τ ≤ 2μ − τ`. Two cases. If `τ ≥ μ`, then `2μ − τ ≤ μ`, and log-concavity of `F` on `(−∞, μ]` gives `f(−τ)/F(−τ) ≥ f(2μ−τ)/F(2μ−τ)` directly. If `0 ≤ τ ≤ μ`, then `f(−τ)/F(−τ) ≥ f(τ)/F(τ)` by log-concavity, and `f(τ)/F(τ) ≥ f(τ)/(1−F(τ))` since `F(τ) ≤ ½`. Either way the monotonicity condition holds, so `A_F(τ)` is monotone *increasing*. And if `A_F(0) ≤ ½`, the same argument with the inequality flipped gives monotone *decreasing*.

There it is in black and white, and it's exactly the trap. Whether deferring low-confidence points *helps or hurts* a group is decided by whether that group's full-coverage accuracy is above or below 50%. On the kind of high-stakes tabular data where the model leans on a spurious correlation, the worst subgroup — the one where the spurious cue fails — sits below 50% accuracy at full coverage. So for that subgroup, raising the global threshold doesn't just help less, it actively moves selective accuracy the *wrong way*. Its margin distribution is shifted left, packed with confident-but-wrong examples, and on the very worst ones confidence is anticorrelated with correctness, so the most confident worst-group points are the ones the model is most confidently wrong about. The average margin distribution, meanwhile, sits above 50% and looks Gaussian-ish, so it's left-log-concave with `A(0) > ½` and selective accuracy climbs nicely. One global threshold, one curve going up on average and a curve going *down* on the subgroup I most need to protect. That's the wall: the global rule isn't merely suboptimal on the worst subgroup, it can be anti-optimal.

Let me not over-rely on perfect symmetry, because the real margin distributions are skewed. I'll extend with skew-symmetric densities: `h_α(τ) = 2 h(τ − μ) G(α(τ − μ))`, where `h` is symmetric about 0 and `G` is a symmetric CDF, `α` the skew (positive = right skew, `α = 0` recovers symmetric `h`). I want two facts. First, more right skew gives higher selective accuracy. The skew-symmetry identities are `h_α(x) = h_{−α}(2μ − x)` and `H_α(x) = 1 − H_{−α}(2μ − x)` — the second from `∫_{−∞}^x h_α = ∫_{−∞}^x h_{−α}(2μ − t)dt = ∫_{2μ−x}^∞ h_{−α} = 1 − H_{−α}(2μ−x)`. Using these, `A_{h_α}(τ) = 1/(1 + H_α(−τ)/H_α(2μ−τ))`, a decreasing function of `H_α(−τ)/H_α(2μ−τ)`. There's a stochastic ordering: for `α₁ ≤ α₂`, `F_{α₁} ≥ F_{α₂}` pointwise (more right-skew puts less mass low) — verified by splitting `x ≤ 0` (where `G(α₁ t) ≥ G(α₂ t)` for `t ≤ 0`) and `x ≥ 0` by reflection. So as `α` rises the numerator `H_α(−τ)` falls and the denominator `H_α(2μ−τ)` rises, the ratio drops, accuracy rises. Right skew helps. Second, skewing in the *same* direction preserves monotonicity. There's a log-gradient ordering `h_α(τ)/H_α(τ) ≥ h_0(τ)/H_0(τ) ≥ h_{−α}(τ)/H_{−α}(τ)` for `α, τ ≥ 0`, because `h_α(τ)/H_α(τ) = h(τ)/∫_{−∞}^τ h(t)·G(αt)/G(ατ) dt` and for `t ≤ τ` the factor `G(αt)/G(ατ) ≤ 1` shrinks the denominator, raising the ratio. Chaining this with the symmetric monotonicity condition shows: if the symmetric version `A_{h_{0,μ}}` is increasing, then `A_{h_{α,μ}}` is increasing for every `α > 0`. So a right-skewed Gaussian-like average margin still climbs; a left-skewed below-50% worst group still falls. The skew story doesn't rescue the worst subgroup — it confirms the trap survives realistic asymmetry. (And the odd-monotone-transform invariance I proved earlier means all of this is unchanged by reparameterizing the confidence.)

So I've diagnosed *why* the global threshold magnifies the gap. But diagnosis isn't a rule. I need a reference for what "fair across subgroups" should even mean for an accept/defer decision, so I have something concrete to aim my rule at. Here's the framing that unlocks it: treat "predict vs. defer" as a meta-classification problem. A *true positive* is when I choose to predict and I'm correct; a *false positive* is when I choose to predict and I'm wrong. Then for a selective classifier define the fractions predicted-and-correct/incorrect at threshold `τ`:

  C(τ) = P(ŷ = y ∧ ĉ ≥ τ),     I(τ) = P(ŷ ≠ y ∧ ĉ ≥ τ),

with group versions `C_g(τ)`, `I_g(τ)`, and the meta true-positive and false-positive rates `R^TP(τ) = C(τ)/C(0)`, `R^FP(τ) = I(τ)/I(0)`. Equalized odds (Hardt, Price, Srebro) for this meta-task says: every subgroup should have the same true-positive and false-positive rate — i.e. each subgroup should be predicted on at the same rate separately conditional on correctness and on incorrectness. That's the fairness target.

Now build the reference that achieves it. Take any selective classifier and, at each threshold `τ`, keep its total counts `C(τ)` and `I(τ)` unchanged — so its *average* accuracy-coverage curve is preserved — but redistribute those correct and incorrect predictions across subgroups *in proportion to each subgroup's full-coverage share*:

  C̃_g(τ) = C_g(0) · C(τ)/C(0),     Ĩ_g(τ) = I_g(0) · I(τ)/I(0).

Check what this does to the meta-rates per subgroup: `R̃^TP_g(τ) = C̃_g(τ)/C̃_g(0) = [C_g(0)C(τ)/C(0)] / [C_g(0)C(0)/C(0)] = C(τ)/C(0) = R̃^TP(τ)`, the same for *every* subgroup; identically `R̃^FP_g(τ) = I(τ)/I(0)` for all `g`. So this reference satisfies equalized odds by construction: acceptance is group-agnostic separately inside the correct and incorrect pools. Its accuracy on subgroup `g` is

  Ã_g(τ) = [C_g(0)C(τ)/C(0)] / [C_g(0)C(τ)/C(0) + I_g(0)I(τ)/I(0)].

This is the "abstain without regard to group membership" reference. The catch: as written it needs the labels (it sorts points into correct/incorrect) and a random redistribution, so it isn't directly implementable as a test-time rule. But it tells me what the shared threshold is failing to do: it is not distributing acceptance pressure evenly across the subgroup structure. The task metric I can control directly is the overall deferral rate per subgroup, so equal subgroup coverage becomes the implementable coverage analogue.

First the sobering question: is the global SR rule even close to this reference, or is the gap fundamental? Let me decompose the margin distribution into the worst subgroup and the rest, `F = p·F_wg + (1−p)·F_bg` with `p` the worst-group mass and `A_wg(0) < A_bg(0)`. Define the share of each subgroup among correct/incorrect predictions, `CF_g(τ) = p_g·cor_g(τ)/cor(τ)` and `IF_g(τ) = p_g·inc_g(τ)/inc(τ)` where `cor_g(τ) = 1 − F_g(τ)`, `inc_g(τ) = F_g(−τ)`. Then the SR worst-group accuracy is

  A_wg(τ) = 1 / (1 + (IF_wg(τ)/CF_wg(τ)) · (I(τ)/C(τ))),

and the reference worst-group accuracy, by the same algebra with the shares *frozen at* `τ = 0`, is

  Ã_wg(τ) = 1 / (1 + (IF_wg(0)/CF_wg(0)) · (I(τ)/C(τ))).

(The reference freezes `IF_wg/CF_wg` at its full-coverage value precisely because it keeps each subgroup's share of correct/incorrect predictions equal to its share at full coverage.) So the SR rule beats the reference on the worst subgroup, `A_wg(τ) ≥ Ã_wg(τ)` for all `τ ≥ 0`, exactly when `IF_wg(τ)/CF_wg(τ) ≤ IF_wg(0)/CF_wg(0)` for all `τ`, which — looking at the neighborhood of `τ = 0` — needs

  d/dτ [ IF_wg(τ)/CF_wg(τ) ] |_{τ=0} ≤ 0.

Let me compute that derivative. I'll first bound `d/dτ(1/CF_wg)|_0`. Write `1/CF_wg = (p·cor_wg + (1−p)cor_bg)/(p·cor_wg) = 1 + ((1−p)/p)·(1−F_bg)/(1−F_wg)`. Differentiate the ratio `(1−F_bg)/(1−F_wg)`: its derivative is `[f_wg(1−F_bg) − f_bg(1−F_wg)]/(1−F_wg)²`. At `τ = 0`, factoring,

  d/dτ(1/CF_wg)|_0 = ((1−p)/p) · (1/(1−F_wg(0))) · ( f_wg(0)(1−F_bg(0))/(1−F_wg(0)) − f_bg(0) ).

Since the worst group is worse, `A_wg(0) < A_bg(0)` means `F_wg(0) ≥ F_bg(0)`, so `(1−F_bg(0))/(1−F_wg(0)) ≥ F_bg(0)/F_wg(0)` — wait, let me be careful and use the bound the structure actually gives: with `0 < F_bg(0) ≤ F_wg(0) < 1`, I can lower-bound `(1−F_bg(0))/(1−F_wg(0))` by `F_bg(0)/F_wg(0)`? Let me check: `(1−F_bg)/(1−F_wg) ≥ F_bg/F_wg ⇔ F_wg(1−F_bg) ≥ F_bg(1−F_wg) ⇔ F_wg − F_wg F_bg ≥ F_bg − F_bg F_wg ⇔ F_wg ≥ F_bg`, true. Good, so

  d/dτ(1/CF_wg)|_0 ≥ ((1−p)/p) · (F_wg(0)/(1−F_wg(0))) · ( f_wg(0)F_bg(0) − f_bg(0)F_wg(0) )/F_wg(0)².

Now the full ratio derivative. By the product rule, `d/dτ(IF_wg/CF_wg)|_0 = (d/dτ IF_wg|_0)/CF_wg(0) + IF_wg(0)·d/dτ(1/CF_wg)|_0`. Substituting the exact derivative of `IF_wg` and the lower bound on `d/dτ(1/CF_wg)|_0`, then collecting the positive prefactors, gives

  d/dτ [ IF_wg/CF_wg ] |_0 ≥ C · ( f_bg(0)F_wg(0) − f_wg(0)F_bg(0) )

for a strictly positive constant `C` (a product of positive prefactors, with the bracket `[(1 + ((1−p)/p)(1−F_bg(0))/(1−F_wg(0)))/(1 + ((1−p)/p)F_bg(0)/F_wg(0)) − F_wg(0)/(1−F_wg(0))]` non-negative because `A_bg(0) ≥ A_wg(0)` makes the first fraction `≥ 1` and `A_wg(0) > ½` makes `F_wg(0)/(1−F_wg(0)) < 1`). So if the SR rule is to even match the group-agnostic reference on the worst group, I need `d/dτ(IF_wg/CF_wg)|_0 ≤ 0`, which combined with the bound forces

  f_bg(0)F_wg(0) − f_wg(0)F_bg(0) ≤ 0,    i.e.    f_bg(0)/f_wg(0) ≤ F_bg(0)/F_wg(0).

Rewrite the right side in accuracy terms: `F_g(0) = 1 − A_g(0)` is the full-coverage *error*, so the necessary condition is `f_bg(0)/f_wg(0) ≤ (1 − A_bg(0))/(1 − A_wg(0))` — the ratio of full-coverage errors. The better group has the smaller error, so the right side is below 1 and shrinks as the full-coverage disparity grows. A bigger disparity makes the condition *harder* to satisfy. So when subgroup disparity is large — exactly the regime I'm worried about — the global SR rule has a hard time even matching the group-agnostic reference, let alone beating it.

Let me pin down how bad it is in the common case. Suppose `F_wg` and `F_bg` are log-concave and the better group is a translate of the worst one, `f_bg(x) = f_wg(x − d)` with `d > 0` (the cleanest model of "one group is just shifted right"). First, this really does make `wg` the worse group: `A_wg(0) = 1 − F_wg(0) ≤ 1 − F_wg(−d) = 1 − F_bg(0) = A_bg(0)`. Now I claim `CF_wg(τ)` is *decreasing* in `τ`. Writing `CF_wg(τ) = 1/(1 + ((1−p)/p)·F_wg(2μ_wg − τ + d)/F_wg(2μ_wg − τ))` (using symmetry to flip `1−F` into `F` of the reflected point), its derivative has the sign of `−[f_wg(a+d)F_wg(a) − f_wg(a)F_wg(a+d)]` evaluated at `a = 2μ_wg − τ`, which is `≤ 0` iff `f_wg(a+d)/F_wg(a+d) ≤ f_wg(a)/F_wg(a)` — true by log-concavity since `d > 0` and `f/F` is decreasing. So `CF_wg` decreases. By the mirror argument `IF_wg(τ) = 1/(1 + ((1−p)/p)·F_wg(−τ−d)/F_wg(−τ))` is *increasing* (same log-concavity inequality, opposite sign). Decreasing `CF_wg`, increasing `IF_wg` ⇒ the ratio `IF_wg/CF_wg` is increasing in `τ` (`d/dτ(IF/CF) = (CF·IF' − IF·CF')/CF² ≥ 0` since `IF' ≥ 0`, `CF' ≤ 0`). And an increasing `IF_wg/CF_wg` plugged into

  A_wg(τ) = 1/(1 + (IF_wg(τ)/CF_wg(τ))(I(τ)/C(τ))) ≤ 1/(1 + (IF_wg(0)/CF_wg(0))(I(τ)/C(τ))) = Ã_wg(τ)

for *all* `τ ≥ 0`. So in the translated-log-concave regime the global SR rule underperforms the group-agnostic reference at *every* threshold. Not a corner case — the typical one. (And if instead the groups differ by a *scale* `v`, `f_bg(τ) = v·f_wg(v(τ−μ_bg)+μ_wg)`, the necessary condition reduces, via log-concavity, to `v < 1`: the worst group would have to have *smaller* variance to win — but empirically the worst group has *larger* variance, `v > 1`, so it loses.)

The lesson is now unambiguous: any rule that picks a *single decision boundary shared across subgroups* — global SR, marginal conformal, a single learned correctness predictor — inherits this. The shared boundary can cover the left-shifted worst subgroup at a lower rate, over-defer it, and underperform a group-agnostic redistribution that ignores group identity when it chooses which correct and incorrect points to keep. The metric I can directly repair in the available policy interface is the subgroup deferral gap. So instead of one boundary for everyone, give each subgroup its own boundary, set so that each subgroup is covered at the same rate.

Concretely: I already have the score `s(x) = max(probs)` and I want each subgroup to keep coverage equal to the target. For a single global threshold I set it to the `(1 − target_coverage)` quantile of `s` over all calibration points, which makes the global coverage equal to the target. The fix is to do that *per subgroup*: for each subgroup `g`, set its threshold to the `(1 − target_coverage)` quantile of `s` over the calibration points *in that subgroup*. Then by definition each subgroup has a `target_coverage` fraction of its own points above its own threshold — every subgroup is covered at the same rate. Accept iff `s(x) ≥ threshold_group(g)`. This is the implementable, group-aware coverage rule suggested by the reference's failure mode: I don't redistribute predictions and I don't need labels at decision time; I just calibrate one quantile per subgroup so the coverage comes out equal. Equal per-subgroup coverage directly minimizes the deferral-rate gap, since every subgroup defers a `1 − target_coverage` fraction on calibration. It is not the same as the label-using equalized-odds construction, because equalized odds would require matching acceptance separately on correct and incorrect points.

Why a *quantile* and not, say, a learned per-group cost or a fitted threshold? Because the quantile pins coverage to the target *exactly* and offline — no extra model to fit, trivial compute, robust to the fixed-pipeline constraint that I can't retrain anything. The cost-based Chow view would require me to know the misclassification-versus-deferral cost; the coverage-target view sidesteps that and is what the benchmark actually asks for. And I should handle the degenerate case: a subgroup that never appeared in calibration has no per-group quantile, so I fall back to the global quantile — the best coverage-matched default I have for an unseen subgroup.

One more property I get for free, and it matters for the AUROC metric. Per-subgroup thresholding only *moves the decision boundary* per subgroup; it does not change the underlying score `s(x) = max(probs)` that I rank correctness by. So the AUROC of the acceptance score as a predictor of correctness is just the base softmax-response score's AUROC — preserved by construction, because re-thresholding is a monotone, group-local shift that doesn't touch the global ranking the metric reads. I keep the SR score's ranking quality and *additionally* equalize coverage.

Let me also be honest about the tension this rule carries. Equalizing coverage does not, by itself, equalize *selective accuracy*. If the worst subgroup is genuinely hard, moving thresholds cannot repair the base classifier's full-coverage error. The honest scope of this rule is: it removes the *deferral-rate* disparity that the global threshold can inject (small `deferral_rate_gap`), it keeps the SR score's correctness ranking (AUROC), and it stops the worst subgroup from being silently over-deferred. Closing the full-coverage *accuracy* gap is a training-time problem (group DRO and friends), which this rule deliberately doesn't touch because the base model is fixed.

So let me write the rule into the one empty slot of the selective-policy harness — the acceptance rule: the SR score, the per-subgroup quantile calibration, the per-subgroup accept test, with the global-quantile fallback.

```python
import numpy as np

TARGET_COVERAGE_DEFAULT = 0.8


class SelectivePolicy:
    """Subgroup-specific thresholds tuned on calibration data."""

    def __init__(self, target_coverage: float = TARGET_COVERAGE_DEFAULT, random_state: int = 0):
        self.target_coverage = float(target_coverage)
        self.random_state = int(random_state)
        self.threshold_: float = 0.5
        self.group_thresholds_: dict[int, float] = {}
        self.meta_model_ = None
        self.strategy_name = "groupwise_thresholding"

    def fit(self, probs, y_true, groups, X=None):
        scores = self.acceptance_score(probs, groups, X)
        quantile = float(np.clip(1.0 - self.target_coverage, 0.0, 1.0))
        self.threshold_ = float(np.quantile(scores, quantile))
        self.group_thresholds_ = {}
        for group_id in np.unique(groups):
            mask = groups == group_id
            if not np.any(mask):
                continue
            self.group_thresholds_[int(group_id)] = float(np.quantile(scores[mask], quantile))
        self.meta_model_ = None
        return self

    def acceptance_score(self, probs, groups, X=None):
        return np.max(probs, axis=1)

    def predict_accept(self, probs, groups, X=None):
        scores = self.acceptance_score(probs, groups, X)
        thresholds = np.asarray(
            [self.group_thresholds_.get(int(g), self.threshold_) for g in groups], dtype=float)
        return scores >= thresholds

    def calibration_summary(self):
        summary = {"threshold": float(self.threshold_)}
        for group_id, threshold in self.group_thresholds_.items():
            summary[f"threshold_group_{group_id}"] = float(threshold)
        return summary
```

Let me trace the causal chain back. I started fixed: a frozen base classifier and one knob, the accept/defer rule at a target coverage, and a worry that under subgroup shift one subgroup gets over-deferred. Chow's rule plus softmax response gave the obvious global-threshold answer, and the odd-monotone-transform invariance let me use the bare max-probability as the score. But differentiating selective accuracy gave the monotonicity test `f(−τ)/F(−τ) ≥ f(τ)/(1−F(τ))`, and for symmetric left-log-concave margins — which even bimodal mixtures of Gaussians satisfy — that test says monotonicity direction is set by whether full-coverage accuracy is above or below 50%; the skew analysis confirmed it survives realistic asymmetry. So a single global threshold helps the above-50% average while *hurting* the below-50% worst subgroup, and pours deferrals onto it. To say what the shared threshold should be compared against I frame predict-vs-defer as a meta-classification and build the group-agnostic reference that satisfies equalized odds by redistributing correct and incorrect predictions according to their subgroup shares at `τ = 0`.

The necessary-condition analysis then shows that any single shared-boundary SR rule struggles to match that reference when the full-coverage error ratio is unfavorable, and in the translated-log-concave regime underperforms it at every threshold. The implementable thing the harness lets me enforce is equal total coverage per subgroup, so the rule is one threshold per subgroup, each set to the same within-subgroup `(1 − target_coverage)` quantile — equal coverage, minimal deferral-rate gap on calibration, the SR score's AUROC preserved because the returned score is still `max(probs)` — with a global-quantile fallback for unseen subgroups. The slot in the harness fills with exactly that.
