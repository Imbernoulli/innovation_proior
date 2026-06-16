Let me start from what actually goes wrong. I have a deep network — non-convex, several layers, somewhere between tens of thousands and millions of parameters — and I train it on crowdsourced data that contains people's records. The trained model gets shipped, sometimes onto a phone, where someone can read its parameters. And these models can memorize: there are reconstruction attacks that pull recognizable training images back out of a face recognizer with only query access, and distributed deep-learning systems already have to think about adversaries who see sanitized parameter updates. So I have to assume the worst adversary — one who knows my whole training procedure, can read the final weights, and may even control every training record except the single one I am trying to protect. I want a guarantee that holds against that adversary, and I want it to be a real, worst-case mathematical guarantee, not a heuristic. The honest yardstick for "real guarantee" here is differential privacy: a randomized mechanism `M` is `(ε, δ)`-private if for any two datasets `d, d'` where one is obtained from the other by adding or removing one record, and any set of outcomes `S`, `Pr[M(d) ∈ S] ≤ e^ε Pr[M(d') ∈ S] + δ`. The whole point of that inequality is deniability — whatever the released model reveals, it would have revealed almost as readily had any one person not been in the data. And the prize I'm reaching for is a *small* `ε`, single-digit, on a real deep net. People have gotten small `ε` on convex models with a handful of parameters, and people have trained genuine neural nets but at a privacy loss so large the guarantee says nothing. Nobody has both.

First instinct: treat training as a black box and privatize only the output. I run ordinary SGD, get final parameters `θ`, and add noise to `θ` before releasing. The Gaussian mechanism tells me exactly how much noise: if I have a vector-valued function `f` with `L2`-sensitivity `S_f = max_{d∼d'} ‖f(d) − f(d')‖₂`, then `f(d) + N(0, S_f² σ² I)` is `(ε, δ)`-private for `σ` on the order of `√(2 ln(1.25/δ))/ε`. Clean. But what is the sensitivity of "the final weights of SGD on a non-convex loss"? If I change one training example, how far can `θ_final` move? For a convex problem you can bound this — the minimizer is a stable function of the data — and that's exactly what the private-ERM line (Chaudhuri–Monteleoni–Sarwate, Bassily–Smith–Thakurta) exploits with output and objective perturbation, getting tight excess-risk bounds like `Õ(√p/ε)`. But my loss is non-convex with millions of parameters; SGD is a long chaotic trajectory and one changed example, amplified through thousands of steps, could in principle send `θ_final` anywhere. I have no tight characterization of that dependence, so the only honest sensitivity bound is a hopeless worst case, and noise calibrated to it would obliterate the model. Output perturbation is out for non-convex deep nets. Wall.

So I can't privatize the endpoint. The alternative is to privatize the *process* — control the influence of the data at every step of training, the way Song–Chaudhuri–Sarwate and the stochastic variant in Bassily et al. do: add noise inside the SGD loop. That's promising because the thing the data touches at each step is a single object I understand — the gradient. At step `t` I form a batch and compute `g_B = (1/|B|) Σ_{x∈B} ∇_θ L(θ, x)`, an unbiased estimate of the true gradient, and step `θ ← θ − η g_B`. If I add noise to the gradient before stepping, and if I can bound how much one example moves that gradient, then each step is a Gaussian mechanism with a sensitivity I can actually compute, and composition handles the rest. The released object at each step is the noisy gradient (and hence, by post-processing, the next `θ`).

So what is the sensitivity of the gradient estimate to one example? Here's the trouble: there is no a-priori bound on `‖∇_θ L(θ, x)‖`. An outlier or a misclassified example early in training can produce an enormous gradient. So the sum `Σ_x ∇_θ L(θ, x)` has unbounded sensitivity — swap one example and the sum can change by an arbitrary amount — and I cannot calibrate Gaussian noise to "infinity." I need to *make* the sensitivity finite, by force. The move is to cap each example's contribution: replace each per-example gradient `g` by a version whose norm is at most some threshold `C`. The cleanest cap in `L2` — which is the norm the Gaussian mechanism wants — is to rescale: `ḡ = g / max(1, ‖g‖₂ / C)`. If `‖g‖₂ ≤ C` this leaves `g` untouched; if `‖g‖₂ > C` it scales it down to exactly norm `C`, keeping its direction. Now every clipped per-example gradient lives in the ball of radius `C`.

I have to be careful about *where* I clip, and this is exactly the place the non-private habit would lead me astray. In ordinary deep learning, gradient clipping is already standard — but there you clip the *averaged* batch gradient, after summing, just to keep a step from exploding. If I do that here, I've bounded the norm of the whole batch gradient, which tells me nothing about any *individual's* contribution: I could have one giant per-example gradient hidden inside a batch whose average happens to be small. Privacy needs to bound *each person's* influence on the released sum. So I must clip *per example, before averaging*. That's the non-negotiable difference from the non-private trick, and it costs me — I now need the per-example gradients, which plain batched autodiff doesn't hand me, though they can be computed efficiently. With per-example clipping in place, let me actually compute the sensitivity. The released aggregate over a lot of `L` examples is `Σ_i ḡ_i` (I'll divide by `L` at the end, that's just post-processing scaling). If I used replacement adjacency, changing one record would replace `ḡ_old` by `ḡ_new`, so the triangle inequality would give `‖ḡ_new − ḡ_old‖₂ ≤ 2C`; that is a different convention and only changes constants. The subsampling analysis I want uses the add/remove convention, `d = d' ∪ {one example}`. Under that convention the two sums differ by exactly one clipped vector, whose norm is at most `C`. So the `L2`-sensitivity of `Σ_i ḡ_i` is `C`, and the sensitivity of the mean is `C/L`. Good — clipping bought me exactly the finite sensitivity I needed, and its value is the knob `C`.

Now noise. Gaussian mechanism: to privatize a function of sensitivity `C`, add `N(0, σ² C² I)`. So the released aggregate is `Σ_i ḡ_i + N(0, σ² C² I)`, and dividing by `L` to get an average gradient gives noise of standard deviation `σC/L` per coordinate. Notice how nicely `σ` and `C` factor: `C` is the sensitivity, set by the geometry of the gradients; `σ` is a unitless *noise multiplier* that, together with `C`, sets the privacy. The std of the noise on the sum is `σC`, on the mean `σC/L`. I'll keep `σ` as the privacy knob and `C` as the clipping knob, deliberately decoupled — calibrate `σ` to the budget, choose `C` from the gradient scale. The descent step is then ordinary SGD on this noised average: `θ_{t+1} = θ_t − η_t g̃_t`. That's the algorithm: at each step, take a random subset of examples, compute their per-example gradients, clip each in `L2` to `C`, sum, add `N(0, σ²C²I)`, divide by `L`, and step.

Before I worry about accounting, one structural choice deserves attention: the random subset. I could use all `N` examples each step, but I don't — I sample a lot of expected size `L = qN` with per-example inclusion probability `q = L/N`. Partly this is the ordinary SGD reason (a sample estimates the gradient and the variance falls with `L`). But there's a privacy reason that's at least as important: privacy amplification by subsampling. If a mechanism is `(ε, δ)`-private on the full data, running it on a uniformly random `q`-fraction makes it roughly `(O(qε), qδ)`-private, because an example that probably isn't even looked at this step is, in expectation, better hidden. So subsampling isn't just for compute — it's privacy leverage, and it means the per-step cost scales with `q`. (There's a secondary engineering split worth noting: the *lot* `L`, the group over which I add noise, need not equal the *batch* the hardware processes at once. I can compute gradients in small batches to fit memory, then group several into a lot before noising. The analysis only cares about the lot.)

Now the part that actually decides whether `ε` is small or vacuous: the accounting. I run `T` steps, each a subsampled Gaussian mechanism, and I need the total privacy of the composition. The standard tool is the advanced (strong) composition theorem of Dwork–Rothblum–Vadhan: composing `k` mechanisms that are each `(ε, δ)`-private gives `(ε̃, kδ + δ')` with `ε̃ = ε√(2k ln(1/δ')) + kε(e^ε − 1)/(e^ε + 1) ≈ ε√(2k ln(1/δ'))`. The `√k` instead of `k` is the whole reason iterative private algorithms are feasible at all. So let me just plug in. Each step, after subsampling, is about `(qε₀, qδ₀)` for the per-step `(ε₀, δ₀)` of the Gaussian, and I compose `T = k` of them. I get a total `ε` growing like `qε₀ √(T log(1/δ'))`, with a `δ` part accumulating `T q δ₀`. Let me put numbers to it the way I actually care about — sampling rate `q = 0.01`, noise multiplier `σ = 4`, target `δ = 10⁻⁵`, `T = 10⁴` steps. Running this through strong composition I get `ε ≈ 9.3`. That is too close to ten for the kind of guarantee I want, and it is far from the privacy cost the Gaussian structure seems to suggest. Wall.

So why is `9.3` so much worse than what the structure should allow? Strong composition is a bound for *arbitrary* `(ε, δ)` mechanisms — it only knows each step's `(ε, δ)` pair, i.e. each step's *tail*. It is blind to the specific shape of the noise I'm composing. And I'm composing the same, very well-understood thing every step: a Gaussian on a subsample. The Gaussian's privacy loss isn't just "bounded with high probability"; its entire distribution is known. Let me look at what I'm throwing away. The privacy loss of a mechanism at output `o`, for neighbors `d, d'`, is the log-likelihood ratio `c(o) = log( Pr[M(d) = o] / Pr[M(d') = o] )` — a random variable over the mechanism's own coins. Differential privacy is exactly a *tail* statement about `c`: `(ε, δ)`-DP means `Pr[c > ε] ≤ δ` up to the usual set-splitting argument. For a pure Gaussian mechanism of sensitivity `Δ`, I can compute `c` explicitly: taking the differing example to shift one coordinate's mean by `Δ`, so `M(d) ∼ N(Δ, σ²)` and `M(d') ∼ N(0, σ²)`, the log-likelihood ratio at a point is `c(z) = log(N(Δ)/N(0)) = (2Δz − Δ²)/(2σ²)`. When `z ∼ N(Δ, σ²)`, the distribution used in the moment definition for the numerator database, this is Gaussian with mean `(2Δ·Δ − Δ²)/(2σ²) = Δ²/(2σ²)` and variance `(Δ/σ²)² σ² = Δ²/σ²`. When the same likelihood-ratio direction is evaluated at `z ∼ N(0, σ²)`, the mean is `−Δ²/(2σ²)` and the variance stays `Δ²/σ²`; maximizing over neighboring directions covers both signs. So the privacy loss is a tame, concentrated bell curve — the sign is controlled by which neighbor generates the output, and the scale is tiny when `σ ≫ Δ`. Strong composition takes that concentrated distribution, summarizes it by a single `(ε, δ)` tail point, and then composes the tails — pessimistically, because composing tails forces a union-bound-flavored `√(log(1/δ'))` slack and a `kδ` accumulation. I'm certain I can do better by composing the *distributions* directly.

How do I compose a loss that is a sum of per-step losses cleanly, even when subsequent steps are chosen from earlier public outputs? Moment generating functions are the right shape: if I can condition on the public past and bound the conditional MGF of each fresh-noise step, then the product telescopes into a sum after taking logs. That's the linearity I want — track a quantity per step that simply sums across steps. So let me define, for a mechanism `M` at order `λ`, the log of the moment-generating function of its privacy loss:

  `α_M(λ; aux, d, d') = log E_{o ∼ M(aux, d)}[ exp(λ · c(o; M, aux, d, d')) ]`,

and to make it a property of the mechanism alone, take the worst case over auxiliary inputs and neighbors, `α_M(λ) = max_{aux, d, d'} α_M(λ; aux, d, d')`. The `aux` is there because my steps are *adaptive* — each step's mechanism depends on the previous outputs (the current `θ` is a function of all past noisy gradients) — so I model the `k`-th mechanism as taking the previous outputs as auxiliary input.

I need two things from this `α` to make it an accountant: that it composes (so I can sum per-step bounds), and that I can convert a bound on it back into an `(ε, δ)` statement (so the final answer is in the currency I promised). Let me prove both.

Composition first. Take a sequence of adaptive mechanisms `M_1, …, M_k`, `M_i` mapping the previous outputs and the database to `R_i`. Write `M_{1:i}` for the prefix and `o_{1:i}` for an outcome prefix. The privacy loss of the whole sequence at an outcome sequence factors, because the joint density factors into conditionals:

  `c(o_{1:k}; M_{1:k}, d, d') = log ∏_{i=1}^k [ Pr(M_i(d) = o_i | M_{1:i−1}(d) = o_{1:i−1}) / Pr(M_i(d') = o_i | M_{1:i−1}(d') = o_{1:i−1}) ]`
  `= Σ_{i=1}^k c(o_i; M_i, o_{1:i−1}, d, d')`.

So the total loss is the sum of per-step losses, each conditioned on the realized past. Now take the MGF and peel it from the last step backward:

  `E_{o_{1:k} ∼ M_{1:k}(d)}[ exp(λ c(o_{1:k}; M_{1:k}, d, d')) ]`
  `= E[ exp(λ Σ_i c(o_i; M_i, o_{1:i−1}, d, d')) ]`
  `= E[ ∏_i exp(λ c(o_i; M_i, o_{1:i−1}, d, d')) ]`.

Condition on the realized prefix `o_{1:k−1}`. The only randomness left in the inner expectation is the fresh draw of `M_k`, and the prefix is exactly the auxiliary input allowed in the definition of `α`, so

  `E[ exp(λ c(o_k; M_k, o_{1:k−1}, d, d')) | o_{1:k−1} ] ≤ exp(α_{M_k}(λ))`.

Pull that factor out, then repeat the same argument for `k−1, k−2, ...`. I do not need the unconditioned privacy-loss variables to be independent; I only need each fresh-noise conditional MGF, given the public past, to be bounded by the worst-case log moment for that step. The product of those bounds is `∏_i exp(α_{M_i}(λ)) = exp(Σ_i α_{M_i}(λ))`. Taking logs: `α_M(λ) ≤ Σ_{i=1}^k α_{M_i}(λ)`. Exactly the linear composition I wanted — and it holds even when each mechanism is *chosen* based on the public outputs of the earlier ones, because the conditioning is built in.

Now the conversion back to `(ε, δ)`. This is just Markov's inequality on the exponentiated loss. For any `λ > 0`,

  `Pr_{o ∼ M(d)}[ c(o) ≥ ε ] = Pr[ exp(λ c(o)) ≥ exp(λε) ] ≤ E[exp(λ c(o))] / exp(λε) ≤ exp(α_M(λ) − λε)`.

To turn a tail bound on the loss into the differential-privacy inequality, split any output set `S` by the bad event `B = {o : c(o) ≥ ε}`: `Pr[M(d) ∈ S] = Pr[M(d) ∈ S ∩ Bᶜ] + Pr[M(d) ∈ S ∩ B] ≤ e^ε Pr[M(d') ∈ S ∩ Bᶜ] + Pr[M(d) ∈ B] ≤ e^ε Pr[M(d') ∈ S] + exp(α_M(λ) − λε)`. So `M` is `(ε, δ)`-private with `δ = exp(α_M(λ) − λε)`, and I get to *minimize over `λ`*: `δ = min_λ exp(α_M(λ) − λε)`. That's the accountant: bound `α(λ)` at each step, sum the bounds, then optimize `λ` to read off the best `(ε, δ)`.

The only thing left — and it's the heart of the matter — is to bound `α(λ)` for a single subsampled-Gaussian step. Set the sensitivity to `1` (clipping already rescaled everything to `C`, so I work in units of `C`; the noise has std `σ` in those units). The neighbor relation: `d = d' ∪ {one example}`. With one differing coordinate after the reduction, this is a one-dimensional problem. Let `μ₀` be the density of `N(0, σ²)` and `μ₁` the density of `N(1, σ²)` — the differing example shifts the mean of one coordinate by its (unit) clipped gradient. Because the example is included only with probability `q`, the mechanism's output distribution under `d` is the *mixture* `μ = (1−q)μ₀ + qμ₁`, while under `d'` it's `μ₀`. So I have to bound two expectations — one for each direction of the likelihood ratio:

  `E₁ = E_{z ∼ μ₀}[ (μ₀(z)/μ(z))^λ ]`  and  `E₂ = E_{z ∼ μ}[ (μ(z)/μ₀(z))^λ ]`,

and `α(λ) = log max(E₁, E₂)`. Both have the same shape: bound `E_{z∼ν₀}[(ν₀/ν₁)^λ] = E_{z∼ν₁}[(ν₀/ν₁)^{λ+1}]` for a pair `(ν₀, ν₁)`. Let me expand the inner ratio around `1`, since `ν₀` and `ν₁` differ only by the `q`-weighted mean shift and are close. Write `ν₀/ν₁ = 1 + (ν₀ − ν₁)/ν₁` and binomially expand:

  `E_{z∼ν₁}[ (1 + (ν₀−ν₁)/ν₁)^{λ+1} ] = Σ_{t=0}^{λ+1} C(λ+1, t) · E_{z∼ν₁}[ ((ν₀−ν₁)/ν₁)^t ]`.

The `t = 0` term is `1`. The `t = 1` term is `E_{z∼ν₁}[(ν₀−ν₁)/ν₁] = ∫(ν₀ − ν₁) = 1 − 1 = 0` — the first-order term vanishes, which is the whole reason this is going to be `O(q²)` and not `O(q)`. So the leading contribution is the `t = 2` term, and I need to show it's the dominant one and bound it. Take the harder direction, `ν₀ = μ₀, ν₁ = μ`. Since `μ ≥ (1−q)μ₀` (the mixture is at least its `μ₀` part), and `μ₀ − μ = q(μ₀ − μ₁)`,

  `E_{z∼μ}[((μ₀−μ)/μ)²] = q² E_{z∼μ}[((μ₀−μ₁)/μ)²] = q² ∫ (μ₀−μ₁)²/μ ≤ (q²/(1−q)) ∫ (μ₀−μ₁)²/μ₀ = (q²/(1−q)) E_{z∼μ₀}[((μ₀−μ₁)/μ₀)²]`.

So I've reduced everything to one clean Gaussian expectation, `E_{z∼μ₀}[((μ₀−μ₁)/μ₀)²]`. Now `μ₁(z)/μ₀(z) = exp((2z−1)/2σ²)` (ratio of the two Gaussians), so `(μ₀−μ₁)/μ₀ = 1 − exp((2z−1)/2σ²)` and

  `E_{z∼μ₀}[(1 − exp((2z−1)/2σ²))²] = 1 − 2 E_{μ₀}[exp((2z−1)/2σ²)] + E_{μ₀}[exp((4z−2)/2σ²)]`.

Using the elementary Gaussian fact `E_{z∼μ₀}[exp(az/σ²)] = exp(a²/2σ²)`: the middle term is `exp(1/2σ²)·exp(−1/2σ²) = 1`, so `−2·1 = −2`; the last term is `exp(4/2σ²)·exp(−2/2σ²) = exp(1/σ²)`. Total: `1 − 2 + exp(1/σ²) = exp(1/σ²) − 1`. Beautiful — it collapses to `exp(1/σ²) − 1 ≈ 1/σ²`. So the `t = 2` term, with the binomial coefficient `C(λ+1, 2) = λ(λ+1)/2`, is

  `C(λ+1, 2) · E_{z∼μ}[((μ₀−μ)/μ)²] ≤ (λ(λ+1)/2) · (q²/(1−q)) · (exp(1/σ²) − 1) ≤ λ(λ+1) q² / ((1−q) σ²)`.

There it is: the second-order contribution is at most `q²λ(λ+1)/((1−q)σ²)`. I still owe myself that the higher terms `t ≥ 3` don't ruin this. The clean way is to bound each `t`-th term and check it drops off geometrically. I use `μ₀ − μ = q(μ₀ − μ₁)`, `μ ≥ (1−q)μ₀`, and the Gaussian absolute-moment bound `E_{μ₀}[|z|^t] ≤ σ^t (t−1)!!`. I also need three pointwise comparisons for `|μ₀ − μ₁|`: on `z ≤ 0`, it is at most `−(z−1)μ₀/σ²`; on `0 ≤ z ≤ 1`, it is at most `μ₀(exp(1/(2σ²)) − 1) ≤ μ₀/σ²`; on `z ≥ 1`, it is at most `z μ₁/σ²`. Splitting the `t`-th integral into those three regions gives the actual cases. The left tail is bounded by

  `(2q)^t (t−1)!! / (2(1−q)^{t−1} σ^t)`.

The middle interval is bounded by

  `q^t / ((1−q)^t σ^{2t})`.

The right tail picks up the likelihood-ratio exponential, so it is bounded by

  `(2q)^t exp((t²−t)/(2σ²)) (σ^t(t−1)!! + t^t) / (2(1−q)^{t−1} σ^{2t})`.

After multiplying by the binomial coefficient `{λ+1 \choose t}`, the assumptions `q < 1/(16σ)` and `λ ≤ σ² ln(1/(qσ))` make the terms for `t > 3` decrease by a fixed geometric factor. So the whole tail of the binomial expansion is dominated by the `t = 3` contribution, `O(q³λ³/σ³)`. The expansion bounds each raw moment by `1 + r`; since the accountant stores the log moment, `log(1+r) ≤ r` converts the same excess bound into an `α(λ)` bound. The leading, explicit contribution is therefore

  `α(λ) ≤ q²λ(λ+1)/((1−q)σ²) + O(q³λ³/σ³)`.

In the range where the lemma is used, the remainder is smaller than a constant multiple of that explicit `q²λ(λ+1)/((1−q)σ²)` term, so the theorem can absorb it into the constants. Now I just turn the crank with composition. Over `T` steps, `α(λ) ≤ T · q²λ(λ+1)/((1−q)σ²) + T·O(q³λ³/σ³)`, which in the intended range is controlled by a constant multiple of `T q² λ² / σ²`. Feed it into the tail bound: I want `(ε, δ)`-privacy, so I need `α(λ) ≤ λε/2` and `exp(−λε/2) ≤ δ` (splitting `exp(α − λε)` into a part controlled by the moment and a remaining tail), plus the validity condition `λ ≤ σ² log(1/(qσ))`. The first condition is `T q² λ²/σ² ≤ λε/2`, i.e. `λ ≤ σ²ε/(2Tq²)` up to constants; the second is `λ ≥ 2 log(1/δ)/ε`. Both can hold simultaneously — pick `λ ≈ 2 log(1/δ)/ε` and require `σ` large enough that `λ` also satisfies the first — and the constants can be chosen so that, for every `ε < c₁ q²T`, all three conditions are met by choosing

  `σ = c₂ · q √(T log(1/δ)) / ε`

for explicit constants `c₁, c₂`. So this is the theorem: with the moments accountant, `σ ≥ c₂ q√(T log(1/δ))/ε` buys `(ε, δ)`-privacy. Compare with what strong composition demanded — there I'd have needed `σ = Ω(q √(T log(1/δ) log(T/δ))/ε)`, an extra factor of `√(log(T/δ))`. That `√log` factor, plus the `Tq` term in `δ` that the moments route never accumulates, is precisely the slack I felt was being wasted, and it's exactly the gap between `ε ≈ 9.3` and `ε ≈ 1.3` on those `q = 0.01, σ = 4, δ = 10⁻⁵, T = 10⁴` numbers. The win came from composing the whole privacy-loss distribution's moments instead of just its tail.

A couple of practical points about computing `α(λ)`. I don't have to use only the asymptotic bound — I can numerically integrate `E₁` and `E₂` directly for each `λ`, which is tighter, and the asymptotic `q²λ(λ+1)/((1−q)σ²)` is the analytic fallback that proves the theorem. And I don't need many orders: the optimal `λ` for the regimes I care about is small, so computing `α(λ)` for integer `λ` up to about `32` and minimizing the tail bound over that grid is enough. One more subtlety I should respect: because I'm fixing the number of iterations and the privacy parameters *ahead of time* and reading off the spent `ε` at the end, I am operating with a fixed budget rather than changing the privacy target mid-run.

So now let me write the mechanism I'd actually run, filling the one empty slot — the transformation from a lot's per-example gradients to the single noised aggregate the optimizer steps on. Flatten each parameter's per-example gradient, combine the per-parameter norms into one per-example `L2` norm, form the clip factor `min(1, C/(‖g‖+10⁻⁶))`, and use that same factor on every parameter tensor for the same example. Then I sum the clipped gradients for each parameter, add Gaussian noise with standard deviation `σC` to that sum, and only then divide by the expected lot size when the training loss is a mean. That ordering is important: the Gaussian mechanism is calibrated to the sensitivity of the sum, and the mean is just post-processing. The tiny denominator floor only makes the clip factor slightly smaller, so it cannot increase sensitivity. And the noise multiplier `σ` reported to the accountant is just the calibrated constant.

```python
import torch


class PrivateGradientMechanism:
    """Clip per-example gradients to C, add Gaussian noise to the summed gradients.

    The structure mirrors the optimizer wrapper pattern: compute one clipping factor
    per sample, accumulate clipped sums, add N(0, (sigma*C)^2) to each sum, then
    scale only as post-processing when the loss reduction is a mean.
    """

    def __init__(self, max_grad_norm, noise_multiplier, expected_lot_size,
                 loss_reduction="mean", generator=None):
        self.max_grad_norm = max_grad_norm          # C: per-example L2 clip threshold = sensitivity
        self.noise_multiplier = noise_multiplier    # sigma: calibrated up-front to (eps, delta)
        self.expected_lot_size = expected_lot_size
        self.loss_reduction = loss_reduction
        self.generator = generator

    def _generate_noise(self, reference):
        return torch.normal(
            mean=0.0,
            std=self.noise_multiplier * self.max_grad_norm,
            size=reference.shape,
            device=reference.device,
            dtype=reference.dtype,
            generator=self.generator,
        )

    def aggregate(self, per_sample_grads, step, epoch):
        batch_size = per_sample_grads[0].shape[0]

        # Per-sample norm across all parameters, matching the flat clipping rule.
        per_param_norms = [g.reshape(batch_size, -1).norm(2, dim=1) for g in per_sample_grads]
        norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)        # [B]

        # Opacus-style clip factor: min(1, C / (||g|| + small floor)).
        clip_factor = (self.max_grad_norm / (norms + 1e-6)).clamp(max=1.0)

        noised_grads = []
        for g in per_sample_grads:
            clip_factor_on_device = clip_factor.to(g.device).to(g.dtype)
            summed = torch.einsum("i,i...", clip_factor_on_device, g)

            # Gaussian mechanism: sensitivity of the sum is C, so std on the sum is sigma*C.
            grad = summed + self._generate_noise(summed)
            if self.loss_reduction == "mean":
                grad = grad / self.expected_lot_size
            noised_grads.append(grad)

        return noised_grads

    def get_effective_sigma(self, step, epoch):
        # constant noise multiplier; the accountant composes T identical subsampled-Gaussian steps
        return self.noise_multiplier
```

And the calibration that runs before training — bound the accumulated log-moments over `T` steps at sampling rate `q`, convert to `ε` at the target `δ`, and binary-search the smallest `σ` that fits the budget:

```python
import math


def compute_epsilon(steps, sigma, q, delta, orders=None):
    """Accumulated epsilon from the sampled-Gaussian log-moment bound."""
    if orders is None:
        orders = range(1, 64)
    best_eps = float("inf")
    for lam in orders:
        if lam <= 0:
            continue
        per_step_alpha = q * q * lam * (lam + 1) / ((1.0 - q) * sigma * sigma)
        total_alpha = steps * per_step_alpha
        # delta = exp(alpha - lambda*epsilon), so epsilon = (alpha + log(1/delta))/lambda.
        eps = (total_alpha + math.log(1.0 / delta)) / lam
        best_eps = min(best_eps, eps)
    return best_eps


def calibrate_noise_to_epsilon(target_epsilon, steps, q, delta, tol=1e-3):
    """Smallest sigma whose composed budget spends at most target_epsilon."""
    lo, hi = 0.01, 100.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if compute_epsilon(steps, mid, q, delta) > target_epsilon:
            lo = mid            # too little noise -> too much epsilon -> need more noise
        else:
            hi = mid
    return (lo + hi) / 2
```

Let me trace the causal chain back. I needed a worst-case privacy guarantee for a deep, non-convex network against an adversary who reads the weights, and output perturbation died because a non-convex SGD trajectory has no usable sensitivity for its endpoint. So I privatized the process instead, at the one object the data touches per step — the gradient — but the gradient has unbounded per-example norm, so I forced finite sensitivity by clipping each example's gradient in `L2` to `C`, and I had to do it per example before averaging (the non-private clip-after-average bounds nobody's individual influence), giving the summed aggregate sensitivity exactly `C`. Then the Gaussian mechanism added `N(0, σ²C²)` to the sum, with `σ` a unitless noise multiplier decoupled from `C`, and dividing by the expected lot size made the mean noise `σC/L`; subsampling each lot at rate `q` both reduced gradient variance and amplified privacy. Accounting the `T` composed steps with the generic strong composition theorem was too loose — it composes only each step's tail and pays a `√(log(T/δ))` factor, sending `ε` near ten where the structure should be much smaller — so I tracked the moments of the privacy-loss random variable: defined `α(λ)` as the log-MGF of the loss, proved it composes linearly by conditioning on the public past and peeling off each fresh-noise step, converted it to `(ε, δ)` via a Markov tail bound minimized over `λ`, and bounded the single-step subsampled-Gaussian moment by `q²λ(λ+1)/((1−q)σ²)` through a binomial expansion whose first-order term vanishes and whose `t = 2` term collapses to `exp(1/σ²) − 1`. Optimizing the composed bound over `λ` gave `σ ≥ c₂ q√(T log(1/δ))/ε`, single-digit `ε` on a real network. The mechanism is then just: clip per example to `C`, sum, add `N(0, σ²C²)`, divide by `L`, step — with `σ` calibrated to the budget up front by inverting the accountant.
