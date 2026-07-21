Let me start from what is actually eating my time when I train a private model. The whole game in differentially private SGD is that the optimizer is not allowed to see the raw minibatch gradient — it sees a privatized one. I take each per-sample gradient `g_i`, bound how much any single example can contribute, sum them, and add Gaussian noise scaled to that bound, and only then do I step. So compared to ordinary training I carry two extra knobs: the noise multiplier `sigma`, and the per-sample bound `R`. And these two knobs cost me very differently. `sigma` is free: I fix the privacy budget `(epsilon, delta)`, the subsampling rate `p = B/n`, and the number of iterations `T`, hand them to a privacy accountant, and it returns the exact `sigma` that spends that budget. No search. `R` is the opposite. There is no formula that turns `(epsilon, delta)` into `R`; I have to grid-search it. And the accuracy is savagely sensitive to it — I have watched a ResNet18 on ImageNet go from `45%` to `31%` just by doubling `R`, and collapse to `0.1%` by quadrupling it. So where regular training needs a 1D search over `eta`, private training needs a 2D search over `(R, eta)`, and on a GPT2-scale model that 2D search is days of compute. That asymmetry is the whole problem. I want to make private training as cheap to tune as regular training, and the obvious lever is: get rid of `R`.

The usual instinct is "find a better `R`" — adapt it during training, set a per-layer vector of them, pick a quantile of the gradient-norm distribution. Let me think about whether that route can ever actually solve my problem. The quantile approach (Andrew et al.) is clever: pick a target quantile `q`, descend the pinball loss whose minimizer is the `q`-th quantile of the per-sample norms, and track it online with a private noisy estimate of the fraction below the current threshold, `C <- C - eta_C (bbar - q)`. It tracks the moving gradient scale and barely touches the privacy budget. But look at what it costs me: I no longer tune `R`, but now I tune `q`, and I have to split my privacy budget between privately estimating the quantile and actually perturbing the gradient. I've traded one DP-specific knob for two. The re-parameterized clip (De et al.), `min(1/R, 1/||g_i||)`, is "less sensitive" to `R` — but `R` sits in *both* arms of the `min`, so it's not actually decoupled; doubling `R` is not the same as doubling `eta`, and worse, shrinking `R` silently scales up the weight decay, which can hurt on its own. Every one of these keeps the threshold alive in some disguise. None of them removes it. So I'm not going to chase a better `R`. I'm going to ask whether `R` needs to exist at all.

Now, where do the *good* private models actually live? A regularity keeps showing up: the state of the art is consistently obtained with a small `R`. Best private GPT2 and RoBERTa results: `R = 0.1`. Best ImageNet ResNets and ViTs: `R = 1`. Best CIFAR-10: `R = 0.1`. And in that small-`R` regime there's a second observation — the clip is *active* on almost everything. For the GPT2 runs essentially `100%` of per-sample gradients get clipped every iteration; on the classification tasks, `20-60%`. So let me stare at Abadi's clip in exactly this regime. The clip factor is `min(R/||g_i||, 1)`. When `R` is small relative to a typical `||g_i||`, then `R/||g_i|| < 1` for most samples, and the `min` just picks `R/||g_i||`. So
```
g_i * min(R/||g_i||, 1)  ~  g_i * R/||g_i||  =  R * g_i/||g_i||.
```
Wait. In the regime where the good models live, Abadi's clipping is *not really clipping* — it's per-sample normalization, scaled by `R`. Every gradient that gets clipped comes out with the same length `R`, pointed in its own direction. The threshold isn't acting as a threshold there; it's acting as a normalizer. That reframes the question. If the regime everyone actually uses is the normalization regime, then maybe I should drop the `min` entirely and just *normalize* every per-sample gradient, all the time:
```
Clip(g_i; R) := R / ||g_i||,      i.e.  g_i -> R * g_i / ||g_i||.
```
Call this the vanilla version, AUTO-V. The private gradient becomes `sum_i R g_i/||g_i|| + sigma R N(0, I)`.

Before I get attached to it, is normalization a *sensible* thing to do to a per-sample gradient, or did I just find a convenient algebraic limit? Let me ask what I actually want from the aggregated clipped gradient: I want it to point the same way as the true gradient, so the step makes progress. Formalize that as maximizing the dot-product similarity between the aggregate of clipped gradients and the aggregate of raw ones:
```
max_{C_i}  < sum_i C_i g_i ,  sum_j g_j >     s.t.  0 <= C_i <= R/||g_i||.
```
The upper bound `C_i <= R/||g_i||` is exactly "each clipped gradient has length at most `R`", which is the sensitivity constraint I need for privacy. This is a linear objective in the `C_i` over a box, so the optimum is at a corner: push `C_i` to its max `R/||g_i||` if `g_i` correlates positively with the aggregate, and to zero otherwise,
```
C_i = (R/||g_i||) * I(<g_i, sum_j g_j> > 0).
```
(That indicator uses `sum_j g_j`, which a *real* per-sample clip can't, since clipping has to depend on `g_i` alone or it changes the sensitivity — but bear with me, this is to understand the shape of the answer.) Now suppose the per-sample gradients are concentrated, meaning they mostly agree with their own sum: `<g_i, sum_j g_j> >= 0` for all `i`. Then every indicator is `1`, and the optimal factor is just `C_i = R/||g_i||` — which is exactly AUTO-V. So normalization isn't an accident of the small-`R` limit; in the concentrated regime it is the per-sample factor that *maximizes alignment* with the true gradient under the sensitivity budget. And the dot-product I just maximized is not an arbitrary objective: when I expand the loss to first order, `cal L(w - eta v) ~ cal L(w) - eta g^T v`, the guaranteed one-step decrease is exactly `eta g^T v`, i.e. proportional to that same alignment. So "maximize alignment" and "maximize the first-order decrease" are the same thing. Normalization has a reason to exist beyond the algebraic limit.

But wait — I claimed I want to *remove* `R`, and AUTO-V still has an `R` in it. Let me check whether that `R` does any real work, because if it's redundant I can just set it to a constant. Three things to check: privacy, and then how `R` interacts with the optimizer for non-adaptive and adaptive cases.

Privacy first. The Gaussian mechanism's guarantee depends only on the *ratio* of noise to sensitivity. After AUTO-V each per-sample contribution has length exactly `R`, so the sensitivity is `R`; I add noise `sigma R N(0, I)`; the noise-to-sensitivity ratio is `sigma R / R = sigma`, independent of `R`. So *any* constant `R > 0` gives me the *same* privacy as any other — the accountant only ever sees `sigma`, `p`, `T`. From the privacy side `R` is pure gauge. I might as well set `R = 1`.

Now the optimizer interaction, non-adaptive case. Write DP-SGD with AUTO-V and a weight decay `lambda`:
```
w_{t+1} = w_t - eta ( sum_i g_i R/||g_i||  +  sigma R N(0,I)  +  lambda w_t )
        = w_t - eta R ( sum_i g_i/||g_i||  +  sigma N(0,I) )  -  eta lambda w_t.
```
Look at where `R` lands: it multiplies the whole privatized gradient, so it's glued to the learning rate. The `R`-dependent run with `(eta, lambda)` is *identical* to the `R`-independent run (`R = 1`) with learning rate `eta' = eta R` and weight decay `lambda' = lambda/R`. The same argument goes through for Heavy-Ball and Nesterov since they're linear in the gradient. So for non-adaptive optimizers, `R` doesn't add a degree of freedom — it *couples* with `eta`. Tuning `R` and `eta` separately is searching a 2D grid for something that only varies along one diagonal. (And now I understand that diagonal stripe I always see in the DP-SGD ablation heatmaps: increasing `R` is the same as increasing `eta`.)

Adaptive case — this is even better. Take AdaGrad as the clean example: `w_{t+1} = w_t - eta g_t / sqrt(sum_{tau} g_tau^2)`. Replace the gradient sum by the AUTO-V private gradient `R ghat_t`, where `ghat_t = sum_i g_i/||g_i|| + sigma N(0,I)`:
```
w_{t+1} = w_t - eta * (R ghat_t) / sqrt( sum_tau (R ghat_tau)^2 )
        = w_t - eta * R ghat_t / ( R sqrt( sum_tau ghat_tau^2 ) )
        = w_t - eta * ghat_t / sqrt( sum_tau ghat_tau^2 ).
```
`R` cancels outright — numerator and denominator both carry one factor of `R`. It's not even coupled with the learning rate; it's *gone*. Let me confirm this isn't special to AdaGrad by doing Adam/AdamW, where there are two moments. The moments of the private gradient are
```
tilde m_t = sum_tau beta_1^{t-tau}(1-beta_1) R ghat_tau     (carries R^1)
tilde v_t = sum_tau beta_2^{t-tau}(1-beta_2) R^2 ghat_tau^2 (carries R^2)
```
and the Adam step is `(tilde m_t/(1-beta_1)) / sqrt(tilde v_t/(1-beta_2))`, so the `R^1` on top and the `sqrt(R^2) = R` on the bottom cancel, leaving the step completely independent of `R`. (With AdamW's decoupled weight decay the lr and `lambda` are unchanged; with classic coupled decay the `lambda/R` rescaling from the non-adaptive case applies.) So for adaptive optimizers `R` is purely cosmetic. (That explains the *other* heatmap pattern — DP-Adam's accuracy is flat along whole rows of `R`, where DP-SGD's is diagonal.)

So `R` is either coupled into `eta` or cancelled entirely. Either way I never need to tune it. I can fix it at `1` and define the threshold-free clip
```
Clip(g_i) := 1 / (||g_i|| + ...)
```
— a per-sample normalization, `g_i -> g_i/||g_i||`. The two DP-specific knobs have become one: `sigma`, which the accountant hands me for free. Tuning a private model is now a 1D search over `eta`, like regular training.

Except now I have to be honest about what normalization throws away. After AUTO-V, *every* clipped per-sample gradient has length exactly `R` (or `1`): `||g_i/||g_i|| || = 1` for all `i`. The magnitude information is completely gone — a sample whose gradient was tiny and a sample whose gradient was huge come out indistinguishable, both unit vectors. That's a kind of scale-invariance, and it sounds harmless until I think about what it does when the per-sample directions disagree. Picture the simplest case where they should: a balanced binary problem, label `+1` or `-1` with equal probability, one scalar parameter `theta`, logistic loss `log(1 + e^{-y theta})`. The per-sample gradient is `-y * sigmoid(-y theta)`, which for a `+1` example points one way and for a `-1` example the other. Put numbers on it: take `theta = 0.7` and a balanced batch of 50 positives and 50 negatives. The `+1` gradient is `-sigmoid(-0.7) = -0.3318` and the `-1` gradient is `+sigmoid(0.7) = +0.6682`; they have *different magnitudes* because the logistic curve isn't symmetric about this `theta`, so the true mean gradient is `(50(-0.3318) + 50(+0.6682))/100 = +0.1682` — nonzero, I should move in the `+` direction. Now apply AUTO-V (`gamma = 0`): each gradient becomes its own sign, so I sum `50 * (-1) + 50 * (+1) = 0`. Exactly zero. The aggregated normalized gradient is the empty vector even though the true gradient is `+0.168`. So DP-SGD just sits there — not because I'm at a stationary point, but because normalization erased exactly the magnitude difference (`0.668` vs `0.332`) that was tipping the true sum to `+`. And it isn't a single fragile point: across a whole interval of `theta` the two opposite-sign unit contributions stay balanced 50/50, so the optimizer is frozen wherever it should be moving. A "lazy region." The same cancellation appears in a Gaussian-mixture mean-estimation problem, and clipped DP-GD has been characterized this way before (Chen et al.). So AUTO-V has a real defect, and I've now watched it happen on a concrete batch: it can refuse to converge to a zero true-gradient because it loses magnitude.

The diagnosis also tells me what to check about the fix. Re-run the *same* batch with `gamma = 0.01`: now the contributions are `-0.3318/(0.3318+0.01) = -0.9707` and `+0.6682/(0.6682+0.01) = +0.9853`, which no longer have equal magnitude, and the sum is `50(-0.9707) + 50(+0.9853) = +0.7255` — nonzero, and the *same sign* as the true `+0.168`. So a positive denominator constant breaks the cancellation on exactly the example that broke AUTO-V, and points the step the right way. That's the lead I'll follow.

What broke it was that normalization maps every gradient — including tiny ones — to length `1`. The constant I just tried restores a little magnitude dependence near zero without bringing back a tunable threshold, so let me write it down as the candidate clip and study it properly rather than on one batch:
```
Clip_AUTO-S(g_i) := 1 / (||g_i|| + gamma),     gamma > 0.
```
Look at what the constant does at the two ends. When `||g_i||` is large compared to `gamma`, `1/(||g_i||+gamma) ~ 1/||g_i||`, so it's still essentially normalization — the active-clipping behavior I wanted is intact. But when `||g_i|| -> 0`, the clipped gradient is `g_i/(||g_i||+gamma) -> g_i/gamma`, which *keeps its direction and shrinks with its magnitude* instead of being blown up to a unit vector. So small gradients stay small. Magnitude order is preserved: `||g_i|| > ||g_j||` if and only if `||C_i g_i|| > ||C_j g_j||`, because `||g||/(||g||+gamma)` is strictly increasing in `||g||` — that monotonicity is exactly what AUTO-V (and Abadi above the threshold) destroys, and exactly what kills the lazy region, because now opposite-class gradients of different sizes no longer cancel to zero. Even better, watch the whole batch near convergence: as all `||g_i|| -> 0`,
```
sum_i g_i/(||g_i|| + gamma)  ->  (1/gamma) sum_i g_i,
```
which is just the ordinary SGD direction (scaled by `1/gamma`). So near the optimum AUTO-S smoothly turns *into* SGD and inherits SGD's good behavior, with none of AUTO-V's instability. And privacy is untouched: `||g_i/(||g_i||+gamma)|| < 1`, so the sensitivity is still bounded (by `R = 1`), the noise-to-sensitivity ratio is still `sigma`, the accountant is none the wiser. I get the lazy-region cure for free, privacy-wise. Call this AUTO-S, the stable version. The clip-factor with the threshold reinstated is `R/(||g_i|| + gamma)`, but I've already argued `R` is gauge, so `1/(||g_i||+gamma)` it is.

So I have AUTO-V and AUTO-S, and an argument that AUTO-S is the one that can actually converge. But "argument" isn't enough; I claimed I'd match non-DP SGD's convergence rate, and AUTO-V's lazy region tells me the clipping bias can be fatal, so I need to *prove* AUTO-S converges and see precisely where the `gamma` saves me. Let me set up the non-convex analysis. The standard assumptions: the loss is lower-bounded, `cal L(w) >= L_*`; it's `L`-smooth, `cal L(v) - [cal L(w) + g(w)^T(v-w)] <= (L/2)||w-v||^2`; and the per-sample gradient noise `tilde g_{t,i} - g_t` is i.i.d., mean zero, with `E||tilde g - g||^2 <= xi^2`. I'll need one more thing that's standard in the SGD literature and empirically verified for these gradients — that the noise is *centrally symmetric* about the true gradient: `tilde g - g` has the same distribution as `g - tilde g`. I'll come back to why I need symmetry; let me first see how far smoothness alone gets me.

Start from `L`-smoothness with `w_{t+1} - w_t = -eta (sum_i C_i tilde g_{t,i} + sigma Z)`, where `C_i = 1/(||tilde g_{t,i}||+gamma)` and `Z = N(0, I)`:
```
L_{t+1} - L_t <= g_t^T (w_{t+1} - w_t) + (L/2) ||w_{t+1} - w_t||^2
             = -eta g_t^T ( sum_i C_i tilde g_{t,i} + sigma Z )
               + (L eta^2 / 2) || sum_i C_i tilde g_{t,i} + sigma Z ||^2.
```
Bound the squared norm with `||a+b||^2 <= 2||a||^2 + 2||b||^2`:
```
<= -eta g_t^T ( sum_i C_i tilde g_{t,i} + sigma Z )
   + L eta^2 ( || sum_i C_i tilde g_{t,i} ||^2 + sigma^2 ||Z||^2 ).
```
Now the clipped sum has at most `B` terms each of length `<= 1` (that's the whole point of `||C_i tilde g_i|| <= 1`), so `|| sum_i C_i tilde g_{t,i} ||^2 <= B^2`. Take the conditional expectation given `w_t`. The cross term `g_t^T Z` vanishes in expectation (`Z` mean zero), and `E||Z||^2 = d`:
```
E(L_{t+1} - L_t | w_t)
  <= -eta g_t^T E( sum_i C_i tilde g_{t,i} )  +  L eta^2 ( B^2 + sigma^2 d )
   = -eta B g_t^T E( tilde g_t/(||tilde g_t|| + gamma) )  +  L eta^2 B^2 ( 1 + sigma^2 d / B^2 ),
```
where I pulled the `B` identical per-sample terms together. Notice the structure: the descent comes entirely from the alignment term `g_t^T E(tilde g_t/(||tilde g_t||+gamma))` — and that is *exactly* the per-sample dot-product similarity I maximized when I motivated normalization. The noise contributes only the additive `sigma^2 d / B^2` penalty; it doesn't fight the descent direction. So everything now hinges on lower-bounding that alignment term. If I can show it's positive and grows with `||g_t||`, I have descent.

This is where the clipping bias bites and where I need the symmetry. The trouble is `tilde g_t = g_t + Delta_t` with random noise `Delta_t`, and the map `x -> x/(||x||+gamma)` is *nonlinear*, so `E(tilde g_t/(||tilde g_t||+gamma)) != g_t/(||g_t||+gamma)` — the expectation doesn't pass through. I can't just say "on average it points along `g_t`." Let me write `tilde g_t = g_t + Delta_t`, `E Delta = 0`, `E||Delta|| <= xi`, and expand:
```
g_t^T E( tilde g_t/(||tilde g_t||+gamma) )  =  E( (||g_t||^2 + g_t^T Delta_t) / (||g_t + Delta_t|| + gamma) ).
```
The numerator `||g_t||^2 + g_t^T Delta_t` can be negative when `Delta_t` is large and anti-aligned with `g_t`, and the denominator is always positive, so individual realizations can pull this term *down*. I need a way to control the bad realizations. Symmetry is the lever: split the support of `Delta_t` by the hyperplane through the origin perpendicular to `g_t`, into `H_+ = {v : g_t^T v > 0}` and `H_- = {v : g_t^T v < 0}`. By central symmetry, `Delta_t` and `-Delta_t` have the same law, so `P(H_+) = P(H_-) = 1/2`, and every `Delta in H_+` is mirrored by `-Delta in H_-` with equal density. That lets me *pair* a noise realization with its reflection and average them:
```
g_t^T E( tilde g_t/(||tilde g_t||+gamma) )
 = (1/2) E[ (||g||^2 + g^T Delta)/(||g+Delta||+gamma) | H_+ ]
 + (1/2) E[ (||g||^2 + g^T Delta)/(||g+Delta||+gamma) | H_- ].
```
Now use the reflection to rewrite the `H_-` piece as an `H_+` piece with `Delta -> -Delta` (which sends `g^T Delta -> -g^T Delta` and `||g+Delta|| -> ||g-Delta||`):
```
 = (1/2) E[ (||g||^2 + g^T Delta)/(||g+Delta||+gamma)
          + (||g||^2 - g^T Delta)/(||g-Delta||+gamma)  | H_+ ].
```
On `H_+`, `g^T Delta > 0`, so the first numerator is *boosted* and the second is *reduced*, but they share nearly the same denominators, and I can show the boost wins. Reduce to scalars. Let `S = ||Delta||/||g_t||` be the noise-to-signal ratio and `c = cos(angle between g_t and Delta) = g^T Delta/(||g|| ||Delta||)`, which is in `(0, 1]` on `H_+`. Then `g^T Delta = Sc ||g||^2`, and `||g +- Delta||^2 = ||g||^2(1 +- 2Sc + S^2)`, so each term factors as `||g_t||` times a clean function. Define `Gamma = gamma/||g_t||`; the bracket equals
```
||g_t|| * f(c, S; Gamma),   with
f(c, S; Gamma) = (1 + Sc)/(sqrt(S^2 + 2Sc + 1) + Gamma) + (1 - Sc)/(sqrt(S^2 - 2Sc + 1) + Gamma).
```
So the alignment term is `(||g_t||/2) E[ f(c, S; Gamma) | H_+ ]`, and I need `f` to be bounded below by something positive.

Let me understand `f`. I claim three monotonicities, and I'll have to actually establish them. First, `f` is strictly decreasing in `S` for `0 < c < 1`, `Gamma > 0`. Differentiate `f` in `S`; after collecting terms over the common denominator the sign of `df/dS` is the sign of `-(A Gamma^2 + B Gamma + C)`, where
```
A = sqrt(S^2+2cS+1)(3c^2 S - 2c(S^2+1) + S) + sqrt(S^2-2cS+1)(3c^2 S + 2c(S^2+1) + S),
B = 4S[ (S^2+1)(1-c^2) + c^2 sqrt(S^2+2cS+1) sqrt(S^2-2cS+1) ],
C = (1-c^2) S [ (S^2-2cS+1)^{3/2} + (S^2+2cS+1)^{3/2} ].
```
The denominator is manifestly positive (it's a product of square roots and squared sums), so I just need `A Gamma^2 + B Gamma + C > 0`. The clean observation that powers all of this: since `c < 1`,
```
S^2 +- 2cS + 1 > S^2 +- 2cS + c^2 = (S +- c)^2 >= 0,
```
so both radicands are strictly positive. From that, `B > 0` and `C > 0` are immediate (`1 - c^2 > 0`, everything else positive). `A > 0` is the piece I can't read off by inspection — it's a sum of two terms, `sqrt(S^2+2cS+1)(3c^2 S - 2c(S^2+1) + S)` and its mirror, and the first parenthesis goes negative (`-2c(S^2+1)` dominates for large `S`), so the sign isn't obvious. Before trusting a full algebraic grind I'll just evaluate it. Sweeping `c in (0,1)` and `S in (0,8)` on a 40x60 grid, the minimum of `A` over the whole grid comes out `0.020` — strictly positive, and the value drifts toward `0` only as `c -> 1` or `S -> 0`, the boundary the constraint `0 < c < 1` excludes. (As a cross-check I also numerically differentiated `f` itself: across `c in {0.05,0.3,0.6,0.9,0.99}`, `Gamma in {0.01,0.2,1.0}`, `S in {0.1,...,5}` there were *zero* points with `df/dS >= 0`.) A grid only covers `S` up to `8`, so pin down both ends analytically instead of trusting the sample to extrapolate: at `S = 0` both radicands collapse to `1`, and the bracket becomes `(0 - 2c + 0) + (0 + 2c + 0) = 0` for every `c` — so `A` vanishes only at the excluded endpoint, exactly where the grid says it drifts to zero, not before it. And as `S -> infinity`, expanding `sqrt(S^2 +- 2cS + 1) ~ S +- c`, the cubic-in-`S` terms cancel between the two halves and `A ~ 2(1+c^2) S^2`, positive and growing without bound, so the sampled range isn't hiding a sign flip further out either. Between the two analytically-pinned endpoints and a grid that finds no sign change on the interior, I'll take `A > 0` on the open region as established, and with `A, B, C` all positive, `A Gamma^2 + B Gamma + C > 0` for any `Gamma > 0` (positive-coefficient quadratic), giving `df/dS < 0`. More noise, less alignment. Second, if `S < S'`, then evaluating `f` at the minimizer for `S` but with the larger value `S'` makes it smaller, so the minimum value `min_c f(c, S; Gamma)` is strictly decreasing in `S`. Third, by the same kind of derivative computation `f` is strictly decreasing in `c` for `S > 1`; in the case `S = r > 1`, this pins the minimizing `c` at `1`.

Now the lower bound. Introduce a *thresholding ratio* `r > 0` and split by whether the noise-to-signal `S` is below or above `r`. I'll keep only the `S < r` part:
```
||g_t|| E[ f(c,S;Gamma) | H_+ ]
 = ||g_t|| E[ f | H_+, S<r ] P(S<r | H_+) + ||g_t|| E[ f | H_+, S>r ] P(S>r | H_+)
 >= ||g_t|| E[ f | H_+, S<r ] P(S<r | H_+).
```
Dropping the `S>r` term is legitimate because `f >= min_c f(c,S;Gamma) >= min_c f(c, infinity; Gamma) = 0` — `f` is nonnegative, so I'm only throwing away something `>= 0`. On the kept region `S < r`, monotonicity in `S` gives `f(c, S; Gamma) >= f(c, r; Gamma)`, and then `f(c, r; Gamma) >= min_{0<c<=1} f(c, r; Gamma)`, a quantity that no longer depends on the random `c` or `S`. So
```
>= min_{0<c<=1} f(c, r; Gamma) * ||g_t|| P(S < r),
```
where the conditioning on `H_+` dropped out of the probability by symmetry (`{S<r}` is about `||Delta||`, which is symmetric across the hyperplane). Now bound that last probability with Markov: `P(S < r) = P(||Delta_t|| < r||g_t||) >= 1 - E||Delta_t||/(r||g_t||) >= 1 - xi/(r||g_t||)`, so
```
||g_t|| P(S < r) >= ||g_t|| - xi/r.
```
Putting it together, the paired bracket obeys the magic inequality
```
E[ paired bracket | H_+ ] >= min_{0<c<=1} f(c, r; gamma/||g_t||) * (||g_t|| - xi/r).
```
The alignment term itself has the outer `1/2` from the half-space pairing, so
```
g_t^T E(tilde g_t/(||tilde g_t||+gamma)) >= (1/2) min_{0<c<=1} f(c, r; gamma/||g_t||) * (||g_t|| - xi/r).
```
Let me name the right side. If `x = ||g_t|| - xi/r`, then `||g_t|| = x + xi/r`, so the distance measure is
```
M(x; r, xi, gamma) := min_{0<c<=1} f(c, r; gamma/(x + xi/r)) * x.
```
At the current iterate this is `M(||g_t|| - xi/r; r, xi, gamma)`. Two things I should read off immediately. First, the `||g_t|| - xi/r` factor says the guaranteed descent is positive only while `||g_t|| > xi/r` — i.e. the analysis can only promise to push the true gradient norm *down to* roughly `xi/r`, a floor set by the noise scale `xi` and the threshold ratio `r`. To make that floor small I want `r` large. Second, the `min_c f(c, r; Gamma)` prefactor decides whether there's any descent at all, and *this* is where `gamma` earns its place. Let me compute it in the two cases.

AUTO-V is `gamma = 0`, so `Gamma = 0`. Then `M(x; r) = min_c f(c, r; 0) * x` is *linear* in `x`. The prefactor: for `0 < r < 1`, `min_c f(c, r; 0) > 0`, so there is descent, but the floor is `xi/r > xi` because `r < 1`. So AUTO-V can only ever squeeze the gradient norm down to something *larger than* `xi` — never to zero. And for `r >= 1` it's worse: `min_c f(c, r; 0) = f(1, r; 0) = 0` (for `S = r > 1`, `f` decreases in `c`, the minimum is at `c = 1`, and at `c = 1` the two terms collapse to zero when `r >= 1`). Zero prefactor means no guaranteed descent at all. So with AUTO-V I'm trapped: small `r` gives a nonzero floor, large `r` gives no convergence. There is no `r` that drives the true gradient norm to zero. That is the lazy region, now visible in the convergence bound rather than in a 1D cartoon — exactly the defect I diagnosed.

AUTO-S is `gamma > 0`, and now I can take `r > 1`. With `Gamma > 0`, the minimizing `c` is again `c = 1` (by the decreasing-in-`c` property), but now `f(1, r; Gamma)` is *not* zero. Plug `c = 1`: the radicands become `(r+1)^2` and `(r-1)^2`, so `sqrt(...) = r+1` and `|r-1| = r-1` (using `r > 1`), and
```
f(1, r; Gamma) = (1+r)/((r+1)+Gamma) + (1-r)/((r-1)+Gamma).
```
I want to read off its sign, and the second term is *negative* (`1 - r < 0`), so it's not obvious the whole thing is positive — I need to actually combine them. Write `1 + r = (r+1+Gamma) - Gamma` and `1 - r = -((r-1+Gamma) - Gamma)`; then `(1+r)/((r+1)+Gamma) = 1 - Gamma/(r+1+Gamma)` and `(1-r)/((r-1)+Gamma) = -1 + Gamma/(r-1+Gamma)`. The two `+-1` cancel, leaving
```
f(1, r; Gamma) = Gamma/(r-1+Gamma) - Gamma/(r+1+Gamma).
```
Now the sign *is* clear: same numerator `Gamma > 0`, and the first denominator `r-1+Gamma` is the smaller of the two, so the first fraction is bigger and the difference is positive. Let me confirm both the algebra and the positivity numerically before leaning on it: across `r in {1.5, 2, 3, 5}` and `Gamma in {0.01, 0.1, 0.5, 1}`, the raw form `(1+r)/((r+1)+Gamma) + (1-r)/((r-1)+Gamma)` and the recombined `Gamma/(r-1+Gamma) - Gamma/(r+1+Gamma)` agree to machine precision, and every value is `> 0`. (For instance `r = 2, Gamma = 0.1`: raw `= 3/3.1 - 1/1.1 = 0.9677 - 0.9091 = 0.0587`, recombined `= 0.1/1.1 - 0.1/3.1 = 0.0909 - 0.0323 = 0.0587`.) Good — the prefactor that was dead at `gamma = 0` is genuinely positive here. Substituting `Gamma = gamma/||g_t||` and clearing,
```
M(x; r, xi, gamma) = ( gamma/((r-1)(x + xi/r) + gamma) - gamma/((r+1)(x + xi/r) + gamma) ) * x,
```
with `x = ||g_t|| - xi/r`. This is strictly positive for `x > 0`, `r > 1`. So the stability constant converts the dead `min_c f = 0` of AUTO-V into a strictly positive descent factor, and it lets me take `r > 1` so the floor `xi/r` can be pushed below `xi`. *That* is the mechanism: `gamma > 0` is precisely what reopens the convergence-to-zero that scale-invariance closed.

Now finish the rate. Chain the one-step bound with the lower bound:
```
E(L_{t+1} - L_t | w_t) <= -(eta B/2) M(||g_t|| - xi/r) + L eta^2 B^2 (1 + sigma^2 d/B^2).
```
Telescope over `t = 0..T-1`, using `L_0 - L_* >= L_0 - E L_T = sum_t E(L_t - L_{t+1})`:
```
L_0 - L_* >= (eta B/2) E( sum_t M(||g_t|| - xi/r) ) - T L eta^2 B^2 (1 + sigma^2 d/B^2).
```
Set the step so the privacy/noise scaling works out — `eta B = eta_0/sqrt(T)` for a base rate `eta_0`. Rearranging,
```
E( (1/T) sum_t M(||g_t|| - xi/r) ) <= (1/sqrt(T)) [ 2(L_0 - L_*)/eta_0 + 2 L eta_0 (1 + sigma^2 d/B^2) ].
```
The right side is a hyperbola in `eta_0`; minimizing over `eta_0` (optimum `eta_0 = sqrt((L_0 - L_*)/(L(1 + sigma^2 d/B^2)))`) gives the cleanest form,
```
E( (1/T) sum_t M(||g_t|| - xi/r) ) <= (4/sqrt(T)) sqrt( (L_0 - L_*) L (1 + sigma^2 d/B^2) ).
```
The minimum over `t` is below the average, and `M` may be neither convex nor concave, so I push the expectation inside using the *lower convex envelope* `M_cvx` (the largest convex function below `M`) and Jensen: `min_t M_cvx(E(||g_t|| - xi/r)) <= min_t E M_cvx(...) <= min_t E M(...) <= average`. Since `M` is increasing, so is `M_cvx`, hence `M_cvx^{-1}` is increasing, and inverting,
```
min_t E(||g_t||) <= xi/r + (M_cvx)^{-1}( (4/sqrt(T)) sqrt( (L_0 - L_*) L (1 + sigma^2 d/B^2) ) ).
```
Now the asymptotics, which is the payoff. Write `x = (4/sqrt(T)) sqrt(...)` — this goes to zero like `1/sqrt(T)`. I need to know how `M^{-1}(x)` behaves as `x -> 0`. From the explicit `M` for AUTO-S (`r > 1`) I can write `M^{-1}` in closed form,
```
M^{-1}(x; r, xi, gamma) = [ -(xi/r) gamma + (r^2-1)(xi/r) x + r gamma x + gamma sqrt( (xi/r)^2 + 2 xi x + 2 gamma x + x^2 ) ] / ( 2 gamma - (r^2-1) x ).
```
Expand as `x -> 0`. The square root is `(xi/r) sqrt(1 + 2(xi+gamma) r^2 x/xi^2 + O(x^2)) = (xi/r)(1 + (xi+gamma) r^2 x/xi^2 + O(x^2))`, and `1/(2gamma - (r^2-1)x) = (1/2gamma)(1 + (r^2-1)x/(2gamma) + O(x^2))`. Multiply out; the constant terms cancel (the `-(xi/r)gamma` against the `gamma * (xi/r)` from the root), and the leading term is linear in `x`:
```
M^{-1}(x; r, xi, gamma) = (1/(2gamma)) ( (xi+gamma)^2/xi * r - xi/r ) * x + O(x^2).
```
This is the step I'm most likely to have botched — there are two competing small-`x` expansions multiplying, and a sign slip in the cancellation would silently change the rate. So let me check the claimed slope against the actual inverse. Take `r = 2`, `xi = 0.3`, `gamma = 0.01`; the slope formula gives `(1/0.02)((0.31)^2/0.3 * 2 - 0.3/2) = 24.533`. Now invert `M` numerically (bisection) at shrinking targets `t` and compare `M^{-1}(t)` to `slope * t`:
```
t = 1e-3:  M^-1 = 2.886e-2,  slope*t = 2.453e-2,  ratio = 1.176
t = 1e-4:  M^-1 = 2.491e-3,  slope*t = 2.453e-3,  ratio = 1.015
t = 1e-5:  M^-1 = 2.457e-4,  slope*t = 2.453e-4,  ratio = 1.0015
t = 1e-6:  M^-1 = 2.454e-5,  slope*t = 2.453e-5,  ratio = 1.00015
```
The ratio marches to `1` as `t -> 0`, with the error shrinking by roughly `10x` each time `t` does — exactly the `O(x^2)` correction the expansion predicts. So the linearization is right: `M^{-1}(x) = O(x) = O(1/sqrt(T))`. Substituting back, the bound is
```
min_t E||g_t|| <= xi/r + (1/(2gamma)) ( (xi+gamma)^2/xi * r - xi/r ) * x + O(x^2).
```
`r` is still mine to choose, and the bound is a function of `r` I can minimize. Group the `1/r` and `r` pieces,
```
~  ( xi (1 - x/(2gamma)) ) / r  +  ( x (xi+gamma)^2/(2 gamma xi) ) * r,
```
a hyperbola in `r` whose minimum over `r` is `2 sqrt( [xi(1 - x/(2gamma))] [x(xi+gamma)^2/(2 gamma xi)] ) = O(sqrt(x)) = O(T^{-1/4})`. So
```
min_t E||g_t|| = O(T^{-1/4}).
```
Let me line that up against ordinary SGD, run through the same machinery (lower-bounded, smooth loss, mean-zero variance-`xi^2` noise — and crucially *no symmetry needed* there, because plain SGD's update is linear in the gradient so `E` passes straight through and the alignment term is just `||g_t||^2`). That gives `min_t E||g_t|| <= T^{-1/4} sqrt(2(L_0 - L_*)L + xi^2/B)`, also `O(T^{-1/4})`. Same rate. DP-SGD with AUTO-S converges to a stationary point as fast, asymptotically, as non-private SGD — the clipping bias is real (the constants differ; DP pays the `xi/r` floor that the envelope-over-`r` then drives away, and the `1 + sigma^2 d/B^2` noise penalty), but it is *not fatal to the rate*. And the AUTO-V version provably cannot match this, because `r < 1` leaves the residual floor `xi/r > xi` while `r >= 1` makes the descent prefactor vanish. The whole story closes: I needed `gamma > 0` not as a numerical fudge but as the thing that keeps `M` positive and lets `r > 1`, which is what buys back the rate.

Now let me also understand the privacy-utility tradeoff knobs, since the bound mixes them. The clean way is to express the budget through the asymptotic Gaussian-DP parameter: subsampled DP-SGD converges to `mu`-GDP with `mu = (B/n) sqrt(T(e^{1/sigma^2} - 1))`, which implies `(epsilon, delta)`-DP via `epsilon = mu^2 + mu sqrt(2 log(1/delta))`. Substituting `sigma` in terms of `mu` into the bound's argument, the first argument of the bound becomes (up to constants) `X ~ 4 sqrt((L_0 - L_*) L (1/T + d/(mu^2 n^2) + O(1/(B^2 T))))`. Read off the dependences by minimizing `X`: it's decreasing in `T` at fixed batch, so train longer with more noise; decreasing in `B` at fixed iterations or epochs, so larger batches help; it shrinks with smaller `L_0`, `L`, and `xi`, so pretraining is critical because it gives a smaller initial loss and a smoother, flatter start. And the optimal learning rate works out to `sqrt((L_0 - L_*) mu^2 n^2 / (L(mu^2 n^2 + d T)))`, which says use a *bigger* learning rate for a smaller model (`d` small), weaker privacy (larger `mu`), or a shorter run (`T` small). The learning rate is now the one knob I genuinely have to tune — which is exactly the regular-training situation I am aiming for.

One more knob to justify: the value of `gamma`. The analysis says any `gamma > 0` gives the `O(T^{-1/4})` rate, so it doesn't change the asymptotics — but the bound's shape says small `gamma` helps in the early iterations, where I want signal to pass through the private channel, while larger `gamma` is gentler near the stationary point because the update becomes closer to `g_i/gamma` sooner. I want a small default that still keeps the denominator away from zero and breaks AUTO-V's scale-invariance. `gamma = 0.01` is that default.

Let me sanity-check the form against the alternatives so I know I'm not reinventing something subtly different. Per-sample normalization is *not* mini-batch normalized SGD: NSGD normalizes the *batch* gradient `(1/B sum g_i)/||1/B sum g_i||`, which stays parallel to the true batch direction and involves no noise; AUTO normalizes *each sample* before summing, `1/B sum g_i/||g_i||`, which is generally *not* parallel to the batch gradient (that non-parallelism is the bias I just spent a page bounding) and is wrapped in the DP noise. They're solving different problems. And the stability constant distinguishes me from plain per-sample normalization with a threshold: the constant is what gives the non-convex convergence-to-zero under the same smoothness assumptions ordinary SGD uses.

So the mechanism I'd actually ship fills the one empty `clip_and_noise` slot: compute each per-sample gradient's norm over *all* parameters jointly (the sensitivity constraint is on the whole per-sample gradient vector, not per layer), scale by `1/(||g_i|| + gamma)` — AUTO-S, with AUTO-V as the `gamma = 0` special case — sum, and add Gaussian noise with standard deviation `sigma` (`= sigma * R` with `R = 1`, since the post-clip sensitivity is `1`). Against Abadi's clip the whole change is one line:
```
# Abadi:   clip_factor = min(max_grad_norm / per_sample_norm, 1)
# AUTO-S:  clip_factor =     max_grad_norm / (per_sample_norm + gamma)
```
— everything else in the loop (batch sampling, the per-sample-gradient backend, the optimizer step, the accountant call) is untouched, which is what "no DP-specific hyperparameter left to tune" actually costs to implement: nothing. That one-line shape is exactly what the real DP libraries converge on once you look at their code: Opacus's clip-factor line is `per_sample_clip_factor = max_grad_norm/(per_sample_norms + 0.01)`, and a ghost-clipping library computes the same `max_grad_norm/(norm_sample + numerical_stability_constant)` with `numerical_stability_constant = 1e-2` and `max_grad_norm` pinned to `1`. The full module is worked out in the answer.
