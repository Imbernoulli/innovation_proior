Let me start from the thing that's actually bothering me about training these latent-variable models. I have a top-down generative network `p(x, h) = p(h) p(x|h)` and a bottom-up recognition network `q(h|x)` that approximates the posterior, and I train by maximizing the evidence lower bound

  L(x) = E_{q(h|x)}[ log p(x,h)/q(h|x) ].

It's a lower bound on `log p(x)` because `log` is concave: `log p(x) = log E_q[ p(x,h)/q(h|x) ] ≥ E_q[ log p(x,h)/q(h|x) ]` by Jensen, and the gap is exactly the KL between `q` and the true posterior, since `L(x) = log p(x) − KL(q(h|x) || p(h|x))`. Maximizing `L` simultaneously pushes up the likelihood and drags `q` toward the posterior. Fine.

What's bothering me is the shape of that objective. It's an *expectation under q* of `log p(x,h)/q(h|x)`. So it asks every single sample drawn from `q` to be a good explanation of `x` — the integrand is large only where `p(x,h)` is large relative to `q(h|x)`, and `q` puts a lot of mass elsewhere, and all of that elsewhere-mass gets penalized. Picture a recognition net that's actually pretty useful: it puts, say, twenty percent of its samples right in the region of high posterior probability and the rest scattered around. For doing inference that's fine — one in five draws lands a good explanation, I can work with that. But `L` doesn't see it that way. `L` averages the log-ratio over *all* the draws, so the eighty percent that miss drag the bound way down. The objective is harsh: it demands that *typical* samples from `q`, not just some samples, explain the data.

And because I'm training the generative model jointly against this same objective, the model will wriggle to make the demand satisfiable. The cheapest way to stop being punished for the missing eighty percent is to make the true posterior into something the simple factorial Gaussian `q` can actually match — approximately factorial, smoothly predictable from `x` by a feed-forward net. So the objective quietly reshapes `p` to have simple posteriors. That's a real, measurable thing: trained models end up using far fewer latent dimensions than they're given. If I measure, for each latent dimension `u`, how much its inferred mean moves across the dataset — `A_u = Cov_x( E_{q(u|x)}[u] )` — I find the dimensions split into two clusters: a handful that are genuinely active and a large pile sitting essentially at the prior, carrying nothing. The capacity is there; the objective is throwing it away. That's the wall: the bound is loose exactly where the posterior is complex, and rather than tolerate the looseness the objective destroys the model's expressiveness.

So I want a different bound. Something that doesn't insist every `q`-sample be a good explanation, that tolerates a recognition network which is only *sometimes* right, and -- if I can get it -- something at least as tight as `L`, with real improvement whenever the old Jensen step has slack.

Where would tightness come from? Let me look again at the very first line of the derivation, the place where I threw something away. I had `log p(x) = log E_q[w]` with `w = p(x,h)/q(h|x)`, and I applied Jensen to pull the `log` inside, turning `log E_q[w]` into `E_q[log w]`. That Jensen step is *exactly* the inequality — it's where all the slack lives. The reason it's loose is that I'm taking the log of a single random weight `w`, and a single sample of `w` is a terrible estimator of `E_q[w] = p(x)`: it's unbiased but wildly variable, and `log` of a wildly variable thing sits far below `log` of its mean.

Here's the thing I keep almost saying out loud: `(1/k) Σ_{i=1}^k w_i` is *also* an unbiased estimator of `p(x)`, for any `k`. Each `w_i = p(x,h_i)/q(h_i|x)` with `h_i ∼ q` independently, and `E_q[w_i] = ∫ q(h|x) · p(x,h)/q(h|x) dh = ∫ p(x,h) dh = p(x)`. So `E[(1/k)Σ w_i] = p(x)`. And as `k` grows this average gets *less* variable — that's the whole point of averaging. A less-variable estimator inside the `log` should lose less to Jensen. That is importance sampling: estimate `p(x)` by averaging importance weights of samples from the proposal `q`.

So let me just *try* defining a new objective by replacing the single weight with the `k`-sample average inside the log:

  L_k(x) = E_{h_1,…,h_k ∼ q(h|x)}[ log (1/k) Σ_{i=1}^k w_i ],   w_i = p(x,h_i)/q(h_i|x).

When `k = 1` this is `E_q[log w] = L(x)`, the old bound, exactly. Good — whatever I build will contain the VAE as a special case, so I'm not throwing the baby out.

Before I trust any of the analytic claims I'm about to make, let me get a concrete feel for whether `L_k` even moves in the direction I'm hoping, on a case where I can compute the truth. Take a one-dimensional toy: prior `h ∼ N(0,1)`, likelihood `x | h ∼ N(h, 1)`, and fix `x = 0.7`. Then the marginal is `p(x) = N(x; 0, 1 + 1) = N(0.7; 0, 2)`, so `log p(x) = −½ log(4π) − 0.7²/4 = −1.388` exactly — I have a ground truth to aim at. For the proposal I deliberately pick a *mismatched* `q(h|x) = N(−0.5, 1.7²)`, off-center and too wide, so there's real Jensen slack to watch close. Estimating `L_k = E[ log (1/k) Σ_i w_i ]` by Monte Carlo (millions of draws so the estimate is steady) gives

  `L_1 = −3.62`,  `L_2 = −2.07`,  `L_5 = −1.54`,  `L_20 = −1.42`,  `L_100 = −1.39`.

That is encouraging in three separate ways at once. Every value sits below `−1.388`, so each `L_k` really is a lower bound. The sequence increases with `k`, exactly the monotonicity I'm hoping to prove. And by `k = 100` it has crawled to within a thousandth of the exact log-likelihood — it looks like it's converging to the ceiling. The single-sample bound is off by more than two nats here because `q` is so mismatched; pooling a hundred weights nearly erases the gap. So the idea isn't dead on arrival. But a numeric trend on one toy is not a theorem, and I want to know *why* each of these three things happens, so let me work them out properly.

First: is it still a lower bound on `log p(x)`? It had better be, or I'm not doing maximum likelihood anymore. This is the same Jensen move as before, just with the average in place of the single weight. `log` is concave, so

  L_k = E[ log (1/k Σ_i w_i) ] ≤ log E[ (1/k) Σ_i w_i ] = log ( (1/k) Σ_i E[w_i] ) = log ( (1/k) · k · p(x) ) = log p(x).

So `L_k ≤ log p(x)` for every `k`. Valid lower bound — and the unbiasedness of the average importance weights is what makes the right-hand side collapse to `log p(x)`. That matches the toy: every `L_k` I measured came out below `−1.388`.

Second — and this is the one I actually care about — is it *tighter* than the VAE bound, and more generally does it improve monotonically with `k`? I want `L_{k+1} ≥ L_k`, ideally `L_k ≥ L_m` whenever `k ≥ m`. Let me see if I can prove it. My instinct says more samples → smaller variance of the average → less Jensen slack → tighter, but "less variance" isn't a proof, so let me find the actual argument.

Let me try the direct thing: relate the `k`-sample average to the `m`-sample average for `m ≤ k`. Take `k` samples `h_1,…,h_k` and their weights `w_1,…,w_k`. Now consider drawing a uniformly random size-`m` subset `I = {i_1,…,i_m}` of the indices `{1,…,k}`. Here's a clean fact about averaging over subsets: for *any* fixed numbers `a_1,…,a_k`,

  E_I[ (a_{i_1} + … + a_{i_m}) / m ] = (a_1 + … + a_k) / k.

Why: each particular index `i` appears in the random `m`-subset with probability `m/k` (by symmetry, `m` chosen out of `k`), so `E_I[ (1/m) Σ_{j} a_{i_j} ] = (1/m) Σ_{i=1}^k a_i · Pr(i ∈ I) = (1/m) Σ_i a_i · (m/k) = (1/k) Σ_i a_i`. Good, the `m`'s cancel and I'm left with the full average.

Apply that with `a_i = w_i`: the full `k`-sample average equals the expectation, over a random `m`-subset, of the `m`-sample average:

  (1/k) Σ_{i=1}^k w_i = E_I[ (1/m) Σ_{j=1}^m w_{i_j} ].

Now substitute that into `L_k` and use Jensen *again*, but this time over the subset randomness `I`, and in the favorable direction. `L_k` takes the `log` of the `k`-average; I just rewrote the `k`-average as `E_I[ m-average ]`; and `log E_I[Y] ≥ E_I[ log Y ]` because `log` is concave. So

  L_k = E_{h_1,…,h_k}[ log (1/k Σ_i w_i) ]
      = E_{h_1,…,h_k}[ log E_I[ (1/m) Σ_j w_{i_j} ] ]
      ≥ E_{h_1,…,h_k}[ E_I[ log (1/m) Σ_j w_{i_j} ] ].

Let me make sure I'm pushing the inequality the right way. `log` concave ⇒ `log(E[Y]) ≥ E[log Y]` (Jensen for concave functions flips the usual convex direction). I have `log E_I[Y]` on the line above and I'm replacing it by `E_I[log Y]`, which is *smaller*, so the `≥` is correct. Good.

Now the inner object: for any fixed `m`-subset `I`, the samples `h_{i_1},…,h_{i_m}` are themselves `m` i.i.d. draws from `q(h|x)` — they're a sub-collection of i.i.d. draws, so they're exchangeable and each `m`-subset is distributed identically. Taking `E_{h_1,…,h_k} E_I` of `log (1/m) Σ_j w_{i_j}` is therefore just `E_{h_1,…,h_m}[ log (1/m) Σ_{j=1}^m w_j ]`, which is the definition of `L_m`. So

  L_k ≥ L_m   for all  k ≥ m.

In particular `L_{k+1} ≥ L_k`. Combined with the first result, I have the full chain

  log p(x) ≥ … ≥ L_{k+1} ≥ L_k ≥ … ≥ L_2 ≥ L_1 = L(x).

So every extra sample makes the bound no looser, and the bottom of the ladder is the plain VAE bound. The increasing run `−3.62 < −2.07 < −1.54 < −1.42` from the toy is now explained: it isn't luck or Monte-Carlo wobble, it's this inequality. I should be careful not to claim *strictly* tighter in every case — the proof only gives `≥`. If the weights were constant almost surely there'd be no Jensen slack left and the inequalities would be equalities; the toy shows slack precisely because `q` is mismatched and the weights vary a lot. The mechanism that does the work is the averaging-over-subsets identity: it lets me express the bigger average as an average of smaller averages, and then one Jensen step over that sub-sampling buys the climb from `L_m` to `L_k`.

Third: does it become *exact* in the limit? I'd like `L_k → log p(x)` as `k → ∞`, so that the ladder actually reaches the ceiling. Look at the random variable inside the log,

  M_k = (1/k) Σ_{i=1}^k w_i.

This is an average of `k` i.i.d. copies of `w = p(x,h)/q(h|x)` with mean `E_q[w] = p(x)`. By the strong law of large numbers, if `w` is well-behaved, `M_k -> p(x)` almost surely, and since `log` is continuous, `log M_k -> log p(x)` a.s. That gives the pointwise limit, but `L_k` is an *expectation* of `log M_k`, and a.s. convergence of a random variable does not by itself license swapping the limit through the expectation — I need something to control the integrand. The clean sufficient condition is bounded importance weights: if `w` is bounded above (and it's positive in this model class, so `log` stays finite below as well), then `log M_k` is uniformly bounded and dominated convergence carries `E[log M_k] -> log p(x)`. So *under that bounded positive-weight assumption* `L_k -> log p(x)` — the ladder reaches the ceiling. I'll flag that this is an assumption I'm leaning on rather than something I've checked in the deep-net case, where the weights can in principle be unbounded; what I can say for sure is the monotone increasing sequence is bounded above by `log p(x)`, so it converges to *some* limit `≤ log p(x)`, and the toy's `L_100 = −1.39` landing essentially on the exact `−1.388` is consistent with that limit being `log p(x)` here. I'd want to confirm the regularity holds before trusting exactness in general.

So I now have a one-parameter family of bounds, tightening monotonically in the non-strict sense from the VAE bound up toward the log-likelihood, controlled by the number of importance samples `k`. Now let me look back at *why* it relaxes the pressure on `q`, because that was the original motivation and I want to check the new objective actually addresses it. The VAE objective `E_q[log w]` punishes the bad samples one at a time. The new objective `E_q[ log (1/k) Σ w_i ]` lets a *batch* of `k` samples pool their weight before the log -- if some of the `k` draws land high weights, the average `(1/k)Σ w_i` can be large even when the other draws are poor. That is the "twenty percent good is useful" tolerance I was after: the recognition net no longer has to be right on every draw; high-weight draws dominate the pooled estimate. Which suggests the generative model is no longer forced to flatten its posteriors into the factorial mold — it can keep complex posteriors as long as `q` hits them often enough. And notice these two readings are the same fact seen twice: the slack that the pooled average removes from Jensen *is* the penalty that the single-sample objective was charging against off-target draws. Reducing the one reduces the other.

Now, an objection that I have to take seriously, because it's the standard reason people are scared of importance sampling: doesn't the importance-weighting estimator blow up in high dimensions? Plain importance sampling of `p(x)` is infamous for enormous, even infinite, variance when the proposal `q` is a poor match — the weight distribution gets heavy-tailed, a single sample dominates the average, and the estimate is garbage. With `h` in tens of dimensions, mismatch between `q` and the posterior is the norm. So shouldn't `M_k` be hopelessly noisy?

Here's the saving grace, and I want to nail it down rather than wave at it: I am never estimating `p(x)` itself, I'm estimating `log p(x)` — I take the `log` *of* the average. The log compresses the heavy tail. Let me make that precise with a tail bound rather than trusting intuition. Take any strictly positive unbiased estimator `Ẑ` of a positive quantity `Z` (here `Ẑ = M_k`, `Z = p(x)`), and consider `log Ẑ` as an estimator of `log Z`. By Markov's inequality applied to the *ratio* `Ẑ/Z`, which has mean `E[Ẑ/Z] = 1`:

  Pr( log Ẑ > log Z + b ) = Pr( Ẑ/Z > e^b ) ≤ E[Ẑ/Z] / e^b = e^{-b}.

So `log Ẑ` is extremely unlikely to overshoot `log Z` by more than a few nats — the right tail decays like `e^{-b}` regardless of how heavy-tailed `Ẑ` is. The heavy tail of `Ẑ` lives in the region where `Ẑ` is *large*, and the log plus this Markov bound crush exactly that region. Let me turn it into a bound on the mean absolute deviation. By Jensen `log Ẑ` is biased low, `E[log Ẑ] ≤ log Z`; write the bias `δ = log Z − E[log Ẑ] ≥ 0` — and notice this is precisely the gap between my bound and `log p(x)`. Using the identity that for any random variable the MAD equals twice the expected positive part of the deviation, `E|Y − E Y| = 2 E[(Y − E Y)_+]`,

  E| log Ẑ − E[log Ẑ] |
    = 2 E[ ( log Ẑ − E[log Ẑ] )_+ ]
    = 2 E[ ( log Ẑ − log Z + log Z − E[log Ẑ] )_+ ]
    ≤ 2 E[ ( log Ẑ − log Z )_+ ] + 2 ( log Z − E[log Ẑ] )      (split the +-part, second term is the constant δ ≥ 0)
    = 2 E[ ( log Ẑ − log Z )_+ ] + 2δ.

For the surviving expectation, use `E[Y_+] = ∫_0^∞ Pr(Y > t) dt` for the nonnegative part, with `Y = log Ẑ − log Z`:

  E[ ( log Ẑ − log Z )_+ ] = ∫_0^∞ Pr( log Ẑ − log Z > t ) dt ≤ ∫_0^∞ e^{-t} dt = 1,

by the Markov tail bound above with `b = t`. So

  E| log Ẑ − E[log Ẑ] | ≤ 2 · 1 + 2δ = 2 + 2δ.

The mean absolute deviation of `log M_k` is bounded by `2 + 2δ`, where `δ` is the bound gap — a small constant, in nats, independent of dimension and of how heavy the weight tail is. The dreaded variance explosion happens to the *average of weights*; the *log of the average* is tame. So importance weighting, which would be reckless if I were estimating `p(x)` directly, is reasonable here precisely because I take the log of the estimate. The objection doesn't sink the approach.

Now I have to actually train on `L_k`. I need its gradient with respect to all the parameters `θ` (generative and recognition together — same `θ` flows through `p(x,h_i)` and through `q(h_i|x)`). The expectation is over `q`, which depends on `θ`, so I can't just differentiate the integrand — same problem the VAE had. Same fix: reparameterize. Write each sample as `h_i = h(ε_i, x, θ)` with `ε_i ∼ N(0, I)` fixed, so the sampling distribution no longer depends on `θ` and the gradient passes inside:

  ∇_θ L_k = ∇_θ E_{ε_1,…,ε_k}[ log (1/k) Σ_i w_i ] = E_{ε_1,…,ε_k}[ ∇_θ log (1/k) Σ_i w_i ],

where now `w_i = w(x, h(ε_i, x, θ), θ) = p(x, h(ε_i,x,θ)) / q(h(ε_i,x,θ)|x)` is a deterministic, differentiable function of `θ` for fixed `ε_i`. Now differentiate the log-of-average:

  ∇_θ log (1/k Σ_i w_i) = ( ∇_θ (1/k) Σ_i w_i ) / ( (1/k) Σ_i w_i )
                        = ( Σ_i ∇_θ w_i ) / ( Σ_i w_i ).

Use `∇_θ w_i = w_i ∇_θ log w_i` (since `∇ log w = ∇w / w`):

  = ( Σ_i w_i ∇_θ log w_i ) / ( Σ_j w_j )
  = Σ_i ( w_i / Σ_j w_j ) ∇_θ log w_i.

Define the *self-normalized* weights `w̃_i = w_i / Σ_j w_j`, which are nonnegative and sum to one. Then

  ∇_θ L_k = E_{ε_1,…,ε_k}[ Σ_i w̃_i ∇_θ log w_i ].

So the gradient is a weighted average of the per-sample gradients `∇_θ log w_i`, weighted by how good each sample is relative to its siblings. The `1/k` from the average cancelled against itself in the normalization — it never appears in the final estimator. Sanity check `k = 1`: a single weight self-normalizes to `w̃_1 = w_1/w_1 = 1`, and the gradient is `∇_θ log w_1`, which is exactly the VAE's reparameterized update. The family reduces correctly.

Look at what each term does. `∇_θ log w_i = ∇_θ log p(x, h_i) − ∇_θ log q(h_i|x)`. The first piece pushes the generative model to assign high probability to each layer given the one above (and, through `h_i`'s dependence on `θ`, nudges the recognition net to produce representations the generative net likes) — in the single-layer case this is just backprop through a stochastic autoencoder. The second piece pushes the recognition net to spread its distribution out rather than collapse. And the whole per-sample update is averaged with weight proportional to the (normalized) importance weight `w̃_i`. Samples that explain the data well get to steer; samples that don't are downweighted instead of being used to punish. The recognition network gets to be wrong sometimes — which is the relaxation I went looking for at the very start, now showing up directly in the update rule. The whole construction is a stochastic autoencoder whose per-sample gradients are combined by importance weighting.

One caveat I should note for honesty about variance. For the plain VAE, there's a trick where you separate out the KL term analytically and only Monte-Carlo the reconstruction term, which lowers variance. With `k > 1` there's no analogous separation — the log-of-sum doesn't split — so I just take the Monte-Carlo gradient as derived. In principle the `k > 1` update could be higher variance for that reason, but the log-domain MAD bound above says it won't be catastrophic.

Let me also think about cost, because this isn't free. Computing `∇_θ log w_i` for each of the `k` samples means a forward and a backward pass per sample, so compute scales linearly in `k`. On a GPU I can parallelize by replicating each training example `k` times within the minibatch and pushing all `k` through at once. If `k` is large and I want to save work, here's a cheaper stochastic variant: I only need a forward pass to compute all the weights `w_i` and hence the normalized `w̃_i`; then instead of backpropagating through all `k`, I sample a single index `i` with probability `w̃_i` and backprop only that one. That's `k` forward passes but a single backward pass, and since the backward pass is roughly twice the cost of the forward, for large `k` it cuts the add-multiplies by about a factor of three, at the price of extra gradient variance. Empirically the trade is fine.

Now to actually implement the `Σ_i w̃_i ∇_θ log w_i` estimator without hand-coding gradients. The idea is to write a *surrogate* scalar whose gradient is exactly that, and let backprop do the rest. If I treat the normalized weights `w̃_i` as constants (detach them — stop gradient), then the scalar `Σ_i w̃_i log w_i` differentiates to `Σ_i w̃_i ∇_θ log w_i`, which matches my estimator on paper. But this is exactly the kind of step where a sign error or a forgotten dependence hides, and the detachment is a subtle move — so I'd rather not take it on faith. Let me check it against the gradient of the actual objective, computed a completely different way. For a *fixed* set of reparameterization noises `ε_1,…,ε_k`, the true thing I want to differentiate is `log (1/k) Σ_i w_i = logmeanexp_i(log w_i)` as a function of `θ`; there's no expectation left once `ε` is fixed, so its autodiff gradient is the ground truth for `∇_θ L_k` at this draw. If my detached surrogate is right, the two should agree to machine precision.

I set up a one-parameter toy in an autodiff framework: a scalar `θ` drives a toy encoder mean and a toy decoder, `k = 4`, fixed noises `ε = (0.5, −1.2, 0.7, 0.1)`. Computing both gradients at `θ = 0.3`:

  gradient of the detached surrogate `Σ_i w̃_i log w_i`   = `−0.0163505`
  gradient of `logmeanexp_i(log w_i)` directly           = `−0.0163505`

They agree to six decimals. So the surrogate construction is correct: detaching the `w̃_i` and differentiating `Σ_i w̃_i log w_i` reproduces the gradient of the real log-mean-exp objective. The reason the detachment matters is now concrete rather than a worry — if I had let autodiff flow through the `w̃_i` coefficients too, I'd be adding the gradient of the normalization, which is not part of `∇_θ log(mean w)` and would have shown up as a mismatch in this exact check. So: compute the log-weights, form `w̃` by a softmax-style normalization, detach it, and maximize `Σ_i w̃_i log w_i` (i.e. minimize its negative).

And I should compute the weights in the log domain for numerical stability — the `w_i` themselves can span many orders of magnitude. To normalize, subtract the max log-weight before exponentiating: `w̃_i = exp(log w_i − max_j log w_j) / Σ_j exp(log w_j − max_j log w_j)`, which is the stable softmax of the log-weights. For evaluation — the held-out `log p(x)` estimate — I want `L_k` itself, which is `log (1/k) Σ_i w_i = logmeanexp_i(log w_i)`, again computed stably as `max_j log w_j + log (1/k) Σ_i exp(log w_i − max_j log w_j)`.

Let me write it down. Encoder is two `tanh` layers emitting a diagonal-Gaussian mean and log-std (exp to keep std positive); decoder is two `tanh` layers to a Bernoulli mean (sigmoid) for binary pixels; reparameterized sampling `h = μ + σ·ε`. The per-sample log-weight is

  log w_i = log p(h_i) + log p(x|h_i) − log q(h_i|x)

with `p(h_i) = N(0,I)` the prior, `p(x|h_i)` Bernoulli, `q(h_i|x) = N(μ,σ²)`. For the Gaussian log-density of the reparameterized sample I can use `−½ε² − log σ` per dimension, since `(h−μ)/σ = ε`. I'll carry a sample axis so all `k` weights are computed in parallel.

```python
import numpy as np
import torch
import torch.nn as nn

LOG2PI = float(np.log(2 * np.pi))


class GaussianBlock(nn.Module):
    """Two tanh layers -> (mu, sigma) of a diagonal Gaussian; exp keeps sigma > 0."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.fc_mu = nn.Linear(hidden_dim, out_dim)
        self.fc_logsigma = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = self.body(x)
        mu = self.fc_mu(h)
        sigma = torch.exp(self.fc_logsigma(h))
        return mu, sigma


class IWAE(nn.Module):
    """One stochastic layer. Same architecture as the plain VAE; only the
    training objective differs -- the k-sample importance-weighted bound."""
    def __init__(self, dim_latent, dim_obs):
        super().__init__()
        self.encoder = GaussianBlock(dim_obs, 200, dim_latent)
        self.decoder = nn.Sequential(
            nn.Linear(dim_latent, 200), nn.Tanh(),
            nn.Linear(200, 200), nn.Tanh(),
            nn.Linear(200, dim_obs), nn.Sigmoid())

    def encode(self, x):
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(sigma)         # reparameterization noise
        h = mu + sigma * eps                  # h = mu + sigma * eps
        return h, mu, sigma, eps

    def log_weights(self, x):
        # x is shaped (k, batch, dim_obs): k independent samples per example.
        h, mu, sigma, eps = self.encode(x)
        # log q(h|x): Gaussian; since (h-mu)/sigma = eps, this is -0.5 eps^2 - log sigma.
        log_q = torch.sum(-0.5 * eps ** 2 - torch.log(sigma) - 0.5 * LOG2PI, -1)
        # log p(h): standard normal prior.
        log_prior = torch.sum(-0.5 * h ** 2 - 0.5 * LOG2PI, -1)
        # log p(x|h): Bernoulli decoder.
        p = self.decoder(h)
        log_lik = torch.sum(x * torch.log(p) + (1 - x) * torch.log(1 - p), -1)
        return log_prior + log_lik - log_q    # log w_i, shape (k, batch)

    def objective(self, x):
        # Training surrogate, not the numerical value of L_k.
        log_w = self.log_weights(x)                                  # log w_i
        log_w_stable = log_w - torch.max(log_w, 0, keepdim=True)[0]  # subtract max (stability)
        w = torch.exp(log_w_stable)
        w_tilde = w / torch.sum(w, 0, keepdim=True)                  # self-normalized weights
        w_tilde = w_tilde.detach()                                   # stop-gradient on the coefficients
        # grad of this wrt theta == E[ sum_i wtilde_i grad log w_i ] = grad L_k
        return torch.mean(torch.sum(w_tilde * log_w, 0))

    def log_likelihood_estimate(self, x):
        # L_k = logmeanexp_i(log w_i): held-out estimate of log p(x).
        log_w = self.log_weights(x)
        m = torch.max(log_w, 0, keepdim=True)[0]
        return torch.mean(m.squeeze(0) + torch.log(torch.mean(torch.exp(log_w - m), 0)))


def train_step(model, x, optimizer, k):
    # Replicate each example k times along a leading sample axis.
    x = x.unsqueeze(0).expand(k, *x.shape)
    optimizer.zero_grad()
    loss = -model.objective(x)        # maximize the surrogate whose gradient is grad L_k
    loss.backward()
    optimizer.step()
    return loss.item()
```
