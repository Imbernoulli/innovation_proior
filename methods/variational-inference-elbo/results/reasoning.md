I keep coming back to the same wall, so let me put it square in front of me. I have a model — say the medical-diagnosis network, six hundred diseases feeding four thousand symptoms through noisy-OR links — and somebody hands me a set of observed symptoms and asks the only question that matters: given these findings, which diseases are probable? That is the posterior, `p(z | x)`, the distribution over the hidden diseases `z` given the observed `x`. And I know exactly how to write it down:

    p(z | x) = p(x, z) / p(x).

The numerator is no trouble at all. `p(x, z)` is just the product of the local factors the graph hands me — each disease's prior, each finding's noisy-OR conditional — and I can evaluate that for any single configuration in linear time. The trouble is the denominator. `p(x) = Σ_z p(x, z)`, summing over every joint configuration of the six hundred binary diseases. That is `2^600` terms. And it is not a problem I can sidestep by being clever about the graph structure, because the structure is the problem: once you observe a finding, all the diseases that could have caused it become coupled — "explaining away" — and when I moralize and triangulate this graph to run exact junction-tree inference, the largest clique is around a hundred and fifty nodes. Exact inference is exponential in that clique size. So `2^150` either way. The evidence is the wall, and the posterior is locked behind it, because I cannot normalize what I cannot sum.

What do I actually have on the shelf? Exact inference is out — that is what just failed. Pruning and bounded-conditioning are tied to the exact algorithm, so they ride the same exponential up a cliff; they postpone the blow-up, they do not remove it. Then there is sampling — build a Markov chain whose stationary distribution is the posterior, run Gibbs, collect samples. That actually works in the limit, and I respect it, but it is stochastic, it can crawl, and worst of all I can never quite tell when it has converged; I am staring at a trace wondering if it has mixed. And it hands me a bag of samples, not a density, and not a number for `p(x)`. I want something deterministic. I want something I can run, watch a single number climb, and stop. And if it could spit out an estimate of `p(x)` along the way, so much the better, because that number is the likelihood and I need it to compare models too.

So let me stop trying to *compute* the posterior and ask a different question. What if I never reach `p(z | x)` exactly, but instead I pick some distribution `q(z)` that I *can* handle — something tractable, something I control — and I try to make `q` as close to the true posterior as I can? Then inference stops being a summation problem and becomes a search problem: search over a family of `q`'s for the one that best matches `p(z | x)`. Optimization, not integration. That reframing feels like the whole game, so let me see if it survives contact with the details.

"Close" needs a meaning. The natural measure of how far one distribution is from another is the Kullback–Leibler divergence,

    KL(q ‖ p(z|x)) = Σ_z q(z) log [ q(z) / p(z|x) ].

It is non-negative, it is zero exactly when `q` equals the posterior, so minimizing it drives `q` toward the truth. Good. So my problem is now

    q* = argmin_q KL(q(z) ‖ p(z|x)).

And immediately I hit the same wall again, because look what is inside that KL. Expand it:

    KL(q ‖ p(z|x)) = E_q[log q(z)] − E_q[log p(z|x)].

That second term has `log p(z|x) = log p(x,z) − log p(x)` buried in it. The `log p(x)` is back. I cannot even *evaluate* my objective, let alone minimize it, because evaluating it needs the evidence I came here to avoid. So the naive reframing does not save me yet. Let me sit with that `log p(x)` term, because the thing that is blocking me is also the only thing standing between me and a clean objective, and maybe its structure is exactly what I should exploit.

But `log p(x)` does not depend on `q` at all — it is a fixed number, whatever it is. So let me carry it along as a constant and write the divergence out fully. Take the KL and substitute `p(z|x) = p(x,z)/p(x)`:

    KL(q ‖ p(z|x)) = E_q[log q(z)] − E_q[log p(z|x)]
                    = E_q[log q(z)] − E_q[log p(x,z)] + E_q[log p(x)]
                    = E_q[log q(z)] − E_q[log p(x,z)] + log p(x),

where the last step is because `log p(x)` is constant under the expectation, so `E_q[log p(x)] = log p(x)`. Now rearrange to isolate the constant:

    log p(x) = E_q[log p(x,z)] − E_q[log q(z)]  +  KL(q ‖ p(z|x)).

Stare at that. The left side is the fixed log-evidence. On the right, the KL term is the thing I want to minimize and cannot evaluate. But the other two terms — `E_q[log p(x,z)] − E_q[log q(z)]` — I *can* evaluate, because they only ever ask me to take expectations of the cheap log-joint and of my own chosen `q` under `q`. Call that piece

    L(q) = E_q[log p(x,z)] − E_q[log q(z)].

So I have an identity:

    log p(x) = L(q) + KL(q ‖ p(z|x)).

This is the hinge of the whole thing. Since KL is always non-negative, `L(q) ≤ log p(x)` for *every* `q`, so `L(q)` is a lower bound on the log-evidence. The name almost writes itself: evidence lower bound. And because `log p(x)` is a constant, the right side `L(q) + KL` is constant too. Pushing `L(q)` up pushes `KL` down by exactly the same amount, so maximizing the bound I can compute is identical to minimizing the divergence I cannot compute. I never have to touch `log p(x)` or `p(z|x)` again. The gap between my bound and the truth is precisely the KL, so the tightness of the bound is the quality of my approximation — when I maximize `L(q)` inside my chosen family, I simultaneously find that family's closest `q` and get its best lower-bound surrogate for the log-evidence. The two things I wanted are the same optimization.

And `L(q)` only ever uses the cheap joint and the entropy of my own distribution. Let me make that explicit by splitting it:

    L(q) = E_q[log p(x,z)] − E_q[log q(z)] = E_q[log p(x,z)] + H(q),

where `H(q) = −E_q[log q]` is the entropy of `q`. So the objective reads "expected log-joint plus entropy of `q`." The first term wants `q` to pile its mass where the joint is large — configurations that explain the data well. The second term, the entropy, fights collapse — it wants `q` spread out, hedged. Maximizing their sum balances fitting the data against not overcommitting. That is a sensible thing to be doing on its own terms, and it never references the intractable normalizer. The thing I was scared of — the evidence — has been quietly converted from an obstacle into a constant I can ignore.

Let me double-check the bound a second way, because if the identity is right I should be able to get the inequality directly without ever invoking `p(z|x)`. Start from the evidence and smuggle in `q`:

    log p(x) = log Σ_z p(x,z) = log Σ_z q(z) · [ p(x,z) / q(z) ] = log E_q[ p(x,z) / q(z) ].

Now `log` is concave, so by Jensen's inequality `log E_q[Y] ≥ E_q[log Y]`:

    log p(x) ≥ E_q[ log ( p(x,z) / q(z) ) ] = E_q[log p(x,z)] − E_q[log q(z)] = L(q).

Same bound, and the slack in Jensen is exactly the KL I computed before — consistent. Jensen even tells me when the bound is tight: Jensen is exact when the thing inside, `p(x,z)/q(z)`, is constant across `z`, i.e. when `q(z) ∝ p(x,z)`, i.e. when `q` *is* the posterior. So if my family were rich enough to contain `p(z|x)`, maximizing `L(q)` would recover the exact evidence. The bound is loose only to the extent my family cannot reach the true posterior. That is a clean statement of what I am giving up and why.

There is a third way to see it that I want to note, because it tells me the bound is not an accident of Jensen but something more rigid. Treat the numbers `{log p(x,z)}_z`, one per configuration, as a vector `u`. Then `log p(x) = log Σ_z exp(u_z)` is a log-sum-exp — a convex function of `u`. Its conjugate is worth deriving, because this is where the entropy enters rather than being guessed. For `f(u) = log Σ_z exp(u_z)`,

    f*(q) = sup_u { Σ_z q(z) u_z − log Σ_z exp(u_z) }.

If some `q(z)` is negative, I can make the supremum unbounded by moving the corresponding coordinate of `u`; if the entries of `q` do not sum to one, shifting every `u_z` by the same constant makes the supremum unbounded. So the finite domain is the probability simplex. Inside that simplex, the stationarity condition is

    q(z) = exp(u_z) / Σ_{z'} exp(u_{z'}),

which means `u_z = log q(z) + c` up to a shared constant, with the usual `0 log 0 = 0` boundary convention by limit. Plugging that back in gives

    f*(q) = Σ_z q(z)(log q(z) + c) − log Σ_z exp(log q(z) + c)
          = Σ_z q(z) log q(z),

the negative entropy. Fenchel duality therefore gives

    log p(x) = max_{q ∈ Δ} { Σ_z q(z) log p(x,z) − Σ_z q(z) log q(z) } = max_{q ∈ Δ} L(q),

and the maximizing `q` is the posterior. The lower bound is the dual representation of the evidence; the variational parameter *is* the distribution `q`; and ranging `q` over a restricted, tractable family instead of all distributions is precisely what trades exactness for tractability. Three roads — the KL identity, Jensen, convex duality — all arrive at the same `L(q)`. I trust it now.

One subtlety I should not gloss over: I chose to minimize `KL(q ‖ p(z|x))` and not the other direction, `KL(p(z|x) ‖ q)`. Is that the right choice or just a habit? Look at what each one asks me to compute. `KL(p ‖ q) = Σ_z p(z|x) log[p(z|x)/q(z)]` is an expectation *under the true posterior* `p(z|x)` — the exact object I cannot evaluate or sample from cheaply. It is computationally dead on arrival. `KL(q ‖ p) = E_q[…]` is an expectation under `q`, the distribution *I designed to be tractable*. That is the whole reason this direction works: the expectations are over something I can handle. So the choice is forced by computability, not aesthetics. I should remember the price, though: this direction is the one whose `q` gets penalized heavily for putting mass where `p` has none (the `log q − log p` blows up when `p ≈ 0`), so the optimal `q` tucks itself inside the support of `p`, hugs a mode, and tends to *underestimate* the posterior variance. I will accept that — a confident, slightly too-narrow deterministic answer beats a sampler I cannot certify has converged — but I will not pretend it captures the full spread.

So the objective is settled: maximize `L(q) = E_q[log p(x,z)] + H(q)` over a family of `q`'s. Now the family. I have been writing `q(z)` as if I knew its shape; I do not. What should it be? Two pressures pull against each other. It has to be rich enough that some member sits close to the true posterior — otherwise the bound is hopelessly loose. And it has to be simple enough that the expectations `E_q[log p(x,z)]` and `H(q)` are actually computable and the optimization is tractable — otherwise I have just moved the intractability from a sum to a search. I need the simplest family that still buys me a flexible approximation.

What is making the true posterior hard is the coupling — the diseases explain each other away, every variable depends on every other through the data. So the boldest possible simplification is to *cut every dependency* and assume the hidden variables are mutually independent under `q`:

    q(z) = ∏_j q_j(z_j).

A fully factorized approximation. This is exactly the mean-field idea out of statistical physics — when you cannot handle a system of interacting spins, you replace each spin's coupling to its neighbors with a coupling to the *average* field they produce, and the system decouples into independent single-site problems whose self-consistent averages solve a set of fixed-point equations. Peterson and Anderson carried this into Boltzmann machines and found the deterministic mean-field computation ran an order of magnitude faster than Gibbs sampling at comparable accuracy. The intuition that makes me believe it can be accurate and not just fast: in a densely connected model, each variable feels the *sum* of many influences, and sums of many terms concentrate — each node becomes relatively insensitive to the precise setting of any one neighbor, sensitive mainly to the average. Where that averaging holds, pretending the variables are independent and tracking only their means costs little. Where it fails — sparse graphs, frustrated couplings that cannot be jointly satisfied — mean field will break by missing the correlations that matter.

And factorizing is exactly the simplification my objective was begging for. With `q(z) = ∏_j q_j(z_j)`, the entropy splits into a sum, `H(q) = Σ_j H(q_j)`, and the expectation `E_q[log p(x,z)]` becomes an integral against a product of one-dimensional factors. Notice that the factors `q_j` can each be *any* shape — I have not constrained the marginals, only forbidden cross-dependence. So the family is genuinely flexible per-coordinate while being trivial to compute against. The one thing it structurally cannot represent is correlation between the latent variables; the price of independence is exactly the lost off-diagonal structure, which is the same variance-underestimation I already signed up for. Consistent. Fine.

Now, how do I actually *maximize* `L(q)` over `q(z) = ∏_j q_j(z_j)`? `L` is not convex in the whole collection of factors at once — there will be local optima, and I will have to live with restarts. But it has a structure that is begging to be exploited: it is a sum and product over the factors, so if I freeze all factors but one, the dependence on that single factor `q_j` should be simple. Optimize one factor at a time, holding the rest fixed, and cycle. Coordinate ascent. Each step can only raise `L`, so the bound climbs monotonically to a local optimum, and there is no step size to tune because each coordinate subproblem I will solve exactly. Let me work out that subproblem, because the whole method lives or dies on whether the per-factor update has a closed form.

Fix every factor except `q_j`. Write `L` as a function of `q_j` alone and throw everything not depending on `q_j` into a constant. Start from `L(q) = E_q[log p(x,z)] − E_q[log q(z)]`. The second term, using the product form, is `Σ_k E_{q_k}[log q_k(z_k)]`; only the `k = j` piece depends on `q_j`, the rest is constant. The first term, by iterated expectation, is `E_{q_j}[ E_{−j}[ log p(x,z) ] ]`, where `E_{−j}` averages over all the *other* factors with `z_j` held fixed. So, up to a constant in `q_j`,

    L(q_j) = E_{q_j}[ E_{−j}[log p(x,z)] ] − E_{q_j}[log q_j(z_j)] + const.

Now I want to recognize this. Define a distribution proportional to the exponential of that inner expectation:

    log r_j(z_j) = E_{−j}[ log p(x,z) ] − log Z_j,   so   r_j(z_j) = (1/Z_j) exp{ E_{−j}[log p(x,z)] }.

Then `E_{q_j}[E_{−j}[log p(x,z)]] = E_{q_j}[log r_j(z_j)] + log Z_j`, and substituting,

    L(q_j) = E_{q_j}[log r_j(z_j)] − E_{q_j}[log q_j(z_j)] + const
           = − Σ_{z_j} q_j(z_j) log [ q_j(z_j) / r_j(z_j) ] + const
           = − KL( q_j ‖ r_j ) + const.

There it is. Maximizing `L` over the single factor `q_j` is *minimizing a KL divergence* between `q_j` and that fixed reference distribution `r_j` — and a KL is minimized, to zero, exactly when the two distributions are equal. So the optimal factor is just `r_j` itself:

    q_j*(z_j) ∝ exp{ E_{−j}[ log p(x,z) ] }.

No gradient, no line search — the coordinate-optimal factor is given in closed form as the exponentiated expected log-joint, the expectation taken over all the other factors. That is the mean-field coordinate-ascent update. And I can simplify what is inside the exponential, because `log p(x,z) = log p(z_j | z_{−j}, x) + log p(z_{−j}, x)`, and the second piece does not depend on `z_j`, so it is absorbed into the normalizer. Therefore, equivalently,

    q_j*(z_j) ∝ exp{ E_{−j}[ log p(z_j | z_{−j}, x) ] }.

The update for one factor is the exponentiated expected log of that variable's complete conditional — its distribution given all the others and the data. I have seen that object before: the complete conditional is exactly what the Gibbs sampler draws from. Gibbs samples `z_j` from `p(z_j | z_{−j}, x)`; mean-field variational inference takes the *expected log* of the same complete conditional and uses it to set `z_j`'s entire variational factor. The stochastic step becomes a deterministic one. I find that satisfying — the two methods are reading off the same local structure, one by sampling it, one by averaging its log.

Cycle through `j = 1, 2, …, m`, replacing each factor by this optimum, recompute, repeat. Each update is an exact coordinate maximization, so `L` never decreases; it climbs to a local optimum. Track `L(q)` itself — the bound — as the convergence monitor, and stop when it stops moving. I have a complete algorithm. But "compute `E_{−j}[log p(z_j | z_{−j}, x)]` and exponentiate" is only useful if that expectation has a closed form. Does it?

The remaining obstacle is practical: the update is only useful when that expected log conditional can be computed. Suppose each complete conditional lies in the exponential family,

    p(z_j | z_{−j}, x) = h(z_j) · exp{ η_j(z_{−j}, x)ᵀ z_j − a(η_j) },

with `z_j` as its own sufficient statistic, `h` the base measure, `a` the log-normalizer, and `η_j` the natural parameter that depends on the conditioning variables. Push this through the update. Take the log, take `E_{−j}`, exponentiate:

    q_j*(z_j) ∝ exp{ E_{−j}[ log h(z_j) + η_j(z_{−j},x)ᵀ z_j − a(η_j) ] }
              ∝ h(z_j) · exp{ E_{−j}[η_j(z_{−j},x)]ᵀ z_j },

because `log h(z_j)` comes straight out as `log h(z_j)`, the `a(η_j)` term does not involve `z_j` so it folds into the normalizer, and the only `z_j`-bearing term left is linear in `z_j` with coefficient `E_{−j}[η_j]`. Look at the result: `q_j*` has the *same* base measure `h` and the *same* exponential-family form as the complete conditional, with natural parameter

    ν_j = E_{−j}[ η_j(z_{−j}, x) ].

So the update is no longer a functional optimization over densities — it is a parameter formula. I do not search for `q_j`; I know it is in a fixed family, and I just set its natural parameter to the expected natural parameter of the complete conditional. For conjugate–exponential models this expectation is itself closed-form, because the variational factors of the *other* variables are in conjugate families and their expected sufficient statistics are exactly what `E_{−j}[η_j]` needs. The whole loop reduces to bookkeeping of expected sufficient statistics. That is what makes mean-field variational inference *practical* and not merely well-defined.

I want to push on the conjugate-Bayesian case specifically, because that is where this will earn its keep. Split the latents into a global `β` (parameters that touch all the data) and locals `z_i` (one per data point), with the joint `p(β, z, x) = p(β) ∏_i p(z_i, x_i | β)`. Take each pairwise likelihood `p(z_i, x_i | β)` in the exponential family with sufficient statistic `t(z_i, x_i)`, and take the prior on `β` to be its conjugate prior. Then the complete conditional of `β` is in the same family as the prior, with natural parameter built by *adding up* the data's sufficient statistics: `α + Σ_i t(z_i, x_i)`. Applying my rule, the global variational update is

    λ = α + Σ_i E_{q(z_i)}[ t(z_i, x_i) ],

the conjugate-prior parameter with each data statistic replaced by its expectation under the current local factors. And each local complete conditional `p(z_i | x_i, β)` is in the exponential family with some natural parameter `η(β, x_i)`, so the local update is

    ϕ_i = E_{q(β)}[ η(β, x_i) ],

the expected natural parameter under the current global factor. Iterate locals-then-global. Both are closed-form, both are just expected natural parameters. Inference has become: initialize the variational parameters, then alternate "given my belief about the parameters, update each data point's local belief" and "given the local beliefs, accumulate their expected statistics into the global belief," watching the bound `L` rise until it levels off.

And now something clicks that I was not looking for. This locals-then-global alternation — set the distribution over the hidden variables given the current parameters, then update the parameters given that distribution — is the *exact shape* of the EM algorithm. EM maximizes `Σ_z p(z|x,θ) log p(x,z|θ)`, the expected complete-data log-likelihood, alternating an E step (compute the posterior over the latents) and an M step (re-estimate the parameters). Let me check that it really is the same object and not just a rhyme. Put parameters `θ` back into my bound:

    L(q, θ) = E_q[log p(x, z | θ)] + H(q),

a lower bound on `log p(x | θ)` for *any* `q`, with gap `KL(q ‖ p(z | x, θ))` — same derivation as before, now carrying `θ`. If I maximize this bound over `q` while `θ` is fixed at its current value, the gap is minimized to zero by `q = p(z | x, θ)`. When that posterior is tractable, the optimal `q` is the exact posterior and the bound becomes tight, `L = log p(x|θ)`. That is the E step. If I then hold that `q` fixed and maximize over `θ`, the entropy `H(q)` is constant in `θ`, so the only term left to optimize is `E_q[log p(x,z|θ)]`; with `q = p(z|x,θ_old)`, this is exactly the expected complete-data log-likelihood. That is the M step.

So EM is coordinate ascent on the very same evidence-lower-bound, in the special case where the variational family is *unrestricted* and rich enough to contain the true posterior, so the E step can set `q = p(z|x,θ)` exactly and the bound touches the log-likelihood at every iteration. The reason EM needs that tractable posterior is now obvious: it is the optimal `q`, and EM uses it. My situation is precisely the one where the posterior is intractable, so I *cannot* take the exact E step. But I do not have to abandon the framework — I just restrict `q` to my factorized family and do the *variational* E step, the mean-field coordinate update, which is still honest coordinate ascent on `L(q, θ)`, only over a smaller set of `q`'s. EM falls out as the limiting, tractable-posterior case of the same machinery. That is the unification: maximum-likelihood estimation and approximate posterior inference are the same optimization of "expected log-joint plus entropy," differing only in how big a family of `q` you are allowed to search.

Let me land this on something concrete I can actually code, the simplest model that exercises every piece — a Bayesian mixture of Gaussians. There are `K` mixture components with means `μ = μ_{1:K}`, each drawn from a Gaussian prior `N(0, σ²)`; each of `n` data points `x_i` gets a latent cluster assignment `c_i` (an indicator over the `K` clusters) and is then `x_i ~ N(c_iᵀ μ, 1)`. The hidden variables are the global means `μ` and the local assignments `c_i`. The mean-field family follows the recipe — one factor per latent variable:

    q(μ, c) = ∏_k q(μ_k; m_k, s_k²) · ∏_i q(c_i; ϕ_i),

a Gaussian factor `N(m_k, s_k²)` on each component mean and a `K`-categorical factor `ϕ_i` on each assignment. Now apply the update `q_j* ∝ exp{ E_{−j}[log p(x,z)] }` to each.

For the assignment `c_i`, the terms of the log-joint that involve `c_i` are its (uniform) log-prior plus the expected log-likelihood `E[log p(x_i | c_i, μ)]`. Since `c_i` is an indicator, `log p(x_i | c_i, μ) = Σ_k c_{ik} log N(x_i; μ_k, 1) = −Σ_k c_{ik} (x_i − μ_k)²/2`, and taking the expectation over the component factors `q(μ_k)`,

    E[log p(x_i | c_i, μ)] = Σ_k c_{ik} ( E[μ_k] · x_i − E[μ_k²]/2 ) + const,

dropping the `x_i²/2` that is constant in `c_i`. Exponentiating the part linear in the indicator, the variational assignment update is

    ϕ_{ik} ∝ exp{ E[μ_k] · x_i − E[μ_k²]/2 },

with `E[μ_k] = m_k` and `E[μ_k²] = m_k² + s_k²` read straight off the Gaussian factor. A soft responsibility, exactly as I'd hope.

For the component mean `μ_k`, collect the `μ_k` terms of the log-joint: its prior `−μ_k²/(2σ²)` plus `Σ_i E[c_{ik}] · log N(x_i; μ_k, 1)`, where `E[c_{ik}] = ϕ_{ik}` is the current responsibility. Expand:

    log q(μ_k) = −μ_k²/(2σ²) − Σ_i ϕ_{ik} (x_i − μ_k)²/2 + const
               = ( Σ_i ϕ_{ik} x_i ) μ_k − ( 1/(2σ²) + (Σ_i ϕ_{ik})/2 ) μ_k² + const.

That is quadratic in `μ_k` — a Gaussian — with sufficient statistics `{μ_k, μ_k²}` and natural parameters `{Σ_i ϕ_{ik} x_i, −1/(2σ²) − (Σ_i ϕ_{ik})/2}`. Converting natural parameters to mean and variance,

    m_k = ( Σ_i ϕ_{ik} x_i ) / ( 1/σ² + Σ_i ϕ_{ik} ),    s_k² = 1 / ( 1/σ² + Σ_i ϕ_{ik} ).

This is exactly a posterior Gaussian over `μ_k` given the data — but a *weighted* one, where each point contributes in proportion to its variational probability `ϕ_{ik}` of belonging to component `k`. It is the conjugate Gaussian–Gaussian complete-conditional update with hard membership replaced by soft, expected membership — precisely the `λ = α + Σ_i E[t(z_i, x_i)]` global rule, instantiated. Alternate the `ϕ` updates (the local / E-step-like sweep) with the `(m, s²)` updates (the global / M-step-like sweep), each guaranteed to raise the bound, monitoring `L` until it converges. Because `L` is non-convex I will run several random initializations and keep the best, since better local optima of the bound correspond to smaller KL to the true posterior.

```python
import numpy as np

# CAVI for a Bayesian mixture of K unit-variance Gaussians with a N(0, sigma^2) prior
# on each component mean. Latents: global means mu_k (Gaussian factor q(mu_k)=N(m_k,s2_k)),
# local assignments c_i (categorical factor q(c_i)=Cat(phi_i)).
# Each step is the closed-form mean-field update  q_j ∝ exp{ E_{-j}[ log p(x,z) ] }, which for
# this conjugate-exponential model sets each factor's natural parameter to the expected natural
# parameter of the corresponding complete conditional. We climb the ELBO to a local optimum.

def elbo(x, m, s2, phi, sigma2):
    # ELBO = E_q[log p(x, z)] - E_q[log q(z)] = E_q[log p(x,z)] + H(q): the bound we climb.
    x = np.asarray(x, dtype=float)
    n, K = phi.shape
    Emu  = m                      # E[mu_k]
    Emu2 = m**2 + s2              # E[mu_k^2]
    # E_q[log p(x | c, mu)] = sum_i sum_k phi_ik ( E[mu_k] x_i - E[mu_k^2]/2 ) - x_i^2/2 - 0.5 log 2pi
    e_log_lik = (np.sum(phi * (np.outer(x, Emu) - 0.5 * Emu2[None, :]))
                 - 0.5 * np.sum(x**2) - 0.5 * n * np.log(2*np.pi))
    e_log_prior_c  = -n * np.log(K)                                   # E_q[log p(c)], uniform over K
    e_log_prior_mu = np.sum(-0.5*np.log(2*np.pi*sigma2) - 0.5*Emu2/sigma2)  # E_q[log p(mu)]
    e_log_joint = e_log_lik + e_log_prior_c + e_log_prior_mu
    # H(q) = H(q(mu)) + H(q(c)):
    H_mu = np.sum(0.5 * np.log(2*np.pi*np.e*s2))
    safe_phi = np.where(phi > 0, phi, 1.0)
    H_c  = -np.sum(np.where(phi > 0, phi * np.log(safe_phi), 0.0))
    return e_log_joint + H_mu + H_c

def cavi_gmm(x, K, sigma2=10.0, max_iters=200, tol=1e-6, seed=0, return_history=False):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    # Initialize the variational parameters (random restart point).
    m  = rng.normal(np.mean(x), np.std(x) + 1e-3, size=K)   # component-mean means
    s2 = np.ones(K)                                          # component-mean variances
    phi = rng.dirichlet(np.ones(K), size=n)                 # soft assignments, rows sum to 1
    history = [elbo(x, m, s2, phi, sigma2)]
    for _ in range(max_iters):
        # --- local update (E-step-like): each assignment factor ---
        # phi_ik ∝ exp{ E[mu_k] x_i - E[mu_k^2]/2 },  with E[mu_k]=m_k, E[mu_k^2]=m_k^2+s2_k.
        Emu, Emu2 = m, m**2 + s2
        log_phi = np.outer(x, Emu) - 0.5 * Emu2[None, :]    # n x K
        log_phi -= log_phi.max(axis=1, keepdims=True)       # log-sum-exp stabilization
        phi = np.exp(log_phi)
        phi /= phi.sum(axis=1, keepdims=True)
        # --- global update (M-step-like): each component-mean factor ---
        # weighted conjugate Gaussian posterior; weights are the responsibilities phi_ik.
        Nk = phi.sum(axis=0)                                 # effective counts
        s2 = 1.0 / (1.0/sigma2 + Nk)                         # variational variance
        m  = (phi.T @ x) * s2                                # variational mean = (sum phi_ik x_i) * s2
        # --- monitor the bound; stop when it stops moving ---
        L = elbo(x, m, s2, phi, sigma2)
        improvement = L - history[-1]
        if improvement < -1e-8:
            raise FloatingPointError("ELBO decreased; check the coordinate updates.")
        history.append(L)
        if improvement < tol:
            break
    if return_history:
        return m, s2, phi, np.array(history)
    return m, s2, phi

def fit_cavi_gmm(x, K, sigma2=10.0, max_iters=200, tol=1e-6, n_init=10, seed=0,
                 return_history=False):
    if n_init < 1:
        raise ValueError("n_init must be at least 1")
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        run_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        m, s2, phi, history = cavi_gmm(
            x, K, sigma2=sigma2, max_iters=max_iters, tol=tol,
            seed=run_seed, return_history=True
        )
        score = history[-1]
        if best is None or score > best[0]:
            best = (score, m.copy(), s2.copy(), phi.copy(), history.copy())
    if return_history:
        return best[1], best[2], best[3], best[4]
    return best[1], best[2], best[3]
```

Stepping back: I refused to compute the intractable evidence `p(x)`, recast posterior inference as searching a family of distributions `q` for the one closest in KL to the true posterior, and discovered that because `log p(x) = L(q) + KL(q ‖ p(z|x))` is constant in `q`, maximizing the computable bound `L(q) = E_q[log p(x,z)] + H(q)` is identical to minimizing the intractable divergence. Choosing `q` fully factorized — mean field — made `L` separable, so coordinate ascent gives a closed-form per-factor update `q_j* ∝ exp{E_{−j}[log p(z_j | z_{−j}, x)]}`, which for conjugate-exponential models is just setting each factor's natural parameter to the expected natural parameter of its complete conditional; the exact-posterior limit of the same bound is the EM algorithm; and the mixture-of-Gaussians instance turns into alternating soft-responsibility and weighted-Gaussian updates that climb the bound to a usable approximate posterior.
