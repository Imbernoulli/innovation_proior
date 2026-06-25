Let me start from the thing that actually wastes my time. I have a convex, G-Lipschitz function f on R^p, a starting point x₀, and the subgradient method x_{k+1} = x_k − γ_k g_k. The whole behaviour of this thing rides on one number per step: γ_k. And the theory tells me, annoyingly precisely, what the best constant step is: γ = D/(G√n), where D = ‖x₀ − x*‖ is the distance from where I start to a solution, G the Lipschitz constant, n the horizon. Plug that in and I get f(x̂_n) − f* = O(DG/√n), which is worst-case optimal for this class. I can't do better in the worst case.

So in principle there's nothing to tune — the optimal step is a formula. The catch is the formula contains D, and D = ‖x₀ − x*‖ depends on x*, which is exactly what I'm trying to compute. It's circular. I can't know the distance to the answer before I have the answer.

The G part of the formula I'm not worried about, because that one's been handled. If I use the AdaGrad-Norm step γ_k = D/√(Σ_{i≤k}‖g_i‖²), accumulating the squared gradient norms I actually observe, then I never need to know G in advance — the denominator self-calibrates to the gradient scale. There's a clean fact that makes this work: with γ_k = 1/√(Σ_{i<k}‖g_i‖²),

    Σ_k γ_k ‖g_k‖² ≤ 2 √(Σ_k ‖g_k‖²),

so the accumulated gradient term stays O(√(Σ‖g‖²)) ≈ O(G√n) and the rate comes out right. Good. So the gradient denominator is solved. What's left, every single time, is the numerator: D. And in practice the way people get D is they don't — they grid-search a log-spaced range of step scales and train the model a dozen times. That grid search is the actual cost of "tuning the learning rate." If I could set the numerator automatically, the learning rate hyperparameter would just disappear.

Let me think about what tools are on the table for the D problem and where each one stalls.

Polyak's step: γ_k = (f(x_k) − f*)/‖g_k‖². Beautiful, optimal rate, no log factor. But it needs f*, the optimal value. I've traded "don't know D" for "don't know f*" — same disease, different organ. And estimating f* online is unstable; the restart fixes that exist drag a log factor back in. Not it.

Exact line search gives the optimal rate with no constants — but it costs a line search per step, and without smoothness the approximate version reintroduces constant dependence. Too heavy.

Coin-betting / COCOB: assume G but not D, run the online-learning machinery, get regret O(DG√((n+1)log(1+D))). That's the best regret possible without knowing D — but it's a √log worse than knowing D, the asymptotic rates aren't even known, and worst of all the method bakes in its own implicit schedule. I can't hand it a warmup-then-cosine schedule, which is fatal for transformers. So coin-betting is conceptually clean but practically boxed in.

DoG is the most tempting because it's so simple: estimate the distance by how far I've actually moved, r̄_k = max_{i≤k}‖x_i − x₀‖, and step with r̄_k/√(Σ‖g_i‖²). The trouble is r̄_k isn't guaranteed bounded — there's a convex example where it runs off to infinity — so it needs extra dampening, and the rate it can prove has extra log factors. The instinct "use the distance I've travelled as a proxy for the distance to the solution" is right, but ‖x_k − x₀‖ is the wrong proxy: travelling far doesn't mean the solution is far, and it can be unbounded.

And the closest one, Carmon–Hinder: they nail the ideal step as a fixed point. The optimal η satisfies η = φ(η) with φ(η) = ‖x₀−x*‖/√(Σ‖g_i(η)‖²). The right-hand side is computable except for that one unknown distance, and they find the bracket where η − φ(η) flips sign and bisect on log η. That gives optimal-up-to-loglog. So they've reduced the whole problem to solving one implicit 1-D equation for the step. But it's still a search — a bisection wrapping the optimizer — and it's framed through regret, and there's that residual loglog.

Staring at all of these together, the shape of what I want is: I want the numerator of the step to *be* D, set from quantities I observe, in a single loop, with no search and no extra log factor. Every method above is doing something to dodge the fact that D is unknown — searching for it, proxying it, trading it. None of them just computes a usable value of D from the run itself.

So let me ask the blunt question: is there any quantity I actually observe during the run that is provably ≤ D? Because if I had a guaranteed *lower bound* on D, I could put it in the numerator. A lower bound is the safe direction — if my estimate is too small, my step is too small, so I'm slow but I never overshoot and diverge. Underestimating D can only cost me speed, never stability. That's a much friendlier failure mode than DoG's possibly-unbounded estimate. So I want a certified lower bound on D, computed from the run.

Where would such a thing come from? Here's the thing I keep circling back to: convergence proofs are full of *upper* bounds on the suboptimality, and those upper bounds are written in terms of D. An upper bound says "the error is at most [something with D in it]." But an upper bound on a nonnegative quantity, rearranged, is a *lower* bound on whatever's inside it. If I have "0 ≤ error ≤ (stuff with D)", then "(stuff with D) ≥ 0", and if D sits in there linearly I can solve for D and get D ≥ (observed stuff). Invert the convergence bound. Don't use it to certify convergence — use it backwards to certify a value of D.

Let me try to actually do this. I'll work in dual averaging, because it pairs cleanly with the any-time AdaGrad-Norm step and I'll want that. Keep a weighted gradient sum s_{k+1} = s_k + λ_k g_k with some positive weights λ_k I'll choose later, and set x_{k+1} = x₀ − γ_{k+1} s_{k+1}. The classical DA bound, with weights, looks like

    Σ_k λ_k (f(x_k) − f*) ≤ ½ γ_{n+1}^{-1} D² + Σ_k (γ_k/2) λ_k² ‖g_k‖².

D shows up as D². That's a problem for inverting cleanly — solving a quadratic for D is doable but ugly, and the D² term is also loose. Carmon–Hinder had a nicer idea: replace D² by D·‖movement‖ using the triangle inequality. Let me see if I can get a bound that's *linear* in D, because then inverting is trivial.

Let me re-derive the bound from scratch and watch where D enters, so I can keep it linear. Start from convexity:

    Σ_k λ_k (f(x_k) − f*) ≤ Σ_k λ_k ⟨g_k, x_k − x*⟩.

Split x_k − x* = (x_k − x₀) + (x₀ − x*):

    = Σ_k λ_k ⟨g_k, x_k − x₀⟩ + Σ_k λ_k ⟨g_k, x₀ − x*⟩.

The second sum: Σ_k λ_k g_k = s_{n+1}, so it's ⟨s_{n+1}, x₀ − x*⟩. And in dual averaging x_k − x₀ = −γ_k s_k, so the first sum is −Σ_k λ_k γ_k ⟨g_k, s_k⟩. Cauchy–Schwarz on the second:

    Σ_k λ_k (f(x_k) − f*) ≤ ⟨s_{n+1}, x₀ − x*⟩ − Σ_k λ_k γ_k ⟨g_k, s_k⟩
                          ≤ ‖s_{n+1}‖ ‖x₀ − x*‖ − Σ_k λ_k γ_k ⟨g_k, s_k⟩
                          = D ‖s_{n+1}‖ − Σ_k λ_k γ_k ⟨g_k, s_k⟩.

There it is — D appears *linearly*, as D‖s_{n+1}‖, exactly because I used Cauchy–Schwarz on ⟨s_{n+1}, x₀−x*⟩ instead of completing a square into D². That's the difference from the classical bound. Now I just need to understand the inner-product sum −Σ λ_k γ_k ⟨g_k, s_k⟩.

This is the standard DA telescoping. Expand ½γ_{n+1}‖s_{n+1}‖². Split off the step in γ:

    ½ γ_{n+1} ‖s_{n+1}‖² = ½ γ_n ‖s_{n+1}‖² + ½(γ_{n+1} − γ_n)‖s_{n+1}‖².

Now s_{n+1} = s_n + λ_n g_n, so ‖s_{n+1}‖² = ‖s_n‖² + 2λ_n⟨g_n, s_n⟩ + λ_n²‖g_n‖². Substitute into the first piece:

    ½ γ_n ‖s_{n+1}‖² = ½ γ_n ‖s_n‖² + γ_n λ_n ⟨g_n, s_n⟩ + ½ γ_n λ_n² ‖g_n‖².

Rearranging for the inner product at step n:

    −γ_n λ_n ⟨g_n, s_n⟩ = ½ γ_n ‖s_n‖² − ½ γ_{n+1} ‖s_{n+1}‖² + ½ γ_n λ_n² ‖g_n‖² + ½(γ_{n+1} − γ_n)‖s_{n+1}‖².

Sum over k = 0…n; the ½γ‖s‖² pieces telescope (s₀ = 0 so the k=0 left end vanishes):

    −Σ_k γ_k λ_k ⟨g_k, s_k⟩ = −½ γ_{n+1} ‖s_{n+1}‖² + Σ_k ½ γ_k λ_k² ‖g_k‖² + ½ Σ_k (γ_{k+1} − γ_k) ‖s_{k+1}‖².

That last term is a sum of (γ_{k+1} − γ_k)‖s_{k+1}‖². My step γ_k = 1/√(Σ_{i<k}‖g_i‖²) is nonincreasing, so γ_{k+1} − γ_k ≤ 0, so that whole term is ≤ 0 and I can drop it (it only helps the upper bound). Putting it back into the suboptimality bound:

    Σ_k λ_k (f(x_k) − f*) ≤ D ‖s_{n+1}‖ + Σ_k ½ γ_k λ_k² ‖g_k‖² − ½ γ_{n+1} ‖s_{n+1}‖².

This is exactly the bound I wanted — linear in D, and look, I even picked up a *negative* term −½γ_{n+1}‖s_{n+1}‖² for free out of the telescoping. Compared to the classical ½γ^{-1}D² + Σ½γλ²‖g‖² bound, I've done two things: traded D² for D‖s_{n+1}‖ (Carmon–Hinder's idea), and gained the negative ‖s_{n+1}‖² term. Both make it tighter. Let me sanity-check that the trade is legitimate — that D‖s‖ − ½γ‖s‖² really is ≤ ½γ^{-1}D². Complete the square: ½γ^{-1}D² − (D‖s‖ − ½γ‖s‖²) = ½γ^{-1}(D² − 2Dγ‖s‖ + γ²‖s‖²) = ½γ^{-1}(D − γ‖s‖)² ≥ 0. Yes, and since γ_{n+1}‖s_{n+1}‖ = ‖x₀ − x_{n+1}‖, the two bounds coincide exactly when D = ‖x₀ − x_{n+1}‖. Good, it's a genuine tightening.

Now the inversion. The left side, Σ_k λ_k (f(x_k) − f*), is a sum of nonnegative terms — f(x_k) ≥ f* for every k, and λ_k > 0. So the left side is ≥ 0. Therefore the right side is ≥ 0:

    0 ≤ D ‖s_{n+1}‖ + Σ_k ½ γ_k λ_k² ‖g_k‖² − ½ γ_{n+1} ‖s_{n+1}‖².

Solve for D:

    D ‖s_{n+1}‖ ≥ ½ γ_{n+1} ‖s_{n+1}‖² − Σ_k ½ γ_k λ_k² ‖g_k‖²,

    D ≥ d̂_{n+1} := [ γ_{n+1} ‖s_{n+1}‖² − Σ_k γ_k λ_k² ‖g_k‖² ] / ( 2 ‖s_{n+1}‖ ).

Everything on the right is a quantity I compute during the run: the accumulated weighted gradient sum s, its norm, the gradient norms, the step sizes. No x*, no f*, no true D. So this is a value of D's lower bound built entirely from the convergence proof run backwards — and if it's a usable size, I can put d̂ in the numerator of my step.

But wait — let me stress this d̂ before I get excited, because something is going to break. Look at the numerator: γ_{n+1}‖s_{n+1}‖² − Σ_k γ_k λ_k²‖g_k‖². There is no reason that's positive. Early on, when s is small and I haven't moved much, ‖s_{n+1}‖² can be tiny relative to the accumulated Σγλ²‖g‖², and then the numerator goes *negative* and d̂ is a negative number. A negative distance is meaningless. The bound has gone vacuous.

When does it go vacuous? Precisely when the algorithm is already making fast progress — when s isn't growing because the gradients are starting to cancel (I'm near a good region), the certificate has nothing to say and reports garbage. In fact, once d gets large enough that the steps are good, I should *expect* d̂ to droop and even go negative; that's the bound telling me "you don't need a bigger D estimate right now." So the raw d̂ is not directly usable as the step numerator. I hit a wall here: a per-step lower bound that's only sometimes valid.

The fix is the natural one once I phrase it right. A lower bound is a lower bound: if at some past step d̂ said "D ≥ 0.4", that statement stays true forever, even if the current step's d̂ is −2. So I don't throw away the good certificates when the current one degenerates — I keep the best one I've ever seen:

    d_{k+1} = max( d_k, d̂_{k+1} ),   starting from a small positive d₀ > 0.

Now d_k is nondecreasing, it's a valid lower bound at every step (it's the max of valid lower bounds and the harmless seed d₀), and it can never go above D because every d̂ ≤ D. The negative-d̂ problem just evaporates: when d̂ is negative, the max ignores it and keeps the previous d. When d̂ is a fresh, larger valid bound, d climbs. So d ratchets upward from d₀ and parks somewhere below D.

I should not take it on faith that it climbs *usefully* — a lower bound that ratchets but barely moves off d₀ would be worthless. The hope is a feedback loop: a bigger d makes a bigger step, a bigger step grows s and ‖s‖², a bigger ‖s‖² makes the next d̂ bigger. But "hope" is the operative word; I want to watch it on something concrete before trusting it. The toy instance from the setup is built for exactly this — f(x) = |x| with x₀ = 1, so x* = 0 and D = 1 exactly, and the gradient is ±1 whenever x ≠ 0, so ‖g_k‖² = 1 every step. Let me run the core loop by hand (weights λ_k = d_k so the estimate feeds its own step — I'll justify that choice properly in a moment, but it's the obvious one to try first; Option II, no extra factor, γ_{k+1} = 1/√(1 + Σ_{i≤k}‖g_i‖²) seeded with G = ‖g₀‖ = 1) from d₀ = 0.1 and just read off d, s, d̂ step by step:

    k     x        s        d̂(raw)     d
    0   1.00000   0.10000   0.00000    0.10000
    1   0.92929   0.20000   0.03536    0.10000
    2   0.88453   0.30000   0.06206    0.10000
    3   0.85000   0.40000   0.08405    0.10000
    4   0.82111   0.50000   0.10301    0.10301   ← d̂ first overtakes d₀, ratchet engages
    8   0.71284   1.01852   0.17710    0.17710
   16   0.32816   3.13604   0.40168    0.40168
   32   0.03539   6.14181   0.58872    0.60057
   40  −0.05298   6.13777   0.54968    0.60460   ← x has crossed 0; s stalls, raw d̂ DROPS, ratchet holds
   56  −0.05321   7.34157   0.56561    0.61001

Three things I'd asserted are now things I've seen. First, the bootstrap is real and it's fast: d goes 0.10 → 0.17 → 0.40 → 0.59 in 32 steps, each rise feeding the next — not the linear crawl I'd have feared from a "loose" bound. Second, the negative/dropping-d̂ pathology is real and lands exactly where I argued it would: once x overshoots past 0 (k≈40) the gradients start cancelling, s stops growing, and the raw d̂ falls back (0.589 → 0.550 → 0.566) — and the running max simply ignores those degenerate certificates and keeps d ≈ 0.60. (If I instead read off the raw Option I estimate, its numerator γ‖s‖² − Σγd²‖g‖² is even −0.003 at k=0, i.e. flatly negative, before s has grown — the ratchet is doing real work from the very first step.) Third, d settles around 0.61, comfortably below D = 1 and never exceeding it: the bound stays honest. I'll come back to *why* it stops near 0.6 rather than reaching 1 once I have the machinery; for now the mechanism survived a concrete look, which is what I needed before building on it.

Let me lock down the weights. I have free positive weights λ_k; the cleanest choice is λ_k = d_k — use the current distance estimate as the dual-averaging weight. Why this? Two reasons fall out. First, with λ_k = d_k the suboptimality on the left becomes Σ_k d_k (f(x_k) − f*), a d-weighted average of the gaps, so later steps (where d is bigger and the estimate better) count more — which is what I want, since the early steps with tiny d are nearly useless. Second, it makes the step itself scale with the estimate: the dual-averaging update with these weights effectively steps with magnitude proportional to d_k/√(Σ‖g‖²), i.e. the AdaGrad-Norm step with my estimated D in the numerator, which is the whole point.

So d₀: how small can it be? It seeds the scale of the first steps and then gets bootstrapped away. In the asymptotic rate it should vanish — its contribution must decay as k grows — so it shouldn't be a hyperparameter at all. Let me set it tiny by default, something like 10^-6, and I'd expect even 10^-16 to work (just bootstraps from further down). The only floor is numerical: in float16, 10^-16 underflows, so for fp16 I'd keep it in 10^-8…10^-6. But conceptually d₀ is not a knob.

Now I owe myself the proof that this whole contraption actually achieves the optimal O(DG/√n), because a lower bound that's too loose would give a step that's too small and a slow rate. The danger is real: if d_k crawled up to D too slowly the rate would suffer. Let me chain the pieces.

I need an upper bound on ‖s_{n+1}‖, because it sits in the suboptimality bound. From the definition of d̂_{n+1} and the fact d̂_{n+1} ≤ d_{n+1} (it's dominated by the running max),

    ½ γ_{n+1} ‖s_{n+1}‖² − Σ_k ½ γ_k λ_k² ‖g_k‖² = d̂_{n+1} ‖s_{n+1}‖ ≤ d_{n+1} ‖s_{n+1}‖.

I want to peel ‖s_{n+1}‖ out of the quadratic. Young's inequality, 2αβ ≤ α² + β², with α² = 2d_{n+1}²/γ_{n+1} and β² = ½γ_{n+1}‖s_{n+1}‖², gives 2αβ = 2d_{n+1}‖s_{n+1}‖. So

    2 d_{n+1} ‖s_{n+1}‖ ≤ 2d_{n+1}²/γ_{n+1} + ½γ_{n+1}‖s_{n+1}‖²
                       ≤ 2d_{n+1}²/γ_{n+1} + d_{n+1}‖s_{n+1}‖ + Σ_k ½ γ_k λ_k²‖g_k‖²,

where the last step substituted the bound above for ½γ_{n+1}‖s_{n+1}‖². Cancel one d_{n+1}‖s_{n+1}‖ from each side:

    d_{n+1} ‖s_{n+1}‖ ≤ 2d_{n+1}²/γ_{n+1} + Σ_k ½ γ_k λ_k²‖g_k‖²,

so

    ‖s_{n+1}‖ ≤ 2 d_{n+1}/γ_{n+1} + ( Σ_k γ_k λ_k²‖g_k‖² ) / (2 d_{n+1}).

Good. Now plug this back into the suboptimality bound (dropping the negative −½γ_{n+1}‖s_{n+1}‖² term, which only helps):

    Σ_k λ_k(f−f*) ≤ D‖s_{n+1}‖ + Σ_k ½γ_k λ_k²‖g_k‖²
                 ≤ 2D d_{n+1}/γ_{n+1} + D(Σγ_k λ_k²‖g‖²)/(2d_{n+1}) + Σ_k ½γ_k λ_k²‖g_k‖².

Now use λ_k = d_k ≤ d_{n+1} ≤ D everywhere, and the AdaGrad-Norm identities. With λ_k = d_k, the middle and last terms both have λ_k² ≤ d_{n+1}², and the first term has 1/γ_{n+1} = √(Σ‖g‖²). Pushing it through:

    Σ_k d_k(f−f*) ≤ 2D d_{n+1} √(Σ‖g‖²) + ½ D d_{n+1} Σγ_k‖g‖² + ½ D d_{n+1} Σγ_k‖g‖²
                 = 2D d_{n+1} √(Σ_k‖g_k‖²) + D d_{n+1} Σ_k γ_k‖g_k‖².

This is the key intermediate bound. Notice both terms are O(D d_{n+1} √(Σ‖g‖²)) once I use the AdaGrad-Norm fact Σγ_k‖g_k‖² ≤ 2√(Σ‖g‖²). To turn this into a rate on f(x̂) − f*, I divide by Σ_k d_k and use Jensen on the d-weighted average iterate x̂_n = (Σ d_k x_k)/(Σ d_k).

The whole rate now hinges on one thing: how big is Σ_k d_k relative to d_{n+1}? If d barely grew, Σd_k ≈ (n+1)·(tiny) and dividing helps little; if d grew to within a constant factor and held, Σd_k ≈ (n+1)·d_{n+1}/const and I win. So I need: d_k doesn't stay tiny — it gets within a constant factor of its final value over a constant fraction of the run. Since d_k is nondecreasing and bounded above by D, it converges to some limit d_∞ ≤ D; so there's a step n̂ after which d_k ≥ ½ d_{n+1} for all k, n ≥ n̂. Take n ≥ 2n̂. Then

    Σ_{k=0}^n d_k ≥ Σ_{k=n̂}^n d_k ≥ ½(n − n̂ + 1) d_{n+1} ≥ ¼(n+1) d_{n+1},

using n̂ ≤ n/2. So 1/Σd_k ≤ 4/((n+1)d_{n+1}). Plug in:

    f(x̂_n) − f* ≤ (1/Σd_k) Σ d_k(f−f*) ≤ [4/((n+1)d_{n+1})]·[2D d_{n+1}√(Σ‖g‖²) + D d_{n+1} Σγ‖g‖²]
                = 8D√(Σ‖g‖²)/(n+1) + 4D(Σγ‖g‖²)/(n+1).

The d_{n+1} cancels — that's the magic of the d-weighted average, it makes the rate independent of how high d climbed, as long as it climbed. Now Σγ_k‖g_k‖² ≤ 2√(Σ‖g‖²) + (a lower-order G²/‖g₀‖ term from the first few steps before the AdaGrad denominator is established), and √(Σ‖g‖²) ≤ G√(n+1). So

    f(x̂_n) − f* ≤ 16DG/√(n+1) + (lower-order O(DG²/((n+1)‖g₀‖))) = O(DG/√(n+1)).

That's it — the optimal rate, asymptotically, with no log factor, with no knowledge of D, no line search, no extra evaluations. The asymptotic caveat is real and I should be honest about why: for any fixed horizon n known in advance, an adversary can build an f where d crawls and only reaches D at the last step, so a non-asymptotic bound must pay something. But the something is small. Let me find it.

For the non-asymptotic version I want to bound min_k d_{k+1}/(Σ_{i≤k}d_i) — because if I'm allowed to return the iterate at the step t that minimizes that ratio, I control the rate directly. Claim: for a nondecreasing positive sequence with N+1 ≥ 2log₂(d_{N+1}/d₀),

    min_{n≤N} d_{n+1}/(Σ_{k≤n} d_k) ≤ 4 log_{2+}(d_{N+1}/d₀)/(N+1),

where log_{2+}(x) = max(1, log₂ x). The intuition: a nondecreasing sequence bounded by D can only multiply itself by 2 about log₂(D/d₀) times, so over most of the horizon it's roughly flat, and on a flat stretch the ratio d_{n+1}/Σd_k is ≈ 1/(length of stretch) ≈ log₂(D/d₀)/(N+1). Let me actually prove it by induction on r = ⌈log_{2+}(d_{N+1}/d₀)⌉.

Base case r ≤ 2: then d_{N+1} ≤ 4d₀, so every d_k ≥ d₀ gives Σ_{k≤N} d_k ≥ (N+1)d₀, hence d_{N+1}/Σd_k ≤ 4d₀/((N+1)d₀) = 4/(N+1) ≤ 4 log_{2+}/(N+1). Fine.

Inductive step r > 2. Set n' = ⌈N+1 − (N+1)/log_{2+}(d_{N+1}/d₀)⌉, the point a 1/log-fraction in from the end. Two cases.

Case A: d_{n'} ≥ ½ d_{N+1}. Then by monotonicity d_k ≥ ½d_{N+1} for all k ≥ n', so

    Σ_{k≤N} d_k ≥ ½(N+1−n') d_{N+1} ≥ ½( (N+1)/log_{2+} − 1 ) d_{N+1}.

Using N+1 ≥ 2log₂(d_{N+1}/d₀) so that (N+1)/log_{2+} ≥ 2, the −1 costs at most half: Σd_k ≥ (N+1)d_{N+1}/(4 log_{2+}). Rearrange: d_{N+1}/Σ_{k≤N}d_k ≤ 2 log_{2+}/(N+1), and so the min over n ≤ N (which is ≤ this) is ≤ 4 log_{2+}/(N+1). Done, no induction needed.

Case B: d_{n'} ≤ ½ d_{N+1}. Then ⌈log_{2+}(d_{n'}/d₀)⌉ ≤ ⌈log_{2+}(½ d_{N+1}/d₀)⌉ = r − 1, and one checks n' ≥ 2log₂(d_{n'}/d₀), so the inductive hypothesis applies to d₀…d_{n'}:

    min_{n≤n'−1} d_{n+1}/Σ_{k≤n}d_k ≤ 4 log_{2+}(d_{n'}/d₀)/n'.

And n' = ⌈N+1 − (N+1)/log_{2+}(d_{N+1}/d₀)⌉, so log_{2+}(d_{n'}/d₀)/n' ≤ log_{2+}(d_{N+1}/d₀)/(N+1) after substituting log_{2+}(d_{n'}/d₀) ≤ log_{2+}(d_{N+1}/d₀) − 1 (since d_{n'} ≤ ½d_{N+1} and r > 2) and simplifying the n' denominator. Either way the min over n ≤ N is ≤ 4 log_{2+}(d_{N+1}/d₀)/(N+1).

Now use it. With the modified step γ_{k+1} = 1/√(G² + Σ_{i≤k}‖g_i‖²) (the G² under the root just kills a log on the first few steps; it can be dropped at the cost of a lower-order term), the key bound divided by Σd_k and the AdaGrad-Norm fact give

    (1/Σ_{k≤n}d_k) Σ d_k(f−f*) ≤ 4D d_{n+1}/(Σ_{k≤n}d_k) · √(Σ‖g‖²).

Return x̂_t at t = argmin_{k≤n} d_{k+1}/(Σ_{i≤k}d_i). Then d_{t+1}/Σ_{k≤t}d_k = min, which by the lemma is ≤ 4log_{2+}(d_{n+1}/d₀)/(n+1), so

    f(x̂_t) − f* ≤ 16 log_{2+}(d_{n+1}/d₀)/(n+1) · D √(Σ_{k≤t}‖g_k‖²) ≤ 16 DG log_{2+}(D/d₀)/√(n+1).

So the price for not knowing D is exactly a factor log(1 + D/d₀) over the D-known rate. Compare that to the naive alternative — just use a step proportional to d₀ — which would cost a factor D/d₀. Logarithmic versus linear in the ratio; and since d₀ can be 10^-6 or smaller, log(D/d₀) is a small constant. This is why d₀ isn't a hyperparameter: the rate depends on it only through a log, and a tiny d₀ costs essentially nothing.

Let me pause on one design choice I made without justifying it: why dual averaging and not plain gradient descent? The bound machinery actually works for both — I can run x_{k+1} = x_k − λ_k g_k with λ_k = d_k/√(G²+Σ‖g‖²) and the same inversion gives a lower bound on D. So why did I pick DA? Let me derive the GD rate and see. For GD the inner-product decomposition is cleaner, −Σλ_k⟨g_k,s_k⟩ = ½Σλ_k²‖g_k‖² − ½‖s_{n+1}‖², and the s-bound becomes ‖s_{n+1}‖ ≤ 2d_{n+1} + (Σλ²‖g‖²)/(2d_{n+1}). Chaining through, the suboptimality picks up a Σ‖g_k‖²/(G²+Σ‖g_i‖²) ≤ log(n+2) term — that's the standard integral lemma Σ a_k φ(Σa_i) ≤ ∫φ, with φ(x)=1/x giving a log. So GD with D-Adaptation lands at O(DG log(n+2)/√(n+2)) — an extra log(n) factor. That log isn't from my method; it's the generic penalty for any-time step sizes on top of gradient descent over an unbounded domain. Dual averaging dodges it. Practically the two are nearly identical, but if I want the clean no-log statement, DA it is. So the choice is justified: DA buys the missing log factor in theory at no practical cost.

There's a slicker numerator hiding in the inner-product sum, and I want to flag it because it'll matter for the Adam version. From the telescoping identity, Σ_k γ_k λ_k⟨g_k, s_k⟩ ≥ ½γ_{n+1}‖s_{n+1}‖² − Σ_k ½γ_k λ_k²‖g_k‖², which is exactly the numerator of d̂ (Option I) divided by nothing — meaning the alternative estimate

    d̂_{n+1} = ( 2 Σ_k γ_k λ_k ⟨g_k, s_k⟩ ) / ‖s_{n+1}‖      (Option II)

has a numerator that the telescoping identity makes ≥ Option I's — the gap between them is exactly the non-positive term ½Σ(γ_{k+1}−γ_k)‖s_{k+1}‖² I dropped, so Option II is the tighter (larger) lower bound. Back on the |x| trace I can check this against numbers: at k=0 Option I's numerator is −0.003 (negative — s is too small) while Option II's is 0; the gap persists with the AdaGrad-Norm step (k=1: 0.006 vs 0.014; k=3: 0.044 vs 0.067), Option II always on top. The two collapse to exactly equal only when the dropped term vanishes, i.e. when γ is held constant — I checked a flat-λ, constant-γ run and numI = numII to machine precision at every step there, whereas with the decreasing AdaGrad-Norm γ the gap is genuinely nonzero. So Option II dominates whenever the step is annealing, which is the case I care about. The quantity ⟨g_k, s_k⟩ is the inner product of the gradient with the current step direction — the (negative) hyper-gradient. People have used this before, but only as a *sign*: gradient agrees with the step → bump the learning rate up; disagrees → down, with an extra hyper-learning-rate to tune. What I'm doing is different in kind: I'm using ⟨g_k, s_k⟩ to estimate the *magnitude* of the right step, not just its direction. And crucially, because it feeds a lower bound that's protected by the running max, I can freely impose a decreasing schedule on top — anneal the learning rate down — without the hyper-gradient fighting back and pushing it up, and I can build the D estimate up during a warmup. That schedule-compatibility is the thing coin-betting can't give me.

Now make it real for deep learning, where the theory technically stops applying (stochastic, non-convex) but the mechanism should still bootstrap a sensible step. I want an SGD variant and an Adam variant, both as drop-in PyTorch optimizers, both accepting the problem's usual schedule as a multiplier γ_k with base value 1.0.

For SGD: maintain per-parameter s and the dual-averaging companion z = x₀ − s, with momentum done by primal averaging x ← β x + (1−β) z (this is the clean way to fold momentum into DA). On the first step compute g₀_norm = ‖g₀‖ and use G ≈ ‖g₀‖ — a crude estimate of the Lipschitz constant, but it just sets the denominator scale and works well in practice. The per-step "distance-scaled learning rate" is dlr = d·(schedule)/g₀_norm. I accumulate the hyper-gradient numerator before updating s, accumulate ‖s‖², form d̂ = 2·numerator/√(Σ‖s‖²) — that factor of 2 is a deliberate practical multiplier (the theory is invariant to any constant step scaling, so it's still covered; empirically the larger estimate helps) — and ratchet d = max(d, min(d̂, d·growth_rate)), where growth_rate is an optional cap defaulting to ∞ that can be set near 1.02 to stabilize.

For Adam I have to be more careful, because Adam uses exponential moving averages, and my s and the gradient accumulators have to be re-expressed as EMAs at the right scale. The derivation: take the unnormalized weighting λ_k = √(β₂^{-k}) and the updates s_{k+1} = s_k + λ_k g_k, v_{k+1} = v_k + λ_k² g_k², γ_{k+1} = 1/√((1−β₂)v_{k+1}). Define the EMA companions v̂ = β₂ v̂ + (1−β₂)g², and check v̂_{k+1} = β₂^k(1−β₂)v_{k+1}, so γ_{k+1} = √(β₂^k)/√(v̂_{k+1}) — i.e. the un-weighted step x_{k+1} = x_k − g_k/√(v̂_{k+1}) is exactly Adam's. For s the matching EMA is ŝ = √β₂·ŝ + (1−√β₂)g, with ŝ_{k+1} = β₂^{k/2}(1−√β₂)s_{k+1}. Carry the hyper-gradient sum as r̂ = √β₂·r̂ + (1−√β₂)·⟨g, ŝ⟩/√(v̂), and the bookkeeping collapses to

    d̂_{n+1} = r̂_{n+1} / ( (1−√β₂) ‖ŝ_{n+1}‖₁ ).

The two Adam-specific corrections that drop out: the norms are now weighted (EMA) instead of plain sums, and the (1−√β₂) factor keeps numerator and denominator at the same scale. Adam adapts faster than SGD here, so unlike SGD it needs no extra constant-2 multiplier on d̂. For SGD I'll keep the hyper-gradient form (Option II) — it's the cleanest there, and for plain weights it's exactly equal to Option I anyway. For the Adam code I'll instead carry the Option I bookkeeping directly in the EMA scale: track ‖s‖² and Σ d²‖g‖² in the coordinate-wise Adam denominator and form d̂ = (‖s‖²_{A⁻¹}/(1−β₂) − Σ d²‖g‖²_{A⁻¹})/‖s‖₁. That's the same inverted bound, just expressed with the running EMA accumulators I already have to maintain for Adam's step — no need to also carry the hyper-gradient sum r̂. The /(1−β₂) is the analogue of the (1−√β₂) scale fix, now in the squared-norm bookkeeping.

Let me write the SGD optimizer first, mirroring the dual-averaging-with-D-Adaptation structure exactly.

```python
import torch, math

class DAdaptSGD(torch.optim.Optimizer):
    """SGD that sets its own step size by lower-bounding the distance to the solution."""
    def __init__(self, params, lr=1.0, momentum=0.0, weight_decay=0.0,
                 d0=1e-6, growth_rate=float('inf')):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, k=0,
                        numerator_weighted=0.0, d=d0, growth_rate=growth_rate)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        group = self.param_groups[0]
        lr = max(g['lr'] for g in self.param_groups)          # schedule multiplier, base 1.0
        decay, momentum, k = group['weight_decay'], group['momentum'], group['k']
        ck = 1 - momentum                                     # primal-averaging weight
        numerator_weighted = group['numerator_weighted']
        growth_rate, d = group['growth_rate'], group['d']

        # step 0: G ~= ||g0||  (crude Lipschitz estimate, sets the denominator scale)
        if k == 0:
            g_sq = 0.0
            for grp in self.param_groups:
                for p in grp['params']:
                    if p.grad is None: continue
                    g = p.grad.data
                    if decay != 0: g.add_(p.data, alpha=decay)
                    g_sq += (g * g).sum().item()
            group['g0_norm'] = math.sqrt(g_sq)
        g0_norm = group['g0_norm']

        dlr = d * lr / g0_norm                                 # distance-scaled step: d*sched/G

        sk_sq = 0.0
        delta_numerator = 0.0
        for grp in self.param_groups:
            for p in grp['params']:
                if p.grad is None: continue
                g = p.grad.data
                st = self.state[p]
                if 'z' not in st:                              # dual-averaging state
                    st['z']  = torch.clone(p.data).detach()
                    st['s']  = torch.zeros_like(p.data).detach()
                    st['x0'] = torch.clone(p.data).detach()
                if decay != 0: g.add_(p.data, alpha=decay)
                s = st['s']
                # hyper-gradient numerator <g_k, s_k>, accumulated BEFORE updating s
                delta_numerator += dlr * torch.dot(g.flatten(), s.flatten()).item()
                s.add_(g, alpha=dlr)                           # s_{k+1} = s_k + dlr * g_k
                sk_sq += (s * s).sum().item()                  # ||s_{k+1}||^2

        numerator_weighted += delta_numerator                 # running Sum gamma*lambda*<g,s>
        if lr > 0.0 and sk_sq > 0:
            d_hat = 2 * numerator_weighted / math.sqrt(sk_sq) # Option II * factor 2
            d = max(d, min(d_hat, d * growth_rate))            # running MAX (lower bound ratchet)
        if sk_sq == 0:
            return loss

        for grp in self.param_groups:
            grp['numerator_weighted'], grp['d'], grp['g0_norm'] = numerator_weighted, d, g0_norm
            for p in grp['params']:
                if p.grad is None: continue
                st = self.state[p]
                st['z'].copy_(st['x0'] - st['s'])             # z = x0 - s   (DA iterate)
                p.data.mul_(1 - ck).add_(st['z'], alpha=ck)   # x = (1-ck)x + ck z (momentum)
            grp['k'] = k + 1
        return loss
```

Each block ties back: `s` is the weighted gradient sum, `delta_numerator` accumulates the hyper-gradient ⟨g_k, s_k⟩, `d_hat = 2·numerator/√‖s‖²` is the inverted bound (Option II, ×2), `d = max(d, …)` is the lower-bound ratchet that fixes the negative-d̂ wall, and `z = x0 − s` with the primal-averaging blend is the dual-averaging update with momentum. The user passes only `lr` as a schedule multiplier with base 1.0 and a negligible `d0`.

Now Adam, expressed with the EMA companions I derived (this is the form faithful to the algorithm):

```python
class DAdaptAdam(torch.optim.Optimizer):
    """Adam that sets its own step size by lower-bounding the distance to the solution."""
    def __init__(self, params, lr=1.0, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, decouple=False, d0=1e-6, growth_rate=float('inf')):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        d=d0, k=0, gsq_weighted=0.0, decouple=decouple, growth_rate=growth_rate)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        group = self.param_groups[0]
        beta1, beta2 = group['betas']
        d = group['d']
        lr = max(g['lr'] for g in self.param_groups)
        dlr = d * lr                                           # distance-scaled step
        growth_rate, decouple = group['growth_rate'], group['decouple']
        gsq_weighted = group['gsq_weighted']

        g_sq = 0.0; sksq_weighted = 0.0; sk_l1 = 0.0
        for grp in self.param_groups:
            decay, eps = grp['weight_decay'], grp['eps']
            for p in grp['params']:
                if p.grad is None: continue
                g = p.grad.data
                if decay != 0 and not decouple: g.add_(p.data, alpha=decay)
                st = self.state[p]
                if 'step' not in st:
                    st['step'] = 0
                    st['s']           = torch.zeros_like(p.data).detach()
                    st['exp_avg']     = torch.zeros_like(p.data).detach()   # m
                    st['exp_avg_sq']  = torch.zeros_like(p.data).detach()   # v-hat
                m, v = st['exp_avg'], st['exp_avg_sq']
                gg = g * g
                m.mul_(beta1).add_(g, alpha=dlr * (1 - beta1))             # weighted EMA of g
                v.mul_(beta2).add_(gg, alpha=1 - beta2)                    # Adam v-hat
                denom = v.sqrt().add_(eps)                                  # sqrt(v-hat)+eps
                g_sq += gg.div_(denom).sum().item()                        # ||g||^2_{A^-1}
                s = st['s']
                s.mul_(beta2).add_(g, alpha=dlr * (1 - beta2))            # s EMA
                sksq_weighted += (s * s).div(denom).sum().item()          # ||s||^2_{A^-1}
                sk_l1 += s.abs().sum().item()                              # ||s||_1

        gsq_weighted = beta2 * gsq_weighted + g_sq * (dlr ** 2) * (1 - beta2)
        if sk_l1 == 0:
            return loss
        if lr > 0.0:
            # inverted-bound estimate (Option I, weighted/EMA form)
            d_hat = (sksq_weighted / (1 - beta2) - gsq_weighted) / sk_l1
            d = max(d, min(d_hat, d * growth_rate))                        # lower-bound ratchet

        for grp in self.param_groups:
            grp['gsq_weighted'], grp['d'] = gsq_weighted, d
            decay, eps = grp['weight_decay'], grp['eps']
            for p in grp['params']:
                if p.grad is None: continue
                st = self.state[p]; st['step'] += 1
                m, v = st['exp_avg'], st['exp_avg_sq']
                denom = v.sqrt().add_(eps)
                if decay != 0 and decouple:                                # AdamW-style decay
                    p.data.add_(p.data, alpha=-decay * dlr)
                p.data.addcdiv_(m, denom, value=-1)                       # x -= m / (sqrt(v)+eps)
            grp['k'] = group['k'] + 1
        return loss
```

The Adam blocks map to the EMA derivation: `v` is Adam's v̂ so the un-weighted step `x -= m/(√v̂+eps)` is ordinary Adam; `s` carries the EMA weighted gradient sum; `sksq_weighted = ‖s‖²_{A⁻¹}`, `gsq_weighted = Σ dlr²‖g‖²_{A⁻¹}`, and `d_hat = (sksq_weighted/(1−β₂) − gsq_weighted)/‖s‖₁` is the inverted bound in the coordinate-wise weighted norm, with the (1−β₂) scale correction. Same ratchet `d = max(d, …)`. The coordinate-wise (AdaGrad) version is the same skeleton with `alphak = Σg²` as the per-coordinate denominator and `d_hat = (‖s‖²_{A⁻¹} − Σdlr²‖g‖²_{A⁻¹})/‖s‖₁`.

One more guarantee worth nailing down: where does d actually settle? I claimed it parks below D; can I bound how far below? Suppose x_n → x* in norm. By triangle inequality D ≤ ‖x_n − x*‖ + ‖x_n − x₀‖ = ‖x_n − x*‖ + γ_n‖s_n‖, and the first term → 0, so D ≤ lim γ_n‖s_n‖. I need γ_n‖s_n‖ in terms of d_n. Redo the Young step but with a tunable split θ: 2αβ ≤ α² + β² with α² = θ d_{n+1}²/γ_{n+1}, β² = (γ_{n+1}/θ)‖s_{n+1}‖². Then 2d_{n+1}‖s_{n+1}‖ ≤ θd²/γ + (γ/θ)‖s‖², and bounding (γ/θ)‖s‖² ≤ (2/θ)d‖s‖ + (1/θ)Σγλ²‖g‖² and the gradient term ≤ 2d²/(θγ), I get

    2(1 − 1/θ) d_{n+1}‖s_{n+1}‖ ≤ (θ + 2/θ) d_{n+1}²/γ_{n+1},

so γ_{n+1}‖s_{n+1}‖ ≤ [(θ + 2/θ)/(2(1 − 1/θ))] d_{n+1} = [(θ² + 2)/(2(θ − 1))] d_{n+1}. Minimize the bracket over θ: d/dθ of (θ²+2)/(2(θ−1)) = 0 gives θ² − 2θ − 2 = 0, θ* = 1 + √3, and the optimal value is (θ*² + 2)/(2(θ*−1)) = 1 + √3. So γ_n‖s_n‖ ≤ (1+√3) d_n, hence D ≤ (1+√3) lim d_n, i.e.

    lim_n d_n ≥ D / (1 + √3) ≈ 0.366 D.

So d doesn't have to reach D, and it generally won't — but under a mild convergence assumption it stabilizes at no less than about a third of D, which is plenty: the d-weighted-average rate cancels d_{n+1} anyway, so any constant fraction of D gives the optimal rate. And this is exactly what I watched the |x| trace do earlier: from d₀ = 0.1 it parked at d ≈ 0.61 with D = 1, i.e. d/D ≈ 0.61. The floor I just derived is D/(1+√3) ≈ 0.366D, and 0.61 > 0.366 — so the observed settling point sits where the theorem says it must (above the floor), not at it, which is the right relationship for a lower bound on the limit. It would have been a red flag if the run had settled *below* 0.366; it didn't.

Causal chain, start to finish: the optimal convex step needs D = ‖x₀−x*‖ in its numerator, and D is unknowable because it's the distance to the answer. Every convergence proof gives an upper bound on suboptimality that is *linear* in D once I use Cauchy–Schwarz instead of completing the square; since suboptimality is nonnegative, that upper bound rearranges into a *lower* bound on D built entirely from observed quantities — the inverted convergence bound. That raw lower bound goes vacuous (negative) once progress is fast, so I keep the running maximum of all bounds seen, which is monotone, stays ≤ D, and bootstraps from a tiny seed d₀ up toward D exponentially fast. Feeding this estimate as the dual-averaging weight λ_k = d_k makes the step the AdaGrad-Norm step with the estimated distance in the numerator; the d-weighted average iterate makes the final rate independent of how high d climbed, so it lands at the optimal O(DG/√n) with no log factor (DA, not GD, to avoid the any-time log), and only a log(1+D/d₀) penalty non-asymptotically — meaning d₀ is not a hyperparameter. The same inversion, re-expressed with EMAs, yields drop-in SGD and Adam optimizers that need no learning rate, accept any schedule, and add no per-step gradient or function evaluations.
