Let me start where it actually hurts. I have a finite sample, `ℓ` pairs `(xᵢ,yᵢ)` drawn i.i.d. from some `P(x,y)` I do not know, and a learning machine that can implement a class of functions `f(x,w)`. What I care about is the expected loss on the future,

    R(w) = ∫ L(y, f(x,w)) dP(x,y),

and what I can actually touch is the average loss on the sample,

    R_emp(w) = (1/ℓ) Σᵢ L(yᵢ, f(xᵢ,w)).

The only induction principle that is obviously implementable is: minimize `R_emp`. Pick the `w` that makes the training error smallest and hope its true risk is small too. This is empirical risk minimization, and it is not some special trick — least squares is exactly this for squared loss, maximum likelihood is exactly this for log loss. So whatever I learn about minimizing the training error, I'm learning about all of those at once.

Why would minimizing `R_emp` give me small `R`? For a single, fixed `w`, the law of large numbers says `R_emp(w) → R(w)`. Fine. But the `w` I return is not fixed in advance. I chose it *after* looking at the sample, precisely to drive `R_emp` down. That is a different object entirely, and I have to be honest that the law of large numbers for a fixed function says nothing about it.

Let me make the danger concrete. Take a class rich enough to fit anything — a big network, or a class that can shatter the sample. I can drive `R_emp` to zero. Training error zero. And yet on a held-out test set I make errors all over the place. This is overfitting, and it is not hypothetical; a sufficiently large machine memorizes its training data and predicts garbage elsewhere. So "minimize the training error" cannot be the whole story, because it has a degenerate optimum that is worthless. The training error is a liar exactly when I most want to trust it.

So what is the gap? The thing I should be afraid of is not `R(w*) − R_emp(w*)` for *my particular* returned `w*`, because I can't isolate that without knowing `P`. The honest quantity is the worst gap over the whole class, because I let the data pick `w` from the whole class:

    sup_{w∈W} |R(w) − R_emp(w)|.

If *that* is small, then in particular the gap for whatever ERM returns is small, and a small training error really does buy me a small risk. If it's large, then somewhere in the class there's a function whose empirical risk badly understates its true risk, and ERM, by construction, goes hunting for exactly that kind of function. This is the same trap as a fixed bound being quoted for a chosen rule. The probability of randomly meeting a particular rare event in a population is tiny — but if you went somewhere *because* the event is common there, the tiny number is a lie. A bound that holds for every fixed rule does not hold for the rule you selected because it looked good. ERM is precisely that biased selection. So I cannot get away with a per-function bound; I need a bound on the supremum.

Good — that tells me the object. Now, is the supremum even controllable? It depends on how rich the class is. If the class can realize every possible labeling of the sample, then for any sample I can find a member with `R_emp = 0` regardless of its true risk, and the supremum is as bad as it gets. So the right notion of "rich" is combinatorial: how many distinct labelings of `ℓ` points can the class produce? The largest number of points the class can label in *all* `2^h` ways — shatter — is its VC dimension `h`. And the deep fact is that the supremum's tail is governed by `h`. For indicator functions,

    Prob{ sup_w |R(w) − R_emp(w)| > ε } < (2eℓ/h)^h exp{−ε²ℓ}.

I want to stare at this, because everything follows from reading it correctly. It is distribution-free: `P` doesn't appear, only `h` (how rich the class is) and `ℓ` (how much data I have). The exponential `exp{−ε²ℓ}` is the law of large numbers pushing the gap down. The polynomial-in-disguise prefactor `(2eℓ/h)^h` is the price of taking a supremum over a class of capacity `h` — it pushes the gap *up*, and the bigger `h`, the more it pushes.

Let me invert this into a usable risk bound. Set the right side equal to `η` and solve for `ε`:

    η = (2eℓ/h)^h exp{−ε²ℓ}
    ⟹ ln η = h·ln(2eℓ/h) − ε²ℓ = h(ln(2ℓ/h) + 1) − ε²ℓ
    ⟹ ε²ℓ = h(ln(2ℓ/h) + 1) − ln η
    ⟹ ε = √( ( h(ln(2ℓ/h) + 1) − ln η ) / ℓ ).

So with probability `1 − η`, simultaneously for *all* `w` — including the `w*` ERM hands me — 

    R(w) ≤ R_emp(w) + C₀(ℓ/h, η),   with   C₀(ℓ/h, η) = √( ( h(ln(2ℓ/h) + 1) − ln η ) / ℓ ).

There it is. The true risk is the training error plus a confidence term that I can actually compute, because it only involves `h`, `ℓ`, `η`. This is the lever. I am no longer at the mercy of the training error; I have a *guaranteed* upper bound on the thing I care about.

Now read the confidence term `C₀` the way I read the tail. It grows with `h` and shrinks with `ℓ`; apart from the fixed confidence-level contribution `−ln η / ℓ`, its main dependence is the capacity-to-sample ratio `h/ℓ` through `h(ln(2ℓ/h)+1)/ℓ`. When `ℓ/h` is large — lots of data relative to capacity — `C₀` is tiny, the bound collapses to `R ≈ R_emp`, and minimizing the training error is genuinely justified. When `ℓ/h` is small, `C₀` is large, and even `R_emp = 0` guarantees nothing. As a rule of thumb the sample is "small" exactly when `ℓ/h` is small, say below twenty. So the whole overfitting story is sitting right here in the bound: rich class relative to data ⟹ big gap ⟹ training error meaningless.

Hold on — `C₀` is clumsy when the training error is near zero. The additive form `R ≤ R_emp + C₀` treats the gap as `√(h ln ℓ / ℓ)` regardless of `R_emp`. But the deviation `|R − R_emp|` is largest when `R(w) ≈ 1/2`, because that's where the variance `√(R(1−R))` of the per-example loss peaks; for small `R` the variance is small and the gap should be smaller too. The clean normalized object would divide by `√(R(1−R))`; I do not have that full bound. In the small-risk regime I care about, `√(R(1−R)) ≈ √R`, and there is a one-sided bound for that approximation:

    Prob{ sup_w (R(w) − R_emp(w)) / √(R(w)) > ε } < (2eℓ/h)^h exp{−ε²ℓ/4},

which is aimed exactly at the low-error case instead of the worst case at `R = 1/2`. Let

    A = h(ln(2ℓ/h)+1) − ln η.

Inverting the tail gives `(R − R_emp)/√R ≤ 2√(A/ℓ)`. If I set `t = √R`, then

    t² − 2√(A/ℓ)t − R_emp ≤ 0,

so the positive root gives `t ≤ √(A/ℓ) + √(A/ℓ + R_emp)`. Squaring that root gives, with probability `1 − η`, for all `w`,

    R(w) ≤ R_emp(w) + C₁(ℓ/h, R_emp(w), η),

    C₁ = 2·(A/ℓ)·( 1 + √( 1 + R_emp(w)·ℓ / A ) ).

Sanity check the two regimes. When `R_emp` is order one, the square root term dominates and `C₁` behaves like `√(R_emp · h ln ℓ / ℓ)` — same order as the additive `C₀`. But when `R_emp = 0`, the square root collapses to `1` and the whole term is just

    C₁(ℓ/h, 0, η) = 4·(h(ln(2ℓ/h)+1) − ln η)/ℓ = 4C₀²,

which is `O(h ln ℓ / ℓ)`, not `O(√(h ln ℓ / ℓ))`. The fast rate. So when I actually manage to fit the data, the penalty is of the order of the square of the worst-case one, much smaller. That's the bound I want to be minimizing in the regime of interest. (And for a general bounded loss `0 ≤ L ≤ B` the same shape holds with `ε = 4(h(ln(2ℓ/h)+1) − ln η)/ℓ` and `R ≤ R_emp + (Bε/2)(1 + √(1 + 4R_emp/(Bε)))`; the additive form is the `R_emp` term plus a confidence interval that interpolates between the fast and slow rates exactly as above.)

Now the real question. I have a bound `R ≤ R_emp + (confidence term in h, ℓ)`. If I fix a single class, I'm stuck on a seesaw I can't ride: make the class small and `R_emp` is large (I underfit, the bound is dominated by a big training error); make it big and the confidence term blows up (I overfit, the bound is dominated by a big capacity penalty). A *fixed* class commits me to one point on that seesaw before I've even seen which one is good. That's the actual flaw in plain ERM — not that the bound is loose, but that with one class I have no freedom to move along the tradeoff the bound is describing.

So don't commit. Let `h` itself be a variable I optimize. But `h` is a property of the *class*, not of a function — so to vary `h` I have to vary the class. Give the whole space of functions a *structure*: a sequence of classes

    S₁ ⊂ S₂ ⊂ ⋯ ⊂ Sₙ ⊂ ⋯,   with VC dimensions   h₁ ≤ h₂ ≤ ⋯ ≤ hₙ ≤ ⋯ .

Nested, deliberately. Why nested and not just any collection? Because nesting forces the two halves of the bound to be monotone in opposite directions, which is what makes the tradeoff well-posed with a single ordered scale. Since `Sₖ ⊆ Sₖ₊₁`, the best training error achievable in `Sₖ₊₁` is no worse than in `Sₖ` — the minimal `R_emp` is non-increasing as I climb the structure. And since `hₖ ≤ hₖ₊₁`, the confidence term is non-decreasing as I climb. One quantity falling, the other rising, monotonically: their sum has a clean minimum. If I allow an arbitrary unordered collection, I lose that built-in opposition between fit and capacity.

For each element `Sₖ` of the structure, I run ERM *inside* `Sₖ` to get its empirical minimizer `f(x, αₖ)` with empirical risk `R_emp(αₖ)`. Each fit comes with its guaranteed risk from the bound:

    R(αₖ) ≤ R_emp(αₖ) + Ω(ℓ/hₖ),

where `Ω` is the confidence term, growing in `hₖ`. Now I have a number — the guaranteed risk — for every level of the structure. Choose the level `k*` (and the function inside it) that makes this *guaranteed* risk smallest. Not the smallest training error — that would just send me to the top of the structure where `R_emp = 0` and `Ω` is enormous. The smallest *bound*. The optimal `k*` sits where the falling `R_emp(αₖ)` and the rising `Ω(ℓ/hₖ)` balance.

This is structural risk minimization. I minimize over two things at once: which class, and which function in it. The principle says: of all the (class, function) pairs, return the one minimizing `R_emp + Ω`. As `k` increases the minimal empirical risk decreases but the confidence interval increases, and SRM weighs both — it picks the complexity that minimizes the *guaranteed* generalization, balancing fit against the capacity penalty the bound charges me.

Notice what just happened to the classical model-selection criteria. AIC penalizes fit by `2l`, MDL by something like `(l/2)log l`, the Bayesian Occam factor by yet another function — and in every case `l` is the *number of parameters*. But my bound's penalty is in `h`, the VC dimension, which is *not* the parameter count. A one-parameter family `sign(sin(αx))` shatters arbitrarily many points — infinite VC dimension on one parameter — so a parameter-counting penalty would happily wave it through and overfit catastrophically. Conversely, a class indexed by a continuous margin can have VC dimension far below its parameter count, so parameter-counting would over-penalize a perfectly safe class. The right complexity measure is capacity, and it is distribution-free. From this vantage the comfortable old principle "prefer fewer parameters" — Occam, read literally as parameter count — is simply not always correct. What I penalize is `h`.

For the admissible-structure conditions, I want exactly what the consistency argument will need: the union `∪ₖ Sₖ` should be dense in the full set of functions (so I don't structurally exclude the good predictor), each `hₖ` should be finite (so each level has a real bound), and each `Sₖ` should contain totally bounded loss functions (so the bound applies). Granting those, what does SRM converge to? If I let the structure grow with the data — choose the level `n(ℓ)` as a function of sample size — the risk of the SRM solution approaches the best possible risk for *any* distribution; it is universally, strongly consistent. The asymptotic rate is of the form

    V(ℓ) = r_{n(ℓ)} + B_{n(ℓ)}·√( h_{n(ℓ)} · ln ℓ / ℓ ),

where `r_n` is the approximation error of `Sₙ` (how well the best function in `Sₙ` does — falling as `n` grows) and the second term is the capacity penalty (rising with `hₙ`). The two pieces are the bias and the estimation cost made explicit: pick `n(ℓ)` growing slowly enough that `B²_{n} · h_{n} · ln ℓ / ℓ → 0`, and both terms vanish, so the rate goes to zero — consistency with a guaranteed speed. The bias/variance balance, but with capacity in place of parameter count and with an actual finite-sample bound attached to each side.

The same selection rule still has to pay for the fact that I am comparing several classes. Suppose I attach to the `k`-th class a weight `w(k) ≥ 0` with `Σₖ w(k) ≤ 1` — encoding a mild a-priori preference, e.g. `w(k) = 1/N` if there are finitely many classes, or `w(k) = 6/(π²k²)` for a countable nested family (a convergent series, gently favoring smaller classes). Spend a failure probability `w(k)·δ` on the bound for class `k`. Then by a union bound, with probability `1 − δ`, simultaneously for every class `k` and every `h ∈ Sₖ`,

    |R(h) − R_emp(h)| ≤ εₖ(ℓ, w(k)δ),

because the total failure probability is `Σₖ w(k)δ ≤ δ`. SRM returns `argmin_k [ R_emp(ĥₖ) + εₖ ]` where `ĥₖ` is the ERM solution in `Sₖ`. Now compare against the best class `k*` I *could* have chosen if an oracle told me which one to use. Writing the excess risk through the chosen `ĥ` against the best function `h*ₖ` in class `k`, and using both that the bound holds and that `k̂` was chosen to minimize the very objective `R_emp + ε`:

    R(ĥ) − R(h*ₖ) = [R(ĥ) − R_emp(ĥ)] + [R_emp(ĥ) − R_emp(h*ₖ)] + [R_emp(h*ₖ) − R(h*ₖ)]
                 ≤ εₖ̂ + [R_emp(ĥ) − R_emp(h*ₖ)] + εₖ.

The middle empirical-risk difference is non-positive once I use that `k̂` minimized `R_emp(ĥₖ) + εₖ`: `R_emp(ĥ) + εₖ̂ ≤ R_emp(ĥₖ) + εₖ ≤ R_emp(h*ₖ) + εₖ`, so `R_emp(ĥ) − R_emp(h*ₖ) ≤ εₖ − εₖ̂`. Substituting, the `εₖ̂` cancels and I get

    R(ĥ) ≤ min_k { min_{h∈Sₖ} R(h) + 2·εₖ(ℓ, w(k)δ) }.

This is an oracle inequality: SRM does as well as the best class plus twice that class's confidence width — as if I had known the right complexity in advance. For VC classes, `εₖ = O(√( hₖ·log(ℓ/hₖ)/ℓ ) + √( log(1/(w(k)δ))/ℓ ))`, and if `hₖ` itself grows with `k` the model-selection cost `√(log(1/w(k))/ℓ)` is dominated by the capacity term, so I pay essentially nothing for not knowing the right class. That is the formal payoff of the whole construction: complexity selected from a single training set, with a distribution-free guarantee, competitive with the best class. (This is a different lever than cross-validation, which estimates each candidate's risk on a held-out split — spending data and concentrating only on the few candidates it scores — whereas SRM bounds the risk a priori from the training sample itself.)

The capacity-not-parameters point becomes concrete with separating hyperplanes. In `n`-dimensional input space the class of all hyperplanes `f(x,w) = θ(w·x + b)` has VC dimension exactly `n + 1` — they shatter at most `n+1` points. If `n` is large this is a huge capacity, and the structure `S₁ ⊂ ⋯` built by, say, growing the input dimension just walks me up to large `h`. The class is also rather inflexible; a single linear threshold often can't reach low empirical risk on real data. Two ways to add flexibility: superpose linear units (networks of neurons), or map the inputs into a very high-dimensional feature space and separate there. Both *raise* the ambient dimension, which naively raises `n+1`, which raises the capacity penalty — exactly the wrong direction.

But look again at what governs the bound: it's `h`, and `h` is the capacity of the class of *decision rules I actually allow*, not the dimension of the space they live in. So if I could shrink the class of hyperplanes — restrict to a sub-class with smaller VC dimension — while keeping it expressive, I'd win. What's the right handle? The margin. Call a hyperplane `w·x + b = 0`, `‖w‖ = 1`, a `Δ`-margin separating hyperplane if it not only classifies the points correctly but keeps them all at distance at least `Δ`:

    y = +1 if w·x + b ≥ Δ,   y = −1 if w·x + b ≤ −Δ.

Restricting to large-margin hyperplanes is restricting to a *sub-class*. How small is its capacity? If the data live in a ball of radius `R`, then the class of `Δ`-margin separating hyperplanes has

    h ≤ min( R²/Δ², n ) + 1.

The VC dimension of the *margin-restricted* hyperplanes is bounded by the smaller of `R²/Δ²` and the ambient dimension term, so when the margin `Δ` is large the active capacity can be far below `n + 1`. So the margin `Δ` is the structuring parameter I was looking for: larger margin indexes a smaller-VC element of a structure on hyperplanes. Decreasing the margin admits more hyperplanes (bigger class, bigger `h`); increasing it restricts to fewer (smaller class, smaller `h`). It decouples capacity from dimensionality whenever `R²/Δ² < n`. I can map into a billion-dimensional feature space and still control generalization if the margin-radius ratio stays small, because the active term is `R²/Δ²`, not the feature count.

So in the separable case I should ask for zero training error while sitting in the smallest-VC element of the margin structure. Since larger margin means smaller capacity, that means the largest margin among separating hyperplanes. Rescale `w` and `b` so the closest points satisfy `|w·x + b| = 1`; then every example obeys `yᵢ(w·xᵢ + b) ≥ 1`. The geometric margin — the distance from the hyperplane to the nearest point — is `1/‖w‖` on each side, so the full margin is `2/‖w‖`. Maximizing the margin is minimizing `‖w‖`, equivalently `½‖w‖²`, and the optimization becomes

    minimize  ½ w·w   subject to   yᵢ(w·xᵢ + b) ≥ 1,  i = 1,…,ℓ.

This is a convex quadratic program. Form the Lagrangian with multipliers `αᵢ ≥ 0`,

    L(w, b, α) = ½ w·w − Σᵢ αᵢ [ yᵢ(w·xᵢ + b) − 1 ],

minimize over `(w,b)` and maximize over `α`. Stationarity in `w` and `b`:

    ∂L/∂w = 0  ⟹  w = Σᵢ αᵢ yᵢ xᵢ,
    ∂L/∂b = 0  ⟹  Σᵢ αᵢ yᵢ = 0.

So `w` is a linear combination of the training inputs. Substitute back into `L` to get the dual, which depends on the data only through inner products:

    maximize  W(α) = Σᵢ αᵢ − ½ Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ (xᵢ·xⱼ)
    subject to  Σᵢ αᵢ yᵢ = 0,   αᵢ ≥ 0.

The Karush–Kuhn–Tucker complementarity condition is `αᵢ [ yᵢ(w·xᵢ + b) − 1 ] = 0`. So `αᵢ > 0` only for the points sitting exactly on the margin, `yᵢ(w·xᵢ + b) = 1` — the support vectors. All other multipliers are zero, and

    w = Σ_{support vectors} αᵢ yᵢ xᵢ.

The solution is sparse: it depends only on the points that pin the margin. The decision function `f(x) = sign(w·x + b) = sign(Σᵢ αᵢ yᵢ (xᵢ·x) + b)` uses the data only through inner products `xᵢ·x`.

That last fact is the door to nonlinearity, and it keeps me honest about capacity. To separate non-linearly, map each input through some `φ` into a high-dimensional feature space and build the optimal hyperplane *there*. But the entire optimization — both the dual `W(α)` and the decision rule — touches the feature vectors only through inner products `φ(xᵢ)·φ(xⱼ)`. So I never need `φ` explicitly; I only need the inner-product function

    K(xᵢ, xⱼ) = φ(xᵢ)·φ(xⱼ).

Replace every `xᵢ·xⱼ` by `K(xᵢ, xⱼ)` and I've built an optimal hyperplane in the feature space while computing only in the input space. By Mercer's theorem any symmetric positive-definite `K` is an inner product in *some* feature space, so any such `K` is admissible — e.g. `K(x,xᵢ) = (x·xᵢ + 1)^d` for polynomials of degree `d`, or `K(x,xᵢ) = exp(−‖x − xᵢ‖²/σ²)` for a radial-basis machine. The machine that constructs `f(x) = sign(Σᵢ αᵢ yᵢ K(xᵢ,x) + b)` is the support vector machine. And the reason it generalizes is not that the feature space is small — it's enormous — but that I am maximizing the margin, i.e. choosing the smallest-VC element `R²/Δ²` of the structure on hyperplanes. Capacity control, not dimension control.

If the data aren't separable I don't want to abandon the margin idea, so I relax the constraints with slack `ξᵢ ≥ 0`, allowing `yᵢ(w·xᵢ + b) ≥ 1 − ξᵢ`, and pay for it:

    minimize  ½ w·w + C Σᵢ ξᵢ   subject to   yᵢ(w·xᵢ + b) ≥ 1 − ξᵢ,  ξᵢ ≥ 0.

Here `C` is itself a structuring/tradeoff parameter: small `C` favors a large margin (small `‖w‖`, small VC element) at the cost of more margin violations; large `C` favors fitting the data (small empirical risk) at the cost of a smaller margin. That is the SRM seesaw again — empirical risk against capacity — now turned into a single dial. To see the dual cap rather than just assert it, add multipliers `αᵢ ≥ 0` for the margin constraints and `μᵢ ≥ 0` for `ξᵢ ≥ 0`:

    L(w,b,ξ,α,μ) = ½w·w + CΣᵢξᵢ − Σᵢαᵢ[yᵢ(w·xᵢ+b) − 1 + ξᵢ] − Σᵢμᵢξᵢ.

Stationarity again gives `w = Σᵢαᵢyᵢxᵢ` and `Σᵢαᵢyᵢ = 0`; stationarity in `ξᵢ` gives `C − αᵢ − μᵢ = 0`, hence `0 ≤ αᵢ ≤ C`. Complementarity is

    αᵢ[yᵢ(w·xᵢ+b) − 1 + ξᵢ] = 0,   (C − αᵢ)ξᵢ = 0.

Substituting out `w`, `b`, `ξ`, and `μ`, the soft-margin dual keeps the same quadratic objective and equality constraint, but the feasible interval becomes capped:

    maximize  Σᵢ αᵢ − ½ Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ K(xᵢ,xⱼ)
    subject to  Σᵢ αᵢ yᵢ = 0,   0 ≤ αᵢ ≤ C.

Choosing `C` is choosing where on the structure to sit.

Let me also note the structure doesn't have to be margin. The same principle takes other shapes for other machines. Fix a network architecture and structure by the *learning procedure*: nest by the norm of the weights, `Sₚ = { f(x,w) : ‖w‖ ≤ Cₚ }` with `C₁ < C₂ < ⋯` — smaller norm-ball, smaller capacity — and minimizing empirical risk within `Sₚ` is, for a convex loss, exactly minimizing `R_emp(w) + γₚ‖w‖²` with a Lagrange multiplier `γₚ` decreasing as the ball grows. That is weight decay, and now I can read it not as an arbitrary smoothness prior but as climbing a structure: `γ` selects the VC element. Or structure by *architecture*: nest by the number of hidden units, the class growing as units are added. Or by *preprocessing*: smooth the inputs with a kernel of width `β`, and nest by `β` — a wider smoothing kernel degenerates the input and shrinks the effective class. In every case the move is identical: order the classes by capacity, fit ERM in each, select the element minimizing the guaranteed risk.

So the causal chain, start to finish. I want small true risk from a finite sample; the only computable surrogate is the training error; but the training error, minimized over a class, is a biased estimate of the true risk by an amount that is the worst-case gap over the class — and that gap is governed by capacity `h`, not by the number of parameters. Bounding that gap gives `R ≤ R_emp + Ω(h, ℓ)`, a distribution-free guarantee in which the penalty grows with `h` and shrinks with `ℓ`. A single fixed class is stuck on one point of the resulting tradeoff, so instead I impose a nested structure `S₁ ⊂ S₂ ⊂ ⋯` of increasing capacity, run ERM inside each element, and select the (class, function) pair that minimizes the guaranteed risk `R_emp + Ω(h, ℓ)` — fit falling, capacity penalty rising, the minimum at the right complexity. This is universally consistent if the structure grows slowly with the data, and competitive with the best class via an oracle inequality. Instantiated on hyperplanes, the structuring parameter is the margin (`h ≤ min(R²/Δ²,n)+1`), so a large-margin separator can sit in a much smaller-capacity element than the ambient feature dimension suggests. The smallest-capacity zero-error element is the maximal-margin hyperplane — the support vector machine — which controls generalization by margin rather than by raw dimension, and goes nonlinear through kernels.

```python
import numpy as np

# Structural Risk Minimization: select model complexity by minimizing a
# guaranteed risk = empirical risk + a VC-capacity confidence term.

def empirical_risk(predict, X, y, loss):
    return np.mean([loss(yi, predict(xi)) for xi, yi in zip(X, y)])

def confidence_term(h, ell, eta):
    # C0 = sqrt( (h (ln(2 ell / h) + 1) - ln eta) / ell ): grows with capacity h,
    # shrinks with sample size ell. Distribution-free. This is the price of taking
    # a supremum over a class of VC dimension h (the worst-case gap R - R_emp).
    return np.sqrt((h * (np.log(2.0 * ell / h) + 1.0) - np.log(eta)) / ell)

def bounded_loss_upper_bound(R_emp, h, ell, eta, B=1.0):
    # For 0 <= L <= B, use the R_emp-dependent interval:
    # epsilon = 4(h(ln(2 ell / h) + 1) - ln eta) / ell.
    eps = 4.0 * (h * (np.log(2.0 * ell / h) + 1.0) - np.log(eta)) / ell
    return R_emp + 0.5 * B * eps * (1.0 + np.sqrt(1.0 + 4.0 * R_emp / (B * eps)))

def guaranteed_risk(R_emp, h, ell, eta):
    # The bound R <= R_emp + confidence. SRM minimizes THIS, not R_emp alone:
    # minimizing R_emp alone sends you to the richest class (R_emp = 0, huge penalty).
    return R_emp + confidence_term(h, ell, eta)

class Structure:
    """A nested family S_1 subset S_2 subset ... with h_1 <= h_2 <= ... .
    Each element fits ERM inside itself; nesting makes min-R_emp fall and h rise."""
    def elements(self):
        # yield (class, vc_dimension) ordered by increasing capacity
        raise NotImplementedError

def srm_select(structure, X, y, loss, eta=0.05, use_relative_bound=False, B=1.0):
    ell = len(y)
    best = None
    for S_k, h_k in structure.elements():
        f_k = S_k.fit_erm(X, y, loss)          # minimize empirical risk WITHIN S_k
        R_emp = empirical_risk(f_k, X, y, loss)
        if use_relative_bound:
            bound = bounded_loss_upper_bound(R_emp, h_k, ell, eta, B=B)
        else:
            bound = guaranteed_risk(R_emp, h_k, ell, eta)
        if best is None or bound < best[0]:
            best = (bound, f_k, h_k)            # keep the smallest guaranteed risk
    return best[1]                             # (class, function) minimizing R_emp + Omega

# Structure on hyperplanes: the margin is the capacity knob.
# VC dim of Delta-margin hyperplanes in a ball of radius R: h <= min(R^2/Delta^2, n) + 1,
# which can be << n+1 when R^2/Delta^2 is the active term. Maximal margin is the
# smallest-VC zero-error element, so the optimal hyperplane is the SRM choice on hyperplanes.
def optimal_hyperplane_dual(X, y, kernel, solve_qp, C=np.inf, tol=1e-8):
    # max  sum_i a_i - 1/2 sum_ij a_i a_j y_i y_j K(x_i, x_j)
    # s.t. sum_i a_i y_i = 0,  0 <= a_i <= C       (C = where on the structure to sit)
    # The QP solver is supplied by the implementation; the math is the SVM dual.
    X = np.asarray(X)
    y = np.asarray(y, dtype=float)
    n = len(y)
    K = np.array([[kernel(X[i], X[j]) for j in range(n)] for i in range(n)])
    Q = (y[:, None] * y[None, :]) * K
    ub = np.full(n, C)
    alpha = solve_qp(Q=Q, c=-np.ones(n), Aeq=y[None, :], beq=np.array([0.0]),
                     lb=np.zeros(n), ub=ub)
    sv = alpha > tol                                                     # support vectors
    free = sv & (alpha < C - tol) if np.isfinite(C) else sv
    bias_set = free if np.any(free) else sv
    bias = np.mean(y[bias_set] - K[bias_set] @ (alpha * y))
    # decision rule uses data only through inner products K(x_i, x):
    def predict(x):
        score = sum(alpha[i] * y[i] * kernel(X[i], x) for i in range(n) if sv[i]) + bias
        return 1 if score >= 0.0 else -1
    return predict
```
