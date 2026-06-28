Let me start from what actually stops me. I want to tune a high-dimensional hyperparameter `λ` — thousands of per-example weights, say — by gradient descent on an outer objective `F(λ) = E_S[f_S(ŵ*(λ), λ)]`, where `ŵ*(λ)` is not a clean formula but literally the place where `T` steps of gradient descent on an inner objective `g(w, λ)` land. To move `λ` I need its total derivative, `d_λ f = ∇_λ f + ∇_λ ŵ*(λ) · ∇_{ŵ*} f`. The two partials `∇_λ f` and `∇_{ŵ*} f` are cheap — a single backward pass through `f` hands them to me. The whole difficulty is the middle factor `∇_λ ŵ*(λ)`, the sensitivity of the inner solution to the hyperparameter, because `ŵ*` is the end of a long chain of inner updates, each of which depended on `λ`.

So make the chain explicit. The inner optimizer is a dynamical system: `w_{t+1} = Ξ_{t+1}(w_t, λ)`, `w_0 = Ξ_0(λ)`, `ŵ* = w_T`. For plain gradient descent `Ξ_{t+1}(w_t, λ) = w_t - γ ∇_w g(w_t, λ)`. Now `ŵ* = w_T` is a composition of `T` maps, each one a function of both the previous state and `λ`, so I just turn the crank of the chain rule. Differentiating `w_T` totally with respect to `λ` and collecting terms, the exact hypergradient is a sum over the entire trajectory,

  d_λ f = ∇_λ f + Σ_{t=0}^{T} B_t A_{t+1} A_{t+2} ⋯ A_T ∇_{ŵ*} f,

where `A_{t+1} = ∇_{w_t} Ξ_{t+1}(w_t, λ)` is how step `t+1` reacts to a perturbation of the *state*, and `B_{t+1} = ∇_λ Ξ_{t+1}(w_t, λ)` is how it reacts to a perturbation of `λ` directly, with `B_0 = d_λ Ξ_0(λ)` carrying the dependence of the *initialization* on `λ`. For the GD map these are concrete: `A_t = ∇_{w_{t-1}}(w_{t-1} - γ ∇_w g(w_{t-1}, λ)) = I - γ ∇²_w g(w_{t-1}, λ)`, and `B_t = ∇_λ(w_{t-1} - γ ∇_w g(w_{t-1}, λ)) = -γ ∇_{λ,w} g(w_{t-1}, λ)`. So the term for index `t` is: a direct `λ`-kick `B_t` at step `t`, then propagated forward through every subsequent state-Jacobian `A_{t+1} ⋯ A_T`, then dotted into the outer gradient `∇_{ŵ*} f`. Each term is the influence of the hyperparameter *at one moment of the inner run* on the final loss.

How do I compute this sum without forming any of those `M×M` Jacobians? Reverse mode. Carry two accumulators backward. Set `α_T = ∇_{ŵ*} f` and `h_T = ∇_λ f`, then sweep `t` down from `T`:

  h_{t-1} = h_t + B_t α_t,     α_{t-1} = A_t α_t,

and `d_λ f = h_{-1}`. Each step needs only a Jacobian-vector product — `A_t α_t` is "vector-Jacobian of the GD map," which autograd gives me for the price of one inner step, and likewise `B_t α_t`. This is just back-propagation through the unrolled inner optimization; Franceschi and coauthors even derive it as the stationarity conditions of a Lagrangian, attaching a multiplier `α_t` to each constraint `w_t = Ξ_t(w_{t-1}, λ)` — same recursion, cleaner provenance. The time is `O(cT)`, one inner-step cost per backward step. Lovely. But there's the catch I keep hitting: to evaluate `A_t` and `B_t` on the way back I need `w_{t-1}`, the state I was at when I took step `t`. The backward pass visits the states in reverse, so I have to have *all* of them, `w_1, …, w_T`, sitting in memory. That's `O(MT)` storage. With `M` a real network's parameter count and `T` a real inner horizon, that product is the thing that does not fit. This is the documented wall: storing the whole trajectory of a model whose parameter vector is on the order of a gigabyte, across tens of thousands of updates, is hopeless even spilling to disk. The estimator is exact and I cannot afford it.

What are my outs? One: propagate forward instead. Carry `Z_t = ∇_λ w_t` alongside the inner run, `Z_0 = B_0`, `Z_{t+1} = Z_t A_{t+1} + B_{t+1}`, and finish with `d_λ f = Z_T ∇_{ŵ*} f + ∇_λ f`. No trajectory to store — I overwrite `w_t` as I go. But `Z_t` is an `M×N` matrix, so it costs `O(MN)` memory and the propagation is `N` times slower than reverse mode, because I'm pushing `N` columns through at once. That's fine if `N` (hyperparameters) is tiny, but my whole premise is high-dimensional `λ`. So forward mode trades my memory problem for an equally fatal `MN`-and-`N×`-time problem. Wall.

Two: reconstruct the trajectory instead of storing it. If I run the backward pass while *exactly reversing* the inner dynamics — recover `w_{t-1}` from `w_t` by undoing the update — then I only ever hold one state, `O(M)`. Maclaurin and coauthors do exactly this for SGD-with-momentum. But reversal dies on finite precision: every momentum-decay multiply by `γ < 1` shifts bits off the bottom, and undoing it means repeatedly multiplying by `1/γ`, so rounding error compounds exponentially and the reconstructed `w_{t-1}` drifts from the real one, eventually nonsense. The patch is to stash the discarded low-order bits in an "information buffer," about `log₂(1/γ)` bits per step — which works, but it's fiddly, and it's wedded to that specific momentum update; a different inner optimizer wants a different, hand-built reversal. Too brittle to lean on.

Three: checkpoint. Keep a state only every `√T` steps, recompute the segments in between on the backward pass. Memory `O(M√T)`. But it *doubles* the compute (each segment runs forward twice), and `√T` still grows with the horizon — it postpones the wall, doesn't remove it.

Four, the trajectory-free route: implicit differentiation. If the inner problem were solved to its *exact* minimizer `w*(λ)`, the implicit function theorem gives a closed form with no trajectory at all,

  d_λ f = ∇_λ f - ∇_{λ,w} g · (∇_{w,w} g)^{-1} · ∇_{ŵ*} f,

needing only Hessian-vector products (approximate the inverse with conjugate gradient), `O(M)` memory. But look at what it assumes. It needs `ŵ*` to actually *be* the exact minimizer — and mine is a finite `T`-step run, not the argmin, so the formula's error is uncontrolled and it need not even point downhill. And it structurally cannot tune any hyperparameter that lives *inside* the inner optimizer — the step size, the horizon — because those got abstracted away into "the exact `w*`." That's a real loss: half the point of treating `ŵ*` as a finite optimizer run is to let `λ` shape that run.

So I'm cornered. The exact trajectory sum is `O(MT)` and won't fit. Forward mode is `O(MN)` and slow. Reversal is brittle. Checkpointing only softens it. Implicit diff needs an exactness I don't have. Let me stop hunting for a fifth trick and instead *stare at the sum I already have* and ask whether I really need all of it.

  d_λ f = ∇_λ f + Σ_{t=0}^{T} B_t A_{t+1} ⋯ A_T ∇_{ŵ*} f.

The term at index `t` carries the factor `A_{t+1} A_{t+2} ⋯ A_T` — a product of `T - t` state-Jacobians of the inner map. How big is each `A`? For GD, `A_t = I - γ ∇²_w g(w_{t-1}, λ)`. Suppose the last stretch of the inner run is in a region where `g` is `α`-strongly convex and `β`-smooth in `w`, which is the regime where GD is actually converging — `α I ≼ ∇²_w g ≼ β I`. Pick the step size so that `γ ≤ 1/β`, the natural choice that makes GD a contraction. Then `γ ∇²_w g` has eigenvalues in `[γα, γβ] ⊆ [γα, 1]`, so `I - γ ∇²_w g` has eigenvalues in `[1 - γβ, 1 - γα] ⊆ [0, 1 - γα]`, and therefore `‖A_t‖ ≤ 1 - γα < 1`. Each `A` is a *contraction*. And the term at index `t` is multiplied by `T - t` of them.

Then the terms are anything but uniform in size. A term from early in the trajectory, small `t`, picks up `A_{t+1} ⋯ A_T`, a product of *many* contractions, so its magnitude is killed by roughly `(1 - γα)^{T-t}` — geometrically small. A term from late in the trajectory, `t` near `T`, has almost no contraction factors and survives at full size. The recent steps should carry essentially all of the sum, and the ancient steps a geometrically vanishing tail. Heuristically the inner optimization *forgets* how it got here — a perturbation to `λ` made way back at the start gets washed out by all the contractive steps that follow, the way GD itself forgets its initialization as it converges. But that is a hand-wave; before I act on it I want to see the actual numbers, because the term also carries a `B_t` and the outer gradient, and a geometric *factor* on each term is not yet a statement about the *sum* I'd be discarding.

I take the smallest concrete instance where I can compute the exact sum and compare: the inner quadratic `½(w - λ)^T G (w - λ)` with `G = diag(1, ½)`, so `β = 1`, `α = ½`, `γ = 0.1 ≤ 1/β`, `T = 100`, with an outer `f` whose `∇_{ŵ*} f` I read off at the end. Here `A = I - γG` and `B = γG` are constant, so I can form the full-reverse sum `h_{0}` (all `T` terms, which is exact) and the truncated `h_{T-K}` for several `K`, and look at `‖e_K‖ = ‖d_λ f - h_{T-K}‖`:

```
K=  1  ||e_K||=8.57e-01
K=  5  ||e_K||=6.91e-01
K= 10  ||e_K||=5.30e-01
K= 20  ||e_K||=3.14e-01
K= 50  ||e_K||=6.30e-02
K=100  ||e_K||=0.00e+00
```

Two things land. The error at `K = T = 100` is exactly zero, as it must be — keeping all terms *is* full reverse mode, so my truncation degrades to the exact estimator in the limit, a reassuring consistency check that I haven't dropped a term I needed. And dividing each `‖e_K‖` by `(1 - γα)^K = 0.95^K` gives a nearly constant ratio (0.90, 0.89, 0.89, 0.88, 0.82), i.e. the error really is decaying like `(1 - γα)^K` and not slower — the geometry of one factor does survive into the sum. So the early terms are negligible in aggregate, not just term-by-term. That earns the cut.

If the early terms contribute a geometrically small tail, I am paying to store the early states for nothing. Cut the sum off — keep only the last `K` terms and throw away the rest:

  h_{T-K} := ∇_λ f + Σ_{t=T-K+1}^{T} B_t A_{t+1} ⋯ A_T ∇_{ŵ*} f.

And this maps onto reverse mode for free, because reverse mode produces the terms *in exactly this order* — it sweeps `t` from `T` downward, so `h_T, h_{T-1}, …` appear one by one, and `h_{T-K}` is just the value of the `h` accumulator after `K` backward steps. I don't run the backward pass to the bottom; I stop after `K` steps. The saved window has to be consecutive: to compute the last `K` terms, `t = T-K+1, …, T`, I need the boundary state `w_{T-K}` and the successors `w_{T-K+1}, …, w_T`, i.e. the `K` final transitions. That is still `O(MK)` memory. And the two jobs cleanly separate: I still run the full `T` forward steps so `ŵ* = w_T` is a good inner solution sitting deep in the strongly convex basin (forward steps are cheap and memoryless — overwrite as you go), but I only *differentiate* through the last `K` transitions. Forward depth `T` for solution quality; backward depth `K` for gradient quality; and `K` can be far smaller than `T`.

Now I have to earn this, not just assert it. How wrong is `h_{T-K}`? Define the error `e_K = d_λ f - h_{T-K}`. It's exactly the part I dropped — the early terms — but re-grouped. Each dropped term is `B_t A_{t+1} ⋯ A_T ∇_{ŵ*} f` for `t ≤ T - K`. Pull out the common contraction tail: every such term contains `A_{T-K+1} ⋯ A_T` (the last `K` Jacobians) sitting on the right, applied to `∇_{ŵ*} f`. So factor it:

  e_K = (Σ_{t=0}^{T-K} B_t A_{t+1} ⋯ A_{T-K}) · (A_{T-K+1} ⋯ A_T ∇_{ŵ*} f).

The right factor is the outer gradient pushed back through the last `K` contractions: `‖A_{T-K+1} ⋯ A_T ∇_{ŵ*} f‖ ≤ (1 - γα)^K ‖∇_{ŵ*} f‖`, since I'm hitting it with `K` operators each of norm `≤ 1 - γα`. The left factor is the truncated trajectory sum of the *early* part; its norm is bounded by `M_B` times the appropriate product bound. Put together,

  ‖e_K‖ ≤ (1 - γα)^K · ‖∇_{ŵ*} f‖ · ‖Σ_{t=0}^{T-K} B_t A_{t+1} ⋯ A_{T-K}‖.

So the bias is `(1 - γα)^K` times a bounded factor, and since `0 ≤ 1 - γα < 1` it decays geometrically in `K` — which is precisely the `0.95^K` decay I measured a moment ago. A moderate `K` can make the dropped tail negligible relative to the accuracy I can use. Let me pin the bounded factor in two regimes. If `g` is globally `α`-strongly convex everywhere, then along the whole sum `‖A_{t+1} ⋯ A_{T-K}‖ ≤ (1 - γα)^{T-K-t}`, and with `M_B = max_t ‖B_t‖` the left factor is `≤ M_B Σ_{t=0}^{T-K} (1 - γα)^{T-K-t} ≤ M_B Σ_{k=0}^{∞} (1 - γα)^k = M_B/(γα)`, giving the clean

  ‖e_K‖ ≤ ((1 - γα)^K / (γα)) · ‖∇_{ŵ*} f‖ · M_B.

I should sanity-check that this bound is actually an upper bound on the toy and not a wishful one. There `γα = 0.05`, `M_B = ‖γG‖ = 0.1`, and `‖∇_{ŵ*} f‖ ≈ 0.9`, so the formula predicts `‖e_K‖ ≤ 0.95^K · 0.91 · 0.1 / 0.05 = 1.82 · 0.95^K`. At `K = 1` that is `1.73` against a measured `0.86`; at `K = 20`, `0.65` against `0.31`; at `K = 50`, `0.14` against `0.063`. The bound sits above the truth by a factor of roughly two throughout (the `1/(γα)` geometric-tail constant is loose, as expected) and tracks the same rate — so the derivation is consistent with the experiment, not merely plausible-sounding.

If `g` is only *locally* strongly convex near the final transition window but possibly nonconvex earlier, I can't bound the early `A`'s by a contraction — in the worst case the smallest eigenvalue of `∇²_w g` is `-β`, so `‖A_t‖ = ‖I - γ ∇²_w g‖ ≤ 1 + γβ ≤ 2` for `γ ≤ 1/β`. The early product can then grow like `2^{T-K}`, so the honest bound is `‖e_K‖ ≤ 2^{T-K+1} (1 - γα)^K ‖∇_{ŵ*} f‖ M_B`. That `2^{T-K}` looks alarming, but it's the price of nonconvexity in the *early* part of the run; the useful regime is the one where the locally convex tail is long enough for the geometric term to dominate that loose early-prefix constant. The requirement itself is local: the inner run only has to be well-behaved near where it ends up, which is exactly where a converging optimizer is.

I want to understand *what* `h_{T-K}` is approximating, because the implicit-differentiation formula keeps nagging at me — it's trajectory-free and I'd like to know the relationship. Take the limit where the inner run actually converges, `w_t → w*`. Then `A_t → A_∞ = I - γ ∇_{w,w} g` and `B_t → B_∞ = -γ ∇_{λ,w} g`, both evaluated at `(w*, λ)`. With `γ ≤ 1/β` I have `‖I - γ ∇_{w,w} g‖ < 1`, so the Neumann series converges:

  (∇_{w,w} g)^{-1} = γ · (γ ∇_{w,w} g)^{-1} = γ · (I - (I - γ ∇_{w,w} g))^{-1} = γ Σ_{k=0}^{∞} (I - γ ∇_{w,w} g)^k = γ Σ_{k=0}^{∞} A_∞^k.

Therefore the implicit-diff middle term is

  -∇_{λ,w} g · (∇_{w,w} g)^{-1} = (-γ ∇_{λ,w} g) · ((1/γ)(∇_{w,w} g)^{-1}) = B_∞ Σ_{k=0}^{∞} A_∞^k,

so the exact hypergradient is `d_λ f = ∇_λ f + B_∞ Σ_{k=0}^{∞} A_∞^k ∇_{ŵ*} f`. That is a chain of three matrix identities I do not want to trust by eye, so I check the middle equality numerically on the toy at its fixed point `w* = λ`. There `∇_{w,w} g = G`, `∇_{λ,w} g = -G`, so the implicit-diff middle factor `-∇_{λ,w} g (∇_{w,w} g)^{-1} = G·G^{-1} = I`. Summing the Neumann series `B_∞ Σ A_∞^k = γG Σ (I - γG)^k` to convergence gives `[[1, 0], [0, 1]]` to within `1.2e-15` of the identity — the two expressions for the middle factor agree, so the rearrangement is right.

And my truncated estimator, in this limit, is `h_{T-K} = ∇_λ f + B_∞ Σ_{k=0}^{K-1} A_∞^k ∇_{ŵ*} f` — it captures the *first `K` terms* of that very same series, i.e. an order-`K` Taylor (Neumann) approximation of the inverse Hessian. To see that the truncated series approaches the inverse at the rate I claimed, I form the partial-sum middle factor `B_∞ Σ_{k=0}^{K-1} A_∞^k` for the toy and measure its distance to the identity: `K=1 → 0.95`, `K=5 → 0.77`, `K=20 → 0.36`, `K=100 → 5.9e-3`, matching `(1 - γα)^K = 0.95^K` term for term. So the residual `B_∞ Σ_{k=K}^{∞} A_∞^k ∇_{ŵ*} f` is `O((1 - γα)^K)`, the same geometric decay, now read off the tail of the series rather than the back-prop sum. The two views are the same object truncated two ways: implicit diff truncates the *Neumann series of the inverse*, I truncate the *back-prop sum*, and they coincide at the fixed point.

That comparison also tells me when each is better. Approximating the inverse with `K` steps of conjugate gradient has bias `O((1 - 1/√κ)^K)` where `κ = β/α` is the local condition number, while my `K`-truncation has bias `O((1 - 1/κ)^K)` — CG's `1/√κ` beats my `1/κ`, so if I genuinely had `w*` in hand, CG on the linear system would converge faster per step. But I don't have `w*`; I have a finite-`T` `ŵ*`. My bound needed only that the retained tail sit in a locally strongly convex region — it never required `w_t` to have reached a stationary point — whereas the implicit formula has *no* error control away from `w*` and can fail to be a descent direction there. And I can differentiate the inner optimizer's own hyperparameters, which implicit diff cannot. So the trade is real and it favors truncation in the messy, finite-budget regime I actually live in.

Small bias is reassuring, but it is not yet what I want. For a *biased* gradient method to converge I'd love something stronger than "`h_{T-K}` is close to `d_λ f`": I'd love "`-h_{T-K}` actually points downhill," i.e. it's a sufficient descent direction, `h_{T-K}^T d_λ f ≥ Ω(‖d_λ f‖²)`. Is it? Let me test the extreme case `K = 1`, the cheapest possible, and assume the clean setting where `g` is globally strongly convex and the outer objective doesn't depend on `λ` directly, `∇_λ f = 0` (true for data hyper-cleaning, where the validation loss sees `λ` only through `ŵ*`). Then `h_{T-1} = B_T ∇_{ŵ*} f` — just the last term. Expand the inner product against the full gradient:

  h_{T-1}^T d_λ f = ‖h_{T-1}‖² + (B_T ∇_{ŵ*} f)^T (Σ_{t=0}^{T-1} B_t A_{t+1} ⋯ A_{T-1}) A_T ∇_{ŵ*} f.

The first piece `‖h_{T-1}‖² = ‖B_T ∇_{ŵ*} f‖² ≥ 0` is the "self" term and it's strictly positive when `B_T` has full column rank — that's why I'll assume it. The danger is the cross term, the dot of `h_{T-1}` against everything I dropped: could it be negative and large enough to cancel the self term? Here's where I have to be careful, because `B_T` and `A_t` vary along the trajectory and the cross term is a sum of many of these mixed products. I bound each one. Take a representative `(B_T ∇_{ŵ*} f)^T B_t A_{t+1} ⋯ A_T ∇_{ŵ*} f`. The clean part of it, if the `A`'s were all exactly `(1 - γα) I` and `B_t` were exactly `B_T`, would be `(1 - γα)^{T-t} ‖B_T ∇_{ŵ*} f‖² ≥ 0` — aligned with the self term, good. The damage comes from the *variation*: `B_t ≠ B_T`, `A_k ≠ (1 - γα)I`. So I peel off three error pieces. First, replacing `B_t` by `B_T` costs `C_B ‖B_T ∇_{ŵ*} f‖ ‖∇_{ŵ*} f‖ ‖w_{T-1} - w_{t-1}‖ ‖A_{t+1} ⋯ A_T‖`, using that `B` is Lipschitz with constant `C_B` and changes only as fast as the state moves. Second, telescoping the `A`'s toward `A_T` one at a time costs a sum of similar `C_A`-Lipschitz terms. Third, replacing each `A_T` by `(1 - γα) I` costs `‖A_T - (1 - γα) I‖^{T-t} ≤ (γ(β - α))^{T-t}`, since `A_T - (1 - γα)I = γ(α I - ∇²_w g)` has norm `≤ γ(β - α)`.

And every one of these errors is multiplied by a state-distance `‖w_{T-1} - w_{t-1}‖`, which I can crush with linear convergence of GD: `‖w_t - w*‖ ≤ D e^{-αγ t}` with `D = ‖w_0 - w*‖`, so `‖w_{T-1} - w_{t-1}‖ ≤ 2 D e^{-αγ(t-1)}`. Multiply by the contraction `(1 - γα)^{T-t} ≤ e^{-γα(T-t)}` riding along and the `t`-dependence collapses: `e^{-αγ(t-1)} e^{-γα(T-t)} = e^{-αγ(T-1)}`, *independent of `t`*. So the first two error pieces, summed over `t`, are each `O(e^{-αγ(T-1)})`, possibly with a `1/(1 - e^{-αγ})` from a geometric tail and a benign factor `T`. The third piece sums as `Σ_{t} (γ(β - α))^{T-t} = Σ_{k≥1} (γ(β - α))^k ≤ γ(β - α)/(1 - γ(β - α))`, finite and small when `γ` is small. Putting the self term and the three error sums together:

  h_{T-1}^T d_λ f ≥ ‖B_T ∇_{ŵ*} f‖² (1 + Σ_{t=0}^{T-1}(1 - γα)^{T-t}) - ‖∇_{ŵ*} f‖² O( T e^{-αγ(T-1)}/(1 - e^{-αγ}) + γ(β - α)/(1 - γ(β - α)) ).

For `T` large the `T e^{-αγ(T-1)}` term vanishes; for `γ` small the `γ(β - α)` term vanishes; and `‖B_T ∇_{ŵ*} f‖² ≥ Ω(‖∇_{ŵ*} f‖²)` because `B_T^T B_T` is nonsingular (full column rank). So the positive self term should win and leave `h_{T-1}^T d_λ f ≥ c ‖∇_{ŵ*} f‖²` for some `c > 0`. Since `‖d_λ f‖ ≤ O(‖∇_{ŵ*} f‖)`, that would read as `h_{T-1}^T d_λ f ≥ Ω(‖d_λ f‖²)` — a sufficient descent direction at `K = 1`.

But this argument has a lot of inequalities chained together, each one slack, and a sign error or an over-loose bound could hide the self term being swamped after all. The cleanest way to find out is to compute `h_{T-1}^T d_λ f` directly. On the toy (`∇_λ f = 0`, globally strongly convex), `h_{T-1} = B_T ∇_{ŵ*} f` and `d_λ f` is the full reverse sum; I draw 2000 random `(λ, w_0)`, form both vectors, and look at the cosine between them and at the descent ratio `h_{T-1}^T d_λ f / ‖d_λ f‖²` — the very `Ω(‖d_λ f‖²)` quantity:

```
min cosine(h_{T-1}, d_λ f)         over 2000 trials:  0.944
min  h_{T-1}·d_λ f / ||d_λ f||^2   over 2000 trials:  0.050
```

Both stay strictly positive in every trial. The worst-case cosine is `0.94`, so `-h_{T-1}` is not just downhill but nearly aligned with the true gradient, and the descent ratio never drops below `0.05`, i.e. the constant `c` in `h_{T-1}^T d_λ f ≥ c ‖d_λ f‖²` is real and bounded away from zero. So even the single-step truncation `K = 1` gives a usable descent direction here, and the same peeling extends the argument to `K > 1`. The bias being small and the direction being downhill are two separate facts, and I now have both — under the conditions that the problem be well-conditioned and `∇_λ f = 0`.

Now stitch this into a convergence statement for the outer loop, where I just feed `h_{T-K}` to SGD on `λ`. Two cases. First, the merely-small-bias case, where all I assume is `‖h_{T-K} - d_λ f‖ ≤ ε`. Standard biased-SGD analysis: `F` is `L`-smooth so `F(λ_{τ+1}) ≤ F(λ_τ) + ⟨∇F(λ_τ), λ_{τ+1} - λ_τ⟩ + (L/2)‖λ_{τ+1} - λ_τ‖²`. Plug in `λ_{τ+1} = λ_τ - η_τ h_{T-K,(τ)}`. The inner-product term splits into the true descent `-η_τ ‖∇F‖²` plus a bias leak `η_τ ⟨∇F, e_τ⟩ ≤ η_τ G ε` (Cauchy–Schwarz with `‖∇F‖ ≤ G`, `‖e_τ‖ ≤ ε`), and the squared term is `≤ L η_τ² (3G²/2 + ε²/2)` after expanding `‖h‖² = ‖d_λ f‖² + ‖e‖² - 2⟨d_λ f, h⟩` and bounding. Telescope over `τ = 1..R`, divide by `Σ η_τ`, take `η_τ = O(1/√τ)` so that `Σ 1/τ / Σ 1/√τ = O(log R / √R)`, and I get

  E[ Σ_τ η_τ ‖∇F(λ_τ)‖² / Σ_τ η_τ ] ≤ Õ( ε + (ε² + 1)/√R ).

So the iterates reach an `ε`-approximate stationary point on average, and crucially `ε = O((1 - γα)^K)` — it shrinks geometrically in the truncation depth. To hit accuracy `ε` I only need `K = O(log 1/ε)` back-prop steps. That's the whole bargain: logarithmic reverse depth buys arbitrary gradient accuracy.

Second, the better case: can I kill the bias term entirely? Yes, when the descent property holds, which is the structured setting. Decompose `d_λ f = ∇_λ f + q + r + e`, splitting the trajectory sum into the kept last-`K` terms `q`, a middle band `r`, and an exponentially-small tail `e` (with `‖e‖ ≤ O(e^{-αγH} ‖∇_{ŵ*} f‖)` once the run is in the strongly convex region for the last `H ≥ K` steps). Note `h_{T-K} = ∇_λ f + q` exactly. Then

  d_λ f^T h_{T-K} = (∇_λ f + q + r + e)^T (∇_λ f + q) = ‖∇_λ f‖² + ∇_λ f^T(q + r + e) + q^T ∇_λ f + q^T(q + r) + q^T e.

The cross terms `∇_λ f^T(q + r + e) + q^T ∇_λ f` are exactly `∇_λ f^T(d_λ f + h_{T-K} - ∇_λ f)` once I recognize `d_λ f + h_{T-K} - ∇_λ f = ∇_λ f + q + r + e + q`. If I assume *non-interference* — that this quantity is `≥ Ω(‖∇_λ f‖²)`, i.e. the direct hyperparameter gradient doesn't fight the part computed through the inner trajectory — then those cross terms are controlled, the sufficient-descent lemma gives `q^T(q + r) ≥ Ω(‖∇_{ŵ*} f‖²)`, and `q^T e ≥ -O(e^{-αγH} ‖∇_{ŵ*} f‖²)` is negligible for `H` large. So `d_λ f^T h_{T-K} ≥ Ω(‖∇_λ f‖² + ‖∇_{ŵ*} f‖²)`, and `‖h_{T-K}‖² ≤ O(‖∇_λ f‖² + ‖∇_{ŵ*} f‖²)` too. Feed those two facts into the nonconvex descent lemma (`L`-smooth `F`, step chosen so `-c₁η + Lc₂η²/2 ≤ 0`, telescoping `Σ_t (c₁η - Lc₂η²/2) h_t² < ∞` with `h_t² = ‖∇_λ f‖² + ‖∇_{ŵ*} f‖²`) and `h_t → 0`. Since `‖d_λ f‖ ≤ O(‖∇_λ f‖ + ‖∇_{ŵ*} f‖)`, the *true* hypergradient norm goes to zero — convergence to an **exact** stationary point, for any `K ≥ 1`, even `K = 1`, with no `ε`-floor left over. That is a much stronger conclusion than the first case, and it should not be free, so I want to know exactly what is buying it: the whole thing rests on the non-interference inequality `∇_λ f^T(d_λ f + h_{T-K} - ∇_λ f) ≥ Ω(‖∇_λ f‖²)`. If that conclusion is correct, it would also explain why the one-step heuristics in learning-rate adaptation and first-order MAML — back-prop through the single most recent inner step — worked as well as people reported: in the non-interfering case one back-prop step would already be enough. But that is exactly the kind of clean story I should distrust until I have stress-tested the assumption it hangs on.

The way to stress-test it is to build a case where non-interference *fails* and see whether convergence genuinely breaks — if it doesn't, the assumption was decorative and the exact-stationary claim is overstated. Scalar problem: `min_λ ½(ŵ*)² + φ(λ)`, inner `ŵ* =` `T` GD steps on `½(w - λ)²`, so `w_{t+1} = w_t - γ(w_t - λ)` and the transition derivative is `B_t = γ > 0` (full rank, smooth, strongly convex — every assumption *except* non-interference holds, which is exactly the isolation I want). The closed forms are `h_{T-1} = ∇φ + ŵ* γ` and `d_λ f = ∇φ + ŵ* γ Σ_{t=0}^{T}(1 - γ)^{T-t}`. Write `u = ŵ* γ` and `v = ŵ* γ Σ(1 - γ)^{T-t}`; both share the sign of `ŵ*`, so `u v ≥ 0`, strictly positive when `ŵ* ≠ 0`. Pick `φ(λ) = ½(λ - λ₀)²` with `λ₀` chosen so the stationary points of the *full* `f` have `ŵ* ≠ 0`. At a true stationary point `d_λ f = 0`, i.e. `∇φ = -v`, the non-interference quantity is `∇φ^T(d_λ f + h_{T-1} - ∇φ) = ∇φ · u = -v u < 0` — it violates the assumption — and there `h_{T-1} = ∇φ + u = (d_λ f - v) + u = u - v ≠ 0`.

I make this concrete with `γ = 0.5`, `T = 200`, `λ₀ = 3`. Solving `d_λ f(λ) = 0` numerically gives the true stationary point `λ* = 1.5` (with `ŵ*(λ*) = 1.5 ≠ 0`, as arranged), and `d_λ f(λ*) = 0` to machine precision — confirming I've located an actual stationary point of `F`. Evaluating the truncated gradient there: `h_{T-1}(λ*) = -0.75`, decidedly nonzero. So at the very point the outer loop should stop, the truncated gradient still pushes with magnitude `0.75`. To see what that does to the loop, I run outer SGD on `h_{T-1}` (step `0.001`, 200000 iterations) from `λ = 0`: it does converge, but to `λ = 2.0`, where the *exact* hypergradient is `d_λ f = 1.0 ≠ 0`. The loop settles where `h_{T-1} = 0`, which is a different point from where `d_λ f = 0`, and the gap is a fixed bias that more iterations cannot remove. So convergence to a true stationary point genuinely fails when non-interference is violated — the assumption was load-bearing, not decorative, and the exact-stationary claim is only as strong as it. And it holds automatically whenever `∇_λ f = 0` (then the quantity is trivially `0 = Ω(0)`), which covers hyperparameter optimization, data hyper-cleaning, regularization learning, image denoising — precisely the cases the high-dimensional `λ` problem lives in.

Let me also sanity-check the design choices against this theory, the ones I'd otherwise just inherit as numbers. Why run `T` forward steps but keep only `K` for the backward — why not just run `K` inner steps total? Because the bias bound `(1 - γα)^K` and the descent lemma both demand that the *last* iterates sit in the locally strongly convex basin near the inner minimum; that's what makes the kept `A`'s contractive and the dropped tail negligible. Getting `ŵ*` into that basin is the job of the full horizon `T`, and the toy quantifies how much horizon it takes: starting from the fixed `w_0`, `‖w_t - w*‖` is `2.0`, `0.82`, `1.3e-2`, `1.6e-11` after `5, 20, 100, 500` steps. So a cold `K = 5`-step run differentiates through iterates still order-1 away from `w*` — nowhere near the basin — whereas `T = 500` forward steps put `ŵ*` essentially on top of `w*`. If I differentiated through those early cold-start iterates the contraction estimate `‖A‖ ≤ 1 - γα` wouldn't hold and both guarantees would evaporate; running long forward, short backward keeps the differentiated window inside the regime the bounds need. The forward pass is cheap and needs no stored graph — I overwrite `w_t` — so paying `T` forward steps for a good solution while paying only `K` backward steps for a good gradient is the efficient split. Why `γ ≤ 1/β`? That single condition is what makes `A_t = I - γ∇²g` a contraction, which simultaneously (a) makes GD converge linearly, (b) makes the truncation bias decay, and (c) makes the Neumann series for the inverse Hessian converge — one knob, three jobs. Why decaying outer step `η_τ = O(1/√τ)`? That's exactly the schedule the biased-SGD telescoping needs for the `Σ1/τ / Σ1/√τ = O(log R/√R)` ratio to vanish. And why choose a larger practical window such as `K = 100` inside a `T = 500` forward run? Because the cost is linear in `K`, while the bound shrinks geometrically in `K`; the extra reverse depth is a deliberate accuracy margin without returning to full `O(MT)` storage.

Now let me write it as the code I'd actually run, filling the empty slot in the harness. The reverse routine is the back-prop recursion `h_{t-1} = h_t + B_t α_t`, `α_{t-1} = A_t α_t`; truncation comes from passing it only the last `K+1` consecutive iterates, so it traverses exactly the last `K` transitions:

```python
from collections import deque
import torch
from torch.autograd import grad as torch_grad


def grad_unused_zero(output, inputs, grad_outputs=None, retain_graph=False):
    grads = torch.autograd.grad(output, inputs, grad_outputs=grad_outputs,
                                allow_unused=True, retain_graph=retain_graph)
    return tuple(torch.zeros_like(var) if grad is None else grad
                 for grad, var in zip(grads, inputs))


def update_tensor_grads(hparams, grads):
    for h, g in zip(hparams, grads):
        if h.grad is None:
            h.grad = torch.zeros_like(h)
        h.grad += g


def inner_step(params, hparams, lr_inner, inner_loss_fn):
    # Ξ(w, λ) = w - γ ∇_w g(w, λ); create_graph keeps the step differentiable.
    loss = inner_loss_fn(params, hparams)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    return [p - lr_inner * g for p, g in zip(params, grads)]


def reverse(params_history, hparams, update_map_history, outer_loss, set_grad=True):
    """The hg.reverse recursion, truncated by the history passed to it.

    params_history is consecutive, first to last. If it is
    [w_{T-K}, ..., w_T], this returns h_{T-K}.
    """
    params_history = [[w.detach().requires_grad_(True) for w in params]
                      for params in params_history]
    o_loss = outer_loss(params_history[-1], hparams)
    alphas = grad_unused_zero(o_loss, params_history[-1], retain_graph=True)
    grad_outer_hparams = grad_unused_zero(o_loss, hparams, retain_graph=True)

    grads = [torch.zeros_like(h) for h in hparams]
    K = len(params_history) - 1
    for k in range(-2, -(K + 2), -1):                 # T, T-1, ..., T-K+1
        w_mapped = update_map_history[k + 1](params_history[k], hparams)
        bs = grad_unused_zero(w_mapped, hparams, grad_outputs=alphas,
                              retain_graph=True)
        grads = [g + b for g, b in zip(grads, bs)]    # h_{t-1} += B_t alpha_t
        alphas = torch_grad(w_mapped, params_history[k],
                            grad_outputs=alphas)      # alpha_{t-1} = A_t alpha_t

    grads = [g + v for g, v in zip(grads, grad_outer_hparams)]
    if set_grad:
        update_tensor_grads(hparams, grads)
    return grads


def bilevel_train(hparams, fresh_inner_params, inner_loss_fn, outer_loss_fn,
                  lr_inner, T, K, outer_opt, num_outer):
    K = min(K, T)
    for _ in range(num_outer):
        params = fresh_inner_params()                 # re-initialize the inner problem
        w = params
        kept = deque([w], maxlen=K + 1)               # consecutive tail: w_{T-K}, ..., w_T
        update_maps = []
        for t in range(T):                            # full forward horizon: T GD steps on g
            step_fn = lambda p, h: inner_step(p, h, lr_inner, inner_loss_fn)
            w = step_fn(w, hparams)
            kept.append(w)
            update_maps.append(step_fn)
            if len(update_maps) > K:
                update_maps.pop(0)
        outer_opt.zero_grad()
        reverse(list(kept), hparams, update_maps, outer_loss_fn)  # fills hparams.grad
        outer_opt.step()                              # one outer SGD step on λ (η decaying)
```

The causal chain, start to finish. I needed the hypergradient `d_λ f` to tune a high-dimensional `λ` whose effect runs through a finite `T`-step inner optimizer. The exact value is a sum over the whole inner trajectory; computing it by reverse mode is `O(MT)` in memory and won't fit, forward mode is `O(MN)` and `N`-times slow, exact-reversal is brittle in finite precision, checkpointing only softens the memory growth, and implicit differentiation needs an exact minimizer I don't have and can't tune the inner optimizer. Staring at the trajectory sum, each early term is multiplied by a long product of inner-map Jacobians `A_t = I - γ∇²g`, which are contractions (`‖A‖ ≤ 1 - γα`) once the run is in a strongly convex basin with `γ ≤ 1/β` — so early terms are geometrically suppressed and the recent steps carry the gradient. Truncating the sum to its last `K` terms — equivalently, stopping reverse mode after `K` backward steps and storing the consecutive state window `w_{T-K}, …, w_T` — gives `h_{T-K}` with memory `O(MK)`, and its bias is `(1 - γα)^K` times a bounded factor, geometrically small in `K` — which the toy confirmed, the measured `‖e_K‖` tracking `0.95^K` and sitting under the derived bound. The same series, taken to its limit, is the Neumann expansion of the inverse Hessian (verified to `1e-15` at the toy fixed point), so `h_{T-K}` is an order-`K` Taylor approximation of implicit differentiation, but one that needs only local strong convexity, controls its error, and can differentiate the inner optimizer. Under the rank, regularity, and non-interference assumptions, `-h_{T-K}` is a sufficient descent direction — borne out by a worst-case cosine of `0.94` and a descent ratio bounded below by `0.05` over 2000 random trials — so outer SGD with a `1/√τ` step size converges, to an `ε = O((1-γα)^K)`-approximate stationary point in general and to an *exact* stationary point under the stronger descent conditions for any `K ≥ 1`. The non-interference assumption is necessary, not cosmetic: a scalar counterexample settles outer SGD at `λ = 2.0` where the exact hypergradient is `1.0 ≠ 0`, a fixed bias no number of iterations removes. Run the full horizon `T` forward for a good `ŵ*`, back-propagate only the last `K` transitions for a cheap accurate gradient, and the whole thing drops into the standard bilevel loop with `O(MK)` memory.
