## Research question

Compactness is clear for a single compact domain, but it is less clear for a set of continuous functions. The question is when a family `F` of continuous functions on a compact metric space behaves like a compact set in the uniform metric

`d_infty(f,g)=sup_x d_Y(f(x),g(x))`.

Finite-dimensional intuition suggests checking coordinates. If the values of every function stay in bounded or precompact sets at each point of the domain, then every fixed coordinate looks compact. The difficulty is that a function is not just its values at isolated points. A family can be bounded at every point and still change too sharply between nearby points for any subsequence to settle uniformly.

A useful criterion has to identify the missing finite-dimensional substitute: a way to make finitely many samples of the domain control the whole function. The issue is not pointwise size alone, but uniform control of oscillation between nearby points.

## Background

For a compact metric space `K` and a metric target `Y`, the continuous maps `C(K,Y)` carry the uniform metric. When `Y=R` or `R^n`, boundedness of a single continuous function follows from compactness of `K`; for a whole family, pointwise boundedness means that for each fixed `x`, the set `{f(x): f in F}` is bounded.

Pointwise boundedness is too weak in infinite-dimensional function spaces. On `[0,1]`, the functions `f_n(x)=x^n` are uniformly bounded by `1`, but their transition near `1` becomes sharper as `n` grows. The pointwise limit is `0` on `[0,1)` and `1` at `1`, so no uniformly convergent subsequence can have that pointwise limit. The failure is not large values; it is uncontrolled oscillation on small intervals.

Uniform convergence of continuous functions preserves continuity. A compactness theorem for families of continuous functions therefore needs a condition that prevents discontinuous pointwise limits from appearing after subsequence extraction. Equicontinuity is the natural condition: one choice of local scale must control the oscillation of every function in the family.

Compactness of the domain supplies finite sampling once equicontinuity is available. If nearby points have uniformly nearby values for all functions in the family, a finite net in the domain reduces a uniform-norm question to finitely many value questions. Pointwise precompactness then supplies finite nets in the target at those sampled points.

## Baselines

- **Pointwise Bolzano-Weierstrass.** At a fixed point `x`, if `{f(x): f in F}` has compact closure, any sequence of functions has a subsequence whose values at `x` converge. Gap: one point gives no control at any other point, and repeating this separately for uncountably many points is not a finite compactness argument.

- **Diagonal extraction on a countable dense set.** In a compact metric domain, choose a countable dense set and extract a subsequence whose values converge at each selected point. Gap: convergence on the dense set does not become uniform unless nearby domain points have uniformly nearby function values.

- **Pointwise boundedness.** For real-valued functions, bounded value sets at each `x` are necessary for compactness in the uniform metric. Gap: functions can remain bounded while developing steeper and steeper local changes.

- **Uniform boundedness in sup norm.** A common bound `||f||_infty <= M` is stronger than pointwise boundedness. Gap: it still allows families such as fast oscillations or steep boundary layers; magnitude control does not control shape.

- **Uniform convergence theorem.** A uniform limit of continuous functions is continuous. Gap: this is a closure test after convergence is known; it does not explain how to obtain a uniformly convergent subsequence.

## Evaluation settings

The artifact is a theorem and proof. The natural setting is a compact metric domain `K`, a complete metric target `Y`, and a family `F subset C(K,Y)` with the uniform metric. For real-valued functions, the pointwise precompactness condition specializes to pointwise boundedness.

Stress cases include pointwise bounded but non-equicontinuous families such as `x^n` on `[0,1]`, uniformly bounded rapidly oscillating families such as `sin(nx)` on a compact interval, compact target spaces where pointwise precompactness is automatic, and noncompact domains where finite domain nets need not exist.

Success means a criterion equivalent to relative compactness in the uniform metric. The proof should explain both the finite-net mechanism behind total boundedness and the diagonal subsequence mechanism behind uniform convergence.

## Proof artifact

The final artifact should state the compact-metric form: a family `F subset C(K,Y)` has compact closure in the uniform metric exactly when it is equicontinuous and each point-evaluation set `{f(x): f in F}` has compact closure in `Y`.

The proof should show necessity by applying compactness through the continuous evaluation maps and by using a finite uniform-net of functions to get equicontinuity. It should show sufficiency by choosing a finite net in the domain from equicontinuity, choosing finite target nets at those sample points from pointwise precompactness, and assigning one representative function to each occupied finite pattern. That produces a finite `epsilon`-net in the sup norm.

The sequential version should make diagonal extraction visible: converge on a countable dense subset, then use equicontinuity to upgrade convergence on that dense set to uniform Cauchy convergence on all of `K`.
