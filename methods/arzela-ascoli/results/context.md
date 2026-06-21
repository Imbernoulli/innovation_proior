## Research question

Compactness is well understood for a single compact domain. It is less clear what it means for a set of continuous functions. The question is when a family `F` of continuous functions on a compact metric space behaves like a compact set in the uniform metric

`d_infty(f,g)=sup_x d_Y(f(x),g(x))`.

Finite-dimensional intuition suggests checking coordinates: if the values of every function stay in bounded or precompact sets at each point of the domain, then every fixed coordinate looks compact. But a function is not just its values at isolated points, so it is not obvious what condition on a family of continuous functions is equivalent to relative compactness in the uniform metric.

## Background

For a compact metric space `K` and a metric target `Y`, the continuous maps `C(K,Y)` carry the uniform metric. When `Y=R` or `R^n`, boundedness of a single continuous function follows from compactness of `K`; for a whole family, pointwise boundedness means that for each fixed `x`, the set `{f(x): f in F}` is bounded.

On `[0,1]`, the functions `f_n(x)=x^n` are uniformly bounded by `1`, while their transition near `1` becomes sharper as `n` grows. The pointwise limit is `0` on `[0,1)` and `1` at `1`. The functions `sin(nx)` on a compact interval are likewise uniformly bounded while oscillating ever faster.

Uniform convergence of continuous functions preserves continuity: a uniform limit of continuous functions is continuous. Compactness of the domain supplies finite sampling of `K`: a finite net in the domain reduces a uniform-norm question to finitely many value questions, and pointwise precompactness supplies finite nets in the target at those sampled points.

## Baselines

- **Pointwise Bolzano-Weierstrass.** At a fixed point `x`, if `{f(x): f in F}` has compact closure, any sequence of functions has a subsequence whose values at `x` converge.

- **Diagonal extraction on a countable dense set.** In a compact metric domain, choose a countable dense set and extract a subsequence whose values converge at each selected point.

- **Pointwise boundedness.** For real-valued functions, bounded value sets `{f(x): f in F}` at each `x`.

- **Uniform boundedness in sup norm.** A common bound `||f||_infty <= M`, stronger than pointwise boundedness.

- **Uniform convergence theorem.** A uniform limit of continuous functions is continuous.

## Evaluation settings

The artifact is a theorem and proof. The natural setting is a compact metric domain `K`, a complete metric target `Y`, and a family `F subset C(K,Y)` with the uniform metric. For real-valued functions, the pointwise precompactness condition specializes to pointwise boundedness.

Test cases include pointwise bounded families such as `x^n` on `[0,1]`, uniformly bounded rapidly oscillating families such as `sin(nx)` on a compact interval, compact target spaces where pointwise precompactness is automatic, and noncompact domains.

Success means a criterion equivalent to relative compactness in the uniform metric. The proof should explain both the finite-net mechanism behind total boundedness and the diagonal subsequence mechanism behind uniform convergence.

## Proof artifact

The final artifact should state a criterion, in compact-metric form, for when a family `F subset C(K,Y)` has compact closure in the uniform metric, in terms of conditions on the point-evaluation sets `{f(x): f in F}` and on the joint behavior of the family across nearby domain points.

The proof should establish necessity by applying compactness through the continuous evaluation maps and by using a finite uniform-net of functions. It should establish sufficiency by choosing a finite net in the domain, choosing finite target nets at those sample points from pointwise precompactness, and assigning one representative function to each occupied finite pattern, producing a finite `epsilon`-net in the sup norm.

The sequential version should make diagonal extraction visible: converge on a countable dense subset, then upgrade convergence on that dense set to uniform Cauchy convergence on all of `K`.
