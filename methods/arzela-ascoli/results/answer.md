# Arzela-Ascoli Theorem

Let `K` be a compact metric space and let `Y` be a complete metric space. Give `C(K,Y)` the uniform metric

`d_infty(f,g)=sup_{x in K} d_Y(f(x),g(x))`.

A family `F subset C(K,Y)` has compact closure in `C(K,Y)` if and only if:

1. `F` is equicontinuous: for every `epsilon>0` there is `delta>0` such that `d_K(x,y)<delta` implies `d_Y(f(x),f(y))<epsilon` for every `f in F`.
2. `F` is pointwise relatively compact: for every `x in K`, the set `{f(x): f in F}` has compact closure in `Y`.

For `Y=R^n`, the second condition is pointwise boundedness.

## Proof

Assume first that the closure of `F` is compact in the uniform metric. For each `x in K`, the evaluation map

`ev_x:C(K,Y)->Y`, `ev_x(f)=f(x)`

is continuous. Therefore `ev_x(F)` has compact closure in `Y`, so pointwise relative compactness holds.

To prove equicontinuity, fix `epsilon>0`. Since the closure of `F` is compact, `F` is totally bounded. Choose continuous functions `g_1,...,g_N` such that every `f in F` satisfies `d_infty(f,g_j)<epsilon/3` for some `j`. Each `g_j` is uniformly continuous on compact `K`. Since there are finitely many `g_j`, choose one `delta>0` such that

`d_K(x,y)<delta => d_Y(g_j(x),g_j(y))<epsilon/3`

for every `j`. If `f in F` and `d_infty(f,g_j)<epsilon/3`, then for `d_K(x,y)<delta`,

`d_Y(f(x),f(y)) <= d_Y(f(x),g_j(x)) + d_Y(g_j(x),g_j(y)) + d_Y(g_j(y),f(y)) < epsilon`.

Thus `F` is equicontinuous.

Conversely, assume `F` is equicontinuous and pointwise relatively compact. It is enough to show `F` is totally bounded in `C(K,Y)`. Indeed, `C(K,Y)` is complete when `Y` is complete: a uniform Cauchy sequence has pointwise limits in `Y`, the convergence to that pointwise limit is uniform, and the uniform limit of continuous functions is continuous. A totally bounded set in a complete metric space has compact closure.

Fix `epsilon>0` and put `eta=epsilon/4`. By equicontinuity, choose `delta>0` such that

`d_K(x,y)<delta => d_Y(f(x),f(y))<eta`

for every `f in F`. Compactness of `K` gives points `x_1,...,x_m` such that every `x in K` is within `delta` of some `x_i`.

For each `i`, pointwise relative compactness gives a finite set `E_i subset Y` such that for every `f in F` there is `a_i in E_i` with

`d_Y(f(x_i),a_i)<eta`.

There are finitely many tuples `(a_1,...,a_m) in E_1 x ... x E_m`. For each tuple that is realized by at least one function `g in F` satisfying

`d_Y(g(x_i),a_i)<eta` for all `i`,

choose one such representative `g`. Let `G` be the finite set of all chosen representatives.

Take any `f in F`. Choose a tuple `(a_1,...,a_m)` with `d_Y(f(x_i),a_i)<eta` for all `i`, and let `g in G` be the representative chosen for that tuple. For any `x in K`, choose `x_i` with `d_K(x,x_i)<delta`. Then

`d_Y(f(x),g(x)) <= d_Y(f(x),f(x_i)) + d_Y(f(x_i),a_i) + d_Y(a_i,g(x_i)) + d_Y(g(x_i),g(x))`

`< eta + eta + eta + eta = epsilon`.

So every `f in F` lies within `epsilon` of some member of the finite set `G`. Hence `F` is totally bounded in the uniform metric, and its closure is compact.

The sequential diagonal form follows from the same mechanism. Given a sequence in `F`, choose a countable dense set in `K`. Pointwise relative compactness lets one extract a diagonal subsequence converging at every dense-set point. Equicontinuity upgrades convergence on finitely many dense samples to a uniform Cauchy estimate on all of `K`; completeness of `Y` gives the uniform limit in `C(K,Y)`.
