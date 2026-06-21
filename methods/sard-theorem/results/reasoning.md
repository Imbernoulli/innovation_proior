I keep staring at the wrong set first. The critical points can be enormous. If `f` is constant, every point in the domain is critical, so any theorem that tries to say "the critical points are small" is dead before it starts. But the constant example also says something else: all those critical points collapse to one target value. The domain set can be large while its image is tiny. So the object to measure has to be the set of critical values, not the critical set.

What does differentiability buy me at one critical point? The derivative is a linear map `Df_x : R^n -> R^m` whose image has dimension `< m`. To first order, a small ball around `x` is sent into a thin neighborhood of a proper linear subspace of the target. That is the geometric pressure I need: rank deficiency squeezes target volume. If the derivative is actually zero, the first-order part vanishes altogether and Taylor's formula says the image diameter is controlled by a higher-order remainder. If a box has side length `delta`, then the image of the relevant points in that box should be much smaller than a full `m`-dimensional box of side `delta`.

The tempting false proof is to say: at every critical point the linear image has zero `m`-volume, so the nonlinear image has zero volume. That is only pointwise. A union of infinitely many zero-volume tangent planes can fill volume if I do not control how they vary. I need a covering argument. Cover the domain by small cubes, show that the critical values coming from each cube fit inside target balls or tubes whose total `m`-volume can be made arbitrarily small, then sum over the cover. Differentiability has to enter through a uniform Taylor estimate on compact pieces, not just through the derivative at a point.

I start with the easiest rank-deficient situation: rank zero. Suppose all first derivatives vanish on a compact set `K_1`. If I only know `C^1`, then for `y` near a point `x in K_1`, `f(y)-f(x)` is `o(|y-x|)`. A cube of side `delta` meeting `K_1` has its relevant image inside a target ball of radius `epsilon(delta) delta`, with `epsilon(delta) -> 0`. There are about `delta^{-n}` such cubes. The total `m`-volume estimate looks like

`delta^{-n} (epsilon(delta) delta)^m = epsilon(delta)^m delta^{m-n}`.

This is enough when `m > n`, but it is not enough in general. If `m <= n`, the power of `delta` does not save me. I need more vanishing, or I need to reduce dimension.

Smoothness gives the way out. Define `K_j` to be the set where every positive-order partial derivative of every component of `f` through order `j` vanishes. On `K_j`, Taylor's formula for `f(y)-f(x)` has no terms through order `j`, so on a cube of side `delta` the image radius is controlled by a higher-order remainder. If I choose `j` so large that `j m > n`, then the covering estimate becomes

`delta^{-n} O(delta^{j m})`,

which tends to zero. Thus the deepest flat part, where many derivatives vanish, has critical image of measure zero. This is the cleanest form of the squeeze: enough differentiability turns high-order flatness into more target-volume decay than the number of boxes can spend.

But not every rank-zero critical point lies in that deepest flat set. Suppose a point lies in `K_j` but not in `K_{j+1}`. Then some derivative of order `j+1` is nonzero there. Equivalently, one of the order-`j` derivative functions has nonzero gradient. Since points of `K_j` make that order-`j` derivative equal zero, the implicit function theorem traps `K_j` locally inside a smooth hypersurface. That is a dimension drop. I can parametrize the hypersurface by `n-1` variables and apply the same critical-value statement to the restriction. So the rank-zero part splits into two kinds of pieces: either enough derivatives vanish and Taylor covering kills the image directly, or a first nonvanishing higher derivative cuts the problem down to a hypersurface.

Now I have to handle positive deficient rank, not just rank zero. Suppose the derivative has rank `r` at a point, with `0 < r < m`. I can choose `r` target coordinates and `r` source coordinates so that the corresponding minor is nonzero. By the inverse function theorem applied to the map

`x -> (f_1(x), ..., f_r(x), x_{r+1}, ..., x_n)`,

I can use local coordinates in which `f` has the form

`F(u,v) = (u, G(u,v))`,

where `u in R^r`, `v in R^{n-r}`, and `G(u,v) in R^{m-r}`. In these coordinates the derivative matrix has an identity block in the first `r` target directions:

`DF = [[I, 0], [D_u G, D_v G]]`.

The rank of `DF` is `r + rank(D_v G)`. So `F` is critical exactly when `D_v G(u,v)` fails to be onto `R^{m-r}`. For each fixed `u`, I am looking at a smaller map

`G_u : R^{n-r} -> R^{m-r}`.

The critical values of `G_u` have zero `(m-r)`-dimensional measure by the lower-dimensional statement. Then the corresponding critical values of `F` sit in slices `{u} x R^{m-r}` whose bad part has zero measure inside each slice. Fubini turns the slice-by-slice zero measure into zero `m`-dimensional measure for the whole local critical-value set. This is the other main mechanism: wherever any rank survives, straighten it into harmless coordinates and push the real difficulty into the remaining transverse variables.

The proof is beginning to organize itself. Localize to compact boxes so all estimates are uniform. At each critical point, if the rank is positive, straighten the positive-rank block and reduce by slicing. If the rank is zero, split by derivative vanishing order. The pieces where some higher derivative first appears lie in hypersurfaces, so induction on the source dimension handles them. The piece where sufficiently many derivatives vanish is killed by Taylor expansion and a covering estimate. Countable unions preserve measure zero, and the whole critical-value set is covered.

Let me check the rank-zero Taylor estimate carefully, because that is where the volume-squeezing insight actually earns its keep. Take a compact cube `Q_0` in the domain and a set `E` inside it on which all derivatives of order `< k` vanish, with `k m > n`. Uniform Taylor control says: for every `epsilon > 0`, if a subcube `Q` has side `delta` small enough and meets `E` at `x_Q`, then for every `y in E cap Q`,

`|f(y) - f(x_Q)| <= epsilon delta^k`.

So `f(E cap Q)` lies in one target ball of radius `epsilon delta^k`. The number of subcubes needed to cover `Q_0` is at most `C delta^{-n}`. The total `m`-volume of the target balls is bounded by

`C delta^{-n} (epsilon delta^k)^m = C epsilon^m delta^{k m - n}`.

Since `k m > n`, this can be made arbitrarily small. The image `f(E)` has measure zero. This estimate is the whole theorem in miniature: the domain may need many boxes, but differentiability makes each image box so thin that the total target volume still vanishes.

There is one subtle point in that estimate. I am covering only `E`, not all of `Q_0`, and I use a point `x_Q in E cap Q` as the Taylor center. That matters because the derivatives vanish on `E`, not necessarily everywhere in the cube. Taylor around `x_Q` controls the values of `f` at the other points of `E cap Q`; I do not need to control the image of the entire cube. The critical set can be rough, but the cover only asks for one center from the set in each cube.

Now the manifold version is no longer a separate idea. Criticality and measure-zero target sets are local in charts. Cover the source manifold by countably many coordinate charts and the target by charts. In a chart, the map is Euclidean, so the image of the critical points from that source chart has measure zero in the corresponding target chart. A countable union is still measure zero. Thus for a smooth map between manifolds, the critical values form a measure-zero subset of the target.

The regular-value consequences fall out immediately. If `y` is not a critical value, then every `x in f^{-1}(y)` has `Df_x` onto. The submersion theorem gives `f^{-1}(y)` the structure of a smooth submanifold of dimension `n-m` when `n >= m`; if `n < m`, a regular value simply has empty preimage. Since the bad values have measure zero, almost every target value is regular. And this is exactly why transversality arguments can work by perturbing a parameter or a value: the non-transverse failures are packaged as critical values of an auxiliary map, and this theorem says those failures occupy no volume.

So the final statement is not that singular behavior cannot occupy the domain. It can. The final statement is sharper and more useful: differentiability prevents rank-deficient behavior from occupying target volume. Critical points may be large, but their image is small. The proof is the repeated enforcement of that sentence: positive rank is straightened and sliced away, zero rank is flattened by Taylor expansion, higher-order obstructions either cut dimension or increase the power in the covering estimate, and the countable cover turns local squeezing into measure zero for all critical values.
