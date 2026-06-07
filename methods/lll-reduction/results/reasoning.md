I have a lattice `L` of rank `n` sitting in `R^m`, handed to me as some integer basis `b_1, ..., b_n`, and the basis I get is ugly — long vectors pointing in nearly the same directions, the kind a careless construction spits out. I want to massage it, using only moves that keep it a basis of the *same* lattice, into a basis whose vectors are short and roughly orthogonal, and above all whose *first* vector is short. And I want a guarantee, and I want it in polynomial time. Let me think about what "keeping it a basis" even allows and what "short" can mean.

The legal moves are exactly the unimodular ones: I can add an integer multiple of one basis vector to another, and I can permute (or negate) vectors. Each such move is an integer matrix of determinant `±1`, so the lattice is preserved. That's a lot of freedom — infinitely many bases of the same `L`, related by `GL_n(Z)`. The trouble is that "short" is not invariant under this freedom; only the covolume `d(L) = |det(b_1,...,b_n)|` is invariant. So whatever quality measure I cook up, it has to be one the unimodular group can't cheat.

Where do I even get a notion of length that respects the lattice? The raw lengths `|b_i|` are useless as a target — I can make them huge by adding multiples of one vector to another without changing the lattice at all. I need something more intrinsic. Gram-Schmidt. Take the basis and orthogonalize it: `b*_i = b_i - sum_{j<i} mu_{i,j} b*_j` with `mu_{i,j} = <b_i, b*_j>/<b*_j, b*_j>`, so `b*_i` is the part of `b_i` orthogonal to everything before it. Two things about this sequence grab me. The `b*_i` are orthogonal, so `prod_i |b*_i| = |det(b*_1,...,b*_n)| = |det(b_1,...,b_n)| = d(L)` — the product of the GS lengths *is* the invariant covolume. And if I write any lattice vector `x = sum r_i b_i` with integer `r_i`, and re-express it in the `b*` coordinates `x = sum r'_i b*_i`, the change of basis from `b` to `b*` is upper-triangular with `1`s on the diagonal, so for the *largest* index `i` with `r_i != 0` I get `r'_i = r_i` — an integer, nonzero. Hence `|x|^2 >= r'^2_i |b*_i|^2 >= |b*_i|^2`. So `|b*_i|` is a genuine lower bound on the length of any lattice vector that "reaches" coordinate `i`.

That reframes everything. The `b*_i` are not in the lattice — they're auxiliary — but their *lengths* are the real currency. A basis is good precisely when the GS lengths `|b*_1|, |b*_2|, ...` don't crash downward as `i` increases. If `|b*_i|` stays comparable to `|b*_{i-1}|` all the way along, then by the bound above every nonzero lattice vector is at least roughly `|b*_1|`, so `b_1` (whose length starts at `|b*_1|`) is essentially as short as anything in `L`. Conversely a *bad* basis is one where `|b*_i|` plummets: then there's room for some lattice vector to be far shorter than `b_1`, and I've been fooled.

So the goal sharpens: rearrange the basis so that (a) each `b_i` is reduced against the earlier ones — no wasteful long component lying along `b*_j` for `j<i` — and (b) the GS lengths decrease slowly. Let me handle (a) first because it's the obvious cleanup. The component of `b_i` along `b*_j` is measured by exactly `mu_{i,j}`. If `|mu_{i,j}|` is large, `b_i` has a big slug of `b_j` (well, of `b*_j`) in it that I should subtract off. But I can only subtract *integer* multiples of `b_j`. The best I can do is `b_i <- b_i - round(mu_{i,j}) b_j`, which replaces `mu_{i,j}` by `mu_{i,j} - round(mu_{i,j})`, landing it in `[-1/2, 1/2]`. I can't push it below `1/2` in absolute value with a single integer move — rounding to the nearest integer is optimal. So the natural cleanliness demand is `|mu_{i,j}| <= 1/2` for all `j < i`. Call that *size-reduced*. And note: subtracting a multiple of `b_j` (with `j<i`) doesn't touch any `b*` at all — `b*_i` is already orthogonal to `b*_j`, so its length is unchanged. Size-reduction is "free" in the GS-length currency. Good to know.

Now (b), the part that actually controls shortness: I want `|b*_i|` not to crash relative to `|b*_{i-1}|`. The cleanest demand would be `|b*_i| >= |b*_{i-1}|`, i.e. GS lengths never decrease. But can I always achieve that by local moves? Think about what move could possibly raise `|b*_i|` at the expense of `|b*_{i-1}|`. The only `b*`-changing move available is swapping two adjacent vectors `b_{i-1}` and `b_i`. Let me see exactly what a swap does.

Look at `b_{i-1}` and `b_i` projected onto the orthogonal complement of `span(b_1,...,b_{i-2})`. In that plane, `b_{i-1}` projects to `b*_{i-1}` and `b_i` projects to `b*_i + mu_{i,i-1} b*_{i-1}` (its `b*_{i-1}`-component plus its orthogonal part `b*_i`). If I swap the two vectors, the new first-of-the-pair is the old `b_i`, so the new `b*_{i-1}` becomes the projection of old `b_i` into that plane — that's `b*_i + mu_{i,i-1} b*_{i-1}`. This is a rank-2 picture: in the projected plane I have exactly the Gauss/Lagrange situation, two vectors, and swapping is the Gauss swap.

So here's the thing I keep circling: in rank 2 the Gauss algorithm — size-reduce then swap while the second is shorter than the first — terminates and gives the optimal basis, because each swap strictly shrinks the product of the two lengths. Can't I just run that locally at every adjacent pair? Reduce `b_i` against `b_{i-1}`, and if the new `b*_{i-1}` (after a hypothetical swap) would be shorter than the current `b*_{i-1}`, swap. The condition "swap improves things" is: `|b*_i + mu_{i,i-1} b*_{i-1}|^2 < |b*_{i-1}|^2`.

Let me try demanding the *negation* everywhere as my reducedness condition: `|b*_i + mu_{i,i-1} b*_{i-1}|^2 >= |b*_{i-1}|^2` for all `i`. If this holds, no adjacent swap helps. Does it terminate, though? I worry. The issue is the boundary: when the swap makes things *almost* but not quite no shorter, I could swap, then a size-reduction elsewhere nudges a coefficient, and I swap back, and I oscillate forever, each swap shrinking the product by an amount that tends to zero. The Gauss proof in pure rank 2 dodges this because the lengths live on a discrete lattice and the product can't shrink by arbitrarily little — but here I have `n` interacting GS lengths and size-reductions sloshing components around, and a swap that shrinks `|b*_{i-1}|^2` by a factor arbitrarily close to `1` gives me no quantitative grip. I can't bound the number of swaps. Wall.

So strict improvement (`>=`) is too weak to prove termination, and I can't simply ask for `|b*_i| >= |b*_{i-1}|` either, because that's a global ordering constraint that local swaps may never be able to satisfy simultaneously across all `i` — there may be no reachable basis with monotone GS lengths, or reaching it may take superpolynomially many moves. I need to relax the swap condition: only swap when the improvement is *substantial*, by at least a constant factor. Then every swap shrinks something by a definite amount, and if I can find a global quantity that drops by that same constant factor at each swap and is bounded below, the swaps must stop.

So let me put a slack constant `delta` into the swap criterion. I'll declare adjacent vectors acceptable when

    |b*_i + mu_{i,i-1} b*_{i-1}|^2 >= delta * |b*_{i-1}|^2,

and only swap when this *fails*, i.e. when the swap would shrink the leading GS length of the pair to below `delta` times its current value — a genuine, by-a-factor improvement. This is the Lovász condition. I need `delta < 1` for there to be any slack (so that swaps are by a real factor and the process can stop), and I'll want `delta` not too small or the guarantee at the end will be weak. Let me carry `delta` symbolically and pin it later; `1/4 < delta < 1` is the range I expect to need, and `3/4` looks like a clean middle value. Let me see if it earns its keep.

First, let me get the condition into a form I can compute with. Because `b*_i` is orthogonal to `b*_{i-1}`,

    |b*_i + mu_{i,i-1} b*_{i-1}|^2 = |b*_i|^2 + mu_{i,i-1}^2 |b*_{i-1}|^2.

So the Lovász condition `|b*_i|^2 + mu_{i,i-1}^2 |b*_{i-1}|^2 >= delta |b*_{i-1}|^2` is

    |b*_i|^2 >= (delta - mu_{i,i-1}^2) |b*_{i-1}|^2.

Nice — both sides are things I already track (`|b*_i|^2` and `mu_{i,i-1}`). And notice the interplay with size-reduction: if I've first size-reduced so that `|mu_{i,i-1}| <= 1/2`, then `mu_{i,i-1}^2 <= 1/4`, so when the condition *holds* I get `|b*_i|^2 >= (delta - 1/4)|b*_{i-1}|^2`. With `delta = 3/4` that's `|b*_i|^2 >= (1/2)|b*_{i-1}|^2`: consecutive GS lengths can halve in square but no faster. That's the slow-decay guarantee I wanted in (b), and it's exactly why I size-reduce `mu_{i,i-1}` *before* testing — the test only bites if the orthogonal part `|b*_i|` is genuinely small, not merely because `b_i` had a large removable component along `b_{i-1}`.

Now termination. I need a potential that drops by a constant factor at each swap and is bounded below. The covolume `d(L) = prod|b*_i|` is invariant — useless. But partial products are not invariant. Define, for `0 <= i <= n`, the Gram determinant of the first `i` vectors,

    d_i = det( <b_j, b_l> )_{1<=j,l<=i}.

Since `<b_j,b_l>` in the `b*` coordinates is `sum_p mu_{j,p} mu_{l,p} |b*_p|^2` and the matrix is upper-triangularizable, this determinant collapses to `d_i = prod_{j<=i} |b*_j|^2`. (Check the ends: `d_0 = 1`, `d_n = d(L)^2`.) Each `d_i` is the squared covolume of the rank-`i` sublattice spanned by `b_1,...,b_i`. Now stack them:

    D = prod_{i=1}^{n-1} d_i.

What does a swap do to `D`? A swap at adjacent indices `(k-1, k)` only changes `b*_{k-1}` and `b*_k`, and it changes them so their *product of lengths is preserved* — the pair's covolume `|b*_{k-1}||b*_k|` is invariant (it's the area they span, unchanged by reshuffling them). So every `d_i` with `i != k-1` is untouched: for `i < k-1` the first `i` vectors aren't involved; for `i >= k` the product `prod_{j<=i}|b*_j|^2` includes *both* changed lengths and so is invariant. Only `d_{k-1}` moves, because it includes `|b*_{k-1}|^2` but not `|b*_k|^2`. And the new `|b*_{k-1}|^2` is exactly the post-swap leading length `|b*_k + mu_{k,k-1} b*_{k-1}|^2`, which the swap was triggered to make `< delta |b*_{k-1}|^2`. So `d_{k-1}` drops by a factor `< delta`, hence

    D_new < delta * D_old.

Every swap multiplies `D` by less than `delta < 1`. And size-reduction leaves every `b*_i` alone, so it leaves every `d_i`, hence `D`, completely unchanged. So `D` is a clean monotone potential: it only moves on swaps, and only downward, by at least the factor `delta`.

Is `D` bounded below? If the basis is integral — `b_i in Z^m` — then each `d_i` is a determinant of an integer matrix, so `d_i` is a positive integer, so `d_i >= 1`, so `D >= 1`. (More generally I just need the inner products `<b_i,b_j>` to be integers.) And how big can `D` start? Initially `|b_i|^2 <= B` gives `d_i <= B^i` (it's a product of `i` squared lengths, each at most `B` by Hadamard), so `D <= B^{1+2+...+(n-1)} = B^{n(n-1)/2}`. Therefore the number of swaps is at most

    log_{1/delta}( B^{n(n-1)/2} ) = O(n^2 log B),

a polynomial bound. That's the whole termination argument, and it's why the relaxation by a constant factor `delta` was essential: strict improvement gave swaps of unbounded smallness and no bound; the constant `delta < 1` turns each swap into a definite `>= log(1/delta)` drop in `log D`, and the integrality floor `D >= 1` stops it. (And independent of integrality, there's an intrinsic floor: each `d_i` is the squared covolume of a rank-`i` lattice, which by Minkowski's convex-body bound contains a nonzero vector of squared length `<= (4/3)^{(i-1)/2} d_i^{1/i}`, so `d_i >= (3/4)^{i(i-1)/2} m(L)^i` where `m(L)` is the minimum squared length in `L` — a positive lower bound depending only on `L`. So termination holds even over the reals, but the integer floor is the clean polynomial one.)

Now I have a definition (size-reduced plus the Lovász condition with `delta`) and a potential that proves any swap-and-reduce process terminates polynomially. Let me turn it into a concrete algorithm with a single sweeping pointer rather than rechecking all pairs blindly. Keep a current index `k`, starting at `k=2` (the first nontrivial pair), with the invariant that the prefix `b_1,...,b_{k-1}` is already fully size-reduced and already satisfies the Lovász condition at every internal step. At each `k`:

I first size-reduce `b_k` against its immediate predecessor only — drive `|mu_{k,k-1}| <= 1/2` by `b_k <- b_k - round(mu_{k,k-1}) b_{k-1}` (updating the affected `mu_{k,j}`). I do *only* this one before the test, because the test depends solely on `mu_{k,k-1}` and `|b*_k|^2`, and fully size-reducing `b_k` against `b_{k-2},...,b_1` would touch coefficients that the test ignores — better to defer that work until I know `k` is staying put. (That deferral is a real factor-of-`n` saving in the step count: do the cheap reduction, test, and only finish reducing in the branch where I advance.)

Then the Lovász test, `|b*_k|^2 >= (delta - mu_{k,k-1}^2)|b*_{k-1}|^2`:

- If it *fails* (and `k>1`): swap `b_{k-1}` and `b_k`. This is the substantial-improvement case — `D` drops by `< delta`. After the swap the GS data for indices `k-1, k` and the `mu` entries referencing them must be recomputed; I can do this with explicit update formulae, but the mathematically clean statement is just "recompute Gram-Schmidt", and the load-bearing fact is the one I already used: the new `b*_{k-1}` equals the old `b*_k + mu_{k,k-1} b*_{k-1}`, so the new `|b*_{k-1}|^2` is less than `delta` times the old. Then step `k <- max(k-1, 2)`: I retreat one, because the swap may have broken the Lovász condition at the pair `(k-2, k-1)` that I'd previously certified, so I must recheck there.

- If it *holds* (or `k=1`): now finish size-reducing `b_k` against `b_{k-2}, ..., b_1` so the whole prefix is size-reduced, and advance `k <- k+1`. The prefix invariant is restored at the larger `k`.

Terminate when `k = n+1`: every adjacent pair satisfies Lovász and every `b_i` is size-reduced, so the basis is reduced. Termination is guaranteed because each pass through the swap branch decreases `k` by one and drops `D` by `< delta` (boundedly many, `O(n^2 log B)`, such passes), while each pass through the advance branch increases `k` by one; `k` is trapped in `{1,...,n+1}`, so the number of advances exceeds the number of swaps by at most `n-1`, and the total step count is polynomial.

Let me now cash in the definition for the *quality* guarantees — the payoff that justifies the whole construction. Suppose the basis is reduced with `delta = 3/4`. From size-reduction `mu_{i,i-1}^2 <= 1/4` and the Lovász condition,

    |b*_i|^2 >= (3/4 - 1/4)|b*_{i-1}|^2 = (1/2)|b*_{i-1}|^2,

so by induction `|b*_j|^2 <= 2^{i-j}|b*_i|^2` for `j <= i`. Then for the raw vector `b_i`, since `|b_i|^2 = |b*_i|^2 + sum_{j<i} mu_{i,j}^2 |b*_j|^2` and `mu_{i,j}^2 <= 1/4`,

    |b_i|^2 <= |b*_i|^2 + (1/4) sum_{j<i} |b*_j|^2 <= |b*_i|^2 ( 1 + (1/4) sum_{j<i} 2^{i-j} )
            = |b*_i|^2 ( 1 + (1/4)(2^i - 2) ) <= 2^{i-1} |b*_i|^2,

and combining with `|b*_j|^2 <= 2^{i-j}|b*_i|^2` gives `|b_j|^2 <= 2^{i-1}|b*_i|^2` for `j <= i`. Two consequences fall out.

Take the product over `i=1,...,n` of `|b_i| <= 2^{(i-1)/2}|b*_i|` and use `prod|b*_i| = d(L)`: `prod_i |b_i| <= 2^{n(n-1)/4} d(L)`. And in particular, putting `j=1` in `|b_j|^2 <= 2^{i-1}|b*_i|^2` and taking the product over all `i`, `|b_1|^{2n} <= 2^{0+1+...+(n-1)} prod|b*_i|^2 = 2^{n(n-1)/2} d(L)^2`, so

    |b_1| <= 2^{(n-1)/4} d(L)^{1/n}.

`b_1` is within `2^{(n-1)/4}` of the covolume-normalized scale `d(L)^{1/n}`. Now the headline guarantee, on shortness relative to the *true* shortest vector. Take any nonzero `x in L` and write `x = sum r_i b_i = sum r'_i b*_i` with integer `r_i`. Let `i` be the largest index with `r_i != 0`; as established, `r'_i = r_i`, a nonzero integer, so `|x|^2 >= r'^2_i |b*_i|^2 >= |b*_i|^2`. And `|b_1|^2 <= 2^{i-1}|b*_i|^2 <= 2^{n-1}|b*_i|^2 <= 2^{n-1}|x|^2`. Therefore

    |b_1| <= 2^{(n-1)/2} |x|   for every nonzero x in L,

i.e. `b_1` is within `2^{(n-1)/2}` of the shortest vector `lambda_1(L)`. That is the deliverable: a polynomial-time-computable basis whose first vector approximates SVP to within a factor exponential in `n` but with base `sqrt 2` — and the same argument extends to bound `b_j` by `2^{(n-1)/2}` times the `j`-th successive minimum, so the whole reduced basis is short, not just `b_1`.

This is exactly where `delta`'s value shows its hand. The two guarantees carry the factor `4/(4 delta - 1)` per index — for `delta = 3/4` that's `4/2 = 2`, the clean base-2 bounds above. Pushing `delta` toward `1` shrinks `4/(4 delta - 1)` toward `4/3`, a better approximation factor; but the per-swap potential drop is by the factor `delta`, so `delta -> 1` makes `log(1/delta) -> 0` and the swap bound `O((n^2 log B)/log(1/delta))` blows up — and at `delta = 1` the termination proof collapses entirely (zero guaranteed drop). And `delta -> 1/4` from above makes `4 delta - 1 -> 0`, so the guarantee factor explodes. So `delta` is a genuine quality/speed dial constrained to `(1/4, 1)`, and `3/4` is the convenient sweet spot — strong base-2 bounds, a real constant potential drop. I'll fix `delta = 3/4`.

One more thing I should secure before coding: keeping the arithmetic exact and the integers from blowing up, since I'm calling this as a subroutine. The `mu_{i,j}` and `|b*_i|^2` are rationals, and the natural common denominators are the `d_i` themselves: from `|b*_i|^2 = d_i / d_{i-1}` and the triangular solve for the GS projection, `d_{i-1} b*_i` is an integer vector and `d_j mu_{i,j}` is an integer. So I can run everything over the integers by carrying the `d_i` as denominators; since no `d_i` ever increases, all denominators stay `<= B^i`, and a bound-chase on the `b*_i`, `b_i`, `mu_{i,j}` through the size-reduction and swap steps keeps every integer's bit-length `O(n log B)`. That makes the whole thing `O(n^4 log B)` arithmetic operations on `O(n log B)`-bit integers — polynomial in both `n` and `log B`, as required. For a clean self-contained implementation I'll let exact rationals (`Fraction`) carry the denominators automatically and recompute Gram-Schmidt after each integer move; that is mathematically identical to (1.2)/(1.3) holding throughout and sidesteps transcribing the explicit swap-update formulae, at the cost of a worse constant.

So the implementation mirrors the sweep exactly. Compute `b*` and `mu` by Gram-Schmidt. Hold `k`. Size-reduce `b_k` against `b_{k-1}` (and, when advancing, against the rest). Test `|b*_k|^2 >= (3/4 - mu_{k,k-1}^2)|b*_{k-1}|^2`. Advance on success, swap and retreat on failure. Stop at the end.

```python
from fractions import Fraction


def dot(u, v):
    return sum(Fraction(a) * Fraction(b) for a, b in zip(u, v))


def gram_schmidt(B):
    # b*_i = b_i - sum_{j<i} mu_{i,j} b*_j ;  mu_{i,j} = <b_i,b*_j>/<b*_j,b*_j>
    n = len(B)
    Bstar = [None] * n
    mu = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        Bstar[i] = [Fraction(x) for x in B[i]]
        for j in range(i):
            mu[i][j] = dot(B[i], Bstar[j]) / dot(Bstar[j], Bstar[j])
            Bstar[i] = [a - mu[i][j] * c for a, c in zip(Bstar[i], Bstar[j])]
    return Bstar, mu


def lll(B, delta=Fraction(3, 4)):
    B = [[Fraction(x) for x in row] for row in B]
    n = len(B)
    Bstar, mu = gram_schmidt(B)

    k = 1                                          # pointer (0-based here)
    while k < n:
        # size-reduce b_k against b_{k-1},...,b_0: drive every |mu_{k,j}| <= 1/2
        for j in range(k - 1, -1, -1):
            if abs(mu[k][j]) > Fraction(1, 2):
                r = round(mu[k][j])                # nearest integer
                B[k] = [a - r * b for a, b in zip(B[k], B[j])]
                Bstar, mu = gram_schmidt(B)        # keep (1.2)/(1.3) exact

        # Lovasz condition: |b*_k|^2 >= (delta - mu_{k,k-1}^2) |b*_{k-1}|^2
        if dot(Bstar[k], Bstar[k]) >= (delta - mu[k][k - 1] ** 2) * dot(Bstar[k - 1], Bstar[k - 1]):
            k += 1                                 # substantial decay absent: advance
        else:
            B[k], B[k - 1] = B[k - 1], B[k]        # swap shrinks |b*_{k-1}|^2 by < delta
            Bstar, mu = gram_schmidt(B)
            k = max(k - 1, 1)                       # retreat to recheck the prior pair

    return [[int(x) for x in row] for row in B]
```

The causal chain, end to end: a lattice has no canonical short basis and SVP is hard, but Gram-Schmidt hands me the basis-invariant lengths `|b*_i|` (their product is the covolume, and they lower-bound every lattice vector), so "good basis" means "GS lengths decay slowly"; demanding `|mu_{i,j}| <= 1/2` (size-reduction, the best a single integer subtraction can do) plus the slow-decay condition `|b*_i|^2 >= (delta - mu_{i,i-1}^2)|b*_{i-1}|^2` defines reducedness; insisting on a *constant-factor* `delta < 1` of improvement before swapping — rather than any improvement — is what makes the potential `D = prod_{i<n} d_i = prod_{i<n} prod_{j<=i}|b*_j|^2` drop by the fixed factor `delta` at every swap while size-reduction leaves it fixed, and the integrality floor `D >= 1` then caps the swaps at `O(n^2 log B)`, giving polynomial time; and the same constants that make termination work make `|b*_i|^2 >= (1/2)|b*_{i-1}|^2`, which propagates to `|b_1| <= 2^{(n-1)/2} lambda_1(L)` and `|b_1| <= 2^{(n-1)/4} d(L)^{1/n}`, with `delta = 3/4` chosen as the sweet spot of the quality factor `4/(4 delta - 1)` against the per-swap drop `delta`.
