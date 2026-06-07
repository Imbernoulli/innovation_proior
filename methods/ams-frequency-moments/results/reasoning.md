OK, let me think this through from scratch. I have a stream of values `a_1, ..., a_m`, each one in `{1, ..., n}`, flying past me one at a time, and I'm not allowed to store them. Value `i` shows up `m_i` times, and I want `F_2 = sum_i m_i^2`. Why `F_2` specifically? Because it's the repeat rate — and concretely it's the size of a self-join: if a database relation `r` has the value `i` appearing `m_i` times in its join column, then `r ⋈ r` produces `sum_i m_i^2` tuples, and the query optimizer badly wants that number before it commits to a plan. So `F_2` is the quantity that measures skew, and skew is what decides everything downstream. I'd love to compute it in one pass over the relation without materializing anything.

The obvious thing: keep a counter `m_i` for every value `i`, and at the end sum the squares. That's exact, one pass, done. But it costs one counter per universe element — `Θ(n log m)` bits. When `n` is huge (the domain of a join attribute), that histogram doesn't fit in fast memory; it spills to disk and every update pays a disk access. The whole point was to *avoid* that. So the histogram is out. I need something whose state is tiny — polylog in `n` and `m` — and the only way that's possible is to give up exactness and give up determinism: estimate, randomly, with some controlled error.

So the question becomes: is there a random variable `X` that (a) I can maintain in one pass with tiny memory, (b) has `E[X] = F_2`, and (c) has small enough variance that I can pin it down? If I can find that, Chebyshev does the rest.

Let me think about what "tiny state, one-pass" really forces. If my state is a single scalar `Z`, and I want to update it in `O(1)` per token without remembering anything, the natural thing is for `Z` to be a **linear function** of the `m_i`: `Z = sum_i c_i m_i` for some fixed coefficients `c_i`. Then each token `a_j = i` just does `Z += c_i`, and at the end `Z` is exactly that linear combination, regardless of arrival order. Linear, incremental, mergeable — perfect for streaming. The whole question is: what coefficients `c_i` let me recover `sum_i m_i^2` from such a linear `Z`?

First dumb attempt: `c_i = 1` for all `i`. Then `Z = sum_i m_i = m = F_1`. That's the length of the stream, not the sum of squares. The linear sum just collapses everything additively and I've lost the squaring entirely. A single linear functional of the `m_i` is, by itself, never going to *be* `sum m_i^2` — that's quadratic. So a deterministic linear sketch can't do it.

But here's the thing — I don't need `Z` itself to be `F_2`. I'm allowed randomness, and I'm allowed to *post-process* `Z`. The squaring I want is `(...)^2`. What if I square `Z`? `Z^2 = (sum_i c_i m_i)^2 = sum_i c_i^2 m_i^2 + sum_{i≠j} c_i c_j m_i m_j`. Look at that: the first group `sum_i c_i^2 m_i^2` is exactly `F_2` if every `c_i^2 = 1`. The trouble is the second group, the cross terms `sum_{i≠j} c_i c_j m_i m_j` — garbage I don't want. With fixed deterministic coefficients those cross terms are just *there* and they corrupt the answer.

So I need `c_i^2 = 1` (to make the diagonal give `F_2`) and I need the cross terms to *vanish*. The first says each `c_i ∈ {-1, +1}` — a sign. The second is where randomness has to come in: if I make the signs **random**, then in expectation the cross terms can cancel. Let me set `c_i = ε_i`, a random `±1` for each value `i`, and define `Z = sum_i ε_i m_i`. On each token `i` I do `Z += ε_i`. The `±1` pulls `Z` up or down depending on the sign of the value that arrived — a tug-of-war between values that hash positive and values that hash negative. And `|Z| ≤ m` always, so `Z` fits in `O(log m)` bits. The signs themselves I'll generate from a small seed via a hash `ε_i = h(i)`, costing `O(log n)` bits, so I never store `n` signs.

Now take `X = Z^2` and compute its expectation. `Z^2 = sum_i ε_i^2 m_i^2 + sum_{i≠j} ε_i ε_j m_i m_j`. Since `ε_i ∈ {-1, +1}`, `ε_i^2 = 1` deterministically, so the diagonal is `sum_i m_i^2 = F_2`, no expectation needed. For the cross terms, `E[ε_i ε_j]` with `i ≠ j`: if the signs are **mean-zero** (`E[ε_i] = 0`, i.e. `+1` and `-1` equally likely) and **pairwise independent** (so `E[ε_i ε_j] = E[ε_i] E[ε_j]` for `i ≠ j`), then every cross term has expectation `0 · 0 = 0`. So

```
E[Z^2] = sum_i m_i^2 + 0 = F_2.
```

`Z^2` is an unbiased estimator of `F_2`, and I only needed the signs to be mean-zero and *pairwise* independent. That's already a small miracle: a single scalar `Z`, updated by one addition per token, whose square is `F_2` in expectation. It really does seem to do almost nothing.

Unbiased is necessary but not sufficient — an estimator with enormous variance is useless. I need `Var(X) = Var(Z^2) = E[Z^4] - (E[Z^2])^2 = E[Z^4] - F_2^2`. So I have to understand `E[Z^4]`.

```
Z^4 = (sum_i ε_i m_i)^4 = sum_{i,j,k,l} ε_i ε_j ε_k ε_l · m_i m_j m_k m_l.
```

Take expectations term by term. The key question for each 4-tuple `(i, j, k, l)` is: what is `E[ε_i ε_j ε_k ε_l]`? Suppose some index appears an *odd* number of times in the tuple — say `i` appears exactly once and is different from `j, k, l`. Then if I can factor `E[ε_i ε_j ε_k ε_l] = E[ε_i] · E[ε_j ε_k ε_l]`, the factor `E[ε_i] = 0` kills the whole term. So any 4-tuple with a "lonely" index contributes zero — *provided I'm allowed to pull `ε_i` out of the expectation as an independent factor*. That factorization, for four arbitrary distinct indices, is exactly the statement that the signs are **four-wise independent**. Pairwise independence won't license it: pulling `ε_i` out past three other distinct signs is a statement about four coordinates at once.

So under four-wise independence, the only surviving 4-tuples are the ones where the indices pair up so every `ε` appears to an even power (`ε^2 = 1`, `ε^4 = 1`). If all four indices are equal, `(i, i, i, i)` contributes `ε_i^4 m_i^4 = m_i^4`; summed over `i`, this is `sum_i m_i^4 = F_4`, with coefficient `1`. If two distinct values each appear twice, the multiset is `{i, i, j, j}` with `i ≠ j`; here `ε_i^2 ε_j^2 = 1`, so the contribution is `m_i^2 m_j^2`. How many ordered 4-tuples `(i,j,k,l)` realize this multiset? It is the number of arrangements of `(i, i, j, j)`, which is `4! / (2! 2!) = 6`. So this case contributes `6 sum_{i<j} m_i^2 m_j^2`.

Therefore

```
E[Z^4] = F_4 + 6 sum_{i<j} m_i^2 m_j^2.
```

Let me abbreviate `S = sum_{i<j} m_i^2 m_j^2`. Now I need `E[Z^4] - F_2^2`, so I have to expand `F_2^2`:

```
F_2^2 = (sum_i m_i^2)^2 = sum_i m_i^4 + 2 sum_{i<j} m_i^2 m_j^2 = F_4 + 2S.
```

Subtracting:

```
Var(Z^2) = (F_4 + 6S) - (F_4 + 2S) = 4S.
```

And I want to bound this against `F_2^2`. From `F_2^2 = F_4 + 2S` and `F_4 ≥ 0`, I get `2S ≤ F_2^2`. So

```
Var(Z^2) = 4S = 2 · (2S) ≤ 2 F_2^2.
```

The variance is at most twice the square of the thing I'm estimating. That's the clean bound — `Var(Z^2) ≤ 2 F_2^2` — and it's exactly the place where four-wise independence earned its keep. With only pairwise independence I could compute the *mean* but I'd have no control over `E[Z^4]`, because the lonely-index terms wouldn't provably vanish; the four-wise property is precisely what zeroes them.

Now, why four-wise and not full independence? Full independence over `n` points is `n` independent signs — `Ω(n)` random bits to store, which blows the whole budget; I'd be back to linear space just for the randomness. But four-wise is the *exact* degree of independence the calculation touches: the mean needs pairs, the variance needs quadruples, and nothing higher ever appears. And four-wise independent `±1` families are cheap. I can work over `GF(2^d)` with `2^d > n`, choose a random cubic polynomial `P(x) = a_0 + a_1 x + a_2 x^2 + a_3 x^3`, and define the sign by one nonzero binary coordinate of `P(i)`, say `ε_i = (-1)^{bit(P(i))}`. For any four distinct inputs, the four polynomial values are independent uniform field elements by interpolation through four points; taking the same nonzero linear bit of each value preserves independent unbiased bits, so the resulting signs are exactly four-wise independent and mean-zero. The BCH/orthogonal-array view is the same object in coding-theory language: an orthogonal array of strength `4` gives `±1` rows where every four coordinates realize all `16` sign patterns equally often. Either way the seed is `O(log n)` bits and I never materialize the `n` signs. So four-wise is both *necessary* (the variance proof needs it) and *sufficient* (nothing more is used) and *affordable* (small seed). That's why the design lands exactly there.

So where am I? One scalar `Z`, `O(log n + log m)` bits, `E[Z^2] = F_2`, `Var(Z^2) ≤ 2 F_2^2`. Let me see if a single copy is good enough. The relative standard deviation is `sqrt(Var)/E[X] = sqrt(2 F_2^2)/F_2 = sqrt(2) ≈ 1.41`. That's terrible — the estimator routinely lands at the wrong order of magnitude on any single run. Chebyshev with one copy gives `P(|Z^2 - F_2| > λ F_2) ≤ Var/(λ F_2)^2 = 2/λ^2`, which for any reasonable `λ < 1` is a useless bound bigger than `1`. So one copy is hopeless; I have to drive the variance down.

Standard move: run `s_1` independent copies — independent sign-hashes, independent scalars `Z^{(1)}, ..., Z^{(s_1)}` — and **average** their squares: `Y = (1/s_1) sum_t (Z^{(t)})^2`. Averaging keeps the mean at `F_2` and cuts the variance by `s_1`: `Var(Y) = Var(Z^2)/s_1 ≤ 2 F_2^2 / s_1`. Now Chebyshev:

```
P(|Y - F_2| > λ F_2) ≤ Var(Y)/(λ F_2)^2 ≤ 2 F_2^2 / (s_1 λ^2 F_2^2) = 2/(s_1 λ^2).
```

If I want this at most `1/8`, I set `s_1 = 16/λ^2`. So `O(1/λ^2)` copies buy me a single estimator that's within `λ F_2` with probability at least `7/8`.

Can I just push `s_1` higher to also get the failure probability down to a tiny `δ`? To reach failure probability `δ` purely by averaging, Chebyshev needs `2/(s_1 λ^2) ≤ δ`, i.e. `s_1 = O(1/(λ^2 δ))`. The `1/δ` dependence is bad — to be sure to one part in a thousand I'd pay a factor of a thousand in space. I want `log(1/δ)`, not `1/δ`.

The fix is a second stage. I have an estimator `Y` that's correct (within `λ F_2`) with probability `≥ 7/8` and wrong with probability `≤ 1/8`. Run `s_2` *independent* such averages `Y_1, ..., Y_{s_2}` and output their **median**. The median is outside `[F_2 - λF_2, F_2 + λF_2]` only if at least half of the `Y_i` are outside it. Each `Y_i` is outside independently with probability `≤ 1/8`, well below `1/2`, so "at least half fail" is a large deviation of a sum of independent indicators below their mean gap. A crude bound already gives `P(at least half fail) ≤ 2^{s_2}(1/8)^{s_2/2} = 2^{-s_2/2}`, so `s_2 = 2 log_2(1/δ)` is enough. The median gives me exponential confidence for only a `log(1/δ)` multiplicative cost. This is the median-of-means structure: average to shrink variance to a constant-factor guarantee, median to amplify confidence cheaply.

Tally the space. I keep `s_1 · s_2 = O(λ^{-2} log(1/δ))` scalars, each `O(log m)` bits, plus that many sign-hash seeds, each `O(log n)` bits. Total

```
O( (1/λ^2) log(1/δ) · (log n + log m) ) bits,
```

which is polylogarithmic in `n` and `m` — exactly the regime the histogram couldn't reach. For very large `m` I can even shave the per-scalar cost: maintain `Z` (or rather its positive and negative parts) with Morris-style approximate counters in `O(log log m)` bits each, trading a little accuracy in the running sum for space, and the analysis goes through with Chebyshev as before. So `F_2` falls to essentially logarithmic space. The thing I thought a single linear functional could never compute, a *randomized* linear functional computes after one squaring, because the randomness annihilates the cross terms and four-wise independence tames the fourth moment.

Now let me step back and ask the more general question, because the same idea-shape should give me other `F_k`. For `F_2` I exploited that squaring a signed sum reproduces squares. For general `k` I can't square my way to `k`-th powers, so I need a different unbiased gadget. Here's the move: I want a random variable whose expectation is `sum_i m_i^k`, computable in small space. Pick a uniformly random *position* `p ∈ {1, ..., m}` in the stream; say the value there is `a_p = l`. Then look forward and count `r = |{q ≥ p : a_q = l}|` — how many times `l` occurs from position `p` onward, including `p` itself. Define

```
X = m · (r^k - (r-1)^k).
```

I only need to remember the chosen value `l` (`log n` bits) and the running count `r` (`log m` bits). Why should this have expectation `F_k`? Condition on which *value* got picked. Picking a uniform position is the same as: pick value `i` with probability `m_i/m` (proportional to its frequency), then pick a uniformly random one of its `m_i` occurrences. If I picked the occurrence that is the `c`-th from the end among `i`'s occurrences (so `r = c`, equally likely `c = 1, ..., m_i`), then `X = m(c^k - (c-1)^k)`. Averaging over `c` uniform in `{1,...,m_i}`:

```
E[X | value i] = sum_{c=1}^{m_i} (1/m_i) · m (c^k - (c-1)^k) = (m/m_i) (m_i^k - 0^k) = (m/m_i) m_i^k,
```

because the sum `sum_c (c^k - (c-1)^k)` **telescopes** to `m_i^k - 0 = m_i^k`. Then weighting by the probability `m_i/m` of having picked value `i`:

```
E[X] = sum_i (m_i/m) · (m/m_i) m_i^k = sum_i m_i^k = F_k.
```

Unbiased again, and the telescoping is the engine. (One implementation wrinkle: I don't know `m` in advance to pick the uniform position. Reservoir sampling fixes it — keep the current candidate occurrence and, when the `m`-th token arrives, replace the stored value with it with probability `1/m`, resetting `r` to `1`, else extend `r` if the new token matches the stored value. After processing the whole stream the held position is uniform over `{1,...,m}`.)

For the variance I bound `E[X^2]`. By the same conditioning,

```
E[X^2] = m^2 sum_i (1/m) sum_{c=1}^{m_i} (c^k - (c-1)^k)^2 = m sum_i sum_{c=1}^{m_i} (c^k - (c-1)^k)^2.
```

Now I need to control `(c^k - (c-1)^k)^2`. Use the elementary factorization for `a > b > 0`: `a^k - b^k = (a-b)(a^{k-1} + a^{k-2} b + ... + b^{k-1}) ≤ (a-b) k a^{k-1}`, since each of the `k` terms in the bracket is at most `a^{k-1}`. With `a = c`, `b = c-1`, this gives `c^k - (c-1)^k ≤ k c^{k-1}`. So `(c^k - (c-1)^k)^2 ≤ k c^{k-1} (c^k - (c-1)^k)`, and summing over `c` the second factor telescopes:

```
sum_{c=1}^{m_i} (c^k - (c-1)^k)^2 ≤ k m_i^{k-1} sum_{c=1}^{m_i} (c^k - (c-1)^k) = k m_i^{k-1} · m_i^k = k m_i^{2k-1}.
```

Hence `E[X^2] ≤ k m sum_i m_i^{2k-1} = k F_1 F_{2k-1}`. The remaining job is to relate `F_1 F_{2k-1}` to `F_k^2`, because variance relative to `(E[X])^2 = F_k^2` is what controls the number of copies. Claim: `F_1 F_{2k-1} ≤ n^{1-1/k} F_k^2`, i.e. `(sum_i m_i)(sum_i m_i^{2k-1}) ≤ n^{1-1/k} (sum_i m_i^k)^2`. To see it, let `M = max_i m_i`. Then `M^k ≤ sum_i m_i^k = F_k`, so `M^{k-1} ≤ F_k^{(k-1)/k}`. Bounding `m_i^{2k-1} = m_i^{k-1} m_i^k ≤ M^{k-1} m_i^k`:

```
(sum_i m_i)(sum_i m_i^{2k-1}) ≤ (sum_i m_i) M^{k-1} (sum_i m_i^k)
   ≤ (sum_i m_i) F_k^{(k-1)/k} F_k = (sum_i m_i) F_k^{(2k-1)/k}.
```

And by the power-mean inequality `(sum_i m_i)/n ≤ ((sum_i m_i^k)/n)^{1/k}`, so `sum_i m_i ≤ n^{1-1/k} F_k^{1/k}`. Plugging in:

```
F_1 F_{2k-1} ≤ n^{1-1/k} F_k^{1/k} · F_k^{(2k-1)/k} = n^{1-1/k} F_k^2.
```

So `Var(X) ≤ E[X^2] ≤ k n^{1-1/k} F_k^2`. The relative variance is `k n^{1-1/k}`, so to push the per-estimator failure below `1/8` via Chebyshev I average `s_1 = 8 k n^{1-1/k}/λ^2` copies, then median over `s_2 = O(log(1/δ))` of them. Space `O(k λ^{-2} log(1/δ) · n^{1-1/k} (log n + log m))`. For `k = 2` that's `O(λ^{-2} √n ...)` — a `√n` factor, which is exactly why the squaring trick is special: the tug-of-war estimator gets `F_2` down to `log n`, beating the general scheme's `√n` for the one case where reproducing squares by squaring is available.

I should also handle `F_0` while I'm here, because the hashing tools are right there. Flajolet and Martin estimate the distinct count by watching trailing-zero patterns of hashed values, but their analysis assumed an idealized hash family I can't actually build. Let me show a plain *linear* hash suffices. Take `d` with `2^d > n`, view values as elements of `GF(2^d)`, pick random `a, b ∈ GF(2^d)`, and map each incoming value `x` to `z = a·x + b`. Let `r(z)` be the number of trailing zero bits of `z`, and track `R = max` over the stream of `r(z)`; output `Y = 2^R`. The only properties I need are that for each fixed `x`, `z` is uniform over `GF(2^d)` (so `P(r(z) ≥ r) = 2^{-r}`), and that the map is *pairwise* independent (so for distinct `x, x'`, `P(r(z) ≥ r and r(z') ≥ r) = 2^{-2r}`) — both hold for `a x + b`. Fix a level `r` and let `Z_r` count the distinct values whose hash has `r(z) ≥ r`. By linearity `E[Z_r] = F_0 / 2^r`, and by pairwise independence `Var(Z_r) = F_0 · 2^{-r}(1 - 2^{-r}) < F_0/2^r = E[Z_r]`. If `2^r > c F_0` then `E[Z_r] < 1/c` so by Markov `P(Z_r > 0) < 1/c` — the level is too high to be reached. If `c 2^r < F_0` then by Chebyshev `P(Z_r = 0) ≤ Var(Z_r)/E[Z_r]^2 < 1/E[Z_r] = 2^r/F_0 < 1/c` — the level is reached. Since `Y = 2^R` with `R` the largest reached level, `Y` is within a factor `c` of `F_0` except with probability `< 2/c`. All in `O(log n)` bits — and with an honest, constructible hash.

Now the other half of the story: how far down can space go? I should not just give upper bounds; I want to know `F_2`'s polylog and `F_k`'s `n^{1-1/k}` aren't leaving easy savings on the table — and especially whether the linear-`n` cost is unavoidable for the hard moments. The tool is communication complexity: a small-space streaming algorithm gives a cheap protocol, so if the protocol is provably expensive, the space is provably large.

Start with the easiest case, `F_∞* = max_i m_i`. Two parties hold subsets `x, y ⊆ {1,...,n}`; concatenate their characteristic elements into a stream `A`. The first party runs the streaming algorithm on its part, ships the `s` bits of memory to the second, who finishes. If the sets are disjoint every value occurs at most once, so `F_∞* = 1`; if they intersect, some value occurs twice, so `F_∞* = 2`. A streaming approximation of `F_∞*` within a factor better than `2` therefore decides **Disjointness** using only `s` bits of communication. But Disjointness needs `Ω(n)` bits (Kalyanasundaram–Schnitger; Razborov's distributional version). So `s = Ω(n)`: approximating `F_∞*` needs linear space. Note this used only a *one-way* message, so it even holds for a constant number of passes.

For `F_k` with `k > 5` the same philosophy but a multi-party game. Define `DIS(s,t)`: `s` players, each holding a `t`-subset `A_i` of a universe `N`, with `|N| = (2t-1)s + 1`; the inputs are arranged so the family is either pairwise *disjoint*, or *uniquely intersecting* (all `A_i` share one common element `x` and are otherwise disjoint), and the players must distinguish the two cases by passing messages, each writing the memory once. Map to streaming: concatenate `A_1, ..., A_s` into a stream and run the `F_k` algorithm, passing the `M`-bit memory player to player. If disjoint, all `st` streamed elements are distinct, so `F_k = st`. If uniquely intersecting, the shared `x` occurs `s` times and the other `s(t-1)` elements occur once, so `F_k = s^k + s(t-1)`. I need these two values separated by a constant factor; choosing `t` on the order of `s^{k-1}` does it, because then `st = Θ(s^k)` and `s^k + s(t-1) = Θ(s^k)` with a larger constant. A `0.1`-approximation detects that gap, so a small-space `F_k` algorithm yields a low-communication protocol for `DIS(s,t)`.

Then I lower-bound `DIS(s,t)`. Following Razborov's box/partition method: a fixed communication transcript corresponds to a combinatorial **box** `X^1 × ... × X^s` of inputs, and I want to show no small set of boxes can separate the disjoint from the uniquely-intersecting distribution. Put a distribution `μ`: choose a random partition of `N` into `s` blocks of size `2t-1` plus a special element `x`, let each `A_j` be a random `t`-subset of its block, and with the special twist that either none or all of the `A_j` grab `x` (equiprobably disjoint vs. uniquely-intersecting). The crux (Lemma 3.4): for any box, the probability mass it gives the uniquely-intersecting input is at least `1/(2e)` times the mass it gives the disjoint input, minus an error `s · 2^{-ct/s^3}`. This rests on calling a partition `j`-bad when conditioning on `x` would drop player `j`'s box-probability too much, and an entropy argument (Lemma 3.5) shows a random partition is `j`-bad with probability at most `1/(20s)`; on good partitions the per-player inequalities multiply (Lemma 3.6) using `(1 - 1/(s+1))^s > 1/e`. Summing over boxes, any protocol with fewer than `Ω(t/s^3)` bits errs with constant probability — so `DIS(s,t)` needs `Ω(t/s^3)` communication. Feeding back the streaming reduction: total communication `≤ sM`, so `sM ≥ Ω(t/s^3)`, giving `M ≥ Ω(t/s^4)`. Choosing `s = Θ(n^{1/k})`, `t = Θ(n^{1-1/k})` yields

```
M ≥ Ω(t/s^4) = Ω(n/s^5) = Ω(n^{1 - 5/k}).
```

So for `k > 5`, approximating `F_k` needs `n^{Ω(1)}` space — the trivial near-linear algorithms can't be beaten by much, and this holds even for a constant number of passes. That tells me the cheap polylog world really does stop somewhere above `k = 2`.

Two more sanity checks on what's essential. Is randomness necessary, or was I just unimaginative? Take any *deterministic* approximation of `F_k` for `k ≠ 1`. Build a family `G` of `2^{Ω(n)}` subsets of `N`, each of size `n/4`, any two sharing at most `n/8` elements (a coding-theory packing). Feed the algorithm `G_1` then `G_2`. The memory after reading `G_1` depends only on `G_1`; if memory had fewer than `log|G| = Ω(n)` bits, two distinct `G_1, G_2` collide in memory, forcing the same output on the streams `G_1 G_1` and `G_2 G_1` — but those have very different `F_k` (e.g. for `k ≥ 2`, `F_k = 2^k n/4` when both halves are equal versus `≤ n/4 + 2^k n/8` when they differ), so the answer is off by more than `10%` on one of them. Hence deterministic needs `Ω(n)` — randomness is genuinely required. And is *approximation* necessary, or could randomness alone compute `F_k` exactly cheaply? No: a randomized *exact* `F_k` (`k ≠ 1`) also reduces to Disjointness and needs `Ω(n)`. So both knobs — randomized *and* approximate — are forced; drop either and you're back to linear space.

Finally, the matching lower bounds for the cheap moments confirm I'm at the floor: `Ω(log n)` for `F_0` (an equality argument on the packing family), `Ω(log log m)` for `F_1` (the output must take `Ω(log m)` distinct values since `m` is unknown), and `Ω(log n + log log m)` for `F_2` — so the tug-of-war's logarithmic space is essentially optimal.

Let me write the `F_2` estimator the way I'd actually run it — the tug-of-war scalar, the four-wise sign hash, and median-of-means.

```python
import math, random, statistics

FIELD_BITS = 64
FIELD_MASK = (1 << FIELD_BITS) - 1
REDUCTION = 0x1B  # x^64 + x^4 + x^3 + x + 1


def gf_mul(a, b):
    z = 0
    a &= FIELD_MASK
    b &= FIELD_MASK
    while b:
        if b & 1:
            z ^= a
        b >>= 1
        carry = a >> (FIELD_BITS - 1)
        a = (a << 1) & FIELD_MASK
        if carry:
            a ^= REDUCTION
    return z


class StreamHash:
    # epsilon_i = +/-1 from a degree-3 polynomial over GF(2^64).
    # One output bit of four-wise independent field values gives exact
    # mean-zero, four-wise independent signs for the variance calculation.
    def __init__(self, rng):
        self.a = [rng.getrandbits(FIELD_BITS) for _ in range(4)]

    def __call__(self, x):
        if not 0 <= x <= FIELD_MASK:
            raise ValueError("token is outside the represented universe")
        a0, a1, a2, a3 = self.a
        v = gf_mul(a3, x) ^ a2
        v = gf_mul(v, x) ^ a1
        v = gf_mul(v, x) ^ a0
        return 1 if (v & 1) == 0 else -1


class StreamSummary:
    def __init__(self, eps, delta, seed=0):
        if eps <= 0:
            raise ValueError("eps must be positive")
        if not 0 < delta < 1:
            raise ValueError("delta must be in (0, 1)")
        rng = random.Random(seed)
        self.s1 = max(1, math.ceil(16 / (eps * eps)))         # average: 16/lambda^2
        self.s2 = max(1, math.ceil(2 * math.log2(1 / delta))) # median: 2 log_2(1/delta)
        self.hashes = [[StreamHash(rng) for _ in range(self.s1)] for _ in range(self.s2)]
        self.z = [[0] * self.s1 for _ in range(self.s2)]  # the linear sketch scalars Z

    def update(self, token, count=1):
        # Z += epsilon_token : one signed addition per copy; |Z| <= m so O(log m) bits.
        for r in range(self.s2):
            for c in range(self.s1):
                self.z[r][c] += count * self.hashes[r][c](token)

    def estimate(self):
        # X = Z^2 is unbiased for F2; average each row, take the median of the row-means.
        values = [z * z for row in self.z for z in row]
        return median_of_means(values, self.s1, self.s2)


def median_of_means(per_copy_values, s1, s2):
    rows = [per_copy_values[r * s1:(r + 1) * s1] for r in range(s2)]
    return statistics.median(sum(row) / s1 for row in rows)
```

The causal chain in one breath: the histogram is `Ω(n)`, so keep one linear scalar `Z = sum_i ε_i m_i` instead; a plain sum gives only `F_1`, but squaring a *signed* sum reproduces `sum_i m_i^2` on the diagonal while the cross terms vanish in expectation under mean-zero pairwise-independent signs, making `Z^2` an unbiased `F_2`; the fourth moment's lonely-index terms vanish only under four-wise independence, which is the exact, cheaply-constructible degree of independence the variance touches and yields `Var(Z^2) ≤ 2 F_2^2`; one copy has relative deviation `√2`, so average `O(1/λ^2)` copies to shrink variance and take the median of `O(log(1/δ))` such averages to make the failure probability exponentially small — giving an `(λ, δ)`-estimate of `F_2` in `O(λ^{-2} log(1/δ)(log n + log m))` bits; the same telescoping-and-Chebyshev recipe gives general `F_k` in `O(n^{1-1/k})` space, while communication-complexity reductions to disjointness show `n^{Ω(1)}` is unavoidable for `k > 5` and that both randomness and approximation are necessary — pinning the cheap moments `F_0, F_1, F_2` at their logarithmic floors.
