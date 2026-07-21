Two integer arrays, `A` of length `n` and `B` of length `m` with `n, m <= 500` and entries in `[-1000, 1000]`, and I have to pick equal-length subsequences of each — non-empty, indices strictly increasing in both — and maximize the position-wise dot product. Three features specific to this problem set the whole shape of the solution. The alignment is *in order*, so I cannot freely re-pair the large magnitudes. The pairing is *non-empty*, so unlike a max-subarray there is no fallback to `0` and the answer is genuinely allowed to be negative. And with up to `min(n, m) = 500` terms each as large as `1000*1000 = 10^6`, the score ranges over `[-5*10^8, 5*10^8]` — which fits a signed 32-bit int, but only just (its ceiling is about `2.1*10^9`), and I would rather not have to reason about whether every intermediate or sentinel value stays inside that margin. So I carry everything in `long long`; at `500*500` cells the memory cost is nothing.

Since a length-1 pairing is always legal, the first shortcut that suggests itself is "just return `max_{i,j} A[i]*B[j]`" — with negatives around, two big negatives multiply to a big positive, so one dominant product feels like it ought to win. It doesn't, because pairs stack. On `A = [2, 1, -2]`, `B = [3, 0, -1]` the best single product is `A[0]*B[0] = 6`, but pairing `A[0]` with `B[0]` and `A[2]` with `B[2]` (indices `0 < 2` in both, legal) scores `2*3 + (-2)*(-1) = 8`; a second modest positive simply adds on top, and the single-product rule never sees it.

The natural rescue — sort both arrays by magnitude and pair biggest with biggest — dies on the order constraint. On `A = [3, -5]`, `B = [-5, 3]` it would pair `-5` with `-5` and `3` with `3` for `9 + 25 = 34`, but reaching `(-5, -5)` uses `A[1]` with `B[0]` while `(3, 3)` uses `A[0]` with `B[1]`: the indices cross, which is not an order-preserving alignment. The only legal length-2 alignment is `3*(-5) + (-5)*3 = -30`, and the real optimum here is the length-1 pair `(-5)*(-5) = 25`. Both greedies fail for reasons — stacking, and order — that cannot be patched without rebuilding the alignment DP, so that is what I derive.

Let `dp[i][j]` be the best dot product of a non-empty aligned pairing drawn from `A[0..i-1]` and `B[0..j-1]`. At cell `(i, j)` the last elements `A[i-1]`, `B[j-1]` are either paired together as the final aligned pair — contributing `A[i-1]*B[j-1]` on its own, or `dp[i-1][j-1] + A[i-1]*B[j-1]` if it extends a real pairing on the smaller prefixes — or `A[i-1]` is dropped (`dp[i-1][j]`), or `B[j-1]` is dropped (`dp[i][j-1]`). So

```
dp[i][j] = max( A[i-1]*B[j-1],
                dp[i-1][j-1] + A[i-1]*B[j-1],
                dp[i-1][j],
                dp[i][j-1] )
```

with the borders `dp[*][0]`, `dp[0][*]` set to a sentinel `NEG` meaning "no non-empty pairing yet". The standalone `A[i-1]*B[j-1]` term is what makes a length-1 pairing representable and guarantees every interior cell holds a real value, so `dp[n][m]` is always a valid answer.

The one delicate point is the extend term. `dp[i-1][j-1]` may be the `NEG` sentinel (on a border), and I must not add the product into it: that would encode a falsehood — a non-empty pairing on an empty prefix — and flirt with underflow besides. So the extend is taken only when `dp[i-1][j-1] != NEG`. Related and equally important: there is deliberately no `max(..., 0)` anywhere. The empty pairing is disallowed, so a forced-negative input like `A = [-5]`, `B = [3]` must return `-15`, not `0`.

This DP invites one specific transcription slip: dropping the standalone product and extending unconditionally, like

```
long long best = dp[i-1][j-1] + prod;   // extend
best = max(best, dp[i-1][j]);
best = max(best, dp[i][j-1]);
```

On the smallest input `A = [5]`, `B = [3]`, whose only pairing is `5*3 = 15`, cell `(1,1)` computes `best = dp[0][0] + 15 = NEG + 15 ≈ -2.3*10^18`, then maxes against two more `NEG` borders, and prints a giant negative. A length-1 pairing is never representable on its own; every cell tries to extend, and at the borders there is nothing real to extend. The corrected body makes the fresh single pair a first-class candidate and gates the extend:

```
long long prod = A[i-1] * B[j-1];
long long best = prod;                          // brand-new length-1 pairing
if (dp[i-1][j-1] != NEG)
    best = max(best, dp[i-1][j-1] + prod);      // extend a real pairing
best = max(best, dp[i-1][j]);                   // drop A[i-1]
best = max(best, dp[i][j-1]);                   // drop B[j-1]
dp[i][j] = best;
```

Now `A = [5]`/`B = [3]` gives `15`. Running the recurrence on the sample `A = [2, 1, -2]`, `B = [3, 0, -1]` fills up to `dp[3][3]`, where `prod = (-2)*(-1) = 2` and the diagonal extend `dp[2][2] + 2 = 6 + 2 = 8` beats dropping either element (both `6`): `dp[3][3] = 8`, the expected answer, and the `8` is precisely the two-pair stack the greedies could not see.

A few corners fall straight out of the recurrence. When `n = 1` or `m = 1` the alignment length is capped at `1`, and the DP only ever realizes single pairs. An all-negative `A = [-3, -4]`, `B = [-5, -6]` yields a large positive by stacking two two-negative products: the diagonal extend at `(2,2)` gives `dp[1][1] + (-4)(-6) = 15 + 24 = 39`. Zeros are ordinary candidate values. And the sentinel is safe: `NEG = LLONG_MIN/4 ≈ -2.3*10^18` is only ever read inside a `max` or behind the `!= NEG` guard, never added to, so it cannot underflow, while genuine values stay within `±5*10^8`.

Beyond the hand-traces, I differential-test the recurrence against an independent brute force that enumerates every non-empty equal-length pair of subsequences and takes the max dot product, feasible only at tiny sizes. Across a few hundred random instances biased toward the dangerous regimes — mixed signs, all-negative, all-positive, zero-heavy — with `n, m` up to `8`, plus the explicit corners above, the DP matches the oracle on every one. The largest shape `n = m = 500` at all `+1000` and at all `-1000` both give `5*10^8` with no overflow, in a few milliseconds against the 1-second limit.

So the shipped program is one self-contained `O(nm)` file: the gated recurrence above run over the `(i, j)` grid with borders initialized to `NEG`, reading `n m` then `A` then `B`, and printing `dp[n][m]`.
