The task is a one-line sum, `S = sum_{i<j} a[i] * a[j]`, but the constraints are chosen to turn it into a type-width problem wearing a combinatorics identity. With `n` up to `10^5` and each `a[i]` up to `10^4`, there are up to `C(n,2) ≈ 5*10^9` pairs, each worth at most `10^8`, so `S` can reach about `5*10^17` — roughly eight orders of magnitude past the 32-bit signed ceiling of `~2.1*10^9`. It fits a 64-bit `long long` (`~9.2*10^18`) with room to spare. So the interesting requirement is not the algorithm but that every value on the accumulation path stay 64-bit; I'll settle the algorithm first and then police the widths.

The definition is a double loop over pairs — correct, but `O(n^2)`, and `~5*10^9` multiply-adds inside a tight loop overshoot the 1-second limit badly. I need the closed form. Let `T = sum_i a[i]` and `Q = sum_i a[i]^2`. Expanding the square over all ordered index pairs,

`T^2 = (sum_i a[i])(sum_j a[j]) = sum_{i,j} a[i] a[j] = Q + sum_{i != j} a[i] a[j]`,

and the off-diagonal sum counts each unordered pair `{i,j}` twice, so `sum_{i != j} a[i] a[j] = 2S`. Hence

`S = (T^2 - Q) / 2`,

one `O(n)` pass that accumulates `T` and `Q`, then a constant-time finish. On the sample `[3,1,4,1,5]`: `T = 14`, `Q = 9+1+16+1+25 = 52`, `S = (196 - 52)/2 = 144/2 = 72`, which matches — and note `T^2 - Q = 144` is exactly `2S`, so dropping the `/2` returns `144` instead of `72`. The numerator is literally `2S`, so the division is always exact.

Now the width discipline — the part the constraints are actually testing. The tempting mistake is to accumulate in `int` and only widen the result:

```
int sum = 0, sumsq = 0;
// ... sum += x; sumsq += x*x;
long long answer = (sum * sum - sumsq) / 2;
```

This is a silent wrong answer on the large tests, and the `long long` destination gives a false sense of safety. Take `n = 10^5` with every value `10^4`: `sum` reaches `10^9`, which still fits in `int` — but `sum * sum` is an `int * int` product, evaluated in 32-bit *before* the widening on assignment, and its true value `10^18` wraps to garbage; `sumsq` accumulating toward `10^13` wrapped long before that. Storing into a `long long` fixes nothing when every operand feeding it was computed in 32 bits — the wide type has to sit on the *operands*. So `x`, `sum`, and `sumsq` are all `long long`; then `sum * sum = 10^18`, `sumsq = 10^13`, and the final `S ≈ 5*10^17` all stay under the 64-bit ceiling. What makes this trap nasty is that it is invisible on every small sample, where `sum * sum` stays under `2^31`, and only detonates near the maximum the judge actually weights.

The formula needs no special-casing at the boundaries. For `n = 0` and `n = 1` the numerator `T^2 - Q` is zero (with `n = 1`, `T = a[0]` so `T^2 = a[0]^2 = Q`), giving `S = 0` with no pairs; all-zeros gives `0`; equal values `[5,5,5]` give `(225 - 75)/2 = 75`, three pairs of `25`. The leading `if (!(cin >> n)) return 0;` guards a completely empty stdin, and `cin >>` skips arbitrary whitespace, so the values may span one line or many.

As a transcription check — separate from the width reasoning — I run the `O(n)` solution against a plain Python double-loop brute force over the definition (arbitrary-precision, so the reference itself can never overflow) on a few hundred random small instances, with value caps from `0` up to `10^4` so the small-`n` totals already climb into the millions. They agree on every case. That catches a wrong constant or a sign slip that the algebra alone might hide, while the maximal-case width analysis covers the overflow those tiny brute-force inputs can never reach.

The shipped program is that single `O(n)` pass: read `n`, accumulate `sum` and `sumsq` in `long long`, and print `(sum * sum - sumsq) / 2` — with `x`, `sum`, and `sumsq` all 64-bit, so no multiplication is ever performed in `int`.
