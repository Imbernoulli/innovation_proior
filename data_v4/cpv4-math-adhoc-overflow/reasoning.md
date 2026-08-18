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

**Deriving the identity carefully and a sanity-check on the sample.** I do not want to trust a half-remembered formula, so I re-derive it on the documented sample `a = [3, 1, 4, 1, 5]`, whose stated answer is `72`. First `T = 3+1+4+1+5 = 14`, so `T^2 = 196`. Next `Q = 9 + 1 + 16 + 1 + 25 = 52`. Then `S = (196 - 52)/2 = 144/2 = 72`. That matches the sample exactly, which is strong evidence the identity is `S = (T^2 - Q)/2` and not, say, `(T^2 - Q)` or `(T^2 + Q)/2`. Good — the algebra and the worked example agree.

A second sanity-check on a trivial instance where I can see every term: `a = [2, 3]`. The only pair contributes `2*3 = 6`. Identity: `T = 5`, `T^2 = 25`, `Q = 4 + 9 = 13`, `S = (25 - 13)/2 = 12/2 = 6`. Correct. And the empty/singleton boundary: with `n = 1`, `a = [7]`, there are no pairs so `S = 0`; identity gives `T = 7`, `T^2 = 49`, `Q = 49`, `S = (49 - 49)/2 = 0`. Correct — the formula self-zeroes when there is nothing to pair. That reassures me the closed form handles the small-`n` corners without special-casing.

It looks like a faithful copy of the math, and `answer` is even a `long long`, which gives a false sense of safety. So I trace the worst case rather than a friendly one, because friendly inputs would hide the bug. Take the maximal instance: `n = 100000` with every `a[i] = 10000`. By hand, `T = 100000 * 10000 = 10^9`. The true `T^2 = 10^18`. The true `Q = 100000 * 10^8 = 10^13`. The true `S = (10^18 - 10^13)/2 = (1000000000000000000 - 10000000000000)/2 = 999999990000000000 / 2 = 499999995000000000`, about `5*10^17`.

This is a silent wrong answer on the large tests, and the `long long` destination gives a false sense of safety. Take `n = 10^5` with every value `10^4`: `sum` reaches `10^9`, which still fits in `int` — but `sum * sum` is an `int * int` product, evaluated in 32-bit *before* the widening on assignment, and its true value `10^18` wraps to garbage; `sumsq` accumulating toward `10^13` wrapped long before that. Storing into a `long long` fixes nothing when every operand feeding it was computed in 32 bits — the wide type has to sit on the *operands*. So `x`, `sum`, and `sumsq` are all `long long`; then `sum * sum = 10^18`, `sumsq = 10^13`, and the final `S ≈ 5*10^17` all stay under the 64-bit ceiling. What makes this trap nasty is that it is invisible on every small sample, where `sum * sum` stays under `2^31`, and only detonates near the maximum the judge actually weights.

Re-trace `n = 100000`, all `10000`: `sum = 10^9` (fits ll), `sum * sum = 10^18` (fits ll), `sumsq = 10^13` (fits ll), `answer = (10^18 - 10^13)/2 = 499999995000000000`. I run this exact case through the compiled program and it prints `499999995000000000`, matching the hand computation. The overflow is gone because the wide type now sits on the operands. The bug broke precisely where I predicted — `int * int` — and the fix removes precisely that, which is the evidence I trust.

**Second implementation episode — a subtler arithmetic bug, traced on the sample.** While cleaning up I briefly wrote the finish as `answer = (sum * sum - sumsq)` and forgot the `/2`, reasoning loosely that "the square of the sum minus the squares is the cross terms." Let me trace that on the sample `a = [3, 1, 4, 1, 5]` before trusting it. `sum = 14`, `sum * sum = 196`, `sumsq = 52`, so this buggy line gives `answer = 196 - 52 = 144`. But the documented answer is `72`, and `144 = 2 * 72`. The trace exposes the defect immediately: `T^2 - Q` equals the sum over all *ordered* off-diagonal pairs, which counts `{i,j}` twice (as `(i,j)` and `(j,i)`); the unordered total `S` is exactly half of it. Dropping the `/2` double-counts every handshake. Restoring `(sum * sum - sumsq) / 2` brings the sample back to `72`. And the division is exact: `T^2 - Q = 2S` is always even by construction, so integer division by 2 loses nothing — I verify on the sample (`144/2 = 72`, no remainder) and note it holds in general because the right side is literally `2S`.

**Edge cases, because this is where ad-hoc arithmetic dies.**
- `n = 0`: the loop never runs, `sum = sumsq = 0`, `answer = (0 - 0)/2 = 0`. No attendees, no handshakes — correct. The `if (!(cin >> n)) return 0;` also guards a completely empty stdin.
- `n = 1`, `a = [9999]`: loop adds once, `sum = 9999`, `sumsq = 9999^2 = 99980001`, `answer = (9999^2 - 9999^2)/2 = 0`. One attendee can shake no hands — correct.
- `n = 2`, `a = [2, 3]`: `sum = 5`, `sumsq = 13`, `answer = (25 - 13)/2 = 6`. Single handshake `2*3 = 6` — correct.
- All zeros, `a = [0, 0, 0]`: `sum = 0`, `sumsq = 0`, `answer = 0`. Every product is zero — correct.
- Equal values, `a = [5, 5, 5]`: `sum = 15`, `sumsq = 75`, `answer = (225 - 75)/2 = 75`. Check: three pairs each `5*5 = 25`, total `75` — correct.
- Overflow boundary: intermediate `sum * sum` peaks at `10^18` and the final `S` at `~5*10^17`, both safely under the `long long` ceiling `~9.2*10^18`; with all operands `long long` there is no 32-bit step anywhere. Safe.
- Output: exactly one integer and a newline; `cin >>` skips arbitrary whitespace, so the values may be on one line or many.

The formula needs no special-casing at the boundaries. For `n = 0` and `n = 1` the numerator `T^2 - Q` is zero (with `n = 1`, `T = a[0]` so `T^2 = a[0]^2 = Q`), giving `S = 0` with no pairs; all-zeros gives `0`; equal values `[5,5,5]` give `(225 - 75)/2 = 75`, three pairs of `25`. The leading `if (!(cin >> n)) return 0;` guards a completely empty stdin, and `cin >>` skips arbitrary whitespace, so the values may span one line or many.

As a transcription check — separate from the width reasoning — I run the `O(n)` solution against a plain Python double-loop brute force over the definition (arbitrary-precision, so the reference itself can never overflow) on a few hundred random small instances, with value caps from `0` up to `10^4` so the small-`n` totals already climb into the millions. They agree on every case. That catches a wrong constant or a sign slip that the algebra alone might hide, while the maximal-case width analysis covers the overflow those tiny brute-force inputs can never reach.

The shipped program is that single `O(n)` pass: read `n`, accumulate `sum` and `sumsq` in `long long`, and print `(sum * sum - sumsq) / 2` — with `x`, `sum`, and `sumsq` all 64-bit, so no multiplication is ever performed in `int`.
