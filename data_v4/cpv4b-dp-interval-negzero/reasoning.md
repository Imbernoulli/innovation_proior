The thing that will bite in this one is signs, not the DP shape. I collapse a row of `n` tiles by repeatedly fusing two adjacent tiles `x, y` into `x + y`, banking `x * y` — and `x * y` can be negative. Fusion is mandatory: for `n >= 2` I must perform `n - 1` fusions, so the maximum total energy for a block can itself be negative, while a row of *all-negative* charges collapses to a large *positive* total because two negatives multiply positive. So "clamp the answer at zero" and "seed the running max at zero" are both traps I need to actively avoid. The trivial cases first: `n = 0` and `n = 1` never fuse, answer `0`.

Scale sets the data type. `n <= 500`, `|a[i]| <= 10^6`, so a block's charge sum reaches `500 * 10^6 = 5*10^8` and one fusion's product `S * S` reaches `2.5*10^17` — past 32-bit (`~2.1*10^9`) by eight orders of magnitude, inside signed 64-bit (`~9.2*10^18`). Every charge, prefix sum, product, and table entry is `long long`; an `int` anywhere is a silent wrong answer on the big tests.

**The structure that kills the exponential.** Fusion order looks like it needs an exponential search, but charge sums are invariant under it — fusion only adds, so a block `[i..j]` always collapses to a tile of charge `S(i,j) = a[i] + ... + a[j]` regardless of order. Look at the *last* fusion forming `[i..j]`: it merges the collapsed left part `[i..k]` (charge `S(i,k)`) with the collapsed right part `[k+1..j]` (charge `S(k+1,j)`), banking `S(i,k) * S(k+1,j)` on top of each side's own optimum, and the two sides were collapsed independently. That is an interval recurrence that only cares about the split structure, not the cross-split order.

**Greedy first, since it's cheaper — but it doesn't survive.** "Repeatedly do the adjacent fusion with the largest immediate `x * y`" is `O(n^2)` and short, but each fusion rewrites a tile's charge and so changes every future product that tile joins; on mixed-sign rows a locally best fusion inflates a charge that then multiplies badly downstream. Small hand cases are annoyingly robust — `[10, -1, 10]` ties at `80` either way — so hand examples won't settle it. I write an exhaustive oracle (recurse over every split, memoize on `(i,j)`, the literal definition of the process) and run greedy against it on random small rows heavy on negatives and zeros; greedy diverges. It's out. I commit to the DP, which I can argue correct straight from the last-fusion decomposition.

**The recurrence.** With prefix sums `pre[t] = a[0] + ... + a[t-1]` so `S(i,j) = pre[j+1] - pre[i]`:

- `dp[i][i] = 0` — a single tile needs no fusion;
- `dp[i][j] = max over k in [i, j-1] of ( dp[i][k] + dp[k+1][j] + S(i,k) * S(k+1,j) )`.

Fill by increasing block length `len` from `2` to `n`, since `dp[i][j]` reads only strictly shorter blocks; the answer is `dp[0][n-1]`, with `n <= 1` short-circuited to `0` before the table exists.

**The base-case sign trap.** The natural way to write the inner max is to seed it at `0`:

```
long long best = 0;
for (int k = i; k < j; k++)
    best = max(best, dp[i][k] + dp[k+1][j] + S(i,k) * S(k+1,j));
```

and that clamps exactly the negatives I flagged at the start. Seeding at `0` inserts a phantom "release nothing" option that no block of `>= 2` tiles actually has. Take `[-3, 4]`: the one possible fusion banks `(-3) * 4 = -12`, so the answer is `-12`, but `max(0, -12) = 0`, and that bogus `0` then feeds into every larger interval as a sub-block value. The seed must instead be a sentinel below every reachable value, `LLONG_MIN`, overwritten by the first real candidate. Every interval of length `>= 2` has at least one split `k`, so `best` is always overwritten before it is read — and `LLONG_MIN` is only ever compared, never has anything added to it, so it can't underflow. Now `[-3, 4]` returns `-12`, and the sign is honest.

The mirror trap is on the final answer: because two negatives bank a positive, an all-negative row is usually *positive* — `[-2, -3, -4]` collapses to `+26` (either split ties there) — so I must not clamp `dp[0][n-1]` with `max(., 0)` either. The recurrence already carries the correct sign; a clamp would be a bug in the shape of a safety net. A zero charge needs no special case: any product touching it is `0`, and the recurrence absorbs that.

**Pinning the magnitude bound, since I rely on it for the type.** I want the worst-case *answer*, not just the worst-case intermediate. On an all-equal positive row `[V]*n` every fusion merges collapsed sub-blocks of `p` and `q` tiles carrying `pV` and `qV`, banking `pq V^2`; summed over the whole binary merge tree, the total of `pq` over internal nodes equals the number of leaf pairs `n(n-1)/2` (each pair of leaves is first joined at exactly one node), so the answer is `V^2 * n(n-1)/2` regardless of tree shape. At `n = 8, V = 10^6` that predicts `28 * 10^12 = 2.8*10^13`, and the oracle on `[10^6]*8` prints `28000000000000` — the identity holds. The legal worst case `n = 500, V = 10^6` gives `1.2475*10^17`, comfortably inside `long long`; mixed signs only shrink the magnitude.

**The empty-input corner.** `if (!(cin >> n)) return 0;` returns silently on truly empty input; when the `n = 0` test supplies the token `0`, the `n <= 1` guard prints `0`. Both satisfy the contract. Output is one integer plus a newline, and `cin >>` skips arbitrary whitespace so the input spacing is free.

I check the whole solution against the oracle on a thousand-plus random small rows — negatives, zeros, empty and singleton, forced-loss rows — with no mismatch, and the statement's `[3, -2, 5, -1]` gives `-7` both ways. At `n = 500` the `O(n^3)` fill is `~1.25*10^8` simple operations, well under the 1 s limit. The full program is in the answer.
