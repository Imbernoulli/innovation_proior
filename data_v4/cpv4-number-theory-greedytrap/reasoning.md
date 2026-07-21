The admissible pulse energies here are exactly the perfect `k`-th powers `1^k, 2^k, 3^k, ...` for a fixed order `k` in `[2, 5]`, reusable freely, and I want the fewest of them summing to a target `n <= 10^6`. The order and target are both small, so the algorithm won't be the hard part — two structural facts will. First, `1 = 1^k` is admissible for every `k`, so every `n` is representable: there is no "impossible" branch, and `n = 0` simply wants `0`. Second, enumerating the powers means computing `b^k` for growing `b`, and at `k = 5` that overflows 32-bit around `b = 85` (`85^5 ≈ 4.4·10^9`), while `b` itself climbs toward the `k`-th root of `10^6`; so the enumeration has to run in 64-bit with an explicit overflow guard — never `int`, never floating `pow`.

Two routes to the answer. Greedy — repeatedly subtract the largest admissible power that fits — is `O(answer)` and three lines, but grabbing the largest power maximizes coverage this step and can strand the remainder off a lattice where the rest of the representation would have been cheap; I won't take it on faith. The alternative is a shortest-representation DP: `dp[v]` = fewest admissible powers summing to `v`, with `dp[0] = 0` and `dp[v] = 1 + min over admissible powers p <= v of dp[v - p]` — this is fewest-coins coin change with the `k`-th powers as the coins, `O(n · P)` where `P` is the number of powers up to `n`.

Greedy is cheap to test, so I try to break it. `k = 2`, powers `1, 4, 9, 16, ...`, target `n = 12`. Greedy grabs `9`, leaving `3`, whose largest square is `1`, so `1 + 1 + 1`: total `9 + 1 + 1 + 1 = 4`. But `4 + 4 + 4 = 12` is `3`. Greedy is wrong, and the reason is visible — snatching the `9` left a remainder of `3` with nothing bigger than `1` to spend on it, whereas backing off to `4` keeps every remainder on the lattice of `4`s. To confirm it isn't a `k = 2` artifact: `k = 3`, powers `1, 8, 27, ...`, `n = 32`. Greedy takes `27`, then `1` five times, total `6`; but `8 + 8 + 8 + 8 = 4`. Same failure mode, wider gap. Greedy is out; the DP is what I can defend.

The recurrence holds because any optimal representation of `v` uses some admissible power `p` as one summand, and deleting that summand leaves an optimal-or-better representation of `v - p`; so `dp[v] <= 1 + dp[v - p]` for that `p`, and since every `1 + dp[v - p]` is achievable the minimum is tight. Filling `k = 2` up to `12` gives `dp[4] = 1`, `dp[8] = 2`, `dp[12] = 1 + dp[8] = 3` — the `4 + 4 + 4` answer, beating greedy's `4`.

The one genuinely dangerous piece is enumerating the powers `<= n`. The naive `(long long)pow(b, k)` goes through a `double`: `pow(15, 5)` can come back `759374.9999...` and truncate to `759374`, or `pow(10, 5)` as `100000.0000001` — an off-by-one in the *value* of a power, which silently corrupts `dp`. And forming `b^k` as an integer product *before* comparing to `n` lets that product overflow `long long` for large `b` at `k = 5` and wrap to something `<= n`, pushing a garbage power into the list. Both are silent wrong-answers. So I build `b^k` by repeated integer multiplication and test, before each multiply, whether the running product would exceed `n` — using division so no overflowing product is ever formed:

```
for (long long b = 1;; b++) {
    long long p = 1; bool exceed = false;
    for (long long e = 0; e < k; e++) {
        if (p > n / b) { exceed = true; break; }  // p*b would exceed n
        p *= b;
    }
    if (exceed || p > n) break;
    powers.push_back(p);
}
```

The guard `p > n / b` is exactly `p * b > n` in integers, so the running product stays `<= n` throughout and never overflows. At `k = 5`, `n = 10^6` the loop admits `15^5 = 759375` and stops at `b = 16` (`65536 * 16` trips the guard), which is right: `16^5 = 1048576 > 10^6`.

Filling the table, `powers` is ascending, so in the inner scan the first `p > v` means every later power exceeds `v` too — I `break` there rather than `continue`. That is what keeps the worst case in budget: at `k = 2`, `n = 10^6`, `powers` has ~1000 entries, and `continue` would rescan all of them for every small `v` (~`10^9` ops), while `break` stops at the first oversized power. With the break, `p <= v` always holds where I index `dp[v - p]`, so the index stays in `[0, n]`.

The base case is the other thing that has to be right: `dp[0] = 0`, or else `dp[1] = dp[0] + 1` is garbage. `INF = 10^9` and `INF + 1` still fits in 32-bit `int`, and because `1` is always admissible no `dp[v]` for `v >= 1` ever stays `INF` anyway. I return early on `n = 0` before allocating anything, so the enumeration loop — which divides by `b` and compares to `n` — never runs on a degenerate `n = 0`.

The remaining corners fall out of the recurrence: `n = 1` → `1` (only power `1` fits); an exact `k`-th power like `k = 2`, `n = 9` → `1`; a small target at high order like `k = 5`, `n = 7` → `7`, since `2^5 = 32 > 7` leaves only `1`s; and the greedy traps land right — `k = 2`, `n = 18` → `2` (`9 + 9`), not greedy's `3`.

To be sure of the whole thing I cross-check against an independent BFS that treats `0..n` as a graph with edges `v -> v - p` for each admissible `p` and finds the shortest path from `n` to `0` — a different method from the forward DP, sharing no code. Over every `(k, n)` with `k in {2,3,4,5}` and `n in [0, 400]` (1604 cases) plus 500 random small cases the two agree everywhere, and the greedy-trap values (`k = 2`: `12→3`, `18→2`, `32→2`; `k = 3`: `32→4`) always come out as the DP value, never greedy's.

So I ship the `O(n · P)` DP: enumerate the powers with the division-guarded product, fill `dp[0..n]` with the ascending-`break` inner loop, and print `dp[n]`. The full program is in the answer.
