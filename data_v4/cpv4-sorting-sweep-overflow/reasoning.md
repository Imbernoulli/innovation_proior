The scale is the first thing I price, because it decides the data types before it decides the algorithm. I have `n` streams; stream `i` is active on the half-open interval `[s_i, e_i)` and draws `w_i` bandwidth, and I want the maximum over all instants `t` of the total weight active at `t`. With `n <= 2*10^5` and `w_i <= 10^9`, if every stream covers one common instant the load is `2*10^5 * 10^9 = 2*10^14`. A signed 32-bit `int` tops out near `2.1*10^9` — five orders of magnitude short. So every accumulator that carries a load has to be 64-bit; the natural `int sum, int best` sweep is a silent wrong-answer on the large tests, and that overflow is the trap this problem is built around. Coordinates fit in 32 bits, but I will store them 64-bit too so no stray `time` arithmetic can surprise me. Input is `n` then `n` triples `s e w` on stdin; I print one integer.

The intervals are half-open, `[s_i, e_i)` — closed on the left, open on the right — and that governs the tie rule. If A is `[2,5)` and B is `[5,9)`, then at `t = 5` stream A is already gone (active only for `t < 5`) and B has just begun; they never overlap. A sweep that treats an end and a start at the same time symmetrically will invent an overlap. So the rule "an end at time `t` happens before a start at time `t`" has to be baked in.

The load only changes at stream boundaries, so the natural algorithm is sort-and-sweep: emit a `+w` event at each `s_i` and a `-w` event at each `e_i`, sort the `2n` events by time, scan left to right maintaining a running load `cur`, and track `best = max(best, cur)`. `O(n log n)` time, `O(n)` memory. As an independent oracle I keep a brute force: since all `w_i` are positive, adding a stream never lowers the load, so the peak is attained at some start `s_i`; evaluating the load at each distinct start is `O(n^2)` but obviously correct, and I will stress the sweep against it. (That positive-weights fact is load-bearing for the oracle — with negative weights the peak could sit at an empty instant of load `0`, and evaluating only at starts would miss it.)

Back to the tie-break, and why the rule is exactly ends-before-starts. After processing all events at a time `t`, `cur` equals the load at `t`: streams with `s_i <= t` have fired their `+w`, streams with `e_i > t` have not yet fired their `-w`, an end exactly at `t` must have fired (excluding that stream, matching `e_i = t` ⇒ inactive), and a start exactly at `t` must have fired (including it, matching `s_i = t` ⇒ active). The *net* of all same-time events is order-independent, so the final `cur` at `t` is correct however I order the group. But the running maximum *within* the group is not: if I apply a start's `+w` before an end's `-w` at the same time, `cur` momentarily carries both the departing and the arriving stream — a phantom overlap that exists at no real instant. On the two abutting streams `[2,5) w=10` and `[5,9) w=10`, whose true peak is `10`, processing B's start before A's end at `t=5` makes `cur` jump `10 -> 20` before falling back, latching `best = 20`. So ends must precede starts at equal time.

The clean encoding: store each event as a pair `(time, delta)` with `delta = +w` for a start and `delta = -w` for an end, and sort ascending. At equal time the negative delta sorts first — ends before starts, for free, with no custom comparator to get backwards.

The sweep itself is one loop over `ev`, a `vector<pair<long long,int>>` of `(time, +w)` / `(time, -w)`:

```
int cur = 0, best = 0;
for (auto &p : ev) { cur += p.second; best = max(best, cur); }
```

Trace on the documented sample (`[0,5) w3`, `[2,8) w4`, `[2,6) w2`, `[5,9) w5`): sorted events `(0,+3),(2,+2),(2,+4),(5,-3),(5,+5),(6,-2),(8,-4),(9,-5)`, and `cur` runs `3,5,9,6,11,9,5,0`, so `best = 11` — matching the expected answer, and at `t=5` the `-3` fires before the `+5` (`9 -> 6 -> 11`), so the phantom `14` never appears. The logic is right.

The types are not. Take three streams each `[0,1) w=10^9` over instant `0`: the real load is `3*10^9`, already past `INT_MAX = 2147483647`. But `cur` and `p.second` are `int`, so `cur += p.second` runs in 32-bit: after two events `cur = 2*10^9`, and the third `+10^9` gives `3*10^9`, which wraps to `3000000000 - 2^32 = -1294967296`. `cur` goes negative, `best` keeps the stale `2*10^9`, and the program prints `2000000000` instead of `3000000000`. On the full bound the true `2*10^14` wraps modulo `2^32` to `552894464` — a silent, much-smaller wrong answer with no crash or warning, showing up only on large inputs. Every quantity that touches the load has to be 64-bit — read `w` as `long long`, make the delta `long long`, and `cur`/`best` `long long`, with `ev` now `vector<pair<long long,long long>>`:

```
long long cur = 0, best = 0;
for (auto &p : ev) { cur += p.second; best = max(best, cur); }
```

Re-run the overflow instance in 64-bit and `cur` climbs `10^9 -> 2*10^9 -> 3*10^9` for `best = 3000000000`, the true load. The worst case `2*10^14` sits far under the `~9.2*10^18` ceiling, and since `cur` only rises on a `+w` and falls on a `-w` back toward `0` (the `+w` and `-w` totals are equal), there is no underflow either. The algorithm was already correct; the fix is purely widening the types.

A few boundaries. `n = 0`: the event list is empty and `best` stays `0`. Empty interval `s_i == e_i`: I `continue` and emit no events, since `[s,s)` is active at no instant. Disjoint streams sharing a boundary fall out of the tie rule already fixed — the end fires first, `cur` returns to `0` between them, no spurious sum.

Against the positive-weights brute oracle over several hundred random small instances — deliberately narrow coordinate ranges to force frequent shared boundaries, up to 30 streams — the sweep matches on every one. The shared-boundary cases are exactly what would expose a reversed tie order, and a direct large-weight case is what would expose the overflow. The `O(n log n)` sort of at most `4*10^5` events is comfortably inside the 1 s limit.

Both traps this problem sets — the reversed tie order and the 32-bit accumulator — are now dispatched by the signed-delta encoding and the `long long` widening. The full program is in the answer.
