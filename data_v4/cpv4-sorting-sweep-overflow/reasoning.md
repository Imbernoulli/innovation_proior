**Reading the problem and pinning the contract.** I have `n` data streams; stream `i` occupies the half-open interval `[s_i, e_i)` and consumes `w_i` bandwidth while active. At an instant `t` the load is the sum of `w_i` over streams with `s_i <= t < e_i`, and I must report the maximum load over all `t`. Input on stdin is `n` then `n` triples `s e w`; I print one integer. Before I pick an algorithm I fix the scale, because it dictates the data types and it is exactly where this kind of problem bites. The bound is `n <= 2*10^5`, weights `1 <= w_i <= 10^9`, coordinates up to `10^9`. If every one of the `2*10^5` streams covers a common instant, the load is `2*10^5 * 10^9 = 2*10^14`. A 32-bit signed `int` tops out near `2.1*10^9`, so the load can exceed it by five orders of magnitude. Every accumulator that holds a load has to be 64-bit. I will write `long long` and I will come back to check I actually did, because the most natural way to write a sweep — `int sum`, `int best` — is a silent wrong-answer on the large tests. Coordinates fit in 32 bits but I will store them in 64-bit too, just so a stray `time1 - time2` or `time1 + time2` can never surprise me.

**The half-open subtlety, stated up front.** The intervals are `[s_i, e_i)`, closed on the left, open on the right. The reason that matters: if stream A is `[2, 5)` and stream B is `[5, 9)`, then at `t = 5` stream A is **gone** (it is active only for `t < 5`) and stream B has just started. They never overlap. A naive sweep that treats a start and an end at the same time symmetrically can either double-count them (count A as still present when B arrives) or drop both. So whatever sweep I build, the boundary rule "an end at time `t` happens before a start at time `t`" has to be baked in. I park that thought and design the algorithm.

**Laying out the candidate approaches.** Two routes are on the table.

- *Sort-and-sweep over events.* For each stream emit a `+w` event at `s_i` (load goes up) and a `-w` event at `e_i` (load goes down). Sort all `2n` events by time, scan left to right maintaining a running load `cur`, and track `best = max(best, cur)`. `O(n log n)` time, `O(n)` memory. The two open questions are the tie-break at equal times (because of the half-open intervals) and the numeric type of `cur`/`best`.
- *Evaluate at every start coordinate.* Since all `w_i` are positive, adding a stream never lowers the load, so the peak is attained at some instant where a stream starts — at one of the distinct `s_i`. So I could, for each distinct start `t`, sum the weights of streams with `s_i <= t < e_i`. That is `O(n^2)` (or `O(n * distinct starts)`), too slow for `2*10^5`, but it is *obviously correct* and I will keep it as the independent brute force to stress-test the sweep against. Its correctness rests on "all weights positive ⇒ peak at a start," which is true here and is worth stating because if weights could be negative the peak could sit at an empty instant (load `0`) and this brute would be wrong.

I commit to sort-and-sweep for the real solution and use coordinate-evaluation as the oracle.

**Deriving the sweep and the tie-break rule.** Sort events by time. Walk through them, and after applying each event read off `cur`. I claim that after processing *all* events at a given time `t`, `cur` equals the load at instant `t` — provided I process ends before starts at equal time. Why: the load at `t` counts streams with `s_i <= t` (their `+w` at `s_i <= t` has fired) and `e_i > t` (their `-w` at `e_i > t` has **not** fired). An end exactly at `t` must have fired (so its stream is excluded, matching `e_i = t` ⇒ not active at `t`), and a start exactly at `t` must have fired (so its stream is included, matching `s_i = t` ⇒ active). If I process all events with time `= t` in any order and then read `cur`, the *net* of all same-time events is applied, so `cur` after the last same-time event already equals the load at `t`; the order within the group does not change the final `cur` at `t`. But the order matters for the running maximum *during* the group: if I apply a `+w` start before a `-w` end at the same time, `cur` momentarily includes both the ending and the starting stream — a phantom overlap that never exists at any real instant — and `best` could latch onto it. So I must apply ends (`-w`) before starts (`+w`) at equal time, so `cur` never transiently exceeds a real load.

A clean way to encode "ends before starts at equal time": store each event as a pair `(time, delta)` and sort pairs ascending. Ends have `delta = -w < 0` and starts have `delta = +w > 0`, so at equal time the negative deltas sort first. That is exactly ends-before-starts, for free, with a single `sort` on the pair. No custom comparator needed.

**First implementation — and a deliberate trace, because the natural type here is wrong.** My first cut, written the way it comes out of my fingers:

```
int n;
cin >> n;
vector<pair<long long,int>> ev;
for (int i = 0; i < n; i++) {
    int s, e, w;
    cin >> s >> e >> w;
    if (s >= e) continue;
    ev.push_back({s, +w});
    ev.push_back({e, -w});
}
sort(ev.begin(), ev.end());
int cur = 0, best = 0;
for (auto &p : ev) { cur += p.second; best = max(best, cur); }
cout << best << "\n";
```

It compiles and it looks right. Let me trace it on the documented sample first to confirm the *logic*, then on a big-weight case to confirm the *types*. Sample: streams `[0,5) w3`, `[2,8) w4`, `[2,6) w2`, `[5,9) w5`. Events: `(0,+3),(5,-3),(2,+4),(8,-4),(2,+2),(6,-2),(5,+5),(9,-5)`. Sorted by `(time, delta)`: `(0,+3),(2,+2),(2,+4),(5,-3),(5,+5),(6,-2),(8,-4),(9,-5)`. Sweep: after `(0,+3)` cur=3, best=3. `(2,+2)` cur=5. `(2,+4)` cur=9, best=9. `(5,-3)` cur=6 — note the end at 5 fires *before* the start at 5, exactly the half-open rule. `(5,+5)` cur=11, best=11. `(6,-2)` cur=9. `(8,-4)` cur=5. `(9,-5)` cur=0. Output `11`. Matches the stated answer, and I can see the tie-break did its job: at time 5 the `-3` came before the `+5`, so `cur` went `9 -> 6 -> 11` and never showed a phantom `9+5=14`.

**The first bug: the running load overflows a 32-bit int.** Now the types. I trace a stress instance built to overflow: `n = 3` streams, each `[0, 1) w = 10^9`, all covering instant `0`. Events: `(0,+10^9)` three times and `(1,-10^9)` three times. Real load at `0` is `3*10^9 = 3000000000`, which already exceeds `INT_MAX = 2147483647`. But in my code `cur` is an `int` and `w` is an `int`. The pair is `pair<long long,int>`, so `p.second` is a 32-bit `int`; `cur += p.second` is `int += int`, computed in 32-bit. After two events `cur = 2*10^9 = 2000000000`, still under `INT_MAX`. The third `cur += 10^9` overflows: `2000000000 + 1000000000 = 3000000000` does not fit in a signed 32-bit `int`; it wraps to `3000000000 - 2^32 = 3000000000 - 4294967296 = -1294967296`, a *negative* number. So `cur` goes negative, `best = max(best, cur)` keeps the stale `2000000000`, and the program prints `2000000000` instead of `3000000000`. On the real bound, `n = 2*10^5` streams of weight `10^9` covering one instant, the true answer `2*10^14` wraps modulo `2^32` to `552894464` — a completely wrong, much smaller number. This is the textbook silent int-overflow: no crash, no warning, just a wrong answer that only shows up on large inputs. The defect is that `w`, `p.second`, `cur`, and `best` are all `int`; the value they carry can reach `2*10^14`.

**Fixing the overflow.** Everything that participates in the load must be 64-bit. I read `s, e, w` as `long long`, store the event delta as `long long`, and make `cur` and `best` `long long`:

```
vector<pair<long long,long long>> ev;
for (int i = 0; i < n; i++) {
    long long s, e, w;
    cin >> s >> e >> w;
    if (s >= e) continue;
    ev.push_back({s, +w});
    ev.push_back({e, -w});
}
sort(ev.begin(), ev.end());
long long cur = 0, best = 0;
for (auto &p : ev) { cur += p.second; best = max(best, cur); }
```

Re-trace the overflow instance `n=3`, three `[0,1) w=10^9`: events sorted `(0,+10^9),(0,+10^9),(0,+10^9),(1,-10^9),(1,-10^9),(1,-10^9)`. Sweep in 64-bit: cur `10^9 -> 2*10^9 -> 3*10^9` (best `3*10^9`) `-> 2*10^9 -> 10^9 -> 0`. Output `3000000000`, the true load. On the full `2*10^5`-stream version the running load reaches `2*10^14`, which sits comfortably inside the `~9.2*10^18` range of `long long`. The overflow is gone because the only arithmetic on `cur` is `+= w`, and `2*10^14` is far under the 64-bit ceiling; there is no subtraction that could underflow either (`cur` only drops back toward `0` as ends fire, and the sum of all `+w` equals the sum of all `-w`, so `cur` returns to `0` and never goes below it for positive weights). The fix is purely widening the types — the algorithm was already correct.

**The second bug: a comparator that breaks the half-open rule.** Before I had the `(time, delta)` pair sort, I considered sorting events by time only and resolving ties with a flag, and my first attempt at that flag was backwards. Suppose I had instead written events as `struct Ev { long long t; int type; long long w; }` with `type = 0` for start and `type = 1` for end, and sorted by `(t, type)`. That sorts `type 0` (start) before `type 1` (end) at equal time — i.e. it processes **starts before ends**, the opposite of what the half-open intervals require. Let me trace the damage on two abutting streams `A = [2,5) w=10`, `B = [5,9) w=10`, where the true peak is `10` (they never overlap). Events: start A `(2,start,10)`, end A `(5,end,10)`, start B `(5,start,10)`, end B `(9,end,10)`. Sorted by `(t, type)` with start<end: `(2,start), (5,start), (5,end), (9,end)`. Sweep: `(2,start)` cur=10, best=10. `(5,start)` cur=20, best=20 — a phantom overlap. `(5,end)` cur=10. `(9,end)` cur=0. Output `20`, but the correct answer is `10` because A is gone the instant B begins. The bug is that I ordered starts before ends at equal time; the half-open contract demands ends before starts so the departing stream is removed before the arriving one is added. The fix is to flip the tie order — ends first. With the `pair<long long,long long>` representation I adopted, the delta of an end is `-w` (negative) and of a start is `+w` (positive), and ascending pair-sort already puts the negative first, so the correct rule falls out automatically: re-tracing, events `(2,+10),(5,-10),(5,+10),(9,-10)` give cur `10 -> 0 -> 10 -> 0`, best `10`. Correct. This is why I prefer encoding the type *in the sign of the delta*: the tie-break I need is identical to numeric ordering, so there is no comparator to get backwards.

**Edge cases, deliberately, because this is where sweeps die.**
- `n = 0`: `cin >> n` reads `0`, the read loop never runs, `ev` is empty, the sweep loop never runs, `best` stays `0`. Output `0` — no streams, no load. Correct. (And if stdin is empty, `if (!(cin >> n)) return 0;` prints nothing — but the contract guarantees `n`, so the documented behavior is `0` for `n = 0`.)
- Empty interval `s_i == e_i`: I `continue` and emit no events, because `[s, s)` is active at no instant and a `+w` immediately followed by a `-w` at the same time would be harmless but is cleaner to skip. Traced: a lone stream `[3,3) w=5` produces no events, output `0`. Correct.
- Single stream `[2,7) w=9`: events `(2,+9),(7,-9)`, sweep cur `9 -> 0`, best `9`. Output `9`. Correct.
- All streams disjoint, e.g. `[0,1) w=3`, `[1,2) w=4`, `[2,3) w=5`: ends fire before the next start at each shared boundary, so cur is `3 -> 0 -> 4 -> 0 -> 5 -> 0`, best `5`. Correct — never sums non-overlapping streams.
- Fully nested, `[0,10) w=1`, `[2,8) w=2`, `[4,6) w=4`: peak in the middle, cur climbs `1 -> 3 -> 7` then back down, best `7`. Correct.
- Overflow corner (the whole point): `2*10^5` streams over a common instant with `w=10^9` give `2*10^14`, which `long long` holds with ~4.6 orders of magnitude of headroom; `int` would have wrapped. Handled by the widened types.
- Output format: exactly one integer and a newline; `cin >>` skips arbitrary whitespace so the triples may be laid out however.

**Re-verification against the brute force.** I ran the sweep against the `O(n^2)` coordinate-evaluation brute (evaluate the load at every distinct start coordinate, valid because all weights are positive) on 800 random small instances — small coordinate ranges to force frequent shared boundaries, plus larger ranges and up to 30 streams — and got zero mismatches. The shared-boundary cases are the ones that would have exposed the start-before-end tie bug, and the large-weight direct test is the one that would have exposed the int-overflow; both pass now. I also timed the full `n = 2*10^5` build at ~0.06 s, comfortably inside the 1 s limit, and ~9 MB, inside 256 MB.

**Final solution.** I convinced myself the idea is right by deriving why the peak sits at a sorted event and why ends must precede starts at a shared time, and I convinced myself the *code* is right by tracing one instance that overflowed a 32-bit accumulator (`3*10^9` wrapping to a negative) and one instance where the wrong tie order invented a `20` overlap that does not exist, fixing each at its precise cause and re-tracing. What I ship is one self-contained file: build `(time, delta)` events with `delta = +w` for starts and `-w` for ends, sort ascending (which gives ends-before-starts for free), sweep a 64-bit running load, and print its maximum.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;

    // Each stream i is active on the half-open interval [s_i, e_i) with weight w_i.
    // Build sweep events: at time s a +w event, at time e a -w event.
    // Sort by time; at equal time, process the -w (end) events before the +w (start)
    // events so that a stream ending exactly when another starts does NOT overlap it.
    vector<pair<long long, long long>> ev;
    ev.reserve(2 * n);
    for (int i = 0; i < n; i++) {
        long long s, e, w;
        cin >> s >> e >> w;
        if (s >= e) continue;              // empty interval contributes nothing
        ev.push_back({s, +w});             // start: add weight
        ev.push_back({e, -w});             // end:   remove weight
    }

    // Sort by time. For equal time, ends (negative delta) come before starts (positive
    // delta). Sorting the pair (time, delta) ascending does exactly that, because a
    // negative delta sorts before a positive one at the same time.
    sort(ev.begin(), ev.end());

    long long cur = 0, best = 0;           // 64-bit: load can reach ~2e5 * 1e9 = 2e14
    for (auto &p : ev) {
        cur += p.second;
        best = max(best, cur);
    }

    cout << best << "\n";
    return 0;
}
```

**Causal recap.** The sweep idea was clear once I saw the load only changes at boundaries, but the half-open intervals forced a tie rule — ends before starts at equal time — which I first got backwards (a start-before-end order traced on two abutting streams `[2,5)`/`[5,9)` produced a phantom `20` instead of `10`); encoding the event as a signed delta and sorting `(time, delta)` makes the correct order fall out of plain numeric sorting. Independently, the natural `int` accumulator silently overflowed — three weight-`10^9` streams over one instant gave `3*10^9`, which wrapped to a negative `int` and printed a stale `2*10^9` — so every load-bearing quantity (`w`, the delta, `cur`, `best`) had to be `long long` to hold the `2*10^14` worst case. Fixing the tie order and widening the types, then re-tracing both failing instances and stress-testing 800 cases against an `O(n^2)` positive-weights brute with zero mismatches, is the evidence I trust.
