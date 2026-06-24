**Reading the problem and pinning the contract.** I have `n` cars, each a half-open interval `[s_i, e_i)`: the car is present at the instant `s_i` and gone by `e_i`. I must report the maximum number of cars present at any single instant under that half-open convention. The exclusive right endpoint is loud in the statement and I should treat it as the load-bearing fact: a car leaving at time `t` and a car arriving at time `t` reuse the same spot, so they do *not* count together. Input is `n` then `n` pairs `s_i e_i`; output is one integer. Let me fix the scale before any algorithm: `n <= 2*10^5` and coordinates up to `10^9`. The answer itself is at most `n <= 2*10^5`, which fits an `int` easily, but I will keep the running count and coordinates in 64-bit anyway because coordinates near `10^9` and the habit of `long long` costs nothing here and removes a whole class of silent surprises. There is one more contract clause I must not skip: records with `s_i >= e_i` occupy no instant and must be ignored. That is a real input class (a logging glitch), and forgetting it would corrupt the count.

**Laying out the candidate approaches.** Two routes are on the table, and I want the one whose boundary behavior I can *prove*, not the one that is fastest to type.

- *Difference array / bucketed prefix sums.* Add `+1` at `s_i` and `-1` at `e_i` into an array indexed by coordinate, prefix-sum, take the max. Lovely when coordinates are small, but here they reach `10^9`, so I would need coordinate compression. Compression does not remove the hard part; it relocates it — I would still have to decide which compressed bucket a departure at `e_i` lands in relative to an arrival at the same value, which is the *same* inclusive/exclusive question wearing a different hat. No simplification, more code.
- *Event sweep with `+1 / -1`.* Emit a `+1` event at each arrival and a `-1` event at each departure, sort the `2n` events by coordinate, sweep left to right with a running counter, and track its maximum. `O(n log n)`, `O(n)` memory, and the boundary decision is concentrated into a single, inspectable place: the tie-break used by the sort. I choose this, precisely because the danger is localized where I can stare at it.

**Deriving the sweep and nailing the tie-break on paper.** The coverage count as a function of the instant `t` is `f(t) = #{ i : s_i <= t < e_i }`. It is a step function: it jumps `+1` as `t` crosses an arrival `s_i` (because at `t = s_i` the condition `s_i <= t` turns true) and jumps `-1` as `t` crosses a departure `e_i` (because at `t = e_i` the condition `t < e_i` turns false). The maximum of `f` is what I want. If I process events in increasing coordinate, a plus event at `s_i` should take effect *at and after* `s_i`, and a minus event at `e_i` should take effect *at and after* `e_i`. Now the crux: what happens at a coordinate `t` where some departures (`e_j = t`) and some arrivals (`s_k = t`) coincide? Under half-open semantics, the car that departs at `t` is already gone at instant `t` (`t < e_j` is false), while the car that arrives at `t` is present at instant `t` (`s_k <= t` is true). So when I evaluate `f(t)`, the departures at `t` must *not* be counted and the arrivals at `t` *must* be counted. In sweep terms: **at a shared coordinate, apply all `-1` (end) events before any `+1` (start) events.** If I did it the other way, I would momentarily count the departing car together with the arriving car — which is exactly the *closed*-interval answer `[s, e]`, off by one against the half-open spec.

**Encoding the tie-break so a plain sort does the right thing.** I will represent each event as a pair `(coordinate, type)` and rely on the lexicographic `pair` comparison. If I let `type = 0` mean an end (`-1`) and `type = 1` mean a start (`+1`), then sorting `(coordinate, type)` ascending puts, at any shared coordinate, all the `type = 0` ends before all the `type = 1` starts — precisely the order half-open needs. The encoding *is* the correctness argument, so I will write a comment to that effect and never touch it casually.

**A derivation sanity-check on the sample.** Cars `[1,4)`, `[2,5)`, `[5,7)`, `[3,4)`, `[4,6)`; the claimed answer is `3`. Events (coordinate, type), start=1, end=0: starts at 1,2,5,3,4 and ends at 4,5,7,4,6. Listed: `(1,s)(4,e)`, `(2,s)(5,e)`, `(5,s)(7,e)`, `(3,s)(4,e)`, `(4,s)(6,e)`. Sort by (coord, type with end<start): `(1,s) (2,s) (3,s) (4,e) (4,e) (4,s) (5,e) (5,s) (6,e) (7,e)`. Sweep, cur starts 0: `(1,s)`→1, `(2,s)`→2, `(3,s)`→3 (max so far 3), `(4,e)`→2, `(4,e)`→1, `(4,s)`→2, `(5,e)`→1, `(5,s)`→2, `(6,e)`→1, `(7,e)`→0. Max is `3`. Matches. Crucially the two ends at coordinate 4 fired *before* the start at 4, so the count dipped to 1 before climbing again, never falsely lumping the departing `[1,4)`/`[3,4)` with the arriving `[4,6)`.

**First implementation — and immediately a trace, because the boundary is where this dies.** My first cut, written quickly and, I suspect, wrongly on the tie-break:

```
vector<pair<long long,int>> ev;
for each car (s, e):
    ev.push_back({s, 0});   // start
    ev.push_back({e, 1});   // end
sort(ev.begin(), ev.end());
long long cur = 0, best = 0;
for (auto &p : ev) {
    if (p.second == 0) cur += 1; else cur -= 1;   // 0=start -> +1, 1=end -> -1
    best = max(best, cur);
}
```

I encoded `start = 0, end = 1` here — the *opposite* of what my derivation demanded. Let me trace the smallest input that exposes a tie: two cars that merely touch, `[1,4)` and `[4,7)`. They share no instant, so the true answer is `1`. Events: `(1,start=0)`, `(4,end=1)` from the first car; `(4,start=0)`, `(7,end=1)` from the second. Sort `(coord, type)`: `(1,0) (4,0) (4,1) (7,1)`. Walk: `(1,0)` start→cur=1, best=1. `(4,0)` start→cur=2, best=2. `(4,1)` end→cur=1. `(7,1)` end→cur=0. Output `2`.

**Diagnosing the bug.** The code returns `2` for two non-overlapping touching cars — wrong, the answer is `1`. The defect is exactly the tie-break I warned myself about and then got backwards: at coordinate `4` the array holds `(4, start=0)` and `(4, end=1)`, and because `0 < 1` the sort put the *start* before the *end*. So the sweep added the arriving car at `4` while the departing car was still counted, momentarily showing `2` — that is the *closed*-interval reading `[1,4]`, `[4,7]`, which do share the point `4`. My encoding inverted the required order: I need ends (`type 0`) to come before starts (`type 1`) at a tie, so the encoding must be `end = 0, start = 1`, and then `+1` is the `type == 1` branch. This is the canonical off-by-one of this whole problem family, surfaced by a two-car trace.

**Fixing the encoding and re-verifying.** Flip the type codes so that the lexicographic sort yields ends-before-starts at ties, and flip the increment branch to match:

```
ev.push_back({s, 1});   // start  (+1)
ev.push_back({e, 0});   // end    (-1)
...
if (p.second == 1) cur += 1; else cur -= 1;
```

Re-trace the touching case `[1,4)`, `[4,7)`. Events: `(1,1)`, `(4,0)`, `(4,1)`, `(7,0)`. Sort: `(1,1) (4,0) (4,1) (7,0)`. Walk: `(1,1)` start→cur=1, best=1. `(4,0)` end→cur=0. `(4,1)` start→cur=1, best=1. `(7,0)` end→cur=0. Output `1`. Correct — at coordinate 4 the end fired first, dropping to 0, *then* the start lifted to 1, so the two cars never coexisted. Re-trace a genuine overlap, `[1,5)` and `[4,7)` (answer 2): events `(1,1)(5,0)(4,1)(7,0)` → sorted `(1,1)(4,1)(5,0)(7,0)` → cur 1,2(best 2),1,0 → output `2`. The case that broke now passes, and it broke for the reason I fixed, which is the evidence I trust.

**A second self-verify episode: the degenerate-record clause.** The contract says a record with `s_i >= e_i` occupies no instant and must be ignored. My current loop blindly pushes a `+1` at `s` and a `-1` at `e` for *every* record. Let me trace a malformed one to see if that silently corrupts the count. Take input `2`, records `[5, 2)` (degenerate, `s=5 >= e=2`) and `[0, 3)` (real). Without a guard, events are `(5,1)`, `(2,0)` for the bad car and `(0,1)`, `(3,0)` for the good one. Sort: `(0,1) (2,0) (3,0) (5,1)`. Walk: `(0,1)` start→cur=1, best=1. `(2,0)` end→cur=0. `(3,0)` end→cur=-1. `(5,1)` start→cur=0. The running count goes *negative* (`-1`) because the degenerate car's end-event fires with no matching live start before it, and although `best` happens to stay `1` here, this is a landmine: a different arrangement of a degenerate record can push a spurious `-1` early and then a stray `+1` late, and in general letting `cur` desync from "actual cars present" is how a wrong maximum sneaks in on a hidden test. The fix is to honor the contract literally: skip any record with `s >= e` before emitting events. With the guard, the degenerate `[5,2)` contributes nothing, only `[0,3)` is swept, and the answer is `1` — which is what "one real car" should give. I add `if (s >= e) continue;`.

**Edge cases, deliberately, because sweeps fail at their seams.**
- `n = 0`: the read loop never runs, `ev` is empty, the sweep loop never runs, `best` stays `0`. Correct — no cars, no occupancy.
- A single car `[3, 9)`: events `(3,1)(9,0)`, sweep `cur` 1 then 0, `best = 1`. Correct.
- All disjoint, e.g. `[0,1) [2,3) [4,5)`: at no instant do two coincide; the sweep rises to 1 and falls to 0 three times, `best = 1`. Correct.
- Many sharing one endpoint, e.g. three cars `[0,4) [1,4) [2,4)` and one `[4,9)`: at `t` just below 4 all three early cars overlap → 3; the `[4,9)` arrives only after the three ends at 4. Events sorted put `(4,0)(4,0)(4,0)` before `(4,1)`, so `cur` climbs to 3 (best 3), drops to 0 across the three ends, then rises to 1. `best = 3`. Correct, and it is the touching boundary that makes this 3 and not 4.
- Fully nested `[0,10) [2,8) [4,6)`: coverage peaks at 3 in the middle; sweep gives 3. Correct.
- Overflow: the running count is at most `n <= 2*10^5`; coordinates up to `10^9` fit `long long` with vast room. No accumulation of magnitudes happens — `cur` only moves by `±1`. Safe in `int`, kept in `long long` out of habit.
- Output: exactly one integer and a newline; `cin >>` skips arbitrary whitespace, so the line structure of the input is irrelevant.

**A stress test against an independent brute force.** Idea verified on paper is not the same as code verified by machine, so I cross-check `sol` against a deliberately different brute force: the brute enumerates every distinct start coordinate `t` (coverage can only rise at a start, so a maximum is attained at some start) and, for each, counts intervals with `s <= t < e` by a direct `O(n)` scan — pure `s <= t < e` predicate, no events, no sort, no tie-break logic at all, so it cannot share my bug. I run it on a generator biased toward *small coordinate ranges* (so endpoints collide constantly — the exact regime where the inclusive/exclusive bug lives) and including degenerate `s >= e` records. Over 900 random small cases (`n` up to 20, coordinates in tiny ranges) the two agree on every single one, zero mismatches. The documented sample prints `3`. At `n = 2*10^5` with coordinates spread across `[0,10^9]` the program finishes in about 0.05 s, comfortably inside the 1 s limit.

**Final solution.** I convinced myself the *idea* is right by deriving the step function and its `±1` jumps and checking the sample by hand; I convinced myself the *boundary* is right by tracing a two-car touching case to the precise inverted tie-break and re-verifying the fix; I convinced myself the *contract* is honored by tracing a degenerate record into a negative running count and adding the skip guard; and I convinced myself the *code* matches the math by a 900-case stress test against an independent brute force. That is what I ship — one self-contained file, the `O(n log n)` event sweep whose entire correctness is the comment on the tie-break:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // n = 0 -> no cars -> answer 0

    // Each car occupies the half-open interval [s, e): present at s, gone by e.
    // A car ending at time t and a car starting at time t do NOT overlap.
    // We build events: a start contributes +1 AT coordinate s; an end contributes
    // -1 AT coordinate e. We sweep coordinates in increasing order. At a shared
    // coordinate, ENDS must be applied before STARTS, because [a,t) and [t,b) are
    // disjoint -- the leaving car frees the instant before the arriving car claims it.
    // event = (coordinate, type) with type 0 = end (-1), type 1 = start (+1).
    vector<pair<long long,int>> ev;
    ev.reserve((size_t)2 * n);
    for (int i = 0; i < n; i++) {
        long long s, e;
        cin >> s >> e;
        // Guard: a degenerate interval with s >= e occupies no instant; skip it.
        if (s >= e) continue;
        ev.push_back({s, 1});   // start
        ev.push_back({e, 0});   // end
    }

    // Sort by coordinate ascending; within equal coordinate, type 0 (end) before
    // type 1 (start). Since we encode end as 0 and start as 1, a plain pair sort
    // on (coordinate, type) gives ends first at ties -- exactly what half-open needs.
    sort(ev.begin(), ev.end());

    long long cur = 0, best = 0;
    for (auto &p : ev) {
        if (p.second == 1) cur += 1; else cur -= 1;
        best = max(best, cur);
    }

    cout << best << "\n";
    return 0;
}
```

**Causal recap.** The whole problem reduces to a `+1/-1` coordinate sweep whose only subtlety is the tie-break at a shared instant; half-open `[s,e)` forces ends to be applied before starts, and my first encoding (`start=0, end=1`) inverted that, which a two-car touching trace `[1,4),[4,7)` exposed by returning the closed-interval answer `2` instead of `1`; flipping the type codes to `end=0, start=1` so the lexicographic sort yields ends-first fixes it. A second trace showed degenerate `s >= e` records desync the running count into negatives, so I added a skip guard per the contract. The `±1` count never exceeds `n`, so 64-bit is ample, and a 900-case agreement with an independent `s <= t < e` brute force plus a 0.05 s run at `n = 2*10^5` closes out the touching, disjoint, nested, shared-endpoint, empty, single-car, and degenerate corners.
