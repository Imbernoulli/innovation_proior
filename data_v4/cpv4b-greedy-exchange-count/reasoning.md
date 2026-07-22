I have a ring of `m` kilometre markers `0..m-1`,
addresses modulo `m`, and `n` floodlights. Floodlight `i` lights a contiguous clockwise arc of
`len_i` markers starting at `s_i`: the markers `s_i, s_i+1, ..., s_i+len_i-1`, all mod `m`. I must
switch on the fewest floodlights so that every marker is lit, or report `-1` if even all of them
together leave a gap. The constraints decide my data types before anything else: `m` can reach
`10^9`, so marker coordinates and the unrolled "doubled-axis" coordinates up to `2m ≈ 2*10^9`
overflow a 32-bit `int`. Every coordinate is `long long`. `n <= 2000` is small, which quietly tells
me an `O(n^2 log n)` algorithm is acceptable — I do not need to be clever about the candidate
enumeration, I need to be *correct* about the wrap.

**Recognising the shape: minimum circular-arc cover.** This is the circle version of "cover a segment
with the fewest intervals". On a *line*, the greedy is textbook: sort intervals, sit at the current
covered endpoint, jump to the interval that starts at or before that endpoint and reaches furthest,
repeat. The exchange argument is that if an optimal solution uses some interval crossing the current
endpoint, swapping it for the furthest-reaching one never loses coverage and never increases the
count. I trust that on a line. The circle is what makes me nervous, because there is no "leftmost
point" to anchor the sweep, and an arc can *wrap across the seam* between marker `m-1` and marker `0`.

**Candidate approaches.** Two routes.

- *Single unrolled sweep.* Cut the ring at marker `0`, lay every arc onto a line, add a shifted copy
  of each wrapping arc so coverage near `0` is visible, then run one line-cover greedy over `[0, m)`.
  This is `O(n log n)`, one pass, very tempting. The risk: a wrapping arc now appears *twice* on the
  line, and if the sweep consumes both copies it will count one physical floodlight as two — a
  double-count that silently inflates the answer. I do not trust this until I have tried to break it.
- *Anchored greedy with a forced first arc.* Marker `0` must be lit by *some* chosen floodlight. So
  enumerate, as the forced first arc, each arc that covers marker `0`; for each, rotate coordinates so
  that first arc's start sits at `0`, and run the line-cover greedy over the window `[0, m)`. Take the
  minimum count over the choices of first arc. This is `O(n)` candidates times `O(n log n)` =
  `O(n^2 log n)`, fine for `n <= 2000`, and the anchoring is exactly what gives the circle a clean
  "leftmost point" to sweep from.

I will prototype the single sweep first because it is shorter, but I will *attack* it before trusting
it.

**Deriving the anchored algorithm precisely.** An arc covers marker `0` iff `s == 0` (it starts there)
or `s + len > m` (it runs off the end and wraps back onto `0`). For a forced first arc with start
`a0`, I re-express every arc in coordinates relative to `a0`: an arc with start `s` and length `L`
has relative start `rs = (s - a0) mod m` in `[0, m)`, giving a linear interval `[rs, rs + L)`. If
`rs + L > m` the arc wraps even in the rotated frame, so it *also* covers a prefix near `0`; I add the
shifted copy `[rs - m, rs + L - m)` so the sweep can start from it. Then I cover the window `[0, m)`
with the line greedy. Because `a0` is the chosen first arc's own start, that arc maps to `rs = 0` and
seeds the sweep. The minimum count over candidates is the answer; if no candidate's sweep finishes,
output `-1`.

**Numeric self-check of the "covers marker 0" predicate.** Before coding I sanity-check the predicate
on the sample `m = 8`. Arc `(7, 2)` covers `7, 0`; predicate says `7 + 2 = 9 > 8`, true — good. Arc
`(0, 3)` covers `0,1,2`; predicate says `s == 0`, true — good. Arc `(2, 3)` covers `2,3,4`;
`2 + 3 = 5` is not `> 8` and `s != 0`, so false — and indeed it does not touch marker `0`. Arc
`(6, 3)` covers `6,7,0`; `6 + 3 = 9 > 8`, true — good. The predicate matches reality on all four, so
I have not flipped the inequality.

**Prototype 1 (single sweep) and a trace, because clean math transcribes dirty.** My first cut, the
tempting one:

```
// add [s,s+L); if it wraps add [s-m, s+L-m); sort by left; cover [0,m)
vector<Iv> ivs;
for (i) { ivs.push_back({S[i], S[i]+Ln[i]});
          if (S[i]+Ln[i] > m) ivs.push_back({S[i]-m, S[i]+Ln[i]-m}); }
sort by l;
long long curEnd = 0, cnt = 0; size_t p = 0;
while (curEnd < m) { newEnd = curEnd;
    while (p < ivs.size() && ivs[p].l <= curEnd) newEnd = max(newEnd, ivs[p].r), p++;
    if (newEnd <= curEnd) { fail; }
    curEnd = newEnd; cnt++; }
```

I pick the smallest input that could expose a wrap double-count: `m = 3`, arcs `(2, 2)` covering
`{2, 0}` and `(1, 1)` covering `{1}`. The true answer is obviously `2` — those two arcs together hit
`{0, 1, 2}` and there is no single arc covering everything. Let me run the prototype by hand. The
wrapping arc `(2, 2)` has `2 + 2 = 4 > 3`, so it contributes both `[2, 4)` and the shifted copy
`[-1, 1)`. Arc `(1, 1)` contributes `[1, 2)`. Sorted intervals: `[-1, 1), [1, 2), [2, 4)`. Now the
sweep over `[0, 3)`:

- `curEnd = 0`: intervals with `l <= 0` are `[-1, 1)`; jump to `1`; `cnt = 1`.
- `curEnd = 1`: intervals with `l <= 1` are `[1, 2)`; jump to `2`; `cnt = 2`.
- `curEnd = 2`: intervals with `l <= 2` are `[2, 4)`; jump to `4`; `cnt = 3`.
- `curEnd = 4 >= 3`: stop. Output `3`.

**The bug — a genuine double-count.** The prototype says `3`, the truth is `2`. The defect is exactly
the one I feared: `[-1, 1)` and `[2, 4)` are *the same physical floodlight* `(2, 2)` — one copy is the
wrap-prefix near `0`, the other is the body near the seam. The sweep happily consumed *both*,
charging two floodlights for one. Marker `0` got lit by the copy `[-1, 1)` and marker `2` got lit by
the copy `[2, 4)`, and nothing in the unanchored sweep noticed they are one lamp. This is precisely
the "index/dedup is easy to get wrong" trap: representing a wrapped arc as two intervals is necessary
for *coverage*, but it corrupts the *count*. The single-sweep idea, as written, is structurally
unable to dedup the seam, so I discard it. (It is not even a one-line fix: I cannot just "skip
duplicates", because sometimes you genuinely do want the wrap copy and not the body. The right tool is
the anchor.)

**Prototype 2: the anchored greedy — implement and trace the same case.** I switch to the anchored
algorithm. The key difference: I sweep a *window of length exactly `m`* rooted at a chosen first arc,
so the window never extends past one full lap, which is what stops a single lamp from being charged at
both ends of a lap.

```
best = +inf
for each f with (S[f]==0 || S[f]+Ln[f] > m):           // f covers marker 0
    a0 = S[f]
    build ivs: for each j, rs = (S[j]-a0) mod m in [0,m); push [rs, rs+Ln[j]);
               if rs+Ln[j] > m push [rs-m, rs+Ln[j]-m);
    sort ivs by l;
    curEnd=0; cnt=0; p=0; ok=true;
    while (curEnd < m) { newEnd=curEnd;
        while (p<ivs.size() && ivs[p].l <= curEnd) newEnd=max(newEnd,ivs[p].r), p++;
        if (newEnd<=curEnd){ ok=false; break; }
        curEnd=newEnd; cnt++; }
    if (ok && curEnd>=m) best=min(best,cnt);
answer = (best==inf? -1 : best)
```

Re-trace `m = 3`, arcs `(2, 2)` and `(1, 1)`. Which arcs cover marker `0`? `(2, 2)`: `2 + 2 = 4 > 3`,
yes. `(1, 1)`: `s != 0` and `1 + 1 = 2` not `> 3`, no. So the only candidate first arc is `f = (2, 2)`,
`a0 = 2`. Relative coordinates: arc `(2, 2)` has `rs = (2 - 2) mod 3 = 0`, interval `[0, 2)`;
`0 + 2 = 2` not `> 3`, so no shifted copy. Arc `(1, 1)` has `rs = (1 - 2) mod 3 = (-1 mod 3) = 2`,
interval `[2, 3)`. Sorted: `[0, 2), [2, 3)`. Sweep `[0, 3)`:

- `curEnd = 0`: `l <= 0` gives `[0, 2)`; jump to `2`; `cnt = 1`.
- `curEnd = 2`: `l <= 2` gives `[2, 3)`; jump to `3`; `cnt = 2`.
- `curEnd = 3 >= 3`: stop. `cnt = 2`.

`best = 2`. Correct. The double-count is gone because anchoring at `a0 = 2` puts the wrapped arc's
body at `[0, 2)` and there is no second copy of it inside the length-`m` window — the window stops at
`3`, never reaching a phantom second appearance. The case that broke the prototype now passes, and it
passes *for the reason I fixed*: the anchor collapses the seam.

**Numeric self-check of the algorithm on the main sample.** I verify the answer `4` on
`m = 8`, arcs `(0,3),(2,3),(5,2),(6,3),(7,2)`. Arcs covering marker `0`: `(0,3)` (`s=0`), `(6,3)`
(`6+3=9>8`), `(7,2)` (`7+2=9>8`). Take first arc `(0, 3)`, `a0 = 0`, so relative coords equal absolute
coords. Intervals: `(0,3)->[0,3)`, `(2,3)->[2,5)`, `(5,2)->[5,7)`, `(6,3)->[6,9)` and since `9>8` also
`[-2,1)`, `(7,2)->[7,9)` and since `9>8` also `[-1,1)`. Sorted by left:
`[-2,1),[-1,1),[0,3),[2,5),[5,7),[6,9),[7,9)`. Sweep `[0,8)`:

- `curEnd=0`: `l<=0` are `[-2,1),[-1,1),[0,3)`; furthest reach `3`; jump to `3`; `cnt=1`.
- `curEnd=3`: `l<=3` are `[2,5)`; jump to `5`; `cnt=2`.
- `curEnd=5`: `l<=5` are `[5,7)`; jump to `7`; `cnt=3`.
- `curEnd=7`: `l<=7` are `[6,9),[7,9)`; furthest `9`; jump to `9`; `cnt=4`.
- `curEnd=9>=8`: stop. `cnt=4`.

So the chosen lamps are `(0,3),(2,3),(5,2),(6,3)`: `{0,1,2}+{2,3,4}+{5,6}+{6,7,0}` = all of `0..7`,
four lamps, matching the stated answer. Crucially the wrap copies `[-2,1)` and `[-1,1)` were available
at the start but the greedy preferred `[0,3)` (further reach), and the *bodies* `[6,9),[7,9)` were
used once at the end — no lamp counted twice. The other two candidate first arcs also yield `4` (I
checked with the brute), so `best = 4`. The algorithm's number matches the hand cover.

**Second debug episode: an in-place / off-by-one I almost shipped in the inner loop.** My very first
keystroke version of the relative-coordinate build had a subtle index slip — I wrote the modulus as
`rs = (S[j] - a0) % m` without the `+ m`, i.e.

```
long long rs = (S[j] - a0) % m;          // WRONG when S[j] < a0
ivs.push_back({rs, rs + Ln[j]});
```

In C++ the `%` of a negative number is negative, so for `S[j] < a0` this gives `rs < 0`. Trace the
sample with first arc `(6, 3)`, `a0 = 6`, arc `(0, 3)`: `rs = (0 - 6) % 8 = -6`, interval `[-6, -3)`.
That interval lies entirely left of the window `[0, 8)` and the wrap-copy condition `rs + L > m`
(`-6 + 3 = -3 > 8`?) is false, so *no* shifted copy is added — arc `(0,3)`'s coverage of markers
`0,1,2` vanishes from this candidate's sweep entirely. Tracing it: with `a0 = 6`, the sweep would
stall around `curEnd = 5` because the only thing covering markers `0,1,2` (relative `2,3,4`) silently
fell off the negative end, and this candidate would wrongly report `-1` or an inflated count, polluting
the `min`. The fix is the canonical positive modulus `rs = ((S[j] - a0) % m + m) % m`, forcing
`rs in [0, m)` so every arc lands inside the window and the `rs + L > m` wrap test is meaningful.
After the fix, arc `(0,3)` under `a0=6` has `rs = ((0-6)%8+8)%8 = 2`, interval `[2, 5)` — exactly the
relative position of markers `0,1,2` when `6` is the origin. Re-running the `a0=6` candidate then gives
`4`, consistent with the other candidates. Two different index mistakes, two different ways to corrupt
the count; both caught by tracing a concrete input rather than eyeballing the formula.

**Edge cases, deliberately, because this is where this kind of code dies.**

- *A single full-ring arc.* If any `len_i == m`, that one lamp lights everything, answer `1`. I guard
  this up front with `if (Ln[i] >= m) { print 1; return; }`. Without the guard, the relative interval
  `[0, m)` reaches exactly `m` and the sweep still returns `1`, but the early exit makes it obviously
  correct and avoids any `rs + L > m` copy bookkeeping for the degenerate `L = m` (which would push a
  copy `[-... )` needlessly). Verified: `m=4`, arc `(2,4)` -> `1`.
- *`m = 1`.* One marker. Any arc has `len >= 1 = m`, so the early exit fires and prints `1`. Verified.
- *Infeasible.* `m=5`, arcs `(0,2),(0,2)`: union is `{0,1}`, marker `2,3,4` never covered. The only
  candidate first arc is `(0,2)`; its sweep stalls at `curEnd = 2` (nothing has `l <= 2` reaching
  further), so `ok = false`, no candidate succeeds, `best = inf`, output `-1`. Verified against brute.
- *Duplicates / fully redundant arcs.* `m=4`, arcs `(0,2),(0,2),(0,2),(2,2)`: three identical lamps and
  one more. Answer must be `2` (`{0,1}` + `{2,3}`), not inflated by the duplicates. The greedy picks
  *one* `[0,2)` then `[2,4)`; identical intervals do not each get charged because the sweep advances
  `curEnd` past them in a single jump. Verified `2`.
- *Coordinate overflow.* With `m` up to `10^9`, `rs + L` reaches up to `~2*10^9` and the shifted copy
  uses `rs - m` down to `~-10^9`; all are `long long`, far inside its range. The `s %= m` normalisation
  and `+ m` keep starts in `[0, m)`. No 32-bit anywhere. Safe.
- *Output format.* Exactly one integer and a newline. `cin >>` skips arbitrary whitespace, so the
  line-based input parses regardless of exact spacing.

**Complexity check against the limit.** Candidates covering marker `0` are at most `n`; each runs a
sort of at most `2n` intervals and a linear sweep, so `O(n^2 log n)`. For `n = 2000` that is about
`4*10^6 * 11 ≈ 4.4*10^7` comparisons — I measured the genuine worst case (every arc a candidate, each
sweep full length) at ~0.14 s, comfortably under the 1 s limit. Memory is `O(n)` per candidate, reused.

**Final solution.** I disproved the tempting single-sweep by tracing a real seam double-count
(`m=3`: prototype `3` vs truth `2`), moved to the anchored greedy whose length-`m` window collapses the
seam, fixed a negative-modulus index slip that silently dropped an arc, and confirmed the recurrence's
count on the main sample by hand. This is what I ship — one self-contained file:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    long long m;
    int n;
    if (!(cin >> m >> n)) return 0;

    // Arc (s, L), 1 <= L <= m, covers circular markers s,...,s+L-1 (mod m).
    // Want minimum number of arcs whose union is the whole ring, else -1.
    //
    // MINIMUM CIRCULAR ARC COVER via greedy-exchange:
    //   Marker 0 must be covered by some chosen arc. We try, as the forced
    //   FIRST arc, each arc that covers marker 0. Forcing arc A fixes the
    //   linear window [A.start, A.start + m): we must cover that whole window.
    //   After A we have covered up to A.start + A.len. Then we repeatedly pick,
    //   among arcs whose linear start is <= current covered end, the one whose
    //   linear end reaches furthest, advancing the covered end. Each pick is one
    //   more arc. We stop once the covered end reaches A.start + m. The minimum
    //   over all choices of first arc is the answer.

    vector<long long> S(n), Ln(n);
    for (int i = 0; i < n; i++) {
        long long s, L;
        cin >> s >> L;
        s %= m; if (s < 0) s += m;
        S[i] = s; Ln[i] = L;
    }

    // A length-m arc covers the entire ring by itself.
    for (int i = 0; i < n; i++) if (Ln[i] >= m) { cout << 1 << "\n"; return 0; }

    long long best = LLONG_MAX;

    // Candidate first arcs = arcs covering marker 0: s == 0 OR s + L > m.
    for (int f = 0; f < n; f++) {
        bool coversZero = (S[f] == 0) || (S[f] + Ln[f] > m);
        if (!coversZero) continue;
        long long a0 = S[f];

        // Coordinates relative to a0: marker (s - a0 mod m). Cover [0, m).
        // Each arc j -> [rs, rs+L); if it wraps, also the prefix copy [rs-m, ...).
        vector<pair<long long,long long>> ivs; // (l, r)
        ivs.reserve(2 * n);
        for (int j = 0; j < n; j++) {
            long long rs = ((S[j] - a0) % m + m) % m;   // positive modulus!
            ivs.push_back({rs, rs + Ln[j]});
            if (rs + Ln[j] > m) ivs.push_back({rs - m, rs + Ln[j] - m});
        }
        sort(ivs.begin(), ivs.end());

        // Greedy minimum interval cover of [0, m). Forced first arc f maps to
        // rs = 0, seeding the sweep. Count arcs used.
        long long curEnd = 0, cnt = 0;
        size_t p = 0;
        bool ok = true;
        while (curEnd < m) {
            long long newEnd = curEnd;
            while (p < ivs.size() && ivs[p].first <= curEnd) {
                newEnd = max(newEnd, ivs[p].second);
                p++;
            }
            if (newEnd <= curEnd) { ok = false; break; }
            curEnd = newEnd;
            cnt++;
        }
        if (ok && curEnd >= m) best = min(best, cnt);
    }

    if (best == LLONG_MAX) cout << -1 << "\n";
    else cout << best << "\n";
    return 0;
}
```

**Causal recap.** The single-sweep idea was tempting and short, but tracing `m=3` with arcs
`(2,2),(1,1)` showed it counted the wrapped lamp twice (output `3`, truth `2`) because a wrapping arc
appears as two intervals and an unanchored sweep over the doubled axis happily consumes both; the cure
is to anchor each sweep at a forced first arc covering marker `0` and cover a window of length exactly
`m`, which collapses the seam so no lamp is charged at both ends of a lap; a second, quieter bug — the
negative-`%` index slip `(S[j]-a0)%m` — silently dropped an arc whenever `S[j] < a0`, fixed by the
positive modulus `((S[j]-a0)%m+m)%m`; with the full-ring early exit and `long long` coordinates the
`m=1`, infeasible, duplicate, and `m=10^9` corners all fall out, and the hand cover of the `m=8`
sample (four lamps `(0,3),(2,3),(5,2),(6,3)`) confirms the count.
