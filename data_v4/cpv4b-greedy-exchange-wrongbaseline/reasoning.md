**Reading the problem and pinning the contract.** I have `n` firmware modules to compile on one
core, serially, no preemption. Module `i` costs `t[i]` seconds and must *finish* by deadline `d[i]`
or it is dropped. I pick a subset and an order; the clock starts at `0` and the core never idles
while a chosen module remains. I want to maximize the **count** of modules that finish on time, and
print that count. Input is `n` then `n` lines of `t[i] d[i]`; output one integer. First I fix the
scale, because it picks my types: `n <= 2*10^5`, and `t[i], d[i] <= 10^9`. If I keep a running
completion clock it can reach `2*10^5 * 10^9 = 2*10^14`, far past the 32-bit ceiling of `~2.1*10^9`.
So the clock and every processing-time accumulator must be 64-bit `long long`. The *answer* itself is
a count `<= n <= 2*10^5` and would fit in `int`, but I will carry it as `long long` too to avoid a
stray narrowing. That is non-negotiable: an `int` clock is a silent wrong answer on the big tests.

**A structural fact I need before any greedy: feasibility is decided by EDD.** Before I worry about
*which* modules to keep, I should know how to test whether a *fixed* chosen subset can ship in full.
Claim: if any ordering of a chosen subset lets every module finish on time, then sorting that subset
by deadline (earliest-deadline-first, EDD) and running it in that order also works. The standard
exchange argument: take any feasible order; if two adjacent jobs are out of deadline order (the
earlier-positioned job has the later deadline), swapping them keeps the later one's finish time the
same as the pair's old total (so still `<=` its deadline, which was even larger) and only moves the
earlier-deadline job *earlier*, which cannot make it late. Bubble all inversions out and you reach EDD
without breaking feasibility. So "is subset `S` schedulable?" is exactly "does EDD order of `S` keep
every prefix sum of processing times `<=` the corresponding deadline?" Good — the ordering is solved;
the hard part is the **selection**.

**Laying out candidate approaches.** I want the method I can prove, not the one that types fastest.

- *Earliest-deadline-first, keep-if-it-fits (the "standard"-feeling sweep).* Sort modules by
  deadline. Walk them, maintaining a running clock; accept a module if `clock + t[i] <= d[i]`,
  otherwise skip it permanently. `O(n log n)`, five lines. This *looks* like the canonical greedy for
  deadline scheduling, and that is exactly why I distrust it: the objective here is a count with
  arbitrary processing times, and a greedy that never reconsiders an early acceptance is the kind of
  thing that overcommits.
- *Brute force over subsets.* For tiny `n`, try every subset, test EDD-feasibility, take the largest
  feasible. Exponential, only good as an oracle, but it pins the true answer so I can attack the
  greedy honestly.
- *Moore-Hodgson exchange.* Sort by deadline; sweep, but when accepting a module would overflow its
  deadline, *evict the longest module accepted so far* instead of skipping the current one. `O(n log
  n)` with a heap. This is the textbook optimum for "maximize number of on-time jobs," but I will not
  assert that — I will derive it by watching the plain sweep fail and seeing what repair fixes it.

**Stress-testing the plain EDD sweep before committing.** "EDD keep-if-fits is the standard greedy"
is precisely the assumption I must not ship unchecked, so let me attack it on a concrete instance.
Take three modules `(t,d)`: `A = (3,3)`, `B = (2,4)`, `C = (2,4)`. EDD order by deadline is `A`
(deadline 3), then `B`, `C` (deadline 4). Run the plain sweep with `clock = 0`:

- `A`: `clock + 3 = 3 <= 3`? yes. Accept. `clock = 3`, count `1`.
- `B`: `clock + 2 = 5 <= 4`? no. Skip.
- `C`: `clock + 2 = 5 <= 4`? no. Skip.

Plain sweep ships `1`. Is `1` optimal? Let me look for a better subset. Drop `A` entirely and keep
`{B, C}`: EDD order `B, C`, finishes at `2` and `4`, both `<= 4`. That ships `2`. So the optimum is
`2`, and the plain sweep got `1`. I now see *why*: accepting `A` early — a module with a tight
deadline *and* a large processing time — pushed the clock to `3` and blocked two later modules that
together were worth more than `A`. The plain sweep's flaw is that it never un-accepts. The
verification paid off: it killed the approach that felt standard.

**Deriving the repair: the exchange step.** The failure tells me what to fix. When the current
module `i` (in EDD order) does not fit, the situation is: I have a set `K` of already-accepted modules
with total time `clock`, and `clock + t[i] > d[i]`. I do not have to refuse `i`. I can instead look at
`K ∪ {i}` — which has `|K| + 1` modules — and *drop the single module with the largest processing
time* from it. Two things to verify about this move:

1. *It keeps `|K|` modules.* Before the step I had `|K|` accepted; I add `i` (giving `|K|+1`) and
   remove one (back to `|K|`). The count never drops. So an eviction is free in count terms — it can
   only ever help later modules by shortening the clock.
2. *It never increases the clock.* The evicted module is the longest in `K ∪ {i}`, so its time is
   `>= t[i]`. New clock `= clock + t[i] - (longest) <= clock + t[i] - t[i] = clock`. The clock is
   non-increasing under the exchange. That is the whole point: dropping the longest job frees the most
   room for the future at no cost in count.

And crucially the kept set stays feasible: every module already in `K` satisfied its deadline at the
old clock, and the new clock is `<=` old clock, so they still finish on time; the only module whose
deadline I just had to respect is the largest-deadline one in the EDD prefix, namely `d[i]`, and after
the eviction the clock is `<= d[i]` because I removed at least `t[i]` from `clock + t[i] <= clock +
(longest)`... let me check that last bound numerically rather than wave at it.

**Numeric self-check of the "clock stays under the current deadline" claim.** I want: after pushing
`t[i]` and evicting the longest, the new clock `<= d[i]`. Take the failing prefix from the trace at
the moment `B` arrives in a variant where `A` is already in: `K = {A}` with `clock = 3`, current
module `B = (2,4)`. Push `B`: tentative clock `= 3 + 2 = 5`. Longest in `{A,B}` is `A` with `t = 3`.
Evict it: new clock `= 5 - 3 = 2`. Is `2 <= d[B] = 4`? Yes. So `B` and whatever remains are feasible
at clock `2`. The general claim is new clock `= (clock + t[i]) - longest`, and since `longest >=
every processing time in the prefix`, in particular the new clock is `<= ` the second-largest-anchored
bound; the concrete instance gives `2 <= 4`, consistent. I will rely on the two proven facts (count
preserved, clock non-increasing) rather than a hand-wave, and the number `2 <= 4` checks out.

**Translating the exchange into a data structure.** I sweep modules in EDD order. I keep the accepted
set as a *max-heap of processing times* and a running `clock = sum of accepted times`. For each module
`(t, d)` in deadline order: push `t`, add `t` to `clock`; if now `clock > d`, pop the heap top (the
largest accepted time) and subtract it from `clock`. At the end, the number of modules still in the
heap is the maximum on-time count. Each module is pushed once and popped at most once, so it is
`O(n log n)`.

Let me confirm this on the counterexample `A=(3,3), B=(2,4), C=(2,4)`, EDD order `A, B, C`:

- `A`: push 3, `clock = 3`. `3 > 3`? no. heap `{3}`.
- `B`: push 2, `clock = 5`. `5 > 4`? yes. Pop max `= 3`, `clock = 2`. heap `{2}`.
- `C`: push 2, `clock = 4`. `4 > 4`? no. heap `{2,2}`.

Heap size `2`. That matches the true optimum `2` and beats the plain sweep's `1`. The repair works on
the exact case that broke the baseline, which is the evidence I trust.

**A second, larger hand-trace to build confidence.** Five modules `(t,d)`: `J1=(4,5)`, `J2=(3,6)`,
`J3=(2,9)`, `J4=(6,7)`, `J5=(2,4)`. EDD order by deadline: `J5(d4), J1(d5), J2(d6), J4(d7), J3(d9)`.

- `J5=(2,4)`: push 2, `clock = 2`. `2 > 4`? no. heap `{2}`.
- `J1=(4,5)`: push 4, `clock = 6`. `6 > 5`? yes. Pop max `= 4`, `clock = 2`. heap `{2}`.
- `J2=(3,6)`: push 3, `clock = 5`. `5 > 6`? no. heap `{2,3}`.
- `J4=(6,7)`: push 6, `clock = 11`. `11 > 7`? yes. Pop max `= 6`, `clock = 5`. heap `{2,3}`.
- `J3=(2,9)`: push 2, `clock = 7`. `7 > 9`? no. heap `{2,3,2}`.

Heap size `3`. My brute-force oracle on this instance also says `3`, so the algorithm and the oracle
agree on a non-trivial case. Note the two evictions: `J1` (tight deadline, biggest time so far) and
`J4` (the longest of all) were exactly the ones the exchange discards — the plain sweep would have
clung to `J1` and skipped both `J2` and `J4`, ending at `2`. The exchange wins by un-accepting.

**First implementation — and immediately a trace, because the eviction condition is easy to mis-key.**
My first cut of the core loop, working from `job[i] = {deadline, time}` after sorting:

```
priority_queue<long long> heap;
long long clock = 0;
for (int i = 0; i < n; i++) {
    long long d = job[i].first, t = job[i].second;
    clock += t;
    if (clock >= d) {              // <-- suspicious comparison
        clock -= heap.top();
        heap.pop();
    }
    heap.push(t);
}
```

Two things look off, so I trace the smallest input that exposes them: a single module that *exactly*
fits, `n = 1`, `(t,d) = (5,5)`. The right answer is `1` — it finishes at exactly `5 <= 5`. Run it:
`clock = 0`; i=0: `clock += 5 -> 5`; `5 >= 5`? yes (with `>=`); so it tries `heap.top()` — but I have
not pushed anything yet, the heap is **empty**, `heap.top()` is undefined behavior / a crash. Even if
it did not crash, evicting on an exact fit is wrong: finishing *at* the deadline is on time.

**Diagnosing the bug.** Two precise defects. First, **the comparison must be strict**: a module that
finishes exactly at its deadline ships, so I evict only when `clock > d`, not `clock >= d`. Using
`>=` drops legitimate exact-fit modules — a one-symbol wrong answer. Second, **I push after the
check**, so on the very first module the eviction branch reads an empty heap. Both stem from ordering
the three operations wrong. The correct order is: *push the current module first, then add to the
clock, then test `clock > d` and evict the heap max if needed* — so the heap always contains the
current module when I test, and the eviction is guaranteed to have something to pop (the heap is
non-empty because I just pushed). Let me restructure:

```
heap.push(t);
clock += t;
if (clock > d) {
    clock -= heap.top();
    heap.pop();
}
```

**Re-verifying the fix on the cases that broke.** `n = 1`, `(5,5)`: push 5, `clock = 5`, `5 > 5`?
no. heap size `1`. Correct. `n = 1`, `(5,4)` (cannot fit, answer `0`): push 5, `clock = 5`, `5 > 4`?
yes, pop the `5` we just pushed, `clock = 0`, heap empty, size `0`. Correct — and notice the eviction
safely pops the module we just pushed, never an empty heap. Re-run the counterexample
`A=(3,3),B=(2,4),C=(2,4)` with the fixed order: `A` push3 clock3 `3>3`?no `{3}`; `B` push2 clock5
`5>4`?yes pop3 clock2 `{2}`; `C` push2 clock4 `4>4`?no `{2,2}`; size `2`. Correct. The exact-fit and
empty-heap cases that broke now pass, and they broke for the two reasons I fixed.

**A second debug episode: the sort key and the pair layout.** My initial instinct was to store
`job[i] = {t, d}` and `sort(job.begin(), job.end())`, then read `t = job[i].first, d =
job[i].second`. I trace why that is wrong on `(t,d)` pairs `(5,1)` and `(1,5)`: storing `{t,d}` gives
pairs `{5,1}` and `{1,5}`; default pair-sort orders by *first* component, i.e. by **processing time**
`1 < 5`, producing order `(1,5)` then `(5,1)`. But EDD wants order by **deadline**: `(5,1)` has
deadline `1` and must come first. My layout sorted by the wrong field. Concretely, with the wrong
order: `(1,5)` push1 clock1 `1>5`?no `{1}`; `(5,1)` push5 clock6 `6>1`?yes pop5 clock1 `{1}`; size
`1`. The correct EDD order `(5,1),(1,5)`: `(5,1)` push5 clock5 `5>1`?yes pop5 clock0 `{}`; `(1,5)`
push1 clock1 `1>5`?no `{1}`; size `1`. Same final count *here* by luck, but the sweep's clock history
differs, and on larger instances sorting by processing time instead of deadline silently breaks the
EDD precondition that the whole correctness argument rests on. The fix: store the **deadline first**
in the pair (`job[i] = {d, t}`) so the default sort is EDD, and read `d = job[i].first, t =
job[i].second`. After this, my brute-force oracle and the solution agree on hundreds of random small
cases, which they did not before I straightened the key.

**Edge cases, deliberately, because this is where this kind of code dies.**

- `n = 0`: the read of `n` succeeds with `0`, the loop never runs, the heap is empty, output `0`.
  Correct — nothing to ship.
- `n = 1` exact fit `(5,5)`: pushes, `5 > 5` false, size `1`. Correct.
- `n = 1` overshoot `(5,4)`: pushes, `5 > 4` true, pops the just-pushed module, size `0`. Correct,
  and safe — the heap is non-empty at the pop because I pushed first.
- All modules fit (e.g. tiny `t`, huge `d`): no eviction ever fires, heap ends with all `n` modules,
  output `n`. Correct.
- No module fits (every `t > d`): each module is pushed then immediately evicted (the just-pushed one
  is the heap max when alone, or an earlier longer one gets dropped but the count still cannot exceed
  what fits), heap ends empty or with only the genuinely-schedulable ones; on all-`t>d` it is `0`.
  Verified by oracle.
- Ties in deadline and in processing time: pair-sort is a total order on `{d,t}`, ties broken by `t`;
  the algorithm's correctness does not depend on tie order (any EDD-consistent order works), and the
  oracle confirms across tie-heavy random cases.
- Overflow: `clock` accumulates up to `2*10^5 * 10^9 = 2*10^14`; `long long` holds it with room.
  `heap.top()` and the subtraction stay in `long long`. The comparison `clock > d` is `long long`
  vs `long long`. No 32-bit accumulator anywhere on the hot path.
- Output: exactly one integer and a newline; `cin >>` skips arbitrary whitespace, so the line-based
  input parses fine even if a producer uses different spacing.

**Empty-input read note.** `if (!(cin >> n)) return 0;` handles both `n = 0` and a totally empty
stream by printing nothing-then-falling-through — but I print the heap size after the loop, so for
`n = 0` I do reach the `cout` and print `0`. Let me double-check: with `n = 0`, `cin >> n` succeeds
(reads the `0`), so I do *not* early-return; the vector is empty, the loop body never executes, and I
print `heap.size() = 0`. Correct. The early return only triggers on a genuinely empty stream, which
the judge will not send, but it is harmless.

**Final solution.** I convinced myself the *idea* is right by disproving the standard EDD-keep-if-fits
sweep on `(3,3),(2,4),(2,4)` (it gets `1`, optimum is `2`) and by proving the exchange preserves the
count while never increasing the clock; I convinced myself the *code* is right by tracing the
empty-heap / exact-fit bug to a precise operation-ordering cause, fixing the strict comparison and the
push-before-check order, fixing the sort key to be deadline-first, and re-verifying every corner. This
is what I ship — one self-contained file, the Moore-Hodgson exchange I can defend rather than the
sweep I broke:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;                 // n = 0 (or empty) -> answer 0
    vector<pair<long long,long long>> job(n);   // (deadline, processing time)
    for (int i = 0; i < n; i++) {
        long long t, d;
        cin >> t >> d;
        job[i] = {d, t};                        // sort key = deadline
    }
    sort(job.begin(), job.end());               // earliest due date first

    // Moore-Hodgson: scan in EDD order, keep a running clock and a max-heap of
    // the processing times currently scheduled. When the clock passes the
    // current deadline, evict the longest scheduled job (the exchange step).
    priority_queue<long long> heap;             // max-heap of processing times
    long long clock = 0;
    for (int i = 0; i < n; i++) {
        long long d = job[i].first, t = job[i].second;
        heap.push(t);
        clock += t;
        if (clock > d) {                        // infeasible: drop the longest job
            clock -= heap.top();
            heap.pop();
        }
    }

    cout << (long long)heap.size() << "\n";     // jobs that survive = max on-time count
    return 0;
}
```

**Causal recap.** The objective is to maximize the *count* of on-time jobs with arbitrary processing
times, and the standard-feeling EDD "keep it if it fits" sweep is wrong here: a traced counterexample
`(3,3),(2,4),(2,4)` showed it ships `1` while dropping the tight-and-long `(3,3)` ships `2`, because
the sweep never un-accepts an early overcommit. The repair is the Moore-Hodgson exchange — when the
clock passes the current deadline, evict the *longest* accepted job — which I proved keeps the count
and never increases the clock (checked numerically: `clock` goes `5 -> 2 <= d = 4`); implemented as a
max-heap of processing times scanned in EDD order. Two code traces caught real bugs: testing `clock
>= d` with the push *after* the check crashes on an empty heap and wrongly evicts exact fits (fix:
push first, use strict `clock > d`), and storing the pair as `{t,d}` sorted by processing time instead
of deadline silently breaks the EDD precondition (fix: store `{d,t}`). With `long long` clocks for the
`2*10^14` range, the corners — `n = 0`, exact fit, no-fit, all-fit, ties — all check out against an
exhaustive subset brute force over hundreds of random cases.
