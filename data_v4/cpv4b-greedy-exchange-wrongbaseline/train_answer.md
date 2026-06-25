**Problem.** A single build core compiles firmware modules serially with no preemption. Module `i`
costs `t[i]` seconds and must *finish* by deadline `d[i]` or it is dropped. The clock starts at `0`
and never idles while a chosen module remains. Choose a subset and an order to maximize the **number**
of modules that finish on time, and print that count. Read `n` then `n` lines of `t[i] d[i]` from
stdin; `0 <= n <= 2*10^5`, `1 <= t[i], d[i] <= 10^9`.

**Why the obvious greedy is wrong.** It is tempting to run the "standard" deadline-scheduling sweep:
sort by deadline (EDD), walk the modules, accept each one if it still fits (`clock + t[i] <= d[i]`),
otherwise skip it forever. That is *not* optimal for maximizing a **count** with arbitrary processing
times. On `(t,d) = (3,3), (2,4), (2,4)` the sweep accepts the tight-and-long `(3,3)` first (clock
becomes `3`), then neither `(2,4)` fits, so it ships `1`. But dropping `(3,3)` lets both `(2,4)`
modules ship (finishing at `2` and `4`), for `2`. The sweep never un-accepts an early overcommit, so
a short-deadline but long module can block two later ones worth more. The plain sweep is discarded.

**Key idea — the Moore-Hodgson exchange.** First, a fixed subset is schedulable iff its EDD order
keeps every cumulative finish time within deadline (the standard adjacent-swap argument), so ordering
is free and only *selection* is hard. Sweep modules in EDD order maintaining a running `clock` and a
**max-heap of the processing times currently accepted**. For each module `(t, d)`: push `t`, add `t`
to `clock`; if now `clock > d`, the prefix is infeasible, so **evict the longest accepted module**
(pop the heap max, subtract it from `clock`). The surviving heap size at the end is the maximum
on-time count.

Two facts make the exchange correct. (1) *Count is preserved*: each overflow adds one module then
removes one, so the accepted count never drops — eviction is free. (2) *The clock never increases*:
the evicted module is the longest in the current set, so its time is `>= t[i]`, giving new `clock =
clock + t[i] - longest <= clock`. A shorter clock can only help later modules, and every previously
accepted module stays on time because its deadline was already met at the old (larger) clock.

**Pitfalls.**
1. *The wrong baseline.* EDD keep-if-fits with no eviction is the trap. It feels canonical (it *is*
   canonical for *minimizing maximum lateness*), but for *maximizing the on-time count* you must add
   the eviction step. Verify before trusting: the sweep gives `1` on `(3,3),(2,4),(2,4)` where `2` is
   reachable.
2. *Operation order and a strict comparison.* Push the current module **before** testing, and evict
   only on `clock > d` (strict). Testing `clock >= d` wrongly drops modules that finish exactly at
   their deadline (on time), and checking before the push reads an empty heap on the first overflow
   (undefined behavior). Pushing first guarantees the heap is non-empty when you pop.
3. *Sort key.* Store the pair as `{deadline, time}` so the default sort is EDD. Storing `{time,
   deadline}` sorts by processing time and silently breaks the EDD precondition the whole proof rests
   on.
4. *Overflow.* The clock reaches `2*10^5 * 10^9 = 2*10^14`; use `long long`. An `int` clock is a
   silent wrong answer on the large tests.

**Edge cases.** `n = 0` -> `0` (empty heap). Single module exact fit `(5,5)` -> `1` (strict `>`).
Single module overshoot `(5,4)` -> `0` (pops the just-pushed module, heap safe). All modules fit ->
`n` (no eviction fires). No module fits -> `0`. Ties in deadline/time are harmless because any
EDD-consistent order works. All verified against an exhaustive subset brute force over hundreds of
random small cases.

**Complexity.** Sorting is `O(n log n)`; each module is pushed once and popped at most once, so the
heap sweep is `O(n log n)`. `O(n)` space. Comfortably within the 1-second limit at `n = 2*10^5`.

**Code.**

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
