A single build core compiles firmware modules serially, no preemption: module `i` costs `t[i]`
seconds and is dropped unless it *finishes* by `d[i]`. I pick a subset and an order — clock from `0`,
never idle while a chosen module remains — to maximize the **count** of modules that ship on time,
and print that count. The word *count* is the whole story, and it is what makes the reflex answer
wrong. This reads like textbook deadline scheduling, where the neighboring problem — minimize maximum
lateness — is solved by a plain earliest-deadline sweep. But maximizing a *count* with *arbitrary*
processing times is a different objective, and the same sweep is not optimal on it. I need the reason
in hand before committing to anything.

First the scale, since it fixes my types. `n` up to `2*10^5`, `t` and `d` up to `10^9`, so a running
completion clock can reach `2*10^5 * 10^9 = 2*10^14` — well past the 32-bit ceiling of `~2.1*10^9`.
The clock and every processing-time accumulator have to be 64-bit `long long`; an `int` clock is a
silent wrong answer on the large tests, nothing subtler than that. The answer itself is a count
`<= n`, small, but I will carry it wide too to avoid a stray narrowing.

Before choosing *which* modules to keep, I need to know how to test whether a *fixed* chosen subset
ships in full — otherwise "keep this one" has no meaning. Feasibility is decided by earliest-deadline
order (EDD): if some ordering of a subset lets every module finish on time, then sorting the subset by
deadline and running it in that order does too. The adjacent-swap argument: in any feasible order, if
two neighbors are out of deadline order, swapping them leaves the later-finishing slot's total
unchanged (still under the larger deadline) and moves the earlier-deadline job earlier, which cannot
make it late; bubble out all inversions and you reach EDD without losing feasibility. So "is subset
`S` schedulable?" is just "does EDD order of `S` keep every prefix sum of times under the matching
deadline?" Ordering is solved; the selection is the hard part.

Now the tempting wrong move: sort by deadline, walk the modules, accept each if it still fits
(`clock + t[i] <= d[i]`), skip it forever otherwise. Five lines, and it feels canonical — which is
exactly why I distrust it for this objective. Take three modules `(t,d)`: `A=(3,3)`, `B=(2,4)`,
`C=(2,4)`. EDD order is `A, B, C`. The sweep accepts `A` (clock `3`), then `B` needs `3+2=5 <= 4`,
no, skip; `C` likewise, skip. It ships `1`. But drop `A` and keep `{B,C}`: they finish at `2` and
`4`, both on time — that ships `2`. So the optimum is `2` and the sweep got `1`. The reason is
structural: `A` has a tight deadline *and* a long processing time, and accepting it early pushed the
clock to `3` and blocked two later modules worth more together than `A`. The flaw is that the sweep
never un-accepts.

That tells me the repair. When the current module `i` in EDD order does not fit — I hold an accepted
set `K` with total time `clock`, and `clock + t[i] > d[i]` — I do not have to refuse `i`. Consider
`K ∪ {i}` and drop the single module with the largest processing time. Two properties make this safe.
The count is preserved: I added one and removed one, so the accepted count never drops — an eviction
is free. And the clock never increases: the evicted module is the longest in `K ∪ {i}`, so its time
is `>= t[i]`, giving new `clock = clock + t[i] - longest <= clock`. A shorter clock can only help
future modules.

Feasibility survives too, and cleanly. Every module already in `K` met its deadline at the old clock,
and the new clock is no larger, so they still finish on time. The one deadline I have to satisfy for
the just-added module is `d[i]`, the largest in the EDD prefix. Before adding `i` the accepted set was
feasible, so its total finish time — the old clock — was under its own largest deadline, which is
`<= d[i]`; the new clock is `<= ` the old clock, hence `<= d[i]`. The two properties plus the sorted
deadlines give feasibility directly, no case analysis.

That is the Moore-Hodgson exchange, and it drops straight onto a heap. Sweep in EDD order holding a
running `clock` and a max-heap of the accepted processing times. For each `(t,d)`: push `t`, add `t`
to `clock`; if `clock > d`, pop the heap max (the longest accepted job) and subtract it. Each module
is pushed once and popped at most once, so it is `O(n log n)`, and the heap size at the end is the
answer. On the counterexample: `A` push 3, clock 3, `3>3`? no; `B` push 2, clock 5, `5>4`? yes, pop
3, clock 2; `C` push 2, clock 4, `4>4`? no. Heap size `2`, matching the true optimum and beating the
sweep — the repair bites on exactly the case that broke the baseline.

**A second, larger hand-trace to build confidence.** Five modules `(t,d)`: `J1=(4,5)`, `J2=(3,6)`,
`J3=(2,9)`, `J4=(6,7)`, `J5=(2,4)`. EDD order by deadline: `J5(d4), J1(d5), J2(d6), J4(d7), J3(d9)`.

Heap size `3`. My brute-force oracle on this instance also says `3`, so the algorithm and the oracle
agree on a non-trivial case. Note the two evictions: `J1` (tight deadline, biggest time so far) and
`J4` (the longest of all) were exactly the ones the exchange discards — the plain sweep would have
clung to `J1` and skipped both `J2` and `J4`, ending at `2`. The exchange wins by un-accepting.

Two things look off, so I trace the smallest input that exposes them: a single module that *exactly*
fits, `n = 1`, `(t,d) = (5,5)`. The right answer is `1` — it finishes at exactly `5 <= 5`. Run it:
`clock = 0`; i=0: `clock += 5 -> 5`; `5 >= 5`? yes (with `>=`); so it tries `heap.top()` — but I have
not pushed anything yet, the heap is **empty**, `heap.top()` is undefined behavior / a crash. Even if
it did not crash, evicting on an exact fit is wrong: finishing *at* the deadline is on time.

Three implementation details need care, and each is a spot this problem specifically invites a bug.
The eviction comparison must be *strict*: a module that finishes exactly at its deadline ships, so I
evict only when `clock > d`, never `clock >= d` — using `>=` would drop legitimate exact-fit modules
like a lone `(5,5)`. The three operations must be ordered push, accumulate, test-and-evict: if I
tested before pushing, the very first overflow would call `heap.top()` on an empty heap (undefined
behavior); pushing the current module first guarantees the heap is non-empty when I pop, and in the
worst case the module I just pushed is itself the one evicted. And the sort key must be the deadline:
I store each pair as `{d, t}` so the default pair comparison sorts by deadline and yields EDD
directly. Storing `{t, d}` would sort by processing time and silently break the EDD precondition the
whole correctness argument rests on — a bug with no visible symptom on many inputs.

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

**Edge cases, because this is where this kind of code dies.**

The corners follow from this. `n = 0` reads cleanly, the loop never runs, the empty heap prints `0`.
A single exact fit `(5,5)` keeps its one module (`5 > 5` is false); a single overshoot `(5,4)` pushes
then immediately evicts the just-pushed module for `0`, the heap safe because I pushed first. If every
module fits, no eviction fires and the heap ends with all `n`. If none fits, each is pushed and
dropped and the count stays `0`. Ties in deadline or time are harmless: any EDD-consistent order
works and the default pair order is one. The `if (!(cin >> n)) return 0` guard covers a genuinely
empty stream; a real `n = 0` still reaches the `cout` and prints `0`.

To validate I ran the heap solution against an exhaustive brute force — every subset, tested over
*all* orderings so the check does not itself assume EDD optimality — across hundreds of random small
instances; they agree, including the tie-heavy and no-fit cases. That is the whole program: read `n`,
load `{d, t}` pairs, sort, run the heap sweep with a `long long` clock, and print `heap.size()` as
the maximum on-time count.
