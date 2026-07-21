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
