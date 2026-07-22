# The Wet-Ink Scriptorium: A Two-Timescale Shelving Script

A monastery has a single copying **desk** that holds at most **k** pages at once. A fixed
liturgy of `M` accesses to numbered pages is known **in advance** (you plan offline). Each
access either **reads** a page (`R`) or **inks** it (`W`). A page that has been inked since it
was last brought to the desk is **wet**; a page never inked since arrival is **dry**.

You write a **shelving script**: an offline plan that decides, for every forced eviction, which
resident page leaves the desk, plus any **proactive drying** you choose to perform.

## Charges

- **Fetch** `F`: every access to a page not currently on the desk (a *miss*).
- **Clean-shelve** `Ce`: shelving a **dry** page to free a slot.
- **Wet-shelve** `De`: shelving a **wet** page (its ink must be blotted as it leaves). `De > Ce`.
- **Proactive dry** `Pc`: drying a resident wet page *now*, before it is shelved. It becomes dry.
  Guaranteed `Pc < De - Ce`, so drying early is always cheaper than a wet shelving — **if** the
  page is not re-inked before it leaves.

Bringing a page to the desk on a `W` access arrives it **wet**; on an `R` access, **dry**.
When the liturgy ends, every page still on the desk that is **wet** must be blotted: a final
charge of `De - Ce` each. Dry residents cost nothing at the end. The five charges
`k F Ce De Pc` are given per instance — read them.

## Input (stdin)

```
k F Ce De Pc
M
<op_1>
...
<op_M>
```

Each `op_i` is `R p` or `W p`, with page id `p` an integer `>= 0`. Op indices are `0..M-1`.

## Output (stdout): the shelving script

```
A
<action_1>
...
<action_A>
```

Each action is one of:

- `EVICT t v` — at op index `t`, evict resident page `v`. Required **exactly** at every op that
  is a miss while the desk already holds `k` pages; forbidden at any other op.
- `CLEAN t v` — just **before** op `t` is processed, proactively dry resident page `v` (charge
  `Pc`). `v` must be resident at that moment.

Replay is deterministic: the desk starts empty; misses with a free slot need no `EVICT`; a `W`
sets its page wet, an `R` on a resident page leaves its state unchanged.

## Feasibility (score 0 on any violation)

Non-integer / non-finite tokens; `A` out of range; an action time outside `0..M-1`; a `CLEAN`
of a non-resident page; a missing or duplicate `EVICT` at a forced miss; an `EVICT` whose victim
is not resident; or an `EVICT` at an op that is not a forced full-desk miss.

## Objective

Minimize the **total charge** = fetches + shelvings + proactive dryings + the final blotting.

## Scoring

Let `cost` be your total charge and `B` the charge of the checker's own **least-recently-used,
never-dry** baseline. Score (minimization):

```
Ratio = min(1.0, 0.1 * B / max(1, cost))
```

Reproducing the baseline scores ≈ `0.1`; a 10×-cheaper script caps at `1.0`. Lower cost is
strictly better and there is no known optimal script.

## Why it is hard

The liturgy sweeps loops of `k+2` distinct pages (one more than the desk can hold), so any
recency rule thrashes, and the inking happens in bursts. **Belady's farthest-in-future** rule
minimizes fetches — but it keeps shelving pages while they are wet, paying `De` again and again.
The cheap move is to schedule proactive dryings during the quiet read/idle stretches so that the
pages Belady is about to shelve leave **dry** — a two-timescale plan (choose *what* to keep on
the fast per-miss clock; choose *when* to pay the ink on the slow per-loop clock) that no
single-metric eviction rule discovers.

## Example (illustrative)

`k=1`, `F=5 Ce=1 De=12 Pc=3`, ops: `W 0`, `R 1`, `R 0`. Fetch page 0 wet (5). At `R 1`, page 0
must be shelved wet (12) and 1 fetched (5). At `R 0`, shelve 1 dry (1) and re-fetch 0 (5); it
ends dry. Total `28`. Instead insert `CLEAN 1 0` (dry page 0 for `3` before it leaves): its
shelving becomes `Ce=1` not `De=12`. Total `5+3+1+5+1+5 = 20`. Same evictions, cheaper ink.

## Constraints

`4 <= k <= 7`, `M <= ~30000`, each input `<= 5 MB`. Time limit 5 s, memory 512 MB. Scoring is
exact and deterministic.
