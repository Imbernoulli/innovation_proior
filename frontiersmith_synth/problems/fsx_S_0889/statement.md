# Hot-Loop Scheduler for an In-Order Dual-Issue Core

## Problem

You are hand-scheduling a straight-line block of micro-ops for a tiny
**in-order, 2-wide** core. There are three kinds of micro-op:

- **A** (alu): a register-register operation.
- **L addr** (load): reads memory address `addr` into a register.
- **S addr** (store): writes a register to memory address `addr`.

You are given every **data dependence**: `u v` means op `u`'s result (or,
for a store, its side effect) must be available before op `v` may issue.
Output a permutation of all N ops — a static instruction order —
respecting every dependence.

**How the core executes your order.** Issue is strictly in program order:
each cycle the core only considers the earliest not-yet-issued op for
**slot 0**; a second op may join it in **slot 1** the same cycle only if
slot 0 issued. Nothing jumps ahead of a stalled op. An op may issue only
once all its predecessors' results are already available (every result
takes >=1 cycle, so two dependent ops never issue the same cycle). One
memory port: slot 1 may not carry a load/store if slot 0 already does.

**Latencies.** An alu op's result is available `ALU_LAT = 2` cycles after
issue. A store's effect is available `STORE_LAT = 1` cycle after issue. A
load's result is available `LOAD_HIT_LAT = 3` or `LOAD_MISS_LAT = 15`
cycles after issue, depending on the cache below — a miss costs 5x a hit.

**The cache.** A direct-mapped cache of `S` sets, one line per set. Every
load *and* every store touches the cache: `set = addr % S`; if the set
already holds `addr` it's a **hit**, otherwise a **miss** — either way the
set now holds `addr`, evicting whatever it held before. A store can
silently evict data a later load still needs, and *which* address is
resident in a set at any moment depends entirely on the order you chose,
not on the dependence structure.

## Input (stdin)

```
N M S
<N lines, one per op, in ascending id order>: id A   |   id L addr   |   id S addr
<M lines>: u v
```
Ids are a permutation of `0..N-1` (numeric value carries no structural
meaning). `1 <= S`; `addr` fit a 64-bit signed integer.

## Output (stdout)

N whitespace-separated integers: a permutation of `0..N-1`, your chosen
issue order.

## Feasibility

- Exactly N tokens, each an integer in `[0, N)`, no duplicates, no
  non-finite/garbage token.
- For every dependence `u v`, `u` must appear strictly before `v`.

Any violation scores 0.

## Objective & Scoring

Simulate the core (as described above) over your order to get its total
cycle count `F`. The checker also simulates its own cache-blind,
ascending-id topological-sort order to get a baseline cycle count `B`.

```
Ratio = min(1, 0.1 * B / F)
```
(clamped the usual way — matching the baseline exactly scores 0.1, and a
10x-lower cycle count than the baseline already saturates the score at
1.0). Fewer cycles is better (minimization).

## Constraints

1 ≤ N ≤ 320, 0 ≤ M ≤ 2N, 1 ≤ S ≤ N. Time limit 5s, memory 512MB.

## Worked Example (illustrative shape only, not a real test)

Two independent 5-round accumulate chains (`L,A,S` repeated,
loop-carried through the store; 15 ops each), N=30, M=36, cache size
**S=1** — every load/store in the whole program, from either chain,
competes for the single cache line that exists. Chain 0 uses address
100 throughout; chain 1 uses address 101.

Running chain 0 to completion, then chain 1 (`0 1 2 ... 29`) keeps each
chain's own address resident except for one cold miss per chain: the
simulator reports **F=84** (matching the cache-blind baseline `B` here,
so this order alone already scores the baseline 0.1).

Interleaving the chains one op at a time (`0 15 1 16 2 17 ... 14 29`)
looks better, since it keeps the second issue slot busier — but now
every access evicts the other chain's line first, so almost every load
misses: the same simulator reports **F=95**, *worse* than either the
serial order or the baseline. Hiding pipeline bubbles by interleaving
is not free once the interleaved streams share cache capacity.
