# The Watermark That Survives the Backfill Race

## Problem
You are migrating rows from an old store to a new one **without downtime**.
The application does a **dual write**: every live write lands correctly and
immediately in both stores (no race on that path). Separately, a background
worker **backfills** old rows into the new store: at an earlier instant it
scanned a key in the old store and read some `(version, value)` pair; only
now, at a later tick, does it actually write that pair into the new store.
If the worker writes it in **unconditionally**, a batch that scanned a key
before a newer live write can land its stale pair *after* that live write
and silently erase it -- a lost update. You must produce a **migration
plan**: for every backfill write, decide whether to apply it unconditionally
or **conditionally** (only if its version is strictly newer than whatever
the new store currently holds for that key), and choose one **cutover**
tick: reads before it are served by the (always-correct) old store; reads at
or after it are served by the new store and must be correct.

## Input (stdin)
```
K T M
V_1 V_2 ... V_K
op_1
...
op_T
```
`K` keys (ids `1..K`), `T` ticks (0-indexed `0..T-1`, in execution order),
`M` of which are backfill ticks. `V_i` is key `i`'s baseline value (implicit
version `0`) in the old store before migration starts. Each `op` is one of:
- `L k v x` -- live dual-write: key `k`, new version `v` (`v>=1`, strictly
  increasing per key), new value `x`. Always lands in both stores.
- `B k v x` -- backfill tick: writes `(v,x)`, the `(version,value)` key `k`
  truly held at some earlier instant (`v=0,x=V_k`, or a version/value from an
  earlier `L` on the same key).
- `R k` -- a client reads key `k`.

## Output (stdout)
Two lines: the chosen cutover tick `C` (`0<=C<=T`), then `M` flags
`f_1 ... f_M`, one per backfill tick **in the order the `B` ops appear in
the input**, each `0` (apply unconditionally) or `1` (apply only if `v` is
strictly greater than the version currently stored for that key in the new
store; a first-ever write to a key always succeeds).

## Feasibility
Reject (score `0`) if: the token count isn't exactly `1+M`; any token isn't
a finite number; `C` isn't an integer in `[0,T]`; any flag isn't exactly `0`
or `1`. Otherwise replay the timeline applying your flags. For every `R k`
at tick `i` with `i>=C`: if the new store's current value for `k` differs
from the old store's true current value (or the key was never written to
the new store at all), that read is a **lost update** and the whole test
case scores `0`.

## Objective
On a feasible plan, let `earliness = (T-C)/T` and `completeness` = the
fraction of keys whose new-store value matches the true value at the final
tick. Maximize
```
F = 9 * earliness + 1 * completeness
```

## Scoring
The checker's own reference plan never cuts over (`C=T`) and conditionally
applies every backfill write; it is always feasible and reaches
`completeness=1`, giving baseline `B = 1`.
```
sc = min(1000, 100*F/B);  Ratio = sc/1000
```

## Constraints
`3<=K<=10`, `8<=T<=28`. Time limit 5s, memory 512MB.

## Example
`K=2, T=4`, baseline `V=[10,20]`. Ticks: `0: B 1 0 10` (copy key 1's
baseline), `1: R 1`, `2: L 2 1 55` (live write to key 2), `3: R 2`. Output
`C=1, f_1=0` (the flag doesn't matter here -- it is key 1's only write, so
it lands either way): tick 1 (`R 1`, `1>=C`) is checked and sees `(0,10)`,
matching the true value -- correct; tick 3 (`R 2`, `3>=C`) is checked and
sees `(1,55)` from the live write -- correct. Both reads pass even though
key 1's backfill was the *only* thing that had happened by tick 1, so
`earliness = (4-1)/4 = 0.75`, `completeness = 1`, `F = 9*0.75 + 1 = 7.75`,
`B = 1`, `Ratio = min(1000, 775)/1000 = 0.775`.

This tiny illustrative shape is **not** representative of the harder cases:
larger instances give every key at least one write and several read-checks,
and the real trap is a *stale* backfill landing after a concurrent live
write to the same key -- applying it unconditionally ("wait for the whole
backfill phase to finish, then flip the switch") clobbers the newer value,
which a later read-check exposes as a lost update, zeroing that case.
