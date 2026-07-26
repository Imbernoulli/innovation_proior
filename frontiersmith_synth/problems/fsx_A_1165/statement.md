# Royal Chronicle: Batching Decrees and Ordering the Wax Page

## Problem

The royal chronicler records every decree the crown issues onto a wax-sealed
scroll. Decrees arrive one per tick, `1, 2, ..., N` (decree `i` arrives at
tick `i`). Each decree `i` has a deadline slack `d_i`: it must be sealed
into some page by tick `i + d_i` at the latest (some decrees are routine and
can wait; a few are urgent and must be sealed soon after they arrive).

You choose when to seal a page: a **seal** at tick `t` gathers every
currently-pending (arrived, not yet sealed) decree into ONE page, in an
order **you choose**, and costs `20 + k` where `k` is the number of decrees
in that page. Every decree must be sealed in **exactly one** page, by its
own deadline. Seal ticks must be strictly increasing.

The kingdom also keeps a list of `C` known **fire drills** (crash points).
Fire drill `j`, at tick `c_j`, needs a specific set of decrees restored
(`c_j` is always `>=` every needed decree's own deadline, so any
deadline-respecting schedule has already sealed all of them by then).
Recovery works backward: it takes the **most recently sealed page at or
before `c_j`**, scans its decrees **in the exact order you sealed them**,
and continues into the next-older page (again in its own order) while some
needed decree is still missing. It stops the instant every needed decree
for that drill has been seen — but every decree passed along the way,
needed or not, costs one **replay probe**.

## Input (stdin)

```
N C TMAX
a_1 d_1
a_2 d_2
...
a_N d_N
c_1 m_1
id_1 id_2 ... id_{m_1}
c_2 m_2
id_1 ... id_{m_2}
...
```
`a_i = i` (decree `i` arrives at tick `i`), `d_i >= 1` its deadline slack.
Then `C` fire-drill blocks: crash tick `c_j`, count `m_j`, then the `m_j`
needed decree ids (1-indexed, distinct). Read every field — do not assume a
uniform mix of urgent vs. routine decrees, or that needed ids cluster
predictably; the real structure lives only in the data.

## Output (stdout)

One line per seal, in increasing tick order:
```
tick k id_1 id_2 ... id_k
```
`tick` = when you seal, `k` = page size, then the `k` decree ids **in the
order recovery will scan them** (first listed = scanned first).

## Feasibility

Score `0` if: a line is malformed; a seal tick exceeds `TMAX` or is not
strictly greater than the previous seal's tick; any id is not yet arrived
(`a_i > tick`) or sealed after its own deadline (`tick > a_i + d_i`); an id
appears in two seals or twice in one seal; or some id from `1..N` never
appears in any seal.

## Objective (minimize)

```
cost = sum over seals (20 + k)
     + sum over fire drills of (replay probes charged by that drill)
```

## Scoring

The checker computes your feasible cost `F` and an internal baseline `B`:
seal every decree alone, the instant it arrives (always feasible, but pays
`21` per decree). For minimization, `Ratio = min(1, 0.1 * B / F)`. Lower `F`
scores higher; the cap leaves room above any reference solution.

## Constraints

`N` up to ~1800, `C` up to ~48, `d_i` up to 70, time limit 5s, memory 512MB.

## Example (worked, illustrative shape only)

`N=5, TMAX=50`, decrees `(a,d)`: `(1,30) (2,3) (3,30) (4,3) (5,30)`
(deadlines `31,5,33,7,35`). One fire drill: `c=9, needed={2,4}` (both
deadlines `<=9`, so any valid schedule has sealed them by then). Sealing
everything at once, output line `5 5 1 2 3 4 5` (tick `5`, `k=5`, arrival
order — legal since `5 <=` every deadline here), costs `20+5=25`; recovery
scans `1,2,3,4,5`, finding `2` at probe `2` and `4` at probe `4` (`4`
probes, stops once both are seen): total `= 29`. Reordering the SAME page
as `5 5 2 4 1 3 5` (needed decrees first) costs the same `25` to seal, but
recovery finds both within `2` probes: total `= 27`. Same seal timing, same
page contents — only the internal order changed, and the fire-drill cost
dropped. Whether to seal early and often or batch big, and how to order
each page against the ACTUAL fire-drill set in your input, is yours to
work out.
