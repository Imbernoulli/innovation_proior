# The Principal Scheme

## Problem
A single `let`-bound definition has been type-checked, producing a list of
**definition constraints**: equations between type expressions over local type
variables `t0..t(k-1)`, generated in the order the type checker walked the
definition's body. Types are built from:

- `int`, `bool` — ground base types
- `L X` — list of `X`
- `P X Y` — pair of `X` and `Y`
- `F X Y` — function `X -> Y`
- `tI` — a local type variable (only inside definition constraints / your input)

Solving the definition constraints tells you, for each `tI`, either a fixed
ground type or that `tI` is genuinely unconstrained — free to be used at
*different* types at different call sites, i.e. it belongs in the definition's
**principal (most general) type scheme**. Alongside the definition constraints
you are given a batch of held-out **use sites**: each is a small set of
equations that a correctly-instantiated occurrence of the definition must
satisfy at that call site. A fresh instantiation is created independently
*per use site* — that is the whole point of a type scheme: the same free slot
can be `int` at one call and `L bool` at another.

## Input (stdin)
```
k m u
m lines: "<typeExpr> = <typeExpr>"      -- definition constraints, in generation order
u blocks, each:
  "USE <id> <q>"
  q lines: "<typeExpr> = <typeExpr>"    -- this use's equations (left side always
                                           references t0..t(k-1); right side is
                                           always a closed ground type)
```

## Output (stdout)
Exactly `k` lines (plus a leading count), one per variable `t0..t(k-1)` in order:
```
k
FIX <closed type expr>     -- this variable is bound to exactly this ground type
GEN <label>                -- this variable is universally quantified as <label>;
                               any variable(s) sharing the same <label> are the
                               same generalized slot
```
`<label>` is any token of letters/digits/underscore not starting with a digit.
`FIX` targets must be closed (`int`/`bool`/`L`/`P`/`F` only — no `tI`).

## Feasibility
Substitute your answer into every **definition constraint**: each `tI` becomes
either its `FIX` type or an opaque leaf named by its `GEN` label. Both sides of
every constraint must then be *syntactically identical* trees (ground
constructors match; `GEN` leaves match iff they carry the same label). Any
mismatch, any malformed/truncated/extra output, or any non-ground `FIX`
target scores `Ratio: 0.0`.

## Objective
For each held-out use site, instantiate your scheme **fresh**: every `GEN`
label gets its own new slot for *this* use only (a different use never shares
it), while `FIX` variables keep their one fixed ground type everywhere. The use
**passes** if all of its equations then hold simultaneously (a slot bound
twice within one use must get the same ground type both times). Score scales
with the fraction of held-out uses that pass (see Scoring). At least one use
in every instance is a call site that no scheme, however general, can satisfy
— so a perfect score is never attainable, even by the true principal type.

## Scoring
Let `F` = uses passed, `B = (total held-out uses)/10` (a calibrated baseline: a
solver that ignores every use site and blind-guesses one default type per free
slot clears only a small slice of the batch by chance, landing near `B`).
`Ratio = min(1000, 100*F/B) / 1000`, printed as `Ratio: <float>`.

## Constraints
`k,m,u` are small (definitions and use batches fit comfortably in memory);
time limit 4s; every `.in` well under 5 MB.

## Example (worked score, illustrative form only)
Suppose `k=1`, definitions empty (`t0` unconstrained), and ten uses split
evenly between `t0 = int` and `t0 = L bool`. `FIX int` is sound but passes
only the five `int` uses (`F=5`, `B=u/10=1`, `Ratio` capped at `1.0` — a
lucky guess on a tiny toy). On real instances a free slot is asked for many
different ground types across its uses, so one `FIX` guess matches only some
of them while `GEN a` passes essentially all, since each use instantiates
`a` fresh. This toy is unrelated to the real instances, which mix several
independently-generalizable slots, ground-pinned variables, multi-member
slots tied by compound constraints (e.g. `L t2 = L t5`), and use sites no
scheme can satisfy.
