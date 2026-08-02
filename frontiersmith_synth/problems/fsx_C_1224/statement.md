# Wire-Tag Reservation Across Schema Versions

## Problem
A service ships a binary wire format that evolves over `V` releases,
numbered `1..V`. Every field, once introduced at some version `v0`, stays in
the schema **forever**: old records already written or replicated keep
referencing it, so it can never be dropped or given a different meaning.
Each field has a fixed per-record occurrence count `freq` that applies to
every version in which it is active (i.e. from `v0` through `V`).

Each field must be assigned an integer **tag** `>= 0`. Encoding one
occurrence of a field costs a number of bytes that depends only on its tag,
via a fixed step schedule:

- tag in `[0, T1CAP)`             -> `1` byte  (compact reservation)
- tag in `[T1CAP, T1CAP+T2CAP)`   -> `T2COST` bytes (standard range)
- tag `>= T1CAP+T2CAP`            -> `T3COST` bytes (overflow key -- an
  expensive fallback used once the compact ranges are exhausted)

`T1CAP`, `T2CAP`, `T2COST`, `T3COST` are given in the input. You must decide
the tag for **every** field in the schema's whole future history in a single
pass, before any version ships -- exactly as a real team commits to a wire
format once and lives with it.

**Forward/backward compatibility.** Two different fields can never share a
tag, at any point, even if they are never simultaneously present in the same
single version: an old client reading new data (or new code reading
archived old data) has to be able to tell fields apart purely from the tag,
across the entire version history. Reusing a tag is a hard schema break.

## Input (stdin)
```
V M T1CAP T2CAP T2COST T3COST
field_id group_id v0 freq        (repeated M times, field_id = 0..M-1)
```
`group_id` is a thematic label (which subsystem the field belongs to); it
does not otherwise constrain the encoding. Fields are listed in the order
the schema evolved: all version-1 fields first, then version 2's additions,
and so on.

## Output (stdout)
Exactly `M` whitespace-separated integer pairs, `field_id tag`, one pair per
field (any order), giving every field's chosen tag.

## Feasibility
- The `M` pairs must cover every `field_id` in `0..M-1` exactly once.
- Every `tag` must be an integer in `[0, 999999]`.
- All `M` tags must be pairwise **distinct** (no compatibility break).
Any violation scores `0`.

## Objective
Minimize total encoded bytes across the field's whole lifetime:
`F = sum over fields of freq * (V - v0 + 1) * byte_cost(tag)`.

## Scoring
The checker also builds its own reference assignment: sort all `M` fields by
**ascending** `freq` and hand out tags `0..M-1` in that order (the
least-used fields get the cheap tags), with total cost `B`. Your score is
`Ratio = min(1.0, B / (10 * F))`. Fewer bytes -> higher ratio. Matching the
reference construction scores `Ratio ~= 0.1`; being 10x leaner than it
saturates at `1.0`.

## Constraints
`3 <= V <= 5`, up to a few hundred fields (typically several dozen to
~100), `1 <= freq <= 15000`,
`0 <= T1CAP,T2CAP <= 20`, `1 <= T2COST <= 3`, `T2COST < T3COST <= 20`.
Time limit 5s, memory 512MB.

## Example (worked, illustrative shape only)
Suppose `V=2`, `T1CAP=1, T2CAP=1, T2COST=2, T3COST=10`, and 3 fields: field 0
(`v0=1, freq=5`), field 1 (`v0=1, freq=1`), field 2 (`v0=2, freq=50`).
Field 2 is only active for 1 version but has huge frequency; fields 0/1 are
active for both versions.

The reference sorts by ascending freq (field1=1, field0=5, field2=50), so it
gives field 2 -- the biggest single contributor -- the expensive tag 2 (cost
10): `B = 1*2*1 + 5*2*2 + 50*1*10 = 2 + 20 + 500 = 522`.

A tag choice that instead reserves the single cheap tag for field 2 (weight
`50*1=50`) over field 0 (weight `5*2=10`) or field 1 (weight `1*2=2`) --
e.g. field2->0, field0->1, field1->2 -- gives `F = 50*1*1 + 5*2*2 + 1*2*10 =
50 + 20 + 20 = 90`, i.e. `Ratio = min(1.0, 522/900) = 0.58`, far better than
the ascending-frequency reference despite giving field 0 a *more expensive*
tag than it had at version 1.
