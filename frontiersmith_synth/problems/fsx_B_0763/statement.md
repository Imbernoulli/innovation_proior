# The Innkeeper's Ledger

## Problem

You run an inn with `m` numbered rooms. Each room has exactly two cots: cot 1 (by the
window, the one every regular asks for) and cot 2 (by the door). Each cot holds at most
one regular for the whole season.

You have `n` regulars. Regular `i` has exactly two favorite rooms, `r1[i]` and `r2[i]`
(their first and second choice, in that order — the ledger records which is which). If
neither favorite room has a free cot, the regular can be sent to the annex instead: a
block of extra cots, `s` of them in total, shared by everybody who needs one.

You also have a season's worth of lookups — for each regular, `f[i]` records how many
times that season they actually show up asking for their room (their trace frequency).
Every time regular `i` shows up, the innkeeper has to walk to wherever they were finally
seated:

| seated at                     | probes per visit |
|--------------------------------|-------------------|
| first-choice room, cot 1       | 1 |
| first-choice room, cot 2       | 2 |
| second-choice room, cot 1      | 3 |
| second-choice room, cot 2      | 4 |
| the annex                      | 10 |

Your job: seat every regular somewhere (one of their two favorite rooms' two cots, or the
annex) so that no two regulars ever share a cot, at most `s` regulars use the annex, and
the season's total probe count is as small as possible.

## Input (stdin)

```
n m s
r1_1 r2_1 f_1
r1_2 r2_2 f_2
...
r1_n r2_n f_n
```
`1 <= r1_i, r2_i <= m`, `r1_i != r2_i`, `f_i >= 1`. It is always possible to seat every
regular somewhere.

## Output (stdout)

`n` integers, one per regular in input order, each one of:
- `1` = seated at `r1_i`, cot 1
- `2` = seated at `r1_i`, cot 2
- `3` = seated at `r2_i`, cot 1
- `4` = seated at `r2_i`, cot 2
- `0` = seated in the annex

## Feasibility

- Exactly `n` integer tokens, each in `{0,1,2,3,4}`.
- For every room and every cot (1 or 2), at most one regular is seated there (checked
  across ALL regulars who could reach that cot, whether it's their first or second
  choice — it's the same physical cot).
- At most `s` regulars total are seated in the annex.
- Any violation scores 0.

## Objective

Minimize `F = sum over regulars of f_i * (probes-per-visit for their seat)`, using the
table above (`10` for the annex).

## Scoring

The checker builds its own reference seating `B` by walking regulars in the GIVEN input
order and seating each one in the first open cot among their four options (annex if all
four are taken). Your score is `min(1.0, 0.1 * B / F)`. Matching that naive seating
scores exactly `0.1`; a seating with a tenth of its total probe cost scores `1.0`.

## Example

`n=3, m=2, s=1`. Regulars: `(1,2,10)`, `(1,2,10)`, `(2,1,5)`.

One feasible seating: regular 1 at room 1 cot 1 (code `1`), regular 2 at room 1 cot 2
(code `2`), regular 3 at room 2 cot 1 (its first choice, code `1`... wait regular 3's
`r1=2`, so code `1` means room 2 cot 1). Check: room 1 cots 1&2 used once each, room 2
cot 1 used once — no collisions, annex unused (`0 <= 1`). Cost = `10*1 + 10*2 + 5*1 = 35`.
The naive walk-in-order baseline seats them identically here (`B = 35`), so this
particular seating would score `0.1` — a smarter seating that avoids ever leaving a
high-frequency regular on cot 2 or worse is required to score higher.

## Constraints

`1 <= n <= 300`, `2 <= m <= 350`, `0 <= s <= 100`, `1 <= f_i <= 1000`. Time limit 5s.
