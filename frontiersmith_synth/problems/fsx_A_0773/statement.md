# Restore the Shredded Royal Multiplication Ledger

The royal treasury kept an n x n ledger of conversion codes: row `i`, column `j` held a
code `ledger[i][j]` in `0..n-1`. What nobody alive remembers is that the ledger was
originally built as the multiplication table of a small finite group: each row, each
column, and each code stood for one of `n` abstract "tokens", and the cell at `(i, j)`
recorded the *product* of token `i` and token `j` under that group's operation. Long ago
a clerk relabelled which row stood for which token, which column stood for which token,
and which code stood for which token -- three independent, unknown relabellings -- before
copying the table into the surviving ledger. A fire then destroyed a large fraction of
the cells. You must restore the entire ledger.

Formally: the surviving ledger is a partial Latin square (every fully-known row or
column would contain each code exactly once) that is *isotopic* to the Cayley table of
some finite group of order `n` -- i.e. there exist permutations `row_perm`, `col_perm`,
`sym_perm` and a group multiplication table `M` such that the true, complete ledger
satisfies `ledger[i][j] = sym_perm(M[row_perm[i]][col_perm[j]])` for all `i, j`. You are
given only the surviving fragment; you do not know the group, the relabellings, or which
cells were originally which token.

This structure is far more rigid than an arbitrary Latin square. In particular it
satisfies the *quadrangle criterion*: whenever two surviving cells carry the same code
(possibly in completely different rows and columns), the tokens their positions
represent must combine to the same underlying group product -- this equality holds
table-wide, not just locally, and combined with knowing (or narrowing down) which group
generated the table, it can pin down cells nowhere near any surviving data. Cell-by-cell
reasoning that only looks at "what's missing from this row and this column" has no way
to see this and, once damage is heavy, is left guessing among several locally-consistent
values with no way to tell which is right.

## Public instance (stdin JSON)

```json
{
  "n": 8,
  "grid": [[3, null, null, 7, ...], [null, 0, 5, null, ...], ...]
}
```

`grid` is `n` rows of `n` entries each; a surviving cell holds its integer code
(`0..n-1`); a destroyed cell holds `null`. `n` is between 6 and 10 across instances.

## Answer (stdout JSON)

```json
{"grid": [[3, 6, 2, 7, ...], [4, 0, 5, 1, ...], ...]}
```

An `n x n` grid of integer codes in `0..n-1` for every cell (surviving cells must be
reproduced exactly as given).

## Scoring (maximize)

An answer is rejected outright (score 0 on that instance) if it is malformed, any code
is out of range, or any surviving cell is not reproduced exactly. Otherwise, among the
cells that were destroyed, let `correct` be the number that match the true original
ledger, and let `viol` be the total number of row/column Latin-property violations
(repeated codes) anywhere in your returned grid. The instance score is

```
raw   = correct / (number of destroyed cells)  -  viol / (2 * n * (n - 1))
score = 0.9 * clamp(raw, 0, 1)
```

so a fill that is everywhere Latin-consistent but only guesses at chance level scores
low, a mostly-correct fill scores moderately, and a fully correct, violation-free
reconstruction scores `0.9` (deliberately short of the maximum, leaving headroom).
`Ratio` is the mean instance score over 10 deterministic, seeded instances (some held
out at heavier damage than others).

Your program reads one public instance JSON from stdin and writes one answer JSON to
stdout. It runs in an isolated subprocess and only ever sees the public instance --
never the true ledger, the group, or the relabellings.
