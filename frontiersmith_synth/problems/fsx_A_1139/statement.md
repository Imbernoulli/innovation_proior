# Coprime-Stage Signal Router

## Problem
A relay must apply a fixed length-`n` Number-Theoretic Transform (NTT) to every packet
it forwards: given `x[0..n-1]` in a prime field `GF(q)`, it must produce

```
X[k] = sum_{j=0}^{n-1} x[j] * w^(j*k mod n)  (mod q),   k = 0..n-1
```

where `w` is a given primitive `n`-th root of unity mod `q`. Additions in the field are
cheap (simple routing); every general scalar multiplication costs a paid "multiplier
unit" in the hardware. `n` is always a product of two or three **pairwise coprime**
small primes. Your job is to wire a program that computes `X` **exactly** at the lowest
total hardware cost.

## Input (stdin)
One line: `n q w`.

## Output (stdout)
A straight-line program over registers. Registers `0..n-1` are the (implicit) inputs
`x[0..n-1]`. Then:

```
R L
<L instruction lines>
O o_0 o_1 ... o_{n-1}
```

`R` must equal `n + L`. Instruction `i` (0-indexed, `i = 0..L-1`) defines register `n+i`
as one of:

```
A a b     reg[n+i] = reg[a] + reg[b]   (mod q)     -- costs 0.1
S a b     reg[n+i] = reg[a] - reg[b]   (mod q)     -- costs 0.1
M a c     reg[n+i] = reg[a] * c        (mod q)     -- costs 1 UNLESS c mod q in {0,1,q-1}
```

Additions are cheap but **not free**: over a field, `x+x` literally computes `2*x`, so an
unlimited free `A`/`S` would let any multiplication be laundered into a double-and-add
chain. Charging `0.1` per `A`/`S` makes laundering a single multiplication cost at least
10 additions to break even — real addition chains for the constants here need far more,
so this is never worthwhile.

`a`/`b` in an `A`/`S` instruction must be a register index strictly less than `n+i`
(you may only reference already-defined registers). In an `M` instruction, `c` is any
integer literal (a field constant you choose — it is reduced mod `q`); it is **not** a
register index. No instruction may multiply two data registers together — every
multiplication is data times a constant, so your whole program is linear in the input.
Finally, `O` followed by `n` register indices names which registers hold `X[0..n-1]`.

## Feasibility
Your program is checked against **every** possible input by linearity: for each
`j = 0..n-1`, running it with `x = e_j` (the `j`-th standard basis vector, `x[j]=1`,
all else `0`) must reproduce column `j` of the true transform exactly, i.e. register
`o_k` must equal `w^(j*k mod n) mod q` for every `k`. Since the program only ever adds
two registers or multiplies a register by a fixed constant, this is equivalent to
matching the transform on every input. Any parse error, non-causal register reference,
non-integer token, or mismatch scores `0`.

## Objective
Minimize `F = (# of M instructions whose constant is not in {0,1,q-1}) + 0.1 * (# of A/S
instructions)`.

## Scoring
Let `B` be the number of `(j,k)` pairs with `w^(j*k mod n) mod q` not in `{0,1,q-1}` —
the multiplier count of the naive "one multiplier per matrix entry" implementation
(its additions cost extra on top, but `B` alone is the reference). With `F` as above,

```
Ratio = min(1, 0.05 * B / F)
```

The naive per-entry construction scores a little under `0.05` (its many additions cost
a bit extra). Materially cutting `F` — via fewer paid multiplications, not via a longer
chain of cheap additions — raises the ratio.

## Constraints
- `15 <= n <= 105`, `n` a product of 2 or 3 pairwise coprime primes from `{3,5,7,11,13}`.
- `q` a prime with `q ≡ 1 (mod n)`, `w` an explicit primitive `n`-th root of unity mod `q`.
- `0 <= L <= 400000`; each register index and constant fits in a normal machine integer.
- Deterministic exact modular-integer scoring; no timing.

## Example
For a (much smaller, illustrative only) length-`4` transform with `B=9`: the naive
per-entry program uses 9 multiplications plus 6 additions, `F = 9+0.1*6 = 9.6`, scoring
`min(1, 0.05*9/9.6) ≈ 0.047`. Cutting this to `5` multiplications and `8` additions scores
`min(1, 0.05*9/5.8) ≈ 0.078`. The real instances (`n` up to `105`) reward re-indexing the
computation via the mechanisms above, not shaving a few terms or padding addition chains.
