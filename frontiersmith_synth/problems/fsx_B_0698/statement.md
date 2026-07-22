# Sacrificial Premixes on a Fixed Tank Ladder

You run a fuel-blending terminal with `F` feedstocks and a **fixed ladder of `M` tanks**,
numbered `1..M`, connected in one forward direction only (liquid may move from a
lower-numbered tank to a higher-numbered one, never backward). Each feedstock `i` has an
available volume and three quality attributes (think octane-, sulfur-, density-type
indices). Each tank `j` has a fixed capacity `cap_j` you must fill **exactly**, if you use
it at all.

## Mixing is nonlinear

Quality attributes do **not** average by volume-weighted arithmetic mean. Each attribute
`k` (`k=1,2,3`) has its own fixed exponent `a_k != 1`. When ingredients with weights
(volumes) `w_1..w_n` and attribute-`k` values `y_1..y_n` are combined, the resulting value
is the weighted **power mean**:

```
z_k = ( (w_1*y_1^a_k + ... + w_n*y_n^a_k) / (w_1+...+w_n) ) ^ (1/a_k)
```

applied independently for each `k`. This is the same rule the checker uses to evaluate you,
exactly — no approximation.

## What you may pour into a tank

Each tank may receive, in any combination:
- direct pours of raw feedstock from **at most 2 distinct feedstocks** (any number of pour
  lines, but at most 2 distinct feedstock ids per tank — a 2-inlet manifold), and
- **at most one** wholesale `TRANSFER` of an EARLIER tank's entire finished contents (that
  earlier tank must already be filled to its own capacity). A tank used as a transfer
  source is consumed — it earns no revenue of its own.

If a tank receives anything, its total incoming volume (pours + transfer) must equal
`cap_j` **exactly**. An untouched tank is simply unused (0 revenue, no constraint).
A tank may be the source of at most one outgoing transfer.

Because direct pours are capped at 2 feedstocks, but a transferred-in premix counts as a
*single* ingredient carrying its own bespoke, already-blended value, chaining a premix
into a final tank lets you reach compositions (under the true power-law rule) that no
direct 2-feedstock recipe of the same capacity can reach. How many stages you use — and
which tanks you sacrifice as premixes versus sell — is entirely your call.

## Spec-corridor pricing

`R` product corridors are given, each an interval `[lo_k,hi_k]` on every attribute plus a
price-per-unit-volume. A sold (non-transferred-out, filled) tank earns
`cap_j * price`, where `price` is the **highest** price among corridors whose intervals
all contain the tank's blended attributes, or a small fallback `p0` if none match.
Maximize total revenue over all sold tanks.

## Input (stdin)

```
F M K R
a_1 a_2 a_3
A_1 x_1,1 x_1,2 x_1,3      (F lines: feedstock i, 1-indexed)
...
cap_1
...
cap_M                       (M lines: tank capacities, 1-indexed)
lo_1,1 hi_1,1 lo_1,2 hi_1,2 lo_1,3 hi_1,3 price_1   (R lines: corridors)
...
p0
```
`K=3`. All feedstock/tank/corridor counts and volumes are integers; exponents are floats;
attribute values lie in `(0,100)`.

## Output (stdout)

Any number of lines, each one of:
```
POUR j i v          (pour v units of feedstock i into tank j)
TRANSFER i j         (transfer tank i's ENTIRE contents into tank j; i < j)
```
1-indexed throughout; `v` a positive integer.

## Feasibility

Any violation scores `0`: a feedstock's total poured volume exceeds its availability; a
tank fed directly by more than 2 distinct feedstocks; a tank receiving more than one
transfer, or a tank transferred out more than once, or a `TRANSFER i j` with `i >= j` or
referencing a tank not filled to its own capacity; a used tank whose total incoming
volume != its capacity; out-of-range indices; non-finite/non-integer tokens.

## Example scoring

Suppose exponent `a_1=2.0`. Tank of capacity 10 gets 4 units of feedstock with
attribute-1 value 20 and 6 units of value 80: `z_1 = sqrt((4*20^2+6*80^2)/10) ≈ 62.5`
(far from the linear mean of `56`). If `z` lands inside a corridor priced 40/unit, that
tank earns `10*40=400`. The checker also builds its own single-feedstock-per-tank
baseline `B`; your ratio is `min(1, 0.1*(your total revenue)/B)`.

## Constraints

Time limit 5s, memory 512MB. All scoring is deterministic.
