# Bell-Tower Chime Rack: Minimizing Mallet Travel

## Problem
A bell-ringer owns `N` chimes, numbered `1..N`, that must be hung on pegs of a binary
mounting rack: a rooted binary tree in which every chime occupies exactly one leaf, and
every leaf holds exactly one chime. A chime's **address** is the root-to-leaf path,
written as a string over `{0,1}` (`0` = left branch, `1` = right branch); the rack has at
most `Dmax` rows, so no address may exceed length `Dmax`.

The bell-ringer performs a fixed **peal** — a sequence of `L` chime strikes — read off
stdin. To strike chime `b` right after chime `a`, the mallet must physically travel up
chime `a`'s branch to their lowest common ancestor peg, then down to chime `b`. If `a`'s
address has length `da`, `b`'s has length `db`, and their addresses share a common prefix
of length `p`, this **tree-walk distance** is `da + db - 2p` mallet-lengths (0 if `a=b`).

Because a chime that strikes often shows up in many consecutive pairs, hanging it deep
multiplies travel cost across every one of those strikes — so depth must be weighed
against how often a chime rings. Separately, two chimes that tend to strike back-to-back
should share a long address prefix (sit near each other on the rack) even at the cost of
extra depth elsewhere. Both pressures act on the same tree, and the mounting layout you
choose determines how they trade off.

## Input (stdin)
```
N Dmax
L
s_1 s_2 ... s_L
```
`s_1..s_L` (each in `1..N`) is the peal — the exact strike sequence the mallet must
follow, in order.

## Output (stdout)
Exactly `N` whitespace-separated tokens: `addr_1 addr_2 ... addr_N`, where `addr_i` is
chime `i`'s rack address (a nonempty string over `{0,1}`, length `<= Dmax`).

## Feasibility
All of the following must hold, else the run scores `Ratio: 0.0`:
- exactly `N` tokens are printed, one address per chime `1..N` (in that order);
- every address is nonempty, uses only characters `0`/`1`, and has length `<= Dmax`;
- no address is a prefix of, or equal to, any other address (the addresses are the
  leaves of one valid binary tree).

## Objective
Minimize the total mallet travel over the peal:
```
F = sum_{t=1}^{L-1} walk(addr(s_t), addr(s_{t+1}))
```
where `walk(a,b) = len(a) + len(b) - 2 * (length of the longest common prefix of a,b)`.

## Scoring
The checker builds its own reference rack `B_addr`: the **canonical near-balanced code**
that assigns chime `i` a `ceil(log2 N)`- or `(ceil(log2 N)-1)`-bit address purely by chime
ID (the standard construction with exactly `N` leaves and minimum possible max-depth) —
it uses neither strike frequency nor strike order. Let `B` be the travel cost `F` computed
on that reference rack for the same peal, floored at `1` (only relevant for a degenerate
peal that never leaves one chime, which the generator never emits). With minimization
normalization:
```
B = max(1, B)
sc = min(1000.0, 100.0 * B / max(1e-9, F))
Ratio = sc / 1000.0
```
Reproducing the canonical rack scores `Ratio = 0.1`; a rack with `10x` less travel caps at
`1.0`.

## Constraints
- `8 <= N <= 320`, `Dmax = ceil(log2 N) + 12`.
- `200 <= L <= 30000`.
- Time limit 5s, memory 512MB.

## Example
`N=4, Dmax=4`, peal `1 4 1 4 2` (pairs `(1,4),(4,1),(1,4),(4,2)`).
Canonical rack: `addr_1=00, addr_2=01, addr_3=10, addr_4=11`. Costs:
`walk(1,4)=2+2-0=4` (three times) and `walk(4,2)=2+2-0=4`, so `B = 16`; printing this
rack scores exactly `Ratio = 0.1`.

Since `(1,4)` is the dominant pair, hang `1` and `4` as siblings instead:
`addr_1=00, addr_4=01, addr_2=10, addr_3=11`. Now `walk(1,4)=2+2-2=2` (three times) and
`walk(4,2)=2+2-0=4`, giving `F = 2+2+2+4 = 10`. `sc = 100*16/10 = 160`, so
`Ratio = 0.16` — a real gain from noticing which pair dominates the peal, without
changing any depth.
