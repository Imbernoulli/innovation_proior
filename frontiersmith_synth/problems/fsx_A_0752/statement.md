# Run-Limited Channel Code: Huffman Under a Bit-Run Ceiling

## Problem
You must design a binary **prefix-free code** for `n` source symbols with given frequencies,
to be sent over a channel that forbids long runs of identical bits: no run of more than `d`
consecutive `0`s or `1`s may occur **inside any single codeword** (this is the channel's
run-length-limited, or RLL, constraint). Every symbol gets exactly one codeword, and no
codeword may be a prefix of another (so a stream of codewords can be decoded unambiguously,
symbol by symbol, exactly as in ordinary Huffman coding).

Subject to those two constraints, you want the **frequency-weighted total length to be as
small as possible** — a smaller code should be assigned to a more frequent symbol, exactly
as in Huffman coding, but now the set of codewords you are allowed to choose from is not
"any binary string": it is only the strings that respect the run ceiling `d`. This shrinks
how many short codewords exist and changes which lengths are actually achievable, so blindly
copying an ordinary (unconstrained) Huffman code and patching it up afterwards is not the
same as designing directly for the constrained alphabet.

## Input (stdin)
```
n d
p_1 p_2 ... p_n
```
`n` = number of source symbols, `d` = the run-length ceiling (no more than `d` consecutive
identical bits inside a codeword). `p_i` = positive integer frequency of symbol `i`.

## Output (stdout)
Exactly `n` whitespace-separated tokens: the codewords `w_1 w_2 ... w_n` in symbol order.
Each `w_i` is a non-empty string over `{0,1}`.

## Feasibility
- Every `w_i` must consist only of the characters `0` and `1`.
- Every `w_i` must satisfy the run ceiling: no `d+1` (or more) consecutive equal characters.
- The set `{w_1,...,w_n}` must be **prefix-free**: no `w_i` may be a prefix of any `w_j`
  (`i != j`), including the case of two identical codewords.

Any violation scores `Ratio: 0.0`.

## Objective (minimize)
```
F = sum_i p_i * len(w_i)
```
the frequency-weighted total length of your code.

## Scoring
Let `L` be the smallest length at which at least `n` distinct run-legal codewords of that
length exist (this only depends on `n` and `d`), and let `B = L * sum_i p_i` be the checker's
internal baseline — the cost of the naive flat code that gives every symbol the same length
`L` regardless of frequency. With your feasible `F`,
```
sc    = min(1000, 100 * B / max(1e-9, F))
Ratio = sc / 1000
```
so the flat baseline scores about `0.1`, and a code ten times cheaper than the baseline caps
at `1.0`. Smaller `F` is better.

## Constraints
`1 <= n <= 40`, `2 <= d <= 4`, `1 <= p_i <= 5000`. Runs in well under the time limit for
these sizes.

## Example
`n=4, d=2`, `p = [1, 1, 1, 5]`. A feasible prefix-free, run-legal code is
`w = [100, 101, 11, 0]`: every codeword has runs of at most 2 identical bits, and none is a
prefix of another (sorted: `0, 100, 101, 11` — no adjacent pair has a prefix relation). Its
cost is `F = 1*3 + 1*3 + 1*2 + 5*1 = 13`. The flat baseline uses the smallest length `L` with
at least 4 run-legal codewords: `L=1` only offers `0,1` (2 words, too few), but `L=2` offers
all of `00,01,10,11` (4 words, each with runs `<= 2`), so `L=2` and `B = L * sum(p) = 2*8 = 16`.
`Ratio = min(1000, 100*16/13)/1000 = 0.1231`. A shorter code (e.g. giving the frequent symbol
4 the length-1 word `0` was already done here; further gains come from also shortening the
other symbols' codewords as much as the run ceiling allows) would push the ratio higher.
