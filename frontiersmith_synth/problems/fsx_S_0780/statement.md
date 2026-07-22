# The Guild Shorthand: Amortized Macro-Library Compression of Ritual Scripts

## Problem
The guild's scribes have accumulated a corpus of `N` ritual scripts. Each script `k`
is a straight-line arithmetic program (SLP) over `M` shared ingredients
`x0..x{M-1}`: a sequence of instructions `t_i = OP a b` (`OP` in `ADD`, `SUB`, `MUL`;
`a`, `b` each an ingredient `x<idx>`, a small integer `c<val>`, or an earlier result
`t<idx>` of the same script), ending with `OUT t<idx>` naming the script's result.

Writing every script out in full is wasteful — many scripts secretly perform the
*same ritual shape* on different ingredients, sometimes with the two sides of a
commutative step (`ADD`/`MUL`) written in the opposite order. Your job is to invent a
small **shorthand library**: up to 8 reusable macros (each a small parameterized
SLP with formal parameters `p0..`), and rewrite every script to call macros where
profitable. A macro definition is paid for once; every call to it is cheap. Finding
the shorthand requires recognizing the *abstract* ritual (which operand slots vary,
and that swapped operand order doesn't change a commutative step) — literal
copy-paste detection will miss rituals whose bound ingredients differ each time.

## Input (stdin)
```
N M P SEED
PROGRAM 0 L0
<L0 lines: OP a b>
OUT t<idx>
PROGRAM 1 L1
...
```
`P` is the prime modulus used for equivalence checking (documentation only); `SEED`
is informational. `1<=M<=8`. Each script's temps are numbered `t0..t{Lk-1}` in order;
an operand `t<idx>` must reference an earlier instruction of the *same* script.

## Output (stdout)
```
MACROS K
MACRO name0 arity0 B0
<B0 lines: OP a b>      (a,b each: p<idx> formal param, t<idx> earlier macro-local
                          temp, c<val> constant, or x<idx> a global ingredient)
RET t<idx>
... (K macros total, 0<=K<=8)
PROGRAMS N
PROGRAM 0 L0'
<L0' lines, each either "OP a b" or "CALL macroName arg0 arg1 ...">
OUT t<idx>
... (N programs total, same order as input)
```
Args to `CALL` are evaluated in the *calling* script's environment (so they may be
`x<idx>`, `c<val>`, or an earlier `t<idx>` of that script) and bound to the macro's
`p0..`. Macro bodies may NOT call other macros.

## Feasibility
Every token must match its grammar and stay in range (macro count <=8, arity <=6,
macro body length in `[1,12]`, `|const|<=1000`, program length <=2000, strict
use-before-define). Then, for several independently random assignments of
`x0..x{M-1}` over `Z_P`, every rewritten script must evaluate (mod `P`, using
`ADD/SUB/MUL` mod `P`) to **exactly** the same value as the original script. Any
schema violation or value mismatch scores `Ratio: 0.0`.

## Objective
Minimize `F = sum over scripts of (rewritten instruction count, where each OP line
and each CALL line costs 1) + 3 * (sum of body sizes of macros that are called at
least once)`. Unused macro definitions cost nothing extra but waste one of your 8 slots.

## Scoring
Let `B` = sum of the ORIGINAL (unrewritten) instruction counts. With your cost `F`:
```
Ratio = min(1, 0.1 * B / F)
```
Resubmitting the corpus verbatim (no macros) gives `F = B`, `Ratio = 0.1`.

## Constraints
- `1 <= N <= 100`, `1 <= M <= 8`, each script length `<= 60`.
- Time limit 5s, memory 512MB, deterministic (fixed field, fixed random points).

## Example
Suppose two scripts both compute `p0*p1 + p0 - p1` for different `(p0,p1)`, written
with the multiply's operands in opposite order. A submission defining one arity-2
macro (body size 3) and calling it from both scripts pays `3*3 + 1 + 1 = 11` instead
of inlining `3 + 3 = 6`... but across *dozens* of occurrences the one-time `3*3=9`
definition cost is dwarfed by the calls saved, and the total `F` drops well below `B`.
