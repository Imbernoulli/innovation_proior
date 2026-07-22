# The Bureaucrat's Ritual: Recompressing a Straight-Line Program

A bureaucrat computes one number by grinding through a fixed, several-hundred-step
arithmetic ritual over the field **F_p** (p = 2147483647). The ritual takes **8 inputs**
`x0..x7` and is written as a *straight-line program* (SLP): a list of instructions, each
producing a new value, using only `+`, `-`, `*` and integer constants. It is horribly
bloated. Your job is to compute **the exact same function** with **as few arithmetic
operations as possible**.

You are told nothing about *what* the ritual computes — only its opaque instruction list.

## Input (stdin)
```
p              # the prime 2147483647
L              # number of instructions in the given program
<L instructions>
```
Value indices: `0..7` are the inputs `x0..x7`. The k-th instruction (k = 1..L) produces the
value at index `7+k`. Each instruction is one line, one of:
```
const V        # a literal 0 <= V < p            (index gets value V)
add I J        # value[I] + value[J]  (mod p)
sub I J        # value[I] - value[J]  (mod p)
mul I J        # value[I] * value[J]  (mod p)
```
Operand indices must refer to strictly earlier values. **The program's result is the value of
its LAST instruction.** All arithmetic is modulo p.

## Output (stdout)
Your replacement program, in the same schema (but WITHOUT the leading `p`):
```
M              # number of instructions in your program
<M instructions>
```
Same index convention (`0..7` = inputs, then one new index per instruction), same op set,
same rule that the **result is your last instruction**. `const` values must lie in `[0, p)`.

## Feasibility
Your program is accepted only if it computes **exactly the same function of `x0..x7`** as the
given program over F_p. The checker verifies this by evaluating both programs at a fixed,
instance-seeded battery of points and requiring agreement at every point. Any parse error,
out-of-range index, out-of-range constant, non-integer token, or a single disagreement scores
**0**.

## Objective (MINIMIZE)
The cost of a program is its number of **arithmetic operations** — `add`, `sub`, `mul` each
count as 1; `const` is free. Fewer is better.

## Scoring
Let `B` be the op count of the *given* program (what re-emitting it verbatim costs) and `F` be
your op count. With the given program as baseline,
```
ratio = clamp_[0, 0.98]( 0.10 + 0.280 * ln(B / F) )
```
So copying the ritual scores 0.10; each halving of the op count adds about 0.19; deep
recompressions climb toward — but never reach — the cap, leaving headroom above any known
construction. The ten test cases use rituals of increasing size.

## Constraints
- p = 2147483647; exactly 8 inputs; `1 <= M <= 200000`.
- Time limit 5s, memory 512 MB. The checker is O(points x program length).

## Example (illustrative FORM only — not an instance you will receive)
Suppose the given program were the 4-op ritual
```
p = 2147483647
L = 4
mul 0 1        # index 8 = x0*x1
mul 8 2        # index 9 = x0*x1*x2
const 5
add 9 10       # index 11 = x0*x1*x2 + 5   <- result
```
Then `B = 3` (two `mul`, one `add`). A valid replacement returning the same function is exactly
this program (F = 3, ratio 0.10). There is nothing to save here — this tiny example only shows
the schema, **not** the structure of the real, much larger rituals, which reward discovering the
compact identity they secretly evaluate.
