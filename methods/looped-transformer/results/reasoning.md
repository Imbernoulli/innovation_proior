I start with the depth bottleneck. A feedforward transformer is a fixed circuit: after its last
layer, computation is over. If I ask it to run an algorithm for more steps, the ordinary answer
is to add more layers. That cannot be the right model for long iterative computation. A small
computer does not grow a new arithmetic unit for each instruction; it reuses the same hardware
and lets time carry the length of the computation. So I make the transformer block the hardware
and place it in an outer loop: at step `t`, it receives the whole current state `X_t` and returns
`X_{t+1}`. Now the burden is narrow and concrete. One application of the block has to execute
one instruction correctly, and it has to leave the state in a form the next application can read.

For one instruction to be meaningful, the input has to behave like mutable state. I divide the
columns into a scratch area, a memory area, and an instruction area. The scratch area is where
the current instruction and operands are copied, transformed, and then cleared. The memory area
stores data. The instruction area stores commands. I also need every column to carry an address,
because a command cannot say "read memory cell `a`" unless attention has something to match
against.

The question is what an address should look like so that attention can pick one column out of
`n` cleanly. Attention scores are bilinear, so if I put the address of column `i` in some rows
of `X`, the score between a query address `q` and a key address `p_i` is `q^T p_i`. For a clean
selector I want the score of the intended column to sit well above all the others. A `+/-1`
binary vector `p_i` of length `log n` does this. Its self-inner-product is `log n`. If two
addresses differ in `k >= 1` bit positions, each disagreeing coordinate flips a `+1` product to
`-1`, subtracting `2` from the score, so `p_i^T p_j = log n - 2k`. I want to be sure of that
before I lean on it, so I take `n = 64` (`log n = 6`), a random address `p`, and flip `k` of its
bits for each `k`:

```text
self inner product: 6                  (= log n)
differ in 1 bit:  4   (= 6 - 2)
differ in 2 bits: 2   (= 6 - 4)
differ in 3 bits: 0   (= 6 - 6)
differ in 6 bits: -6  (= 6 - 12)
```

So the matching column scores `log n` and every nonmatch scores at most `log n - 2`: the winner
is separated by a margin of at least two regardless of `n`. That fixed margin is what lets a
single temperature work for any memory size. Set key and query to project onto the address rows,
and the attention score matrix becomes a Gram matrix of address codes.

Whether the margin is enough depends on the softmax temperature. With column-softmax at inverse
temperature `lambda`, the mass leaking off the winning column is bounded by the competitors'
relative weights, roughly `(n-1) e^{-2 lambda}` from the margin of two. Setting this below `eps`
asks for `lambda >= (1/2) log((n-1)/eps)`, which the simplified bound rounds up to
`lambda >= log(n^3/eps)`. I check it is not wishful: for `n = 64`, `eps = 1e-3`, that gives
`lambda ~= 19.4`, and a worst-case score vector with one entry at `log n` and all `n-1` others
at `log n - 2` puts softmax mass

```text
winner mass = 1.000000,  total leakage = 8.9e-16  <<  eps = 1e-3
```

on the winner, with the analytic upper bound `(n-1) e^{-2 lambda} = 9.2e-16` matching the
measured leakage. The bound is in fact generous; the margin to spare is enormous, which is
reassuring because real entries are bounded rather than exactly `+/-1`. With bounded input
entries the fuller read/write error bound carries an additional `G d` factor and asks for
`lambda > log(G d n^3 / eps)`. Either way, the error is a parameter I can drive down by
increasing temperature.

A read operation is then attention-as-dereference. The command or scratchpad contains the
address to fetch; every source column contains its own address; the matching source column gets
nearly all the softmax mass. The value matrix routes the selected contents into a temporary
buffer. Because the transformer has residual connections, I cannot simply "set" a scratchpad
slot by attention alone; attention only adds into the residual stream, so I need the feedforward
sublayer to overwrite. A scratchpad indicator bit supplies the gate. On scratch columns the ReLU
expression adds the new value and cancels the old one; off scratch columns the large negative
gate closes the update. A write is the same idea with the gate firing on the destination memory
column instead of the scratch column. The buffer is cleared with
`v <- v - ReLU(v) + ReLU(-v)`, which sends any sign of `v` to zero.

With read and write in hand, arithmetic can live in the ReLU sublayer. Binary addition over a
fixed number of bits is piecewise linear in the carry chain, so I can build a one-hidden-layer
ReLU network with `8d` hidden activations for adding two `d`-bit nonnegative integers as long as
the sum stays in range. For signed memory I need a representation where the sign is cheap to
test. A two's-complement-style `+/-1` representation works: if the most significant bit `b_N` is
`-1`, the value is nonnegative; if `b_N` is `+1`, the value is negative. Negation is bit flip
plus one. The bit flip is `2 ReLU(-b) - 1`, which maps `+1` to `-1` and `-1` to `+1`, and the
plus one is exactly the binary-add subroutine.

Before I trust subtraction I want to see negation actually invert a value end to end, because a
single off-by-one in the carry would silently corrupt the machine. I enumerate all `4`-bit
values, flip each with `2 ReLU(-b) - 1`, run the add-one, and compare to `-v`:

```text
negation mismatches (excluding -2^(N-1)): 0
```

Every representable value negates correctly; the only excluded case is `-2^{N-1}`, which has no
representable negation in `N` bits, exactly as in ordinary two's complement. Adding the negated
operand to the second operand then gives `mem[b] - mem[a]`.

The branch condition has to be checked case by case, because this is where a sign error would
break the machine: SUBLEQ jumps when the result is `<= 0`, so I need a flag that is `1` for
every negative value and for zero, and `0` for every positive value. A negative value has
`b_N = +1`, so `ReLU(b_N)` contributes exactly one and `ReLU(-b_N)`-type terms stay silent. Zero
is the all-`-1` bit pattern, and the sign bit alone does not catch it, so I need a second term
that fires only on that pattern. The candidate is `ReLU(1 - N - sum_i b_i)`: at all `-1` the
argument is `1 - N - (-N) = 1`; if any bit flips to `+1`, `sum_i b_i` increases by at least two,
so the argument drops to at most `-1` and the ReLU kills it. For a positive value, `b_N = -1`
and at least one lower bit is `+1`, so `ReLU(b_N) = 0` and the second term is also zero. The
proposed flag is

```text
flag = ReLU(b_N) + ReLU(1 - N - sum_i b_i).
```

Reasoning case by case is exactly where I make sign mistakes, so I run all `2^4 = 16` patterns
through the encoding and the flag:

```text
bits=[-1,-1,-1,-1] value= 0  want=1 got=1
bits=[-1,-1,-1, 1] value= 1  want=0 got=0
bits=[-1, 1, 1, 1] value= 7  want=0 got=0
bits=[ 1,-1,-1,-1] value=-8  want=1 got=1
bits=[ 1, 1, 1, 1] value=-1  want=1 got=1
...
mismatches: 0
```

It is correct on all sixteen: `1` on the eight nonpositive values, `0` on the seven positive
ones, with zero handled by the second term and the negatives by the sign bit. The same
expression can be written as the single preactivation `-sum_i b_i - N + 1` plus `ReLU(b_N)`.

While building the zero detector I am tempted by the symmetric-looking `1 + N - sum_i b_i`, which
also has a constant and a `-sum_i b_i`. But I should check it rather than trust the symmetry. On
the all-`-1` (zero) pattern with `N = 4` it gives `1 + 4 - (-4) = 9`, not `1`, and on the
positive patterns `[-1,-1,-1,1]` and `[-1,1,1,1]` it gives `7` and `3`. It is not a unit zero
detector at all — it is large everywhere and largest on zero — so I discard it and keep
`ReLU(1 - N - sum_i b_i)`, whose sign inside the ReLU is what makes it vanish off the all-`-1`
pattern.

Once `flag` is a true `0/1` value, the program-counter update should be a coordinatewise ReLU
multiplexer between the sequential successor `p_inc` and the jump target `p_jump`. The form I
try is

```text
p_next = 2 ReLU(p_inc - flag * 1)
       + 2 ReLU(p_jump - (1 - flag) * 1)
       - 1.
```

The intent: when `flag = 0` the first term should rebuild `p_inc` from its `+/-1` coordinates and
the second should vanish; when `flag = 1` the roles swap. I want to see the `+/-1` reconstruction
actually come out right, because the factor of `2` and the `-1` offset are easy to get wrong. With
random `p_inc`, `p_jump`:

```text
flag=0: p_next == p_inc?  True
flag=1: p_next == p_jump? True
```

So for `flag = 0`, `ReLU(p_inc - 0) = ReLU(p_inc)` keeps the `+1` coordinates and zeros the `-1`
ones; doubling and subtracting `1` restores `+1` and `-1`, while `ReLU(p_jump - 1)` is zero on
`+/-1` inputs. For `flag = 1` the subtractions swap which term survives. A residual-canceling term
subtracts the old program counter before the new one is written.

Softmax reads and writes are approximate, so a column that should hold exactly `+/-1` or `0` will
drift by up to `eps` each step, and over many loop iterations that drift could accumulate past the
margin. I need a cleanup layer that maps small neighborhoods of `-1`, `0`, and `1` back to the
exact values. For `eps < 1/2`, the piecewise-linear correction is

```text
b_clean = (ReLU(b + 1 - eps) - ReLU(b + eps)) / (1 - 2 eps)
        + (ReLU(b - eps) - ReLU(b - 1 + eps)) / (1 - 2 eps)
        - 1.
```

The constants `1/(1 - 2 eps)` are chosen to make the ramps unit-slope between the centers, and
the `-1` offset sets the baseline. Rather than re-derive the centers by hand I evaluate it at
`eps = 0.1`. The three centers map to themselves:

```text
clean(-1) = -1.000000   clean(0) = 0.000000   clean(1) = 1.000000
```

and, more to the point, the function is flat on each `eps`-neighborhood, so the drift it is meant
to absorb is actually absorbed:

```text
clean(-1 +/- 0.05) = -1.000000     clean(0 +/- 0.05) = 0.000000
clean( 1 +/- 0.05) =  1.000000
```

Perturbations of `0.05 < eps` snap back to the center exactly. That is what prevents arbitrarily
many loop iterations from compounding selector error.

These pieces assemble into `SUBLEQ(a,b,c)`: write `mem[b] <- mem[b] - mem[a]`; if the new
`mem[b] <= 0`, jump to instruction `c`; otherwise advance to the next instruction. A command is
just `[p_a; p_b; p_c]`. By merging compatible feedforward work into neighboring attention
layers, I can make one instruction take nine transformer layers and two heads at width
`O(log n + N)`, with `N` bits per integer and valid range `[-2^{N-1}+1, 2^{N-1}-1]`. The range is
not arbitrary: it is exactly the values whose negation is representable, which is what the
negation check above ran into. A more literal unmerged executor would still have the same state
transitions but would spell out the subtraction and branch as separate sublayers.

The end-of-program command also has to be an actual fixed point, or the loop would wander off the
program after the last real instruction. I reserve two memory cells: one stores zero and one
stores `-1`. The EOF command subtracts the zero cell from the `-1` cell, so the memory value is
unchanged (`-1 - 0 = -1`) and the branch condition is true (`-1 <= 0`). Its jump target is EOF
itself. Once the program counter reaches EOF, the flag stays `1`, the multiplexer keeps selecting
`p_jump = p_EOF`, and later loops keep it there — a genuine fixed point of the block, which is
what I needed to verify the loop terminates cleanly.

The one-instruction machine establishes universality, but it is not a pleasant way to express
numerical algorithms — every multiply is a loop of additions. I therefore generalize the
instruction. Instead of always using subtraction as the operation, I allow a small menu of
hardcoded transformer-based function blocks. A `FLEQ` instruction carries pointers to two inputs,
one output, a function selector, a branch flag cell, and a jump target:

```text
mem[c] <- f_m(mem[a], mem[b]); if mem[flag] <= 0 goto p.
```

The function-call and branch halves do not interfere, which I can see from the structure: the
function block writes only `mem[c]`, and the branch reads only `mem[flag]`. If the flag cell is
fixed positive, this is only a function call; if the input and output pointers are dummy
locations, this is only a branch. With function blocks of depths `l_m` and head counts `h_m`,
the unified executor has `9 + max_m l_m` layers, `sum_m h_m` heads, and width `O(Md + log n)`.
The separate scratch region for each function block is not elegant, but it keeps the addressing
argument simple and the number of functions is treated as a small fixed constant.

Now I need the function blocks themselves to be nonlinear without making the ReLU network deep —
deep ReLU stacks would defeat the fixed-depth goal. The unused nonlinearity in the layer is the
softmax. If I feed the softmax a two-entry score vector `[a^T x, 0]`, the first coordinate is
`e^{a^T x} / (e^{a^T x} + 1) = 1 / (1 + e^{-a^T x})`, which is exactly `sigmoid(a^T x)`. I
confirm the identity rather than assume it:

```text
s=-3: softmax_first=0.047426  sigmoid=0.047426   match
s=-1: softmax_first=0.268941  sigmoid=0.268941   match
s= 0: softmax_first=0.500000  sigmoid=0.500000   match
s= 1: softmax_first=0.731059  sigmoid=0.731059   match
s= 3: softmax_first=0.952574  sigmoid=0.952574   match
```

So if the score vector is arranged so one entry is `a^T x` and the remaining normalizing mass
contributes the right constant, the selected softmax coordinate is a sigmoid of `a^T x`. Store
slopes in the key/query construction and coefficients in the value matrices; summing `m` heads
yields `sum_i c_{ji} sigmoid(a_{ji}^T x)` for the selected function index `j`. That is the
classic shallow sigmoid network, so Barron's theorem turns it into an approximation scheme for
every `f` in the stated `Gamma_{C,B}` class with `f(0)=0`, with error `O(m^{-1/2})` once the
sharpness parameter is at least `m^{1/2} log m`. This gives a three-layer, `m`-head function
block, and I can trade heads for dimension in a variant if I want a one-head form.

The same softmax can be made locally linear by padding with a large constant — pushing all
scores toward the flat region of the simplex makes the output an affine function of the inputs.
That gives a two-layer, one-head block for `A^T B + epsilon M` with `||M|| <= 1`, and a
four-layer, one-head transpose block via vectorization and fixed permutation. With
multiplication, subtraction, and transpose available as function blocks, Newton matrix inversion
and power iteration become short programs in the unified instruction language. The depth of the
transformer executor stays fixed; the number of loop invocations carries the number of Newton or
power-iteration steps.

The in-context learning construction follows the same pattern. For linear regression, a program
computes `w^T x_i`, subtracts `y_i`, multiplies by `x_i` and the step size, and subtracts the
gradient step from `w`, while counters and pointer updates move through the data and epochs.
For a two-layer sigmoid network, the forward pass, backpropagated deltas, outer-product
gradients, and parameter updates are written as calls to the same function-block menu. In the
linear and sigmoid-network settings this keeps the executor at 13 layers, one head, and width
`O(log D + d)`, with approximation error controlled by temperature and matrix-multiplication
constants. For arbitrary losses or activations represented through Barron approximation, the
number of heads must grow enough to control the accumulated `O(m^{-1/2})`
function-approximation error.
