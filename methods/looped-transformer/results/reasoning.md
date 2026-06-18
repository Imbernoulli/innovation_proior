I start with the depth bottleneck. A feedforward transformer is a fixed circuit: after its last
layer, computation is over. If I ask it to run an algorithm for more steps, the ordinary answer
is to add more layers. That cannot be the right model for long iterative computation. A small
computer does not grow a new arithmetic unit for each instruction; it reuses the same hardware
and lets time carry the length of the computation. So I make the transformer block the hardware
and place it in an outer loop: at step `t`, it receives the whole current state `X_t` and returns
`X_{t+1}`. Now the burden is narrow and concrete. One application of the block has to execute
one instruction correctly.

For one instruction to be meaningful, the input has to behave like mutable state. I divide the
columns into a scratch area, a memory area, and an instruction area. The scratch area is where
the current instruction and operands are copied, transformed, and then cleared. The memory area
stores data. The instruction area stores commands. I also need every column to carry an address,
because a command cannot say "read memory cell `a`" unless attention has something to match.

The compact address code is a `+/-1` binary vector `p_i` of length `log n` for column `i`. Its
self-inner-product is `log n`. If two addresses differ in `k >= 1` bit positions, their inner
product is `log n - 2k`, so the matching address wins by a margin of at least two. That margin
is the whole reason the code is useful: set key and query to project onto the address rows, and
the attention score matrix becomes a Gram matrix of address codes. With temperature
`lambda >= log(n^3/epsilon)` in the simplified bound, the column-softmax is within `epsilon` of
the selector I want. With bounded input entries, the fuller read/write error bound carries the
additional `G d` factor. Either way, the error is a parameter I can drive down by increasing
temperature.

A read operation is then attention-as-dereference. The command or scratchpad contains the
address to fetch; every source column contains its own address; the matching source column gets
nearly all the softmax mass. The value matrix routes the selected contents into a temporary
buffer. Because the transformer has residual connections, I cannot simply "set" a scratchpad
slot by attention alone; I need the feedforward sublayer to overwrite. A scratchpad indicator
bit supplies the gate. On scratch columns the ReLU expression adds the new value and cancels the
old one; off scratch columns the large negative gate closes the update. A write is the same
idea with the gate firing on the destination memory column instead of the scratch column. The
buffer is cleared with `v <- v - ReLU(v) + ReLU(-v)`.

With read and write in hand, arithmetic can live in the ReLU sublayer. Binary addition over a
fixed number of bits is piecewise linear, so I can build a one-hidden-layer ReLU network with
`8d` hidden activations for adding two `d`-bit nonnegative integers as long as the sum stays in
range. For signed memory I use a two's-complement-style `+/-1` representation: if the most
significant bit `b_N` is `-1`, the value is nonnegative; if `b_N` is `+1`, the value is
negative. Negation is bit flip plus one. The bit flip is
`2 ReLU(-b) - 1`, and the plus one is exactly the binary-add subroutine. Adding that result to
the second operand gives `mem[b] - mem[a]`.

The branch condition has to be checked case by case, because this is where a sign error would
break the machine. A negative value has `b_N = +1`, so `ReLU(b_N)` contributes exactly one.
Zero is the all-`-1` bit pattern. The unit detector for that pattern is
`ReLU(1 - N - sum_i b_i)`: at all `-1`, the argument is `1`; if any bit flips to `+1`, the sum
increases by at least two and the argument is at most `-1`. For every positive value,
`b_N = -1` and at least one lower bit is `+1`, so both terms are zero. Therefore the correct
flag is

```text
flag = ReLU(b_N) + ReLU(1 - N - sum_i b_i).
```

The same expression can be written as the preactivation `-sum_i b_i - N + 1`. The alternative
`1 + N - sum_i b_i` would not be a unit zero detector: on the all-`-1` pattern it produces
`1 + 2N`, so I reject it.

Once `flag` is a true `0/1` value, the program-counter update is a coordinatewise ReLU
multiplexer. With `p_inc` the next sequential address and `p_jump` the command target,

```text
p_next = 2 ReLU(p_inc - flag * 1)
       + 2 ReLU(p_jump - (1 - flag) * 1)
       - 1.
```

If `flag = 0`, the first term rebuilds `p_inc` from `+/-1` coordinates and the second term is
zero. If `flag = 1`, the roles swap and the result is `p_jump`. A residual-canceling term
subtracts the old program counter before the new one is written.

Softmax reads and writes are approximate, so I need a cleanup layer that maps small
neighborhoods of `-1`, `0`, and `1` back to the exact values. For `epsilon < 1/2`, the
piecewise-linear correction is

```text
b_clean = (ReLU(b + 1 - epsilon) - ReLU(b + epsilon)) / (1 - 2 epsilon)
        + (ReLU(b - epsilon) - ReLU(b - 1 + epsilon)) / (1 - 2 epsilon)
        - 1.
```

Checking the three centers confirms the constants: it maps `-1` to `-1`, `0` to `0`, and `1`
to `1`, and it is flat on the relevant neighborhoods. That prevents arbitrarily many loop
iterations from compounding selector error.

These pieces assemble into `SUBLEQ(a,b,c)`: write `mem[b] <- mem[b] - mem[a]`; if the new
`mem[b] <= 0`, jump to instruction `c`; otherwise advance to the next instruction. A command is
just `[p_a; p_b; p_c]`. By merging compatible feedforward work into neighboring attention
layers, I can make one instruction take nine transformer layers and two heads at width
`O(log n + N)`, with `N` bits per integer and valid range `[-2^{N-1}+1, 2^{N-1}-1]`. A more
literal unmerged executor would still have the same state transitions but would spell out the
subtraction and branch as separate sublayers.

The end-of-program command also has to be an actual fixed point. I reserve two memory cells:
one stores zero and one stores `-1`. The EOF command subtracts the zero cell from the `-1` cell,
so the memory value is unchanged and the branch condition is true. Its jump target is EOF
itself. Once the program counter reaches EOF, later loops keep it there.

The one-instruction machine proves universality, but it is not a pleasant way to express
numerical algorithms. I therefore generalize the instruction. Instead of always using
subtraction as the operation, I allow a small menu of hardcoded transformer-based function
blocks. A `FLEQ` instruction carries pointers to two inputs, one output, a function selector, a
branch flag cell, and a jump target:

```text
mem[c] <- f_m(mem[a], mem[b]); if mem[flag] <= 0 goto p.
```

The function-call and branch halves do not interfere. If the flag cell is fixed positive, this
is only a function call; if the input and output pointers are dummy locations, this is only a
branch. With function blocks of depths `l_m` and head counts `h_m`, the unified executor has
`9 + max_m l_m` layers, `sum_m h_m` heads, and width `O(Md + log n)`. The separate
scratch region for each function block is not elegant, but it keeps the addressing argument
simple and the number of functions is treated as a small fixed constant.

The nonlinearity comes from softmax rather than from making the ReLU network deep. If the score
vector is arranged so one entry is `a^T x` and the remaining normalizing mass contributes the
right constant, the selected softmax coordinate is a sigmoid of `a^T x`. Store slopes in the
key/query construction and coefficients in the value matrices; summing `m` heads yields
`sum_i c_{ji} sigmoid(a_{ji}^T x)` for the selected function index `j`. Barron's theorem then
turns this into an approximation scheme for every `f` in the stated `Gamma_{C,B}` class with
`f(0)=0`, with error `O(m^{-1/2})` once the sharpness parameter is at least
`m^{1/2} log m`. This gives a three-layer, `m`-head function block, and I can trade heads for
dimension in a variant if I want a one-head form.

The same softmax can be made locally linear by padding with a large constant. That gives a
two-layer, one-head block for `A^T B + epsilon M` with `||M|| <= 1`, and a four-layer, one-head
transpose block via vectorization and fixed permutation. With multiplication, subtraction, and
transpose available as function blocks, Newton matrix inversion and power iteration become
short programs in the unified instruction language. The depth of the transformer executor stays
fixed; the number of loop invocations carries the number of Newton or power-iteration steps.

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
