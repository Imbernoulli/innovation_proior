# Looped Transformers as Programmable Computers

## Final Method

Use a fixed transformer block as an instruction executor and run it recurrently:

```text
for t = 1..T:
    X <- TF(W; X)
```

The input matrix `X in R^{d x n}` is mutable state. Its columns are divided into scratchpad,
memory, and commands, and every column carries a `+/-1` binary address `p_i` of length `log n`.
If two addresses differ in `k` bits, then `p_i^T p_j = log n - 2k`; the matching address is
therefore separated from every nonmatch by at least two.

One transformer layer is

```text
Attn(X) = X + sum_h V_h X softmax(X^T K_h^T Q_h X)
f(X)    = Attn(X) + W2 ReLU(W1 Attn(X) + b1 1^T) + b2 1^T.
```

Setting `K = Q` to read the address rows makes attention an approximate address selector. With
temperature at least `log(n^3/epsilon)` in the simplified selector bound, the softmax is within
`epsilon` of the intended permutation; the full bounded-input read/write bound uses
`lambda > log(G d n^3 / epsilon)`. Value matrices move selected data into a buffer, and ReLU
gates keyed by a scratchpad indicator overwrite the destination and clear the buffer.

## SUBLEQ Executor

The core instruction is

```text
SUBLEQ(a, b, c):
    mem[b] <- mem[b] - mem[a]
    if mem[b] <= 0: goto c
    else: goto next instruction
```

Each command is `[p_a; p_b; p_c]`. Integers are stored as `N` `+/-1` bits
`[b_N, ..., b_1]`:

```text
if b_N = -1: value = sum_{i=1}^{N-1} 2^{i-1} (b_i + 1)/2
if b_N = +1: value = -2^{N-1} + sum_{i=1}^{N-1} 2^{i-1} (b_i + 1)/2
```

Subtraction uses two's-complement negation: flip bits with `2 ReLU(-b) - 1`, add one using the
binary-add ReLU network, then add to `mem[b]`.

The branch flag must be

```text
flag = ReLU(b_N) + ReLU(1 - N - sum_i b_i).
```

This is `1` for negative values by the sign bit, `1` for zero because zero is the all-`-1`
pattern, and `0` for positive values. The signs matter: the equivalent preactivation
`-sum_i b_i - N + 1` gives the same indicator.

The program-counter update is

```text
p_next = 2 ReLU(p_inc - flag * 1)
       + 2 ReLU(p_jump - (1 - flag) * 1)
       - 1,
```

with residual-cancel terms for the old program counter. Softmax residue is removed by

```text
b_clean = (ReLU(b + 1 - eps) - ReLU(b + eps)) / (1 - 2 eps)
        + (ReLU(b - eps) - ReLU(b - 1 + eps)) / (1 - 2 eps)
        - 1.
```

One `SUBLEQ` instruction is then a 9-layer, 2-head looped transformer of width
`O(log n + N)`, with integer range `[-2^{N-1}+1, 2^{N-1}-1]`. Laid out as a sequence of
fixed-depth sublayers, one instruction step is:

```python
def subleq_step(X):
    X = read_inst(X)              # 1 layer
    X = read_mem(X)               # 1 layer, 2 heads (mem[a], mem[b])
    X = subtract_mem(X)           # 3 layers: flip bits, add one, add/clear
    X = write_mem(X)              # 1 layer
    X = conditional_branching(X)  # 3 layers: flag, PC+1, select/clear
    X = error_correction(X)       # 1 layer
    return X
```

Looping this step for `T` iterations executes a `T`-instruction `SUBLEQ` program; the depth is
fixed at nine and only `T` grows.

## Unified FLEQ Executor

To avoid expressing all numerical work through subtraction alone, replace the subtraction slot
with a menu of transformer-based function blocks:

```text
FLEQ(a, b, c, m, flag, p, d_h, d_w):
    mem[c] <- f_m(mem[a], mem[b])
    if mem[flag] <= 0: goto p
```

If the function blocks have depths `l_m` and head counts `h_m`, the unified executor has
`9 + max_m l_m` layers, `sum_m h_m` heads, and width `O(Md + log n)`. A fixed positive
flag cell gives a pure function call; dummy input/output cells give a pure branch.

## Function Blocks

Nonlinear functions are encoded by forcing softmax to act as a sigmoid. For a selected function
index `j`, the block stores slopes and coefficients so that `m` heads output

```text
sum_{i=1}^m c_{ji} sigmoid(a_{ji}^T x).
```

By Barron's theorem, any `f in Gamma_{C,B}` with `f(0)=0` can be approximated with error
`O(m^{-1/2})` when the sigmoid sharpness satisfies `tau >= m^{1/2} log m`. This gives a 3-layer,
`m`-head function block; the `m` terms can alternatively be packed across dimension to use a
single head.

Matrix transpose is a 4-layer, 1-head function block using vectorization and fixed permutation.
Matrix multiplication is a 2-layer, 1-head block: padding the softmax with a large constant
linearizes it, yielding `A^T B + eps M` with `||M|| <= 1`.

## Consequences

`SUBLEQ` gives a universal one-instruction computer at fixed transformer depth. `FLEQ` makes the
same executor practical for numerical programs. With multiplication, transpose, subtraction,
and sigmoid-style function blocks, the construction gives fixed-depth looped transformers for:

- a basic calculator,
- Newton matrix inversion,
- power iteration with Newton inverse-square-root normalization,
- linear-model SGD over in-context data,
- two-layer sigmoid-network SGD with backpropagation.

The headline depth for these applications is at most 13 layers; runtime is paid in the number
of recurrent invocations, not in a deeper transformer.
