Let me start from what actually bothers me. A transformer with `L` layers does exactly `L` stages of computation and then stops. That's fine for classification, but I keep wanting these networks to *run algorithms* — sweep a dataset a few times, iterate a recurrence, train a small model on data handed to it in the prompt. And the honest situation is that to do `T` steps of anything, the standard move is to stack `T`-ish layers. For long computations that's hopeless, and for the genuinely hard cases it can't even be true: Liu and coauthors showed that if shallow transformers could shortcut arbitrary automata, certain circuit-complexity classes would collapse in ways nobody believes. So depth scaling with the number of steps isn't just inconvenient, it's a wall for the hard problems.

What do I actually know about transformers being computationally powerful? Pérez, Marinković and Barceló, and later with finite precision Wei, Chen and Ma, prove transformers can simulate Turing machines. Great — but those are existence arguments. They tell me *a* transformer exists; they don't tell me how to build one for a *particular* algorithm, and they lean on high precision and recursive links. On the concrete end there's the in-context-learning line: von Oswald shows one linear-self-attention layer's action on the input is exactly one gradient-descent step on a regression loss, and Akyürek constructs a constant-depth network that does one SGD step for a linear model. Those are real, explicit constructions — but they're one step, linear model, one loss. To get `T` steps you stack `T` of them, and there's no control flow and no nonlinear model. And then RASP and Tracr: a little language that compiles into transformer weights, which is the right *spirit*, except the compiled network grows with the program and the language can't express general nonlinearities or clean iteration.

So I have two piles. One pile is general but abstract and deep. The other is concrete but special-purpose and also, secretly, deep — because depth tracks steps. What I want is something that's both: explicit weights, fixed depth, and general. The thing all of these have in common, when I squint, is that they let depth grow. What if I just refuse to let depth grow?

Picture an actual computer. A CPU isn't deep. It has one arithmetic unit, one control unit, a program counter, and RAM, and it reuses that same hardware over and over — one instruction per cycle, cycling for as long as the program runs. The program length lives in the *number of cycles*, not in the size of the silicon. If I want a fixed-size transformer to behave like that, I need the analogue of cycles. A transformer's forward pass is the silicon; the missing thing is the loop. Take a multi-layer transformer, call its action `X ← TF(W; X)`, and just feed its output back in as the next input, `T` times. Now the depth only has to be big enough to execute *one instruction*, and `T` — the number of loops — carries the program length. Universal Transformers already showed weight-tying across depth and repeated application is compatible with attention, so there's nothing architecturally exotic here. The whole game becomes: make "one instruction" small, self-contained, and fixed-depth. That's the lever. Everything below is just earning it.

For this to be CPU-like, the input sequence has to be laid out like memory. Let me partition the `n` columns into three regions: a scratchpad `S`, a memory block `M`, and a command block `C`. Memory holds the program's variables; commands hold the lines of code; the scratchpad is the working area — the cache — where I copy things, operate on them, and copy results back. And I'll work with `d`-dimensional embedding columns rather than tokens, the way Garg and von Oswald and Akyürek do; an embedding layer can always map tokens into these vectors, and reasoning at the level of column vectors lets me argue with block matrices instead of discrete symbols.

Now the first real problem: a CPU reads `mem[a]` — it addresses a specific location. How does attention "address" a column? Attention's score between column `i` and column `j` is some bilinear form of their contents. If I want column `j` to be able to say "fetch whatever is at address `i`," I need every column to carry an *address*, and I need the score to spike when two addresses match. So append a positional code to each column. The obvious choice is one-hot, but that costs width `n`, which is wasteful. Sinusoidal codes are added, not appended, and don't give me a clean "match" test. Let me try binary: encode column index `i` as a `±1` vector `p_i` of length `log n`, where bit `k` is `+1` if bit `k` of `i` is `1` and `−1` otherwise. Then `p_i^T p_i = log n`, and for `i ≠ j`, if the two codes differ in `k ≥ 1` of the `log n` coordinates the inner product is `(log n − k) − k = log n − 2k ≤ log n − 2`. There's the structure I wanted: the self-inner-product is strictly the largest, by a gap of at least `2`, and the width is only `log n`. The same code can be the program counter (which command to run next) and a data pointer (which variable to read). Appending it as a suffix rather than adding it keeps the block-matrix bookkeeping clean.

Now: read. I want attention to copy the contents of the column addressed by some pointer sitting in the scratchpad into the scratchpad itself. So make the attention score depend *only* on whether two columns share an address. Set the key and query matrices equal, both projecting onto the positional-encoding rows, so that `K X = Q X = [p_{i_1} ⋯ p_{i_n}]` and the score matrix is exactly the Gram matrix of the codes, `(K X)^T (Q X)` with entries `p_{i}^T p_{j}`. On each column the largest score is the diagonal `p^T p = log n`, beating every off-diagonal by at least `2`. Now soften that into a selection: apply the column-softmax with a high temperature `λ`. The softmax of a vector whose top entry beats the rest by a fixed margin concentrates on the top entry; with this margin structure, taking `λ ≥ log(n^3/ε)` drives every column of the softmax to within `ε` of a one-hot. So `softmax((KX)^T(QX))` is, up to an `ε M` error with `‖M‖ ≤ 1`, an *approximate permutation matrix* — it selects, per column, the column whose address matches. Multiply by a value matrix `V` that picks out the data rows and routes them into the scratchpad row, and the residual carries everything else along unchanged. The attention has now *copied* the addressed value into the scratchpad. One layer, one head, width `O(log n + d)`.

But copying via the residual just *adds* the fetched value on top of whatever was in the scratchpad slot; I actually want to *overwrite* it, and only in the scratchpad column, not everywhere. This is what the MLP is for. Keep a single indicator bit `b` in the last row of each column — `1` if the column is the scratchpad, `0` otherwise. Then a ReLU gate keyed on `b` does a conditional overwrite. Concretely, to replace the scratchpad's stored value `v_orig` with the newly-fetched `v_new`, I write
`v_orig ← v_orig + ReLU(C(b−1)·1 + 2v_new − 2v_orig) − ReLU(C(b−1)·1 − 2v_new + 2v_orig)`
with `C` a large positive constant. On a non-scratchpad column `b = 0`, so `C(b−1) = −C` swamps the arguments and both ReLUs are zero — no change. On the scratchpad column `b = 1`, the `C(b−1)` term vanishes and the two ReLUs combine to `2v_new − 2v_orig` halved back into the right magnitude, i.e. they execute the subtraction that turns `v_orig` into `v_new`. Then reset the fetched slot, `v_new ← v_new − ReLU(v_new) + ReLU(−v_new) = 0`, so the workspace is clean for the next step. Write is the *same* construction with the gate sign flipped — copy a scratchpad value out to the memory location named by a pointer — using `ReLU(−Cb·1 + …)` so it fires on memory columns instead. Each is one layer.

Good — I can read and write addressed memory. Now arithmetic and control, which both live in the ReLU MLP. Take incrementing the program counter: I store indices as binary, so "next instruction" is binary addition of two integers. A one-hidden-layer ReLU network with `8d` hidden units computes the binary representation of the sum of two `d`-bit numbers (the carries are piecewise-linear in the bits), as long as the sum stays below `2^{d+1}`. So `PC ← PC + 1` is one ReLU layer.

Subtraction — the heart of `SUBLEQ` — I get from two's complement. Represent an integer by bits `b_N … b_1` (MSB first), with the convention that `b_N = +1` means negative. Then `−r` is the two's complement of `r`: flip every bit and add `1`. Flipping a `±1` bit is a single neuron, `b_flip = 2·ReLU(−b) − 1` (sends `+1 ↦ −1`, `−1 ↦ +1`). Add `1` with the binary-add network. Then add `b_s` to `b_{−r}` to get `mem[b] − mem[a]`. That's a couple of ReLU layers; the intervening attention sublayers I make into the identity by zeroing their value matrices.

Control flow needs a truth value. Given the result of the subtraction stored as `N` two's-complement bits `b_N … b_1`, "is it `≤ 0`?" is two disjoint cases: it's strictly negative exactly when the sign bit `b_N = +1`, and it's exactly zero exactly when *every* bit is `−1`. The sign case is one neuron: `ReLU(b_N)` is `1` when `b_N = +1` and `0` when `b_N = −1`, so it already returns `1` on every strictly-negative number and `0` on every non-negative one. I just need to add `1` back in the single remaining case, all-bits-`−1`. Detect that with `Σ_{i=1}^N b_i = −N`: build `ReLU(1 − N − Σ_i b_i)`, whose argument at the all-`−1` point is `1 − N − (−N) = 1`, and for any other pattern at least one bit is `+1` so `Σ_i b_i ≥ −N + 2`, making the argument `≤ −1` and the ReLU `0`. So
`flag = ReLU(b_N) + ReLU(1 − N − Σ_i b_i)`
is a clean `0/1` indicator of `mem[b] − mem[a] ≤ 0`, computed in one ReLU layer. (And for an ordinary integer cell, not bit-packed, the same flag is just `flag = 1 − ReLU(mem[a]) + ReLU(mem[a] − 1)`, which is `1` for `mem[a] ≤ 0` and `0` for `mem[a] ≥ 1`.)

Now the jump. Let the current program counter point to `p_PC`. If `flag = 1` I want the next PC to be the jump target `p_i`; if `flag = 0` I want `p_{PC+1}`, the increment. With both candidate codes sitting in the scratchpad, a ReLU layer selects between them:
`p_next = 2·ReLU(p_{PC+1} − 1·flag) + 2·ReLU(p_i − 1·(1 − flag)) − 1`.
When `flag = 0`: the first term is `2·ReLU(p_{PC+1}) `, the second is `2·ReLU(p_i − 1) = 0` because `p_i ∈ {±1}` so `p_i − 1 ≤ 0`; the `−1` rebuilds the `±1` code, giving `p_{PC+1}`. When `flag = 1`: symmetric, giving `p_i`. To respect the residual I add a cancel term `−ReLU(p_PC) + ReLU(−p_PC)` so the old PC is subtracted off. This whole branch needs three ReLU layers and so two transformer layers, with attention value matrices zeroed where I just want the MLP.

One nuisance: my reads and writes used softmax, not hardmax, so each leaves an `ε` residue. After a few operations the bits are `±1` or `0` plus some `|ε| < 0.5` noise. I can snap them back exactly with a piecewise-linear ReLU "comparator":
`b_clean = 1/(1−2ε) · (ReLU(b + 1 − ε) − ReLU(b + ε)) + 1/(1−2ε) · (ReLU(b − ε) − ReLU(b − 1 + ε)) − 1`.
This is a staircase that maps the `ε`-neighborhood of each of `{−1, 0, 1}` onto the exact value. One layer of error correction and the state is clean again. (If I used hardmax the error would be exactly zero, but hardmax isn't a transformer; softmax keeps the whole thing a genuine attention network, and I clean up afterward.)

Now I have every piece of a one-instruction computer. Mavaddat and Parhami showed a single instruction is enough for universal computation, and the clean choice is `SUBLEQ(a, b, c)`: `mem[b] ← mem[b] − mem[a]`; if the result is `≤ 0`, jump to instruction `c`, else fall through. Looping over `SUBLEQ` instructions is Turing complete given enough memory. Let me assemble one `SUBLEQ` instruction out of my blocks. An instruction is just three addresses, so I encode it as the concatenation of three pointers, `cmd = [p_a; p_b; p_c]`, a `3 log n` vector sitting in the command region. One pass of the looped transformer executes one instruction:

read the instruction pointed to by the PC into the scratchpad — one layer. Read the two operands `mem[a]`, `mem[b]` — one layer, but two heads now, one per operand. Compute `mem[b] − mem[a]` via flip-add-add — folded into the feedforward layers of the previous step, so no extra layer. Write the result back to address `b` — one layer. Compute the `≤ 0` flag — one layer. Conditional branch on the flag, setting the PC to `c` or `PC+1` — two layers. Error-correct the bits — one layer. Tally: nine layers, two heads, width `O(log n + N)` with `N` the bits per integer, integers ranging over `[−2^{N−1}+1, 2^{N−1}−1]`. And the depth is *nine* no matter how long the program is; the program length is the number of loops `T`. To make the machine halt cleanly I reserve an `EOF` command whose three pointers are rigged so that `mem[b] − mem[a] ≤ 0` is always true and the jump target is the `EOF` command itself — once reached, the PC keeps pointing at `EOF`, freezing the state at a fixed point.

That's a universal computer in nine fixed layers. But `SUBLEQ` builds everything out of subtraction, which is brutal for anything numerical — matrix multiply, a sigmoid, a square root would be enormous programs. I want richer primitives without giving up the fixed-depth property. Generalize the single instruction: instead of *only* subtraction, let the operation be any `f_m` drawn from a small menu of `M` functions I can hardcode. Call the instruction
`FLEQ(a, b, c, m, flag, p)`: `mem[c] ← f_m(mem[a], mem[b])`; if `mem[flag] ≤ 0` goto `p`.
Each `f_m` is realized by its own little fixed-depth transformer — a *function block* — that reads its first argument `A` from columns `1:d`, its second `B` from columns `d+1:2d`, and writes `f(A,B)` into columns `2d+1:3d`. Wrap them in the same nine-layer `SUBLEQ` skeleton (fetch instruction, route to the right block, write back, branch, error-correct) and the unified machine has `9 + max_m l_m` layers, `Σ_m h_m` heads, width `O(Md + log n)` — still constant in the program length, since the depth is set by the deepest single function block. And the two halves of the instruction are independent: point `flag` at a cell hardwired to `1` and `FLEQ` is a pure function call; point `a, b, c` at dummy cells and it's a pure conditional jump. So `FLEQ` subsumes both `mem[c] = f_m(mem[a], mem[b])` and `if mem[flag] ≤ 0 goto p` as special cases.

The crucial new ingredient is nonlinear functions, and here's where I want to be clever rather than brute-force. The lazy option is to compute nonlinearities in the MLP — it's a universal approximator. But there's a cost: to approximate even `x^2` to error `ε` with a ReLU MLP takes depth growing like `log(1/ε)`, and I'm trying to keep depth constant. Can the *attention* itself produce the nonlinearity? Stare at the softmax. On a single column, `softmax([z; C])` for a large constant `C` makes the score for `z` come out as `e^z` over a denominator dominated by the `C` term — that's a saturating, sigmoid-shaped response, not a linear one. So the softmax isn't just a selector; with the right padding it's a *sigmoid generator*. Let me force that.

Put the function's parameters into the weight matrices: the slope vectors `a_{ji}` go into the key matrix, the coefficients `c_{ji}` into the value matrix, and the identity into the query, so the score on the data column is `a_{ji}^T x`. Now append to each `a_{ji}` an extra last entry equal to `−log(3d−1)`, and set the last entry of `x` to `1`. Over the `3d` columns the softmax denominator becomes `(3d−1) + e^{a_{ji}^T x}`, so the entry the value matrix reads out is
`e^{a_{ji}^T x} / ((3d−1) + e^{a_{ji}^T x}) = 1 / (1 + e^{−a_{ji}^T x}) = σ(a_{ji}^T x)`.
The softmax *is* a sigmoid. One head computes `c_{ji} σ(a_{ji}^T x)`; summing over `m` heads and adding the residual gives `Σ_{i=1}^m c_{ji} σ(a_{ji}^T x)`. And now Barron's theorem cashes this in: any `f` with bounded Fourier-magnitude integral (`f ∈ Γ_{C,B}`, `f(0) = 0`) is approximated by such a sum of `m` sigmoids with error `O(m^{-1/2})` once `τ ≥ m^{1/2} ln m`. So attention with `m` heads is a universal function approximator — at *constant* depth (a three-layer block), exactly the depth separation I wanted over the `log(1/ε)`-deep ReLU route. To carry a menu of `N` functions, index them with a one-hot `e_j` in the input and store all the `{c_{ji}, a_{ji}}` in the weights; the indicator picks which function fires.

Linear algebra falls out of the same softmax trick read the other way. For matrix multiply I want `A^T B` to land in attention's `V X softmax(...)`. The obstacle is the softmax nonlinearity; but the same large-constant padding that gave me a sigmoid can instead *linearize* it: for a column `[x; C]` with `C` large, `softmax([x; C]) = [x + ε; *]`, i.e. over the non-`C` entries the softmax is locally linear with error controlled by `C`. So with the right padding attention computes the bilinear `A^T B + ε M`, `‖M‖ ≤ 1` — matrix multiplication in two layers, one head, error tunable by `C`. Transpose I get by vectorizing the matrix and applying a fixed permutation of columns (a permutation is just a read/write pattern) — four layers. With multiply and transpose as function blocks, the standard iterations follow: matrix inversion by Newton's iteration and the top eigenvector by power iteration, each a short `FLEQ` program running in a thirteen-layer looped transformer.

And then the payoff I was actually after: training a model in-context is *also* just a `FLEQ` program. Linear SGD does `w ← w − η Σ_i (w^T x_i − y_i) x_i`; as a program that's a handful of instructions — `f_mul` for the inner product `w^T x_i`, `f_sub` for the residual `w^T x_i − y_i`, `f_mul` to multiply by `x_i`, `f_mul` by the step size `η`, `f_sub` to update `w` — plus pointer increments to walk the data, a within-epoch counter and a `≤ 0` branch to cycle back to the first point, and an epoch counter to stop after `T` passes. Thirteen layers, one head, width `O(log D + d)`, and the number of *loops* scales with `T·D`. For a nonlinear model I write backprop out as a `FLEQ` program too: forward pass `z = W_1 x + b_1`, `a = σ(z)` (the sigmoid function block I just built), `o = W_2 a + b_2`; backward `δ_2 = o − y`, `δ_1 = σ'(z) ⊙ W_2 δ_2` with `σ'(z) = σ(z)(1 − σ(z))` assembled from `f_sigmoids`, `f_sub`, `f_mul`; the gradients are outer products `δ a^T`, `δ x^T` via `f_mul`; and the updates subtract `η ·` gradient via `f_mul`, `f_sub`. Still thirteen layers; the network's depth shows up only as more loop iterations (`f_sigmoids` can encode any activation as a sum of sigmoids if I'm willing to pay heads for the approximation error). That's training — including backprop, including a nonlinear model — performed by a fixed-depth attention network at inference time, with weights I can write down.

So the whole thing chains back to one decision. Refusing to let depth grow forced me to add a loop, like a CPU's cycles. The loop forced me to lay the input out as scratchpad/memory/commands and to give every column a binary address. Addressing forced read/write out of attention-as-approximate-permutation plus a ReLU overwrite gate, and arithmetic and control out of the ReLU MLP. Those primitives assemble into one `SUBLEQ` instruction in nine fixed layers — a universal computer whose program length lives in the number of loops, not the depth. Generalizing the single operation to a menu of function blocks gives `FLEQ`; getting nonlinearity *from the softmax* (Barron) and matrix products from *linearizing* the same softmax keeps every block constant-depth; and once arbitrary functions and linear algebra are cheap, a calculator, matrix inversion, power iteration, and SGD-with-backprop are all just short programs the looped, ≤13-layer transformer executes.

```python
import torch
import torch.nn as nn


# A transformer layer with a tunable softmax temperature (lambda) and a position-wise
# ReLU MLP. In the construction every weight below is *set*, not trained.
class TFLayer(nn.Module):
    def __init__(self, d_model, n_heads, lam):
        super().__init__()
        self.d, self.h, self.lam = d_model, n_heads, lam
        # per-head K, Q, V; for read/write we set K = Q (project onto position rows);
        # for a sigmoid block we put slopes a_ji in K, coeffs c_ji in V, identity in Q.
        self.K = nn.ParameterList(nn.Parameter(torch.zeros(d_model, d_model)) for _ in range(n_heads))
        self.Q = nn.ParameterList(nn.Parameter(torch.zeros(d_model, d_model)) for _ in range(n_heads))
        self.V = nn.ParameterList(nn.Parameter(torch.zeros(d_model, d_model)) for _ in range(n_heads))
        self.W1 = nn.Parameter(torch.zeros(4 * d_model, d_model))   # FFN: arithmetic + control
        self.b1 = nn.Parameter(torch.zeros(4 * d_model))
        self.W2 = nn.Parameter(torch.zeros(d_model, 4 * d_model))
        self.b2 = nn.Parameter(torch.zeros(d_model))

    def forward(self, X):                       # X: [d_model, n] (columns = tokens)
        attn = X.clone()
        for K, Q, V in zip(self.K, self.Q, self.V):
            scores = (K @ X).transpose(0, 1) @ (Q @ X)      # entries p_i^T p_j (read) or a^T x
            P = torch.softmax(self.lam * scores, dim=0)     # high lam -> approx permutation /
            attn = attn + V @ X @ P                         #            sigmoid via [z; C] padding
        ones = torch.ones(1, X.shape[1])
        return attn + self.W2 @ torch.relu(self.W1 @ attn + self.b1[:, None] @ ones) + self.b2[:, None] @ ones


def pos_encoding(i, log_n):
    # +/-1 binary code of column index i: p_i^T p_i = log_n, p_i^T p_j <= log_n - 1 (i != j),
    # so high-temperature attention can address a column by matching its code.
    bits = [(i >> k) & 1 for k in range(log_n)]
    return torch.tensor([1.0 if b else -1.0 for b in bits])


def fix_bits(b, eps):
    # snap softmax (not hardmax) residue back to exact {-1, 0, 1}; |b - exact| < eps < 0.5.
    s = 1.0 / (1.0 - 2.0 * eps)
    return s * (torch.relu(b + 1 - eps) - torch.relu(b + eps)) \
         + s * (torch.relu(b - eps) - torch.relu(b - 1 + eps)) - 1.0


# One FLEQ instruction = a fixed-depth stack: 9 layers for the SUBLEQ skeleton
# (read instr; read mem[a], mem[b] with 2 heads; subtract; write back; flag; branch; error-fix)
# plus the layers of the selected function block f_m (sigmoid/matmul/transpose/...).
def build_fleq_computer(d_model, log_n, lam, function_blocks):
    skeleton = [TFLayer(d_model, n_heads=2, lam=lam) for _ in range(9)]  # SUBLEQ control logic
    deepest = max(len(fb) for fb in function_blocks)                     # depth set by one block
    body = [TFLayer(d_model, n_heads=sum(fb_heads), lam=lam)             # heads summed over blocks
            for _ in range(deepest)] if function_blocks else []
    layers = skeleton + body
    return layers                                                       # weights set per the lemmas


# Drive the computation by feeding the output back as the next input: T loops = T instructions.
# Depth of `layers` is constant; program length lives entirely in T.
def run(layers, X, T):
    for _ in range(T):
        for layer in layers:        # one application of the looped block ...
            X = layer(X)            # ... executes exactly one program step
    return X
```
