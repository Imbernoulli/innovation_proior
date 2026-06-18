# Context: making a fixed-size attention network execute arbitrary programs (circa 2022-2023)

## Research question

A transformer with `L` layers does a fixed amount of computation per forward pass: information
flows through exactly `L` attention+MLP stages and stops. Yet many things we want models to do
are *iterative* — run an algorithm for `T` steps, sweep a dataset for several epochs, apply a
recurrence whose length is not known at design time. The precise problem: can a transformer of
**fixed depth**, not growing with the length of the computation, carry out an arbitrary
iterative program — a calculator, numerical linear algebra, an optimizer training a small
network on data given in its own input — where the only thing that scales with the size of the
computation is the *number of times the network is invoked*, not the number of layers it has?

A solution would have to (1) decouple the network's depth from the number of program steps;
(2) give the network a way to address and edit specific locations of its input (read a
variable, write a result back) the way a CPU reads and writes RAM; (3) implement, inside the
fixed attention/MLP primitives, the operations a general program needs — arithmetic, control
flow (a conditional jump), and at least a rich class of nonlinear functions; and (4) do all of
this with weights one can actually exhibit, so the claim is a construction and not just an
existence theorem. Each existing result below achieves some of these; none achieves all four at
fixed depth with explicit weights. Closing that gap is the problem.

## Background

By this time the transformer (Vaswani et al. 2017) is the dominant sequence architecture. A
layer is an attention sublayer followed by a position-wise MLP. Writing the input as a matrix
`X ∈ R^{d×n}` (each of the `n` tokens a `d`-dimensional column), the layer computes

```
Attn(X) = X + Σ_i V_i X · softmax( X^T K_i^T Q_i X )      (per head i, softmax over columns)
f(X)    = Attn(X) + W_2 · ReLU( W_1 Attn(X) + b_1 1^T ) + b_2 1^T
```

with a temperature `λ ≥ 0` available inside the softmax. Two facts about this layer are
load-bearing. First, attention's score matrix `X^T K^T Q X` lets any column's output depend on
*any other column* — a content-addressable, all-to-all mixing. Second, with a high softmax
temperature the column-softmax of a score matrix approaches a hard, one-hot selection, i.e. an
(approximate) permutation matrix; so attention can *move* a chosen column's contents to another
position. The MLP, being a ReLU network, is a universal approximator of continuous functions on
its own.

Several strands of prior work establish the landscape. **Expressivity / Turing-completeness:**
Pérez, Marinković & Barceló (2019) and Pérez et al. (2021) prove transformers can simulate
Turing machines, and Wei, Chen & Ma (2022, "statistically meaningful approximation") give a
finite-precision version; these constructions use unbounded/high precision and recursive links
around attention. **Recursion in architectures:** Universal Transformers (Dehghani et al. 2018)
already tie weights across depth and apply a block repeatedly, showing recurrence is compatible
with attention. **In-context learning of algorithms:** Garg et al. (2022) show a from-scratch
GPT-2 learns in-context to fit linear functions and small nets, working entirely at the level
of vector embeddings rather than language; Akyürek et al. (2022) and von Oswald et al. (2022)
argue trained in-context learners *implement* learning algorithms implicitly, and give a
hardcoded single linear-self-attention layer equal to one gradient-descent step on linear
regression. **Automata:** Liu et al. (2022) show shallow transformers find "shortcut" solutions
that replicate an automaton's computation with far fewer layers than reasoning steps — but only
for restricted families, since a general shortcut would collapse circuit-complexity classes
widely believed distinct. The standing tension across all of this: the constructions that are
*general* (Turing-complete) are abstract and depth- or precision-hungry, while the ones that are
*concrete* (one GD step, one automaton) are shallow but special-purpose and tied to a single
model/loss; depth keeps scaling with the number of computation steps.

Two pieces of classical theory sit in the toolbox. **One-instruction-set computers:** Mavaddat
& Parhami (1988) show a single instruction suffices for universal computation. The canonical
example, `SUBLEQ(a,b,c)`, subtracts `mem[a]` from `mem[b]`, stores the result in `b`, and jumps
to instruction `c` if the result is `≤ 0`; a machine looping over `SUBLEQ` instructions is
Turing complete given enough memory. **Sigmoid approximation:** Barron (1993) proves any
function with a bounded Fourier-magnitude integral (`f ∈ Γ_{C,B}`, `f(0)=0`) can be approximated
by a linear combination of `m` sigmoids `Σ_i c_i σ(a_i^T x)` with error `O(m^{-1/2})` once the
sigmoid sharpness satisfies `τ ≥ m^{1/2} ln m`.

## Baselines

**RASP / Tracr (Weiss, Goldberg & Yahav 2021; Lindner et al. 2023).** RASP is a domain-specific
language whose primitives mirror an encoder's select/aggregate operations; programs (count,
sort, histogram, recognize Dyck-`k`) compile into transformer weights, and Tracr is a compiler
that realizes this. **Gap:** the compiled network's size scales with the program — more lines of
code means a bigger/deeper transformer — and the language's expressivity is limited (no general
nonlinear functions, iteration is awkward) with Turing-completeness unclear, as discussed by the
authors themselves and by Merrill & Sabharwal (2022).

**Hardcoded in-context gradient descent (Akyürek et al. 2022; von Oswald et al. 2022).** Both
exhibit explicit weights making a transformer implement gradient-based learning at inference
time: von Oswald shows a single linear self-attention layer's update on the input equals one
step of gradient descent on a regression loss; Akyürek constructs a constant-depth decoder that
performs one SGD step for a linear model. **Gap:** restricted to linear models and a single loss
(least squares), and crucially a fixed forward pass equals a *fixed number* of update steps — to
run `T` iterations you need depth proportional to `T`, and there is no mechanism for general
control flow or for nonlinear models / backprop.

**Abstract Turing-completeness constructions (Pérez et al. 2019/2021; Wei, Chen & Ma 2022).**
These prove transformers can simulate any Turing machine, sometimes with a recurrent link.
**Gap:** they are existence/simulation arguments at the level of "there is a transformer that
does it," typically needing high or unbounded precision and not yielding a recipe for a
*particular* algorithm or a way to compile a high-level program; they don't bound the depth
needed for a single step of useful computation.

**Universal Transformer (Dehghani et al. 2018).** Shares one block's weights across depth and
applies it for a number of steps, with a halting mechanism. **Gap:** it is a *trained*
architecture aimed at better generalization, not a construction with known weights that
provably executes a specified program; what one block's single application computes is left to
learning.

**Standard deep transformer for an iterative task.** The default alternative to any of the above
is simply: stack enough layers. **Gap:** depth then grows with the number of reasoning/iteration
steps, which is exactly what becomes impractical for long computations and, for the hardest
tasks, would imply circuit-complexity collapses believed false (Liu et al. 2022).

## Evaluation settings

The natural yardsticks are constructive, not statistical — the artifact is a set of explicit
weight matrices, and the question is *what programs they execute and at what size*. The relevant
measurements are: the depth (number of layers), number of heads, and width (embedding
dimension) required to execute one instruction / one program step; whether the depth stays
constant as the *program length* (number of instructions / iterations `T`) grows; and the
approximation error of the realized computation as a function of the softmax temperature `λ` and
the number of sigmoid terms `m`. Concrete targets to demonstrate, drawn from what a "general
computer" should do: execute a `SUBLEQ`/OISC program (with example programs such as factorial,
list reversal, linear search); run a basic calculator; run numerical linear algebra (matrix
transpose, matrix multiplication, matrix inversion, power iteration); and run an in-context
learning algorithm — SGD with backpropagation training a small (e.g. two-layer) network on data
supplied in the input. Inputs are `d`-dimensional embedding vectors rather than tokens (an
embedding layer maps tokens to these), so the construction can be reasoned about as block-matrix
algebra. Integers are stored as fixed-width `±1` binary vectors (`N` bits), in two's complement.

## Code framework

The construction plugs into the standard transformer-layer primitive that already exists, plus
the recurrence that ties one block's output back to its own input. What is *not* yet settled —
and is exactly what has to be designed — is how to format the input so the network can address
its own memory, and what each fixed-depth block must compute so that looping it executes a
program. The substrate below is just the generic pieces: a transformer layer, a way to attach
position information to columns, and an outer loop.

```python
import torch
import torch.nn as nn


class TransformerLayer(nn.Module):
    """One standard layer: multi-head attention (with a softmax temperature) + a
    position-wise ReLU MLP, each with a residual connection. Weights are given, not trained."""

    def __init__(self, d_model, n_heads, lam):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads)   # K, Q, V supplied as weights
        self.lam = lam                                        # softmax temperature
        self.W1 = nn.Linear(d_model, 4 * d_model)
        self.W2 = nn.Linear(4 * d_model, d_model)

    def forward(self, X):
        X = X + self.attn(X, X, X)[0]                         # content-addressed mixing
        X = X + self.W2(torch.relu(self.W1(X)))              # position-wise nonlinearity
        return X


def positional_columns(n, dim):
    """Attach position information to each of the n columns. Encoding scheme TBD."""
    pass  # TODO: how a column's index is represented so attention can address it


def build_block(spec):
    """Assemble a fixed-depth stack of TransformerLayers whose weights make one application
    of the block do one step of the computation. Depth must NOT grow with program length."""
    # TODO: the construction we will design — what one application of the looped block computes,
    #       and the weights that realize it.
    pass


def run(block, X, T):
    """Drive the computation by feeding the block's output back as its next input."""
    for _ in range(T):
        X = block(X)        # one program step per pass
    return X
```

The single open slot is `build_block` (and the column-addressing scheme it relies on): what one
pass of the looped block must compute, and the explicit weights that make it so.
