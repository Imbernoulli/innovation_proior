# Context — Multiplying Two Multidigit Numbers, and Whether n² Is the Floor

## Research question

Two numbers `a` and `b`, each `n` digits long in binary, are placed on the input of a computing automaton; the binary digits of their product must come out. The only multiplication method in universal use — the one taught in school, the one ancient scribes used, the one wired into early machines — forms a partial product of every digit of `a` against every digit of `b` and adds the shifted rows. With `n` digits each that is `n` rows of `n` single-bit multiplications: on the order of `n²` bit operations, plus `O(n²)` carries and additions.

For small numbers this is irrelevant. But the cost is quadratic in the digit length, so doubling the operands quadruples the work, and as soon as one wants products of numbers with hundreds or thousands of digits the `n²` term swamps everything. The precise question is whether `n²` bit operations is *intrinsic* to multiplication or merely an artifact of the row-by-row layout. Formally: writing `M(n)` for the least number of bit operations sufficient to multiply two `n`-digit numbers, is `M(n)` forced to grow like `n²`, or is there an algorithm with `M(n) = o(n²)`?

## Background

**The cost model.** Numbers are written in binary; the symbols `0` and `1` are bits. One bit operation means writing one of the symbols `0, 1, +, −, (, )`, or adding/subtracting/multiplying two bits. An algorithm is realized by an automaton whose links carry bits; its cost is measured by `N`, the number of bit operations, and `T`, the depth (the number of time steps when operations run in parallel). For two `n`-digit operands the inputs occupy about `2n` links and the product about `2n+1` links, which already forces the trivial lower bounds `N ≳ n`, `T ≳ log n`.

**The schoolbook method (OML) and its cost.** Write each operand in base `B` (think `B = 10`, or binary `B = 2`). To multiply `a` and `b`, for every digit `aᵢ` and digit `bⱼ` form the one-digit product `aᵢ·bⱼ`, place it at position `i+j`, and sum all of them with carries. With `n` digits each that is exactly `n²` single-digit products and `O(n²)` additions, so `M(n) = O(n²)` by this method. In the binary squaring estimate, if `a` has at least `n/2` ones, the ordinary table contains at least `n²/2` bits to add and no more than `2n²`; adding the shifted rows costs no more than `8n²` bit operations. Together with the input/output-size lower bound, this gives the crude bracket `4n < M(n) < 8n²`. This method is ancient: a doubling-and-adding form of it appears in the Rhind papyrus (~1800 B.C.).

**The prevailing belief: quadratic is the floor.** Around 1956 (and restated at a 1960 seminar on mathematical problems of cybernetics at Moscow University), the conjecture was put forward that `M(n) = Ω(n²)` — that any multiplication automaton must, in the worst case, perform on the order of `n²` bit operations. The argument behind it was historical as much as mathematical: throughout the history of mankind people have used the ordinary method, whose cost is `O(n²)`, and *if a more economical method existed it would surely have been found already*. It also feels structurally forced, because every digit of `a` genuinely enters the final product alongside every digit of `b`, so it seems as though `n²` digit-interactions cannot be avoided. No proof of this lower bound was known — only trivial bounds of order `n` had ever been established — but it was regarded as very plausible.

**The companion program: lower bounds on discrete-function complexity (Ofman, 1962).** The conjecture lived inside a broader effort, organized by Kolmogorov, to estimate the complexity `S_f(n)` of computing discrete functions on binary automata — to find, for various functions, how the least number of bit operations grows with `n`. Multiplication was the central hard case of this program. Ofman, one of the first to study these complexities, formalized the automaton model above and remarked that the difficulties show up already in estimating the complexity of *ordinary* multiplication of binary `n`-digit numbers — the upper side was as stubborn as the lower side. The field's posture toward `M(n)` was: a trivial `Ω(n)` from below, a stubborn `O(n²)` from above, and a widely believed (unproven) conjecture that the two could not be brought together below `n²`.

**Residue (Czech) number systems.** One known representation is the residue number system of Svoboda and Valach (1955): represent a number by its residues modulo the first `k` primes; then multiplication is performed residue-by-residue, costing about `O(n log n)`. Converting between positional and residue form is expensive, and in residue form one cannot tell which of two numbers is larger without converting back.

**The squaring–multiplication equivalence.** Multiplication and squaring are interchangeable up to a constant factor, via
```
a·b = ¼[(a+b)² − (a−b)²] .
```
Dividing by 4 is trivial in binary (a two-bit shift). So `M(n)` equals, up to a constant, the complexity of squaring a single `n`-digit number; one may study `f(x) = x²` instead of the two-variable product. This reduction lets a would-be analyst work with one operand and its square rather than a general product — a convenience that matters once the recursion below is set up on a single number.

**Divide and conquer on a number, and the four-product expansion.** A number can be cut in half by base-position. With `m = n/2`, in base `B`,
```
a = a₁·Bᵐ + a₂ ,   b = b₁·Bᵐ + b₂ ,
```
where the high parts `a₁, b₁` and low parts `a₂, b₂` each have about `m` digits. Expanding,
```
a·b = a₁b₁·B²ᵐ + (a₁b₂ + a₂b₁)·Bᵐ + a₂b₂ .
```
This rewrites one `n`-digit multiplication as four `m`-digit multiplications — `a₁b₁`, `a₁b₂`, `a₂b₁`, `a₂b₂` — plus shifts (multiplication by powers of `B`, i.e. digit placement) and additions, all `O(n)`. The corresponding squaring split, `a = a₁·Bᵐ + a₂`, gives `a² = a₁²·B²ᵐ + 2a₁a₂·Bᵐ + a₂²`, with the cross term `2a₁a₂` sitting at the middle place value.

## Baselines

**Schoolbook long multiplication (OML).** Core idea: form every digit-by-digit partial product `aᵢbⱼ` and accumulate it at place `i+j` with carries. Algorithm: `n` shifted rows, each with `n` single-digit products, summed. Cost: `Θ(n²)` single-digit products and `Θ(n²)` additions. Exact and simple.

**The bit-group splitting automaton (Ofman's Theorem 1).** Core idea: cut the multiplier into groups of `s` bits; within a group, multiply by the selected digit group and accumulate; across groups, run the partial results in parallel and add. Algorithm: parameterized by `s`, `1 ≤ s ≤ m`. It achieves automaton characteristics `N ≍ m²/s`, `T ≍ s·log²m`. At `s = 1`: `N ≍ m²`, `T ≍ log²m`. At `s = m`: `N ≍ m`, `T ≍ m·log m`. This trades operation count against depth — pushing `s` up lowers `N` while inflating depth.

**Naive recursive halving (four sub-multiplications).** Core idea: apply divide-and-conquer to the four-product expansion above, recursing on each `m`-digit product. Algorithm: split `a, b` into halves; recursively compute `a₁b₁`, `a₁b₂`, `a₂b₁`, `a₂b₂`; recombine with shifts and adds. Running time obeys
```
T(n) = 4·T(n/2) + O(n) ,
```
the `O(n)` being the combine cost. Solving: `log₂ 4 = 2`, so `T(n) = Θ(n²)`.

## Evaluation settings

The yardstick is the asymptotic count of elementary operations — single-bit (or single-digit) multiplications and additions, and automaton depth — as a function of operand length `n`, against the schoolbook reference of `Θ(n²)` single-digit products. Operands are `n`-digit integers in a fixed base `B` (decimal for hand analysis; binary for the automaton model). Two cost measures matter: `N`, the total operation count, and `T`, the parallel depth. Correctness is exact, checked against the true product for all inputs; the regime of interest is large `n`, where asymptotics dominate constant factors. The analytic instrument for turning a divide-and-conquer recurrence `T(n) = a·T(n/b) + f(n)` into a closed-form growth rate is the recursion-tree / master-theorem analysis, with the per-level combine cost `f(n) = O(n)`.

## Code framework

The primitives already available: arbitrary-size integers, the schoolbook product as a base operation, integer floor-division and remainder to split an operand into high/low halves, and shifting by multiplying by powers of the base. The scaffold is a recursive routine with a base case, a half-size split, and one empty algebraic slot for the recursive products and recombination.

```python
BASE = 10

def multiply_candidate(x, y):
    # base case: operands small enough to multiply directly
    if x < BASE or y < BASE:
        return x * y

    # choose the split point (about half the digits)
    n = max(len(str(x)), len(str(y)))
    m = n // 2
    split = BASE ** m

    high1, low1 = divmod(x, split)
    high2, low2 = divmod(y, split)

    # TODO: choose the recursive products and recombine them with powers of BASE.
    return ...  # TODO
```
