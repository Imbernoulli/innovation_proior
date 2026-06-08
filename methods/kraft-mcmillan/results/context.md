# Context: achievable codeword-length profiles for decodable variable-length codes

## Research question

We want to encode the symbols of a source — letters of an alphabet, outputs of a random variable `X` ranging over `X = {x₁,…,x_m}` — as strings over a `D`-ary channel alphabet `D = {0,1,…,D−1}`, and we want the encoding to be *short on average*. The natural idea is the one Morse already used: give frequent symbols short codewords and rare symbols long ones, so that the expected length `L = Σ p_i ℓ_i` (with `ℓ_i` the length of the codeword for `x_i`) is as small as possible.

Two hard requirements collide with that wish. First, we do not send one symbol; we send a *stream*, and we transmit codewords back-to-back with no separators. The receiver sees one long `D`-ary string and must recover the original symbol sequence from it **unambiguously** — no two different source sequences may produce the same channel string. Second, we would very much like to decode *as we go*: recognize each symbol the instant its codeword ends, without scanning ahead or waiting for the rest of the message.

So the precise problem is: **which length profiles `(ℓ_1,…,ℓ_m)` are actually realizable** by a code that can be decoded? If we make all the `ℓ_i` tiny we will surely create ambiguity; there must be a budget. What is that budget, exactly, and does it depend on *how strong* a decodability requirement we impose — instant decoding versus mere eventual unique decoding? And once we know the budget, **how short can `L` be made**, and how close can we get to that floor in practice?

A solution has to (a) state the exact constraint relating the lengths `ℓ_i`, (b) say whether demanding instant decodability throws away length profiles that mere unique decodability would have allowed, and (c) connect the achievable `L` to the statistics `p_i` of the source.

## Background

**Codes and their extension.** A source code `C` assigns to each symbol `x ∈ X` a nonempty finite string over the `D`-ary channel alphabet; `ℓ(x) = |C(x)|`. The expected length is `L(C) = Σ_x p(x) ℓ(x)`. To encode a sequence we use the *extension* `C*`, the concatenation `C*(x_1 x_2 ⋯ x_n) = C(x_1)C(x_2)⋯C(x_n)`.

**The nested hierarchy of code classes.** Several increasingly strict decodability conditions sit one inside the other:
- *Nonsingular*: distinct symbols get distinct codewords (`x ≠ x' ⇒ C(x) ≠ C(x')`). Enough to describe one symbol, not a stream.
- *Uniquely decodable (UD)*: the extension `C*` is nonsingular — no two distinct symbol sequences encode to the same channel string. Decoding may require looking at the *entire* received string before even the first symbol is fixed.
- *Prefix / instantaneous*: no codeword is a prefix of any other. Then the end of each codeword is recognizable immediately; the code is *self-punctuating* and decodes left-to-right with no lookahead.

These form strict containments `prefix ⊂ uniquely decodable ⊂ nonsingular ⊂ all codes`. Concrete witnesses exist that a UD code need not be prefix-free: e.g. over `{0,1}` the assignment `{10, 00, 11, 110}` is uniquely decodable but not instantaneous (one may have to read far ahead — counting a run of `0`s — to decode the first symbol), and Sardinas–Patterson (1953) gives a finite algorithm to test unique decodability by forming and eliminating sets of dangling suffixes.

**The tree representation.** A `D`-ary code lives naturally on a `D`-ary tree: each node has `D` children, the `D` edges out of a node are labelled by the `D` channel symbols, and a codeword is the label-sequence along a root-to-node path. The prefix condition says **no codeword is an ancestor of another** — equivalently, codewords sit at *leaves* of the subtree they span. A node at depth `ℓ_i` has exactly `D^{ℓ_max − ℓ_i}` descendants at any deeper level `ℓ_max`, and the descendant-sets of distinct prefix codewords are disjoint. The total number of nodes at depth `ℓ_max` is `D^{ℓ_max}`. This finite *leaf budget* is the combinatorial fact the whole problem turns on.

**Entropy and the intuition that lengths look like a distribution.** For a source with pmf `p`, the `D`-ary entropy is `H_D(X) = −Σ p_i log_D p_i = Σ p_i log_D(1/p_i)`. Shannon's framing of communication (1948) already suggested that the right "cost" of a symbol of probability `p_i` is about `log_D(1/p_i)` `D`-ary digits, so that a code with `ℓ_i ≈ log_D(1/p_i)` would have `L ≈ H_D(X)`. Relative entropy `D(p‖q) = Σ p_i log_D(p_i/q_i) ≥ 0` (Gibbs' inequality), with equality iff `p = q`, is the standard tool for turning "`L` versus `H`" into a nonnegativity statement. Morse code itself is the motivating pre-existing artifact: short sequences for frequent letters (a single dot for E), but it needs an inter-letter space — a wasted symbol — precisely because it is not self-punctuating, which is exactly the inefficiency a prefix code would remove.

**The discrete-noiseless-channel connection.** Shannon (1948, Part I) computed the capacity of a discrete noiseless channel whose allowed symbols have integer durations `t_1,…,t_m`: the number of admissible strings of length `T` grows like `X_0^T`, where `X_0` is the largest real root of the characteristic equation `Σ_j X^{−t_j} = 1`. The expression `Σ X^{−t_j}` is the same shape as the quantity that will govern codeword lengths, with the channel-symbol durations `t_j` playing the role of codeword lengths `ℓ_j` — a hint that the achievable-length question is a counting/capacity question in disguise.

## Baselines

The prior art is the set of constructions and partial answers available for variable-length lossless codes.

- **Morse code (telegraphy).** Frequency-sensitive variable-length code over a 4-symbol alphabet (dot, dash, letter-space, word-space). Core idea: short codes for frequent letters. Gap: it is *not* self-punctuating among the dot/dash symbols — it needs explicit spaces between letters, wasting alphabet capacity; and many `D`-ary strings are unused, so it is neither prefix-optimal nor analyzed for an exact length budget.

- **Shannon–Fano coding (Shannon 1948; Fano).** Order the symbols by probability; recursively split the ordered list into two (or `D`) groups of nearly equal probability and assign the next channel digit by group. Produces a prefix code with an expected-length guarantee on the order of `H_D(X)+1`. Core idea: top-down balanced splitting approximates `ℓ_i ≈ log_D(1/p_i)`. Gap: it is a *suboptimal* heuristic — the greedy split need not minimize `Σ p_i ℓ_i`, and crucially it gives a construction, not a *characterization* of which length profiles are even possible or a matching lower bound on `L`.

- **Sardinas–Patterson test (1953).** A finite algorithm that *decides whether a given code is uniquely decodable* by iteratively forming the set of "dangling suffixes" and checking whether any codeword reappears among them. Core idea: unique decodability is a decidable structural property of the codeword set. Gap: it tests a *given* code; it says nothing about which *length profiles* `(ℓ_i)` admit a UD code, nor whether instant-decodability shrinks the realizable set, nor about expected length.

- **Fixed-length / block codes and Shannon's source-coding results (1948).** For an i.i.d. source one can block `n` symbols and use roughly `nH_D(X)` channel digits asymptotically (the basis of the noiseless coding theorem). Core idea: typical-set counting gives an asymptotic rate of `H_D(X)`. Gap: it is asymptotic and combinatorial (about counting typical sequences), not a *per-symbol, finite, exact* statement about variable-length codeword lengths and the budget they must obey; it does not isolate the clean inequality on the `ℓ_i`.

The open gap across all of these: nobody has a *tight, exact* condition `Φ(ℓ_1,…,ℓ_m)` that is **necessary and sufficient** for a decodable code with those lengths to exist, nor a clean argument that the instant-decodability restriction is free.

## Evaluation settings

The yardsticks against which a length-budget result and a coding scheme would be assessed are all pre-existing and structural rather than benchmark datasets:

- **The code classes themselves** as test objects: nonsingular, uniquely decodable, prefix/instantaneous — with explicit small examples over `D = 2` (e.g. the four-symbol codes used to separate the classes) to check that any proposed condition correctly admits/rejects each.
- **Source models**: a single random variable `X` with a given pmf `p` over a finite alphabet (the memoryless case), i.i.d. blocks `X^n`, and stationary sources for asymptotic per-symbol length.
- **Metrics**: existence of a code with a prescribed length profile (`yes/no`), and expected codeword length `L = Σ p_i ℓ_i` (in `D`-ary digits/symbol), to be compared against the entropy `H_D(X)` as the candidate floor and `H_D(X)+1` as the candidate per-symbol ceiling; per-symbol length `L_n = (1/n)·E[ℓ(X^n)]` for block coding.
- **The `D`-ary expansion / unit interval** as a checking device: assigning each codeword the sub-interval of `[0,1]` whose `D`-ary expansion starts with that codeword, so disjointness of intervals tests the prefix condition.

## Code framework

Before any exact length-budget theorem exists, the scaffold is just the apparatus for *defining* codes, *measuring* their length, and *checking* decodability, with explicit stubs for the still-unknown achievability test and construction.

```python
from math import log

# --- pre-existing primitives ---

def expected_length(code, p):
    """L(C) = sum_i p_i * len(codeword_i). Known once a code is given."""
    return sum(p[x] * len(code[x]) for x in code)

def entropy_D(p, D):
    """H_D(X) = sum_i p_i log_D(1/p_i). The candidate floor for L."""
    return sum(pi * log(1.0/pi, D) for pi in p.values() if pi > 0)

def is_prefix_free(code):
    """No codeword is a prefix of another (instantaneous / self-punctuating)."""
    words = list(code.values())
    for i, a in enumerate(words):
        for j, b in enumerate(words):
            if i != j and b[:len(a)] == a:
                return False
    return True

def is_uniquely_decodable(code):
    """Sardinas-Patterson: decide if the extension is nonsingular.
    A pre-existing decision procedure on a *given* code."""
    # TODO: dangling-suffix iteration (Sardinas-Patterson 1953)
    pass

# --- open primitives ---

def lengths_are_achievable(lengths, D):
    """Necessary-and-sufficient test on a positive length profile (l_1,...,l_m):
    does there EXIST a decodable D-ary code with exactly these lengths?
    No exact budget Phi(l_1,...,l_m) is available yet."""
    # TODO: achievability condition
    pass

def construct_code(lengths, D):
    """Given an achievable length profile, build an actual decodable code.
    The constructive converse is still unknown."""
    # TODO: place codewords to realize these lengths
    pass

def optimal_lengths(p, D):
    """Choose (l_1,...,l_m) minimizing L = sum p_i l_i subject to
    achievability, and bound the resulting L against H_D(X).
    The length-vs-entropy relation is still open."""
    # TODO: the optimal length rule + its expected-length bound
    pass
```
