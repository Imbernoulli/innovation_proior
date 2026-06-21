# Context: constructing a provably minimum-length prefix code

## Research question

We want to transmit messages drawn from a known ensemble by replacing each message with a string of coding symbols (binary digits in the simplest case), and we want the transmission to be as short as possible *on average*. If message *i* occurs with probability *p_i* and its code has length *l_i* digits, the quantity to minimize is the average code length

    L = Σ p_i l_i.

Two requirements constrain the codes we are allowed to use. First, no two messages may share the same code (the code must be uniquely readable). Second, the code must be *instantaneously decodable*: when message codes are run together into one long stream with no separators, the receiver must be able to chop the stream back into messages knowing only the codebook. This forces the **prefix-free** property: no code may appear, digit for digit, as the front of any longer code.

The goal is *constructive*. It is already known (Shannon) that the average length cannot beat the entropy of the source and that a code within one digit of the entropy exists. The question is how to build, given the probabilities, the code of **minimum** average length over all prefix codes — the minimum-redundancy code — together with a proof that nothing shorter is possible.

## Background

A code over an alphabet of *D* symbols can be drawn as a *D*-ary tree: each edge carries one digit, each message sits at a leaf, and the digits along the root-to-leaf path spell the code. The prefix-free property is exactly the statement that every message is a *leaf* — no message sits on the path to another. So choosing a prefix code is choosing a tree, and the length *l_i* is the depth of leaf *i*.

The trees one can build are limited by the **Kraft inequality**. For any prefix code over a *D*-ary alphabet, the lengths must satisfy

    Σ D^{-l_i} ≤ 1,

and conversely any set of lengths obeying this can be realized by some prefix code. The reason is a counting argument on the tree: fix the deepest level *l_max*; a leaf at depth *l_i* blocks off D^{l_max − l_i} of the level-*l_max* nodes as its descendants, these blocked sets are disjoint across leaves, and there are only D^{l_max} nodes at that level, so Σ D^{l_max − l_i} ≤ D^{l_max}. Kraft turns "find the best tree" into "find the best set of lengths subject to Σ D^{-l_i} ≤ 1."

Treating the lengths as continuous and minimizing L = Σ p_i l_i subject to Σ D^{-l_i} ≤ 1 by Lagrange multipliers gives the ideal lengths l_i* = −log_D p_i, with average length equal to the entropy H_D(X) = −Σ p_i log_D p_i. This yields the **source-coding lower bound**: every prefix code has L ≥ H_D(X), with equality exactly when each probability is a power of 1/D (a *D*-adic distribution). The clean way to see it: writing r_i = D^{−l_i} / Σ_j D^{−l_j} and c = Σ D^{−l_i} ≤ 1,

    L − H = D(p‖r) + log_D(1/c) ≥ 0,

both terms non-negative (relative entropy is non-negative; c ≤ 1 by Kraft). No code can have average length below the entropy.

The matching upper bound says the entropy is essentially reachable. Rounding the ideal lengths up, l_i = ⌈log_D(1/p_i)⌉, keeps Σ D^{−l_i} ≤ Σ p_i = 1 (Kraft is satisfied) and gives, from log_D(1/p_i) ≤ l_i < log_D(1/p_i) + 1, the bound

    H_D(X) ≤ L < H_D(X) + 1.

So a prefix code always exists whose average length is within one digit of the entropy. Between these two bounds — entropy below, entropy-plus-one above — lives the true minimum.

A fact about these "ideal" lengths: the per-symbol Shannon length ⌈log_D(1/p_i)⌉ is *not* always the length that symbol gets in an optimal code — for some distributions an optimal code assigns a symbol a length larger than ⌈log_D(1/p_i)⌉, and the set of optimal lengths need not even be unique. Optimality is a property of the whole tree, not of any single symbol.

## Baselines

**Shannon's code.** Take lengths l_i = ⌈log_D(1/p_i)⌉ directly. This satisfies Kraft and lands within one digit of entropy. Each length is built per-symbol from the rounded ideal length of that symbol alone.

**Shannon–Fano coding** (Shannon 1948; Fano, MIT Technical Report 65, 1949). A constructive procedure that works **top-down**. Sort the messages in decreasing order of probability. Split the sorted list into two consecutive groups whose total probabilities are as nearly equal as possible; assign 0 to one group and 1 to the other. Recurse inside each group, splitting again into near-equal-probability halves, until every group holds a single message. Each split contributes one digit to the codes in that group. Read as a tree, this grows from the root (all messages) down to the leaves (single messages), making at each step the most balanced split available. It achieves L(C) ≤ H + 2.

## Evaluation settings

The natural yardstick is the average code length L = Σ p_i l_i (equivalently the redundancy L − H, the excess over the entropy lower bound), measured against a known message ensemble with given probabilities — for instance a small ordered ensemble of a dozen-or-so symbols with assorted probabilities for hand-tracing, or the relative frequencies of letters or symbols in a body of text for a realistic source. The comparison of interest is against the entropy H (the floor) and against the Shannon and Shannon–Fano constructions on the same ensemble, in both the binary case (D = 2) and the general *D*-ary case. The probabilities/frequencies and the entropy of a source are facts about the source itself, available before any code is built.

## Code framework

The pieces that already exist: a min-priority-queue over weighted items, and a tree-walk that reads a digit off each edge. The construction itself — the procedure that, from the probabilities, produces the minimum tree — is the empty slot.

```python
import heapq

def optimal_prefix_lengths(probs):
    """probs: {symbol: probability}. Return {symbol: code length}
    for a minimum-average-length prefix code. To be derived."""
    # min-heap over current nodes by weight already exists
    heap = [(p, i, sym) for i, (sym, p) in enumerate(probs.items())]
    heapq.heapify(heap)
    # TODO: the construction that yields an optimal tree
    pass

def assign_codewords(tree, D=2):
    """Walk the tree, emitting one of D digits per edge -> {symbol: codeword}."""
    # standard root-to-leaf walk already exists; depends only on the tree above
    pass

def average_length(probs, lengths):
    return sum(probs[s] * lengths[s] for s in probs)
```

The slot to fill is the rule inside `optimal_prefix_lengths` — the procedure that builds the tree from the probabilities, and the argument that it attains the minimum.
