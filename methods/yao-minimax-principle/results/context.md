# Lower bounds for randomized computation

## Research question

For a deterministic algorithm we have a clean lower-bound methodology: identify a worst-case input and prove that no algorithm in the model can solve *that* input quickly. The cost of the algorithm is its cost on its single worst input, so one bad input is a certificate against every deterministic algorithm.

Randomization breaks this. A randomized algorithm is a coin-flipping procedure whose running time (or number of probes, comparisons, queries) on a fixed input is a random variable, and whose intrinsic cost is the expected cost on its *worst* input. The trouble is that for any single fixed input the algorithm may flip its coins so as to be fast on exactly that input; the input that is worst for one setting of the coins need not be worst after the coins are re-randomized. So the deterministic recipe — "exhibit one hard input" — gives nothing: the adversary commits to an input, then the algorithm randomizes against it, and the expected cost can be small even when every individual deterministic strategy is bad somewhere.

Concretely, by the late 1970s randomized algorithms (Rabin's probabilistic algorithms, the Solovay–Strassen primality test, randomized combinatorial search) were demonstrably faster than their deterministic counterparts on some problems, and the natural question — *how fast can the best randomized algorithm possibly be; what is the intrinsic randomized complexity of a problem* — had essentially **no general lower-bound technique**. The goal: a method that converts the hard, quantifier-heavy object "the best over all randomized algorithms of the worst over all inputs of the expected cost" into something one can actually compute or bound by hand.

## Background

Two distinct ways of measuring the "expected running time" of an algorithm were on the table, and they are easy to conflate.

The first is the **distributional / average-case** view, in the tradition of analysis of algorithms (Knuth): one *assumes a "natural" distribution on the inputs* — e.g. in sorting n numbers, all n! initial orderings equally likely — and asks for the algorithm with the smallest *average* cost under that distribution. The cost object here is C(A,d) = Σ_x d(x)·(cost of deterministic A on x), and the difficulty of the problem under d is min over deterministic algorithms A of C(A,d). The standard criticism: real input distributions vary, and often the true distribution is simply unknown, so an average-case bound under an assumed distribution may say nothing about practice.

The second is the **randomized** view, made prominent by Rabin's "Probabilistic Algorithms" and Gill's work on probabilistic Turing machines (the concept was familiar to statisticians far earlier — Luce and Raiffa's *Games and Decisions*). Here the *algorithm*, not the input, carries the randomness: a randomized algorithm is a probability distribution q over deterministic algorithms, its cost on a fixed input x is the q-expected cost, and its intrinsic cost is taken on the *worst* input x. The difficulty of the problem is the infimum over randomized algorithms of this worst-case expected cost.

A clean way to make both precise is the **decision-tree model**, the natural setting for low-order combinatorial complexity. A deterministic algorithm is a binary decision tree; each internal node probes one bit of the input (an adjacency-matrix entry, or a pairwise comparison), each leaf gives the answer, and the cost on an input is the root-to-leaf path length (number of probes). Two canonical problem families live here: testing a graph property P from its adjacency matrix (the model of Rivest–Vuillemin, where worst-case probe lower bounds were known and the evasiveness/Aanderaa–Rosenberg line lives), and partial-order / element-selection problems solved by pairwise comparisons (sorting, median, selecting the k-th smallest — Knuth, Floyd–Rivest, Pohl, Fredman).

The load-bearing background fact comes from game theory. **Von Neumann's minimax theorem (1928)** concerns a finite two-person zero-sum game with payoff matrix U: a maximizing player chooses a row, a minimizing player chooses a column, and the payoff is U(i,j). If a player must move first and announce a mixed strategy, the opponent best-responds. It is immediate that moving second is no worse — more information — so min over the minimizing player's mixed strategies of the max over rows ≥ max over the maximizing player's mixed strategies of the min over columns; one always has minmax(U) ≥ maxmin(U). Von Neumann's theorem is the non-obvious reverse: with *mixed* strategies the two are **equal**, minmax(U) = maxmin(U) = v, the value of the game — going first is no disadvantage. Equivalently (and this is the same fact wearing optimization clothes), the optimal mixed strategies of the two players are the solutions of a primal/dual pair of linear programs, and minmax = maxmin is **strong LP duality**. Borel had earlier obtained only special small cases (up to 5×5) and doubted the general statement; von Neumann's 1928 proof settled it.

## Baselines

- **Worst-case deterministic decision-tree lower bounds (Rivest–Vuillemin 1976; the evasiveness line).** For testing a graph property from the adjacency matrix, one proves that any deterministic decision tree must probe Ω(n²) entries by adversary/topological arguments. Core idea: an adaptive adversary answers probes so as to keep the algorithm uncommitted as long as possible. Gap: this bounds *deterministic* algorithms only. A randomized algorithm escapes the adversary by randomizing which entries it probes, and the worst-case-input adversary cannot pin it down — so these techniques said nothing about randomized complexity, which "was lacking previously."

- **Average-case analysis under an assumed input distribution (Knuth; Floyd–Rivest 1975 for selection; Pohl 1975).** Idea: fix a natural distribution d, compute or bound min_A C(A,d) for deterministic A. Floyd–Rivest, for instance, bound the average comparisons for finding the median under the uniform ordering. This is a fully developed deterministic technology. Gap as a *randomized* tool: on its face it answers a different question — average cost of a deterministic algorithm under one assumed distribution (criticized as distribution-dependent) — and it was not used to say anything about randomized algorithms at all.

- **Direct analysis of specific randomized algorithms (Rabin 1976; Gill 1974; Solovay–Strassen 1977).** These give *upper* bounds — a particular randomized algorithm runs in expected time T — establishing that randomization helps. They are constructions, not impossibility results: they cannot certify that no randomized algorithm beats some threshold. The only concrete demonstration in the literature that randomization lowers complexity by an order of magnitude was Freivalds' result that {ww : w∈{0,1}*} is recognized by a 1-tape Turing machine with error in O(n log n) vs Θ(n²) deterministically — an *upper*-bound-style separation, again not a lower-bound method.

- **The minimax theorem / LP duality itself (von Neumann 1928).** As a baseline it is a finished mathematical fact about games and linear programs, stated and used for economics and decision theory (Luce–Raiffa). Gap: as it stands it is a fact about abstract matrices and linear programs; it has not been connected to decision-tree algorithm complexity, and on its own says nothing about the cost of any algorithm.

## Evaluation settings

The natural yardsticks are the low-order combinatorial complexity problems where exact constants matter:

- **Graph-property testing from the adjacency matrix.** Inputs: undirected graphs on n vertices, 𝒢_n. A property P: 𝒢_n → {true,false}. Cost metric: number of adjacency-matrix entries probed by a decision tree. Properties of interest include connectedness-type / monotone properties, non-planarity, containing a fixed subgraph, being Hamiltonian, containing a perfect matching, containing a clique of size k.

- **Element-selection / partial-order problems by comparisons.** Inputs: linear orderings of an n-element set V. Cost metric: number of pairwise comparisons "x_i : x_j". Tasks: sorting (output the full order), selecting the k-th smallest, finding the median, finding a "mediocre" element (rank in the middle third). The uniform distribution over the n! orderings is the canonical input distribution.

- **Error regimes.** Two-sided: algorithms required to be always correct (errorless) versus algorithms permitted to answer wrongly with probability at most λ (e.g. λ = 5%, or λ ≤ 0.1). The same problems are studied in both regimes, and the comparison between them is itself a measurement target (does allowing error buy an order-of-magnitude speedup?).

## Code framework

The decision-tree harness needs the model primitives and the two cost functionals. The unresolved part is whether these two functionals have any exact relationship.

```python
# --- problem & model primitives ---

def inputs(n):
    """Enumerate all inputs of size n (e.g. graphs on n vertices, or
    linear orderings of n elements)."""
    raise NotImplementedError

def deterministic_algorithms(n):
    """Enumerate the deterministic decision trees in the model
    (pure algorithms: each probes one bit / makes one comparison per node,
    no redundant tests)."""
    raise NotImplementedError

def cost(A, x):
    """r(A, x): number of probes/comparisons the deterministic tree A
    makes on input x (root-to-leaf path length)."""
    raise NotImplementedError

# --- cost functionals ---

def average_cost(A, d, n):
    """C(A, d) = sum_x d(x) * cost(A, x): deterministic A's average cost
    under input distribution d."""
    return sum(d[x] * cost(A, x) for x in inputs(n))

def best_deterministic_under(d, n):
    """min_A C(A, d): the best a deterministic algorithm can do, knowing d."""
    return min(average_cost(A, d, n) for A in deterministic_algorithms(n))

def randomized_worstcase_cost(q, n):
    """A randomized algorithm is a distribution q over deterministic algorithms.
    Its intrinsic cost is the worst-input expected cost:
        max_x  sum_A q(A) * cost(A, x)."""
    return max(sum(q[A] * cost(A, x) for A in deterministic_algorithms(n))
               for x in inputs(n))

# --- complexity quantities ---

def distributional_complexity(n):
    """F1 = sup over input distributions d of best_deterministic_under(d)."""
    # TODO
    pass

def randomized_complexity(n):
    """F2 = inf over randomized algorithms q of randomized_worstcase_cost(q)."""
    # TODO
    pass

# --- unresolved comparison ---

def relate(distributional, randomized):
    """TODO: decide whether and how F1 and F2 are related."""
    pass
```
