# Context: unstructured search and the quantum computing toolkit

## Research question

Given an unsorted collection of `N` items, exactly one of which is "marked," how fast can the marked
item be found when nothing about the structure of the collection can be exploited? Formally: there is
a Boolean condition `C` on the `N` states `S₁,…,S_N`, with `C(S_ν)=1` for a single unknown target
`S_ν` and `C(S)=0` for every other state, and `C(S)` can be evaluated on any state in unit time. The
goal is to identify `ν`.

Classically the answer is bleak and well understood. With no sorting to guide the search, an algorithm
can only probe items one at a time; each probe reveals whether that single item is the target. A
deterministic algorithm needs `N−1` probes in the worst case and `N/2` on average; a randomized one
still needs `Θ(N)` probes to succeed with constant probability, because each query rules out at most
one candidate. So `Θ(N)` is the classical cost, and it feels fundamental — there is simply no
information to act on until you have looked at almost everything.

The question that matters is whether a *quantum* computer — a device whose memory can hold a
superposition of all `N` states at once and whose evolution is unitary — can break the linear barrier
for this most structureless of all search problems, and if so, exactly how far the cost can be pushed
down. This matters because unstructured search is the abstract heart of an enormous range of problems:
deciding satisfiability of a formula over `n` binary variables is exactly searching `N=2ⁿ`
assignments for one that works, so a sub-linear search would bear directly on the hardest problems in
`NP`.

## Background

By the mid-1990s the quantum-computing model was firmly in place. Benioff (1980) gave a quantum
Hamiltonian model of computation; Deutsch (1985) formalized the universal quantum computer and the
quantum Church–Turing principle; Bernstein and Vazirani (1993) constructed an efficient universal
quantum Turing machine and pinned down the basic complexity theory; Yao (1993) proved quantum circuits
and quantum Turing machines polynomially equivalent. The operational picture these established is the
load-bearing background:

- The state of an `n`-bit quantum register is a vector of `2ⁿ` complex **amplitudes**, one per
  classical basis state; measuring yields basis state `i` with probability `|amplitudeᵢ|²`.
- Evolution is by **unitary** matrices. Unitarity is forced by probability conservation: the squared
  amplitudes must keep summing to one. This is the quantum analogue of premultiplying a probability
  vector by a stochastic matrix in a classical probabilistic algorithm — except the entries are
  complex, so amplitudes can carry a *sign or phase* and cancel. That cancellation (interference) is
  the one resource with no classical-probability counterpart.
- Three concrete primitives are available and cheap. (1) The single-bit operation
  `M = (1/√2)·[[1,1],[1,−1]]`, which sends `|0>` to the even superposition `(1/√2, 1/√2)` and `|1>`
  to `(1/√2, −1/√2)`. Applied bitwise to `n` bits it is the **Walsh–Hadamard transform**
  `W_{ij}=2^{-n/2}(−1)^{i·j}`, where `i·j` is the bitwise dot product. Acting on the all-zeros state
  it produces the **uniform superposition**: every one of the `N=2ⁿ` amplitudes equals `1/√N`, built
  in `O(log N)` gates. (2) A **selective phase rotation**: a diagonal unitary `diag(e^{iφ₁},…)` that
  multiplies chosen states by a phase. Crucially it leaves every `|amplitude|²` unchanged, so it has
  no analogue among classical probabilistic operations, yet it can tag a state. (3) The ability to
  query the condition `C` coherently — an **oracle** that acts on the register while leaving no record
  of which state it inspected, so that computational paths leading to the same outcome remain
  indistinguishable and can interfere.

Several facts about these primitives set up the problem. The uniform superposition, by itself, is
worthless for search: it spreads amplitude equally, so a measurement returns a uniformly random
item — success probability `1/N`, no better than a single classical guess. Quantum parallelism
("examine all `N` states at once") is therefore a trap; the difficulty is never breadth, it is getting
the amplitude to *concentrate* on the one state you want before you measure. The available primitives
that act differently on a chosen state — notably the selective phase rotation — change amplitudes by a
sign or phase only, and a measurement sees just `|amplitude|²`, so any such tagging is on its own
invisible at readout. And the classical information-theoretic intuition that "you must look at `Θ(N)` items"
hangs over everything as the benchmark to beat.

The one prior quantum speedup of comparable fame, Shor's (1994) polynomial-time factoring algorithm,
achieves its exponential advantage by exploiting *structure* — the periodicity of modular
exponentiation, extracted by a quantum Fourier transform. Unstructured search has, by definition, no
such periodicity to lever, so the route that works for factoring is unavailable here; a genuinely
different mechanism is needed.

## Baselines

- **Classical exhaustive / randomized search.** Probe items one at a time; stop when `C` returns 1.
  Deterministic worst case `N−1`, average `N/2`; randomized `Θ(N)` for constant success. Core idea:
  each query tests one candidate and rules out at most one. The gap it leaves: it uses no
  superposition and no interference, so it cannot test possibilities in parallel — and information-
  theoretically it appears stuck at `Θ(N)` for unstructured instances.

- **Classical probabilistic algorithms (Markov-chain view).** Maintain a probability distribution over
  states; evolve by premultiplying the probability vector by a stochastic transition matrix (e.g.
  simulated annealing). Core idea: a distribution that can be steered toward good states. The gap:
  probabilities are non-negative and add, so distinct paths can only reinforce, never cancel; there is
  no destructive interference to suppress the `N−1` wrong answers.

- **Quantum parallelism via Walsh–Hadamard alone (Deutsch–Jozsa 1992 style).** Put the register in the
  uniform superposition and evaluate the function on all inputs simultaneously. Core idea: one
  evaluation touches every state. The gap, fatal for search: the result is still a uniform spread, so
  a measurement gives a random state with success `1/N`. Deutsch–Jozsa extracts a single global
  property (a parity) by interference, but that trick reads out one bit about the whole function — it
  does not single out one marked state among `N`.

- **Shor's factoring (1994), as the contrasting quantum algorithm.** Core idea: reduce factoring to
  finding the period of `a^x mod N`, then read the period off with a quantum Fourier transform; cost
  polynomial in `log N`. The gap relative to the present problem: it is entirely structure-dependent.
  Remove the periodic structure — as in a black-box unsorted database — and the QFT has nothing to
  lock onto. It shows quantum computers *can* win big, but says nothing about how to win without
  structure.

## Evaluation settings

The natural yardstick is **query (oracle) complexity**: the number of evaluations of the condition `C`
needed to identify the marked state with at least constant success probability (the convention is
probability `≥ 1/2`). The instance family is parameterized by the database size `N=2ⁿ`, with a single
marked item among `N`; the relevant regime is large `N`. Two reference points frame any result. The
classical baseline is `Θ(N)` queries. The information-theoretic question — "how few queries can *any*
quantum algorithm use, given no structure?" — is posed in the oracle (black-box) model; this is the
setting in which a matching lower bound would be proved. Success is measured purely in asymptotic query
count and in the success probability achieved after a prescribed number of iterations; the precise
iteration count, and not just its order, is reported as part of the protocol.

## Code framework

The implementation skeleton starts with the primitives already available: uniform-superposition
preparation (Walsh–Hadamard on the all-zeros register), a coherent oracle for the condition `C`, the rule
that all operations are unitary, and a final measurement. What to do with the register between
preparation and measurement — the per-iteration unitary processing and how long to run it — is left open.

```python
import numpy as np

def uniform_superposition(N):
    # Walsh-Hadamard on |0...0>: equal amplitude 1/sqrt(N) in every basis state.
    return np.full(N, 1.0 / np.sqrt(N))

def oracle(state, marked):
    # Coherent query of C: leaves no trace of which state was inspected.
    # TODO: how should a query act on the amplitudes so that it is (a) unitary
    #       and (b) actually useful for concentrating amplitude on `marked`?
    pass

def iteration_step(state, marked):
    # TODO: the per-iteration unitary processing applied to the register
    #       (may involve the oracle above). Unknown at this point.
    pass

def search(N, marked, iterations):
    state = uniform_superposition(N)
    for _ in range(iterations):
        state = iteration_step(state, marked)  # TODO
    # measure: probability of outcome i is |state[i]|**2
    probs = np.abs(state) ** 2
    return probs

def optimal_iterations(N):
    # TODO: how many repetitions to run?
    pass
```
