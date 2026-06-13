# Context: minimizing a rugged cost over a huge discrete space (circa 1982)

## Research question

We are handed a combinatorial optimization problem: a finite but astronomically large set of
configurations, and a cost (objective) function `f` that maps each configuration to a real number
measuring how good it is. The goal is to find a configuration of minimum cost — or, realistically,
one close to it. The traveling-salesman problem is the clean stand-in: given `N` cities and the
cost of traveling between any two, find the tour visiting each city once and returning home that
minimizes total distance. The same flavor of problem pervades scheduling, layout, and the physical
design of computers (partitioning gates between chips, placing components, routing wires), where
the number of variables runs into the tens of thousands.

The difficulty is structural. The decision version of the traveling-salesman problem is
NP-complete, and the optimization version is NP-hard: every known exact method needs effort that
grows exponentially with `N`, so exact solution is feasible only up to modest sizes. For problems of
practical size there is no hope of enumerating or exactly solving; one must settle for heuristics
that find good configurations with effort that scales as a small power of `N`. A solution would have
to (1) work on a generic cost function presented as a black box — evaluate `f`, propose
rearrangements — without bespoke problem-specific theory; (2) reliably avoid being trapped in the
poor local minima that defeat the obvious heuristic; and (3) scale to very large `N`. No existing
heuristic meets all three at once: the standard ones either get stuck or are narrowly tailored to one
problem.

## Background

The prevailing approach to such problems is *heuristics*, of which there are two broad strategies.
The first is **divide-and-conquer**: split the problem into manageable subproblems, solve each, and
patch the solutions together. This works well only when the subproblems are naturally disjoint and
the division is a good one, so that errors made at the seams do not eat the gains — for an
intertwined problem the right division is itself unclear. The second is **iterative improvement**,
also called local search or hill-climbing.

The field state, framed by these heuristics, rests on a few load-bearing facts:

- **The cost is a function of a discrete configuration, and its landscape is rugged.** For
  combinatorial problems `f` typically has enormous numbers of local minima — configurations from
  which no single allowed rearrangement lowers the cost, yet which are far above the global
  minimum. This is not incidental; it is why naive search fails.

- **Iterative improvement gets stuck in local, not global, minima — this is the central observed
  failure mode.** One starts from a configuration, applies a standard rearrangement to each part of
  the system in turn, and *accepts only rearrangements that lower the cost*; when no neighboring
  rearrangement improves `f`, the search halts. Because it only ever moves downhill, it halts at a
  local minimum. The customary patch is to run it many times from different random starting
  configurations and keep the best result, which mitigates but does not solve the problem. On the
  traveling-salesman problem a purely greedy construction (from each city, go to the nearest
  not-yet-visited city) gives tours whose length-per-step is about `1.12` on average, well above
  optimal.

- **A system at temperature `T` in thermal equilibrium occupies its configurations with Boltzmann
  probabilities.** From statistical mechanics, the probability of a configuration `{r_i}` of energy
  `E({r_i})` is proportional to its Boltzmann factor `exp(-E({r_i}) / (k_B T))`, where `k_B` is
  Boltzmann's constant. Equilibrium quantities are averages over this distribution. As `T` is
  lowered the distribution concentrates: in the low-temperature limit it collapses onto the lowest-
  energy (ground) state(s). Ground states and near-ground states are exponentially rare among all
  configurations of a macroscopic body, yet they dominate its properties at low temperature
  precisely because of this exponential weighting.

- **The Metropolis Monte Carlo algorithm samples the Boltzmann distribution and obeys detailed
  balance** (Metropolis, Rosenbluth, Rosenbluth, Teller & Teller 1953). To compute canonical-
  ensemble averages `<F> = (∫ F e^{-E/kT}) / (∫ e^{-E/kT})` over a high-dimensional configuration
  space, naive Monte Carlo — sample configurations uniformly and weight each by `e^{-E/kT}` — is
  hopeless, because almost every uniform sample lands where `e^{-E/kT}` is negligibly small. Their
  fix samples configurations *with* probability `∝ e^{-E/kT}` and weights them evenly. The
  mechanism: from the current configuration propose a small random displacement, compute the
  resulting energy change `ΔE`; if `ΔE ≤ 0` accept the move; if `ΔE > 0` accept it with probability
  `e^{-ΔE/kT}` (draw `r` uniform on `(0,1)`, accept iff `r < e^{-ΔE/kT}`), otherwise keep the old
  configuration (and re-count it). With a symmetric proposal (`P_{rs} = P_{sr}`) this rule's
  stationary distribution is exactly Boltzmann: taking `E_r > E_s`, the number of systems moving
  `r→s` is `ν_r P_{rs}` and the number moving `s→r` is `ν_s P_{sr} e^{-(E_r - E_s)/kT}`, so the net
  flow vanishes only when `ν_r / ν_s = e^{-E_r/kT} / e^{-E_s/kT}`; combined with ergodicity (any
  configuration reachable from any other), the chain converges to `ν_r ∝ e^{-E_r/kT}`. A practical
  note from that work: the maximum displacement must be chosen with care — too large and most moves
  are rejected, too small and the configuration barely changes; either extreme slows equilibration.

- **Materials reach low-energy ground states by annealing, not by sudden cooling.** Growing a
  single crystal from a melt is done by careful annealing: first melt the substance, then lower the
  temperature *slowly*, spending a long time near the freezing point. Low temperature alone is not
  sufficient to find a ground state; cooling too quickly (*quenching*) freezes the material into a
  defected, metastable, glass-like state of higher energy. The slow schedule is what lets the
  system stay near equilibrium and settle into the true ground state.

- **Ensemble averages come from the partition function and free energy.** `Z = Tr e^{-E/(k_B T)}`
  sums the Boltzmann factor over all configurations; the free energy `F(T)` (essentially
  `-k_B T ln Z`) yields the average energy `<E(T)>`, the entropy `S(T)` (the logarithm of the
  number of configurations contributing at `T`), and the specific heat `C(T) = d<E>/dT`. A peak in
  `C(T)` marks a change in the order of the system — the onset of freezing or clustering — i.e. a
  phase-transition-like reorganization at a particular temperature.

- **Frustration produces many near-degenerate minima.** When interactions cannot all be satisfied
  at once — e.g. a Hamiltonian combining local random attractive couplings with a long-range
  repulsive (balance) term, as in a spin glass — the ground state is not a single symmetric
  configuration but one of many near-degenerate, disordered configurations. Many practical
  combinatorial cost functions, written with `±1` variables, are exactly such frustrated
  Hamiltonians; transforming one good configuration into another generally requires substantial
  rearrangement.

## Baselines

- **Iterative improvement / local search (hill-climbing).** Start from a configuration; repeatedly
  apply a standard rearrangement operator to each part of the system, accepting a rearrangement
  only if it strictly lowers `f`; stop when no single rearrangement improves the cost. It is a
  downhill walk in configuration space. *Gap:* it terminates at the first local minimum it reaches,
  which for a rugged `f` is generically far from the global minimum; it has no mechanism to climb
  out. The standard mitigation — restart from many random initial configurations and keep the best
  — buys some robustness but no escape from the basins themselves, and its cost grows with the
  number of restarts.

- **Greedy / nearest-neighbor construction (for routing/tour problems).** Build a solution by
  always taking the locally cheapest next step (from the current city, go to the nearest unvisited
  city). *Gap:* myopic — early greedy choices force expensive later ones; the resulting tour is
  systematically above optimal (length-per-step about `1.12` on average for random instances), and
  there is no way to revisit a bad early commitment.

- **Divide-and-conquer.** Partition the problem, solve subproblems independently, patch together.
  *Gap:* only as good as the partition; for problems whose parts interact strongly there is no
  clean way to divide them, and errors introduced at the seams can negate the gains.

- **Specialized exact methods (branch-and-bound, cutting planes, exact tour solvers).** Guarantee
  the true optimum. *Gap:* exponential worst-case effort; usable only on small instances (a few
  hundred cities), and each is engineered for one problem rather than for a generic black-box cost.

## Evaluation settings

The natural testbeds are large combinatorial-optimization instances where the cost is cheap to
evaluate and easy to perturb. The traveling-salesman problem is the standard quantitative yardstick
— it is simply stated, has a long literature, and admits instances of controllable size (from a few
cities up to several thousand), with random uniform city placements and clustered placements as the
two canonical regimes; the metric is total tour length (often normalized to length-per-step so it
is comparable across `N`). The physical-design problems supply realistic large-scale tests: two-way
partitioning of a logic design's gates between chips (metric: number of signals crossing the chip
boundary, plus a balance penalty), component placement (metric: total wire length, with congestion
penalties), and wire routing (metric: wire density / congestion along links). A configuration is
represented compactly — a permuted list of city indices for a tour, a `±1` assignment per gate for
a partition — and a move is a local rearrangement of that representation (swap or reverse a segment
of the tour; flip one gate to the other chip). The natural yardstick is the quality reached by
iterative improvement and greedy heuristics under comparable computational budgets, and how the
required effort scales with `N`.

## Code framework

The primitives that already exist: a mutable representation of a configuration, a cost function
evaluated on it, a random-move operator that perturbs it, a way to copy and restore a configuration,
a uniform random-number source, and optional progress reporting. A generic black-box optimizer
still needs the search loop itself.

```python
import copy

class EnergySearchProblem:
    """A black-box optimization problem over a discrete configuration space.
    The user supplies the representation, the cost, and a random move."""

    copy_strategy = "deepcopy"
    user_exit = False

    def __init__(self, initial_state):
        self.state = self.copy_state(initial_state)
        self.best_state = None
        self.best_energy = None

    def energy(self):
        """Cost of the current configuration. Lower is better."""
        # TODO: problem-specific objective f(state)
        raise NotImplementedError

    def move(self):
        """Perturb self.state in place by one local rearrangement.
        May return the change in energy dE for efficiency, or None."""
        # TODO: problem-specific neighbor proposal
        raise NotImplementedError

    def copy_state(self, state):
        """Return an independent copy of a configuration."""
        if self.copy_strategy == "deepcopy":
            return copy.deepcopy(state)
        if self.copy_strategy == "slice":
            return state[:]
        if self.copy_strategy == "method":
            return state.copy()
        raise RuntimeError("unknown copy strategy")

    def update(self, *args):
        """Optional progress hook."""
        pass

    def search(self):
        """Return the best state and cost found."""
        # TODO: the search loop.
        raise NotImplementedError
```

The open slot is the search loop itself.
