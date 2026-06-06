# Simulated Annealing

**Domain:** Global Optimization (combinatorial / black-box minimization).

## Problem

Minimize a cost function `f` over a huge discrete configuration space (e.g. the traveling-salesman
tour length over all permutations of `N` cities). The space is far too large to enumerate, exact
solution is exponential, and the obvious heuristic — iterative improvement / hill-climbing, which
accepts only moves that lower `f` — gets permanently stuck in a local minimum because it can never
step uphill to escape a basin.

## Key idea

Read the cost as a physical **energy** and borrow the way matter finds its lowest-energy state. A
system at temperature `T` in thermal equilibrium occupies a configuration of energy `E` with the
**Boltzmann probability** `∝ e^{-E/T}` (units chosen so `k_B = 1`); as `T → 0` this distribution
collapses onto the minimum-energy (ground) states. So if we could sample configurations from
`e^{-f/T}` and lower `T` to zero, the sample would land on the minimum-cost configuration.

Sampling that distribution is done by the **Metropolis rule**: from the current configuration,
propose a local move with a symmetric proposal, compute the cost change `ΔE`, and

- accept the move if `ΔE ≤ 0` (downhill always),
- accept it with probability `e^{-ΔE/T}` if `ΔE > 0` (uphill, controlled),
- otherwise keep the current configuration (and count it as the next step).

Detailed balance makes this the unique rule (for symmetric proposals) whose stationary distribution
is Boltzmann: with `E_r > E_s` the net flow between neighbors vanishes only when
`ν_r / ν_s = e^{-(E_r - E_s)/T}`, forcing `ν_r ∝ e^{-E_r/T}`. The temperature is a single dial:
high `T` (`e^{-ΔE/T} ≈ 1`) accepts almost everything and the walk **explores**; `T = 0` accepts only
downhill moves and reduces exactly to greedy hill-climbing (**exploits**); intermediate `T` accepts
small uphill moves readily and large ones rarely, escaping shallow traps without wandering forever.

The remaining choice is the schedule of `T`. A fixed positive `T` never commits to a minimum; an
instant drop to `T = 0` (**quenching**) freezes the walk into a poor local minimum — the same
failure as accept-only-downhill. The fix is the metallurgical analogy: **anneal** rather than
quench. Start at a high `T_max` where the system is "molten" (nearly all moves accepted), then lower
`T` **slowly** toward zero, spending enough Metropolis steps at each temperature for the walk to
re-equilibrate, so the sampled distribution tracks the Boltzmann distribution continuously down to
the ground state. Large-scale structure settles at high `T`, fine detail at low `T`.

## Final algorithm

Four ingredients per problem: a configuration representation, a random local **move**, a cost
(**energy**), and a cooling **schedule**.

1. Initialize a configuration; set `T = T_max`; record it as the best seen.
2. Repeat for the planned number of steps:
   - propose a local move; compute `ΔE` (cheap incremental delta from the touched part);
   - accept iff `ΔE ≤ 0` or `random() < e^{-ΔE/T}`; otherwise restore the previous configuration;
   - update the best-ever configuration if the current cost is lower;
   - lower `T` toward `T_min` on a **geometric** schedule
     `T(s) = T_max · (T_min/T_max)^{s/S} = T_max · e^{-ln(T_max/T_min)·s/S}` (geometric in `T`,
     linear in `log T`; ratio `α = T_{k+1}/T_k` a little below 1, e.g. ~0.9).
3. Stop when `T` reaches `T_min` (system "frozen"); return the **best configuration seen**.

Geometric (not linear) cooling is scale-free and devotes more stages to the low-`T` regime where the
fine structure is resolved; a logarithmic schedule `T_k ∝ c/log k` has an asymptotic global-
convergence guarantee but is impractically slow, so geometric is the practical choice. The cost
variance at fixed `T` estimates the specific heat `C(T) = d⟨E⟩/dT`; its peaks mark the
phase-transition-like temperatures where cooling should be slowest. `T_max` is calibrated so the
initial acceptance rate is near 1; `T_min` so the final improvement rate is near 0.

## Code

A clean, general-purpose implementation: an abstract annealer doing the geometric cooling and the
Metropolis test, with the traveling salesman as a concrete instance.

```python
import math
import random
import copy


class Annealer:
    """Minimize a black-box cost over a discrete configuration space by
    simulated annealing. A subclass supplies move() and energy()."""

    Tmax = 25000.0   # starting "molten" temperature: nearly all moves accepted
    Tmin = 2.5       # final "frozen" temperature: essentially no uphill moves
    steps = 50000    # total Metropolis steps over the whole cooling schedule

    def __init__(self, initial_state):
        self.state = copy.deepcopy(initial_state)
        self.best_state = None
        self.best_energy = None

    def move(self):
        """Perturb self.state by one local rearrangement, in place.
        Return the energy change dE (cheap incremental delta), or None
        to recompute the energy from scratch."""
        raise NotImplementedError

    def energy(self):
        """Cost of the current configuration. Lower is better."""
        raise NotImplementedError

    def anneal(self):
        # Geometric cooling: T = Tmax * exp(Tfactor * step/steps),
        # Tfactor = -ln(Tmax/Tmin), so step 0 -> Tmax, step `steps` -> Tmin.
        if self.Tmin <= 0.0:
            raise ValueError("Tmin must be > 0 for geometric cooling.")
        Tfactor = -math.log(self.Tmax / self.Tmin)

        T = self.Tmax
        E = self.energy()
        prev_state = copy.deepcopy(self.state)
        prev_energy = E
        self.best_state = copy.deepcopy(self.state)
        self.best_energy = E

        step = 0
        while step < self.steps:
            step += 1
            T = self.Tmax * math.exp(Tfactor * step / self.steps)

            dE = self.move()
            if dE is None:
                E = self.energy()
                dE = E - prev_energy
            else:
                E += dE

            # Metropolis: reject an uphill move with prob 1 - exp(-dE/T).
            if dE > 0.0 and math.exp(-dE / T) < random.random():
                self.state = copy.deepcopy(prev_state)   # rejected: restore
                E = prev_energy
            else:                                        # accepted
                prev_state = copy.deepcopy(self.state)
                prev_energy = E
                if E < self.best_energy:
                    self.best_state = copy.deepcopy(self.state)
                    self.best_energy = E

        self.state = copy.deepcopy(self.best_state)
        return self.best_state, self.best_energy


class TravellingSalesmanProblem(Annealer):
    """State is a permuted list of cities; energy is the closed tour length."""

    def __init__(self, state, distance_matrix):
        self.distance_matrix = distance_matrix
        super().__init__(state)

    def move(self):
        # Swap two cities; return the resulting change in tour length.
        a = random.randint(0, len(self.state) - 1)
        b = random.randint(0, len(self.state) - 1)
        initial = self.energy()
        self.state[a], self.state[b] = self.state[b], self.state[a]
        return self.energy() - initial

    def energy(self):
        d = self.distance_matrix
        s = self.state
        return sum(d[s[i - 1]][s[i]] for i in range(len(s)))


if __name__ == "__main__":
    import itertools
    # toy instance: random points, full distance matrix
    pts = [(random.random(), random.random()) for _ in range(30)]
    dm = [[math.dist(p, q) for q in pts] for p in pts]
    state = list(range(len(pts)))
    random.shuffle(state)
    tsp = TravellingSalesmanProblem(state, dm)
    tsp.Tmax, tsp.Tmin, tsp.steps = 10.0, 0.01, 100000
    best_state, best_energy = tsp.anneal()
    print("tour length:", best_energy)
```

The same `Annealer` solves any problem for which one can write a configuration, a random `move`, and
an `energy` — partitioning (flip a gate between chips), placement (swap two components), routing
(re-route a wire), scheduling — by subclassing and filling in those two methods. For a continuous,
generalized variant, `scipy.optimize.dual_annealing` implements generalized (Tsallis) simulated
annealing — a heavy-tailed Cauchy-Lorentz visiting distribution, a generalized acceptance
`p = min{1, [1 - (1 - q_a)βΔE]^{1/(1-q_a)}}`, the temperature law
`T_{q_v}(t) = T_{q_v}(1)(2^{q_v-1}-1)/((1+t)^{q_v-1}-1)`, and a local-search polish at accepted points
— with the classic Metropolis/Boltzmann logic above as its special case.
