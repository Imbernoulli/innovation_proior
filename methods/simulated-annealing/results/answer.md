# Simulated Annealing

**Domain:** Global Optimization (combinatorial / black-box minimization).

## Problem

Minimize a cost function `f` over a huge discrete configuration space (e.g. the traveling-salesman
tour length over all permutations of `N` cities). The space is far too large to enumerate, the
decision version of traveling salesman is NP-complete and the optimization version is NP-hard, and
the obvious heuristic — iterative improvement / hill-climbing, which accepts only moves that lower
`f` — gets permanently stuck in a local minimum because it can never step uphill to escape a basin.

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

With symmetric proposals and downhill moves accepted with probability one, detailed balance fixes
the uphill acceptance probability: with `E_r > E_s` the net flow between neighbors vanishes only when
`ν_r / ν_s = e^{-(E_r - E_s)/T}`, forcing `ν_r ∝ e^{-E_r/T}`. The temperature is a single dial:
high `T` (`e^{-ΔE/T} ≈ 1`) accepts almost everything and the walk **explores**; `T = 0` accepts only
downhill moves and reduces exactly to greedy hill-climbing (**exploits**); intermediate `T` accepts
small uphill moves readily and large ones rarely, escaping shallow traps without wandering forever.

The remaining choice is the schedule of `T`. A fixed positive `T` never commits to a minimum; an
instant drop to `T = 0` (**quenching**) freezes the walk into a poor local minimum — the same
failure as accept-only-downhill. The fix is the metallurgical analogy: **anneal** rather than
quench. Start at a high `T_max` where the system is "molten" (nearly all moves accepted), then lower
`T` **slowly** toward zero, spending enough Metropolis steps at each temperature for the walk to
re-equilibrate. In the ideal quasistatic limit the sampled distribution tracks Boltzmann toward its
zero-temperature mass on ground states; in finite runs the schedule is a practical way to avoid
quenching into the first basin. Large-scale structure settles at high `T`, fine detail at low `T`.

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

Geometric (not linear) cooling is scale-free and makes each stage a comparable fractional change in
`T`; a logarithmic schedule `T_k ∝ c/log k` has an asymptotic global-convergence guarantee but is
impractically slow, so geometric is the practical choice. With `k_B = 1`, the cost variance at fixed
`T` gives the specific heat `C(T) = d⟨E⟩/dT = Var(E)/T^2`; its peaks mark the phase-transition-like
temperatures where cooling should be slowest. `T_max` is calibrated so the initial acceptance rate is
near 1; `T_min` so the final improvement rate is near 0.

## Code

A clean, general-purpose implementation: an abstract annealer doing the geometric cooling and the
Metropolis test, with the traveling salesman as a concrete instance.

```python
import copy
import math
import random
import time
from collections import defaultdict


def round_figures(x, n):
    if x == 0:
        return 0.0
    return round(x, int(n - math.ceil(math.log10(abs(x)))))


class Annealer:
    """Minimize a black-box cost over a discrete configuration space by
    simulated annealing. A subclass supplies move() and energy()."""

    Tmax = 25000.0   # starting "molten" temperature: nearly all moves accepted
    Tmin = 2.5       # final "frozen" temperature: essentially no uphill moves
    steps = 50000    # total Metropolis steps over the whole cooling schedule
    updates = 100
    copy_strategy = "deepcopy"
    user_exit = False

    def __init__(self, initial_state):
        self.state = self.copy_state(initial_state)
        self.best_state = None
        self.best_energy = None
        self.start = None

    def move(self):
        """Perturb self.state by one local rearrangement, in place.
        Return the energy change dE (cheap incremental delta), or None
        to recompute the energy from scratch."""
        raise NotImplementedError

    def energy(self):
        """Cost of the current configuration. Lower is better."""
        raise NotImplementedError

    def copy_state(self, state):
        if self.copy_strategy == "deepcopy":
            return copy.deepcopy(state)
        if self.copy_strategy == "slice":
            return state[:]
        if self.copy_strategy == "method":
            return state.copy()
        raise RuntimeError("unknown copy strategy")

    def set_schedule(self, schedule):
        self.Tmax = schedule["tmax"]
        self.Tmin = schedule["tmin"]
        self.steps = int(schedule["steps"])
        self.updates = int(schedule["updates"])

    def update(self, step, T, E, acceptance, improvement):
        pass

    def anneal(self):
        # Geometric cooling: T = Tmax * exp(Tfactor * step/steps), where
        # Tfactor = -ln(Tmax/Tmin), so step 0 -> Tmax and step `steps` -> Tmin.
        if self.Tmin <= 0.0:
            raise ValueError("geometric cooling requires Tmin > 0")
        Tfactor = -math.log(self.Tmax / self.Tmin)
        self.start = time.time()

        T = self.Tmax
        E = self.energy()
        prev_state = self.copy_state(self.state)
        prev_energy = E
        self.best_state = self.copy_state(self.state)
        self.best_energy = E
        trials = accepts = improves = 0
        if self.updates > 0:
            update_wavelength = self.steps / self.updates
            self.update(0, T, E, None, None)
        else:
            update_wavelength = None

        step = 0
        while step < self.steps and not self.user_exit:
            step += 1
            T = self.Tmax * math.exp(Tfactor * step / self.steps)

            dE = self.move()
            if dE is None:
                E = self.energy()
                dE = E - prev_energy
            else:
                E += dE

            trials += 1
            # Metropolis: reject an uphill move with prob 1 - exp(-dE/T).
            if dE > 0.0 and math.exp(-dE / T) < random.random():
                self.state = self.copy_state(prev_state)   # rejected: restore
                E = prev_energy
            else:                                        # accepted
                accepts += 1
                if dE < 0.0:
                    improves += 1
                prev_state = self.copy_state(self.state)
                prev_energy = E
                if E < self.best_energy:
                    self.best_state = self.copy_state(self.state)
                    self.best_energy = E
            if update_wavelength and self.updates > 1:
                crossed = (step // update_wavelength) > ((step - 1) // update_wavelength)
                if crossed:
                    self.update(step, T, E, accepts / trials, improves / trials)
                    trials = accepts = improves = 0

        self.state = self.copy_state(self.best_state)
        return self.best_state, self.best_energy

    def auto(self, minutes, steps=2000):
        def run(T, steps):
            E = self.energy()
            prev_state = self.copy_state(self.state)
            prev_energy = E
            accepts = improves = 0
            for _ in range(steps):
                dE = self.move()
                if dE is None:
                    E = self.energy()
                    dE = E - prev_energy
                else:
                    E = prev_energy + dE
                if dE > 0.0 and math.exp(-dE / T) < random.random():
                    self.state = self.copy_state(prev_state)
                    E = prev_energy
                else:
                    accepts += 1
                    if dE < 0.0:
                        improves += 1
                    prev_state = self.copy_state(self.state)
                    prev_energy = E
            return E, accepts / float(steps), improves / float(steps)

        step = 0
        self.start = time.time()
        T = 0.0
        E = self.energy()
        self.update(step, T, E, None, None)
        while T == 0.0:
            step += 1
            dE = self.move()
            if dE is None:
                dE = self.energy() - E
            T = abs(dE)

        E, acceptance, improvement = run(T, steps)
        step += steps
        while acceptance > 0.98:
            T = round_figures(T / 1.5, 2)
            E, acceptance, improvement = run(T, steps)
            step += steps
            self.update(step, T, E, acceptance, improvement)
        while acceptance < 0.98:
            T = round_figures(T * 1.5, 2)
            E, acceptance, improvement = run(T, steps)
            step += steps
            self.update(step, T, E, acceptance, improvement)
        Tmax = T

        while improvement > 0.0:
            T = round_figures(T / 1.5, 2)
            E, acceptance, improvement = run(T, steps)
            step += steps
            self.update(step, T, E, acceptance, improvement)
        Tmin = T

        elapsed = time.time() - self.start
        duration = round_figures(int(60.0 * minutes * step / elapsed), 2)
        return {"tmax": Tmax, "tmin": Tmin, "steps": duration, "updates": self.updates}


def distance(a, b):
    R = 3963
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    return math.acos(math.sin(lat1) * math.sin(lat2) +
                     math.cos(lat1) * math.cos(lat2) * math.cos(lon1 - lon2)) * R


class TravellingSalesmanProblem(Annealer):
    """State is a permuted list of cities; energy is the closed tour length."""

    def __init__(self, state, distance_matrix):
        self.distance_matrix = distance_matrix
        super().__init__(state)

    def move(self):
        # Swap two cities; return the resulting change in tour length.
        initial_energy = self.energy()
        a = random.randint(0, len(self.state) - 1)
        b = random.randint(0, len(self.state) - 1)
        self.state[a], self.state[b] = self.state[b], self.state[a]
        return self.energy() - initial_energy

    def energy(self):
        e = 0.0
        for i in range(len(self.state)):
            e += self.distance_matrix[self.state[i - 1]][self.state[i]]
        return e


if __name__ == "__main__":
    cities = {
        "New York City": (40.72, 74.00),
        "Los Angeles": (34.05, 118.25),
        "Chicago": (41.88, 87.63),
        "Houston": (29.77, 95.38),
        "Phoenix": (33.45, 112.07),
    }
    initial_state = list(cities)
    random.shuffle(initial_state)

    distance_matrix = defaultdict(dict)
    for ka, va in cities.items():
        for kb, vb in cities.items():
            distance_matrix[ka][kb] = 0.0 if kb == ka else distance(va, vb)

    tsp = TravellingSalesmanProblem(initial_state, distance_matrix)
    tsp.set_schedule(tsp.auto(minutes=0.02, steps=200))
    tsp.copy_strategy = "slice"
    best_state, best_energy = tsp.anneal()
    print("tour length:", best_energy)
```

The same `Annealer` solves any problem for which one can write a configuration, a random `move`, and
an `energy` — partitioning (flip a gate between chips), placement (swap two components), routing
(re-route a wire), scheduling — by subclassing and filling in those two methods.
