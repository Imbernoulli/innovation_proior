We are handed a combinatorial optimization problem: a finite but astronomically large set of configurations and a cost function $f$ scoring each one, and we want the configuration of minimum cost. Traveling salesman is the clean case — $N$ cities, a tour is a permutation, $f$ is the total length, and there are $(N-1)!/2$ tours, hopeless to enumerate past a handful of cities. The decision problem is NP-complete and the optimization problem is NP-hard, so exact methods cost exponential effort and die at modest sizes; for anything real we need a heuristic whose effort grows like a small power of $N$. The standard heuristic is iterative improvement: start somewhere, define a local move (swap two cities, reverse a segment), and repeatedly apply moves, accepting one only if it lowers $f$, until none does. It is a downhill walk, and that is exactly its undoing. A downhill walk stops the instant it reaches a configuration with no downhill neighbor — a local minimum — and for a rugged cost like a tour length the landscape is riddled with local minima, generically far above the global one. The method has no mechanism to climb out, because climbing out means going *up* in cost first, and that is precisely what it forbids. The usual patch — restart from many random configurations and keep the best — buys more rolls of the dice but never escapes the basins themselves; every run freezes in the first basin it falls into. Greedy nearest-neighbor construction is no better: it is myopic, with early choices forcing expensive later ones, and gives tours systematically above optimal. Divide-and-conquer is only as good as the partition, and breaks at the seams when the parts interact. The defect is shared and precise: the *strictness* of "accept only if it lowers $f$" is what makes the walk monotone, and monotone means it can only descend into wherever it started.

So escaping a local minimum requires, at some point, accepting a move that makes $f$ worse — but in a *disciplined* way, enough uphill motion to clear shallow ridges, not so much that the walk becomes an aimless random walk. I propose Simulated Annealing, which finds that discipline by reading the cost as a physical energy and borrowing the way matter settles into its lowest-energy state. A system with many degrees of freedom at temperature $T$ in thermal equilibrium visits a configuration of energy $E$ with the Boltzmann probability $\propto e^{-E/T}$ (units chosen so $k_B=1$). At high $T$ this is nearly flat — every configuration roughly equally likely, the system roams; at low $T$ it is brutally steep, suppressing configurations even slightly above the minimum, so the mass piles onto the lowest-energy states; and as $T\to 0$ it collapses entirely onto the ground state(s), which are exponentially rare yet dominate purely by exponential weighting. Identify $f$ with $E$: "find the minimum-cost configuration" becomes "find the ground state," and the temperature is exactly the dial I wanted — high $T$ roams and escapes traps, low $T$ settles onto minima. The whole problem reduces to sampling configurations from $e^{-f/T}$ and sliding $T$ to zero.

We cannot sample that distribution directly — its normalizer is the partition function $Z=\sum e^{-f/T}$, the same astronomical sum we cannot enumerate, and uniform-sample-then-reweight is hopeless because at any interesting low $T$ almost every random configuration has weight essentially zero. But we do not need independent samples; we need a *process* whose long-run visiting frequencies are Boltzmann. That is the Metropolis rule, and the load-bearing step is to derive why it converges to exactly the right distribution. From the current configuration propose a local move to a neighbor with a *symmetric* proposal, so the probability of proposing $r\to s$ equals that of proposing $s\to r$ (a random swap or displacement has this built in). To decide acceptance, demand that the stationary distribution be $\nu_r \propto e^{-E_r/T}$. Take two neighbors $r,s$ with $E_r > E_s$, and let $\nu_r$ be the ensemble fraction in $r$. Downhill moves I always accept — that is just ordinary improvement, the part I never wanted to give up — so all $\nu_r P_{rs}$ of the $r\to s$ proposals succeed. For the uphill direction $s\to r$, accept with some probability $A$ to be chosen, giving $\nu_s P_{rs} A$ successful moves. The net flow from $s$ to $r$ is $P_{rs}(\nu_s A - \nu_r)$, which vanishes — making $\nu$ stationary — only when $\nu_s A = \nu_r$, i.e.

$$A = \frac{\nu_r}{\nu_s} = \frac{e^{-E_r/T}}{e^{-E_s/T}} = e^{-(E_r-E_s)/T}.$$

Writing $\Delta E = E_r - E_s > 0$ for the uphill cost, the rule is: accept an uphill move of size $\Delta E$ with probability $e^{-\Delta E/T}$. The downhill case folds in, since for $\Delta E < 0$ we have $e^{-\Delta E/T} > 1$, so one formula covers both — accept with probability $\min(1, e^{-\Delta E/T})$. Operationally: compute $\Delta E$; if $\Delta E \le 0$ accept; else draw $u$ uniform on $(0,1)$ and accept iff $u < e^{-\Delta E/T}$, otherwise stay put. Once downhill moves are always accepted, detailed balance with a symmetric proposal *forces* the uphill probability $e^{-\Delta E/T}$ — any other curve would converge to a distribution I could not control. Two cautions make it trustworthy: the move set must connect the whole space (ergodicity), or the chain samples a Boltzmann distribution restricted to a sub-region; and a *rejected* move must re-count the current configuration as a real step, not skip it, or the ensemble loses exactly the systems that tried and failed to leave $s$, breaking the balance just engineered. Now read the temperature dependence: at high $T$, $e^{-\Delta E/T}\approx 1$ and the walk roams blind to cost; at $T\to 0$, $e^{-\Delta E/T}\to 0$ for every uphill move, so only downhill moves survive and the rule is *exactly* the greedy hill-climbing I started with. The temperature continuously interpolates exploration and greedy exploitation, accepting moves with $\Delta E \lesssim T$ easily and $\Delta E \gg T$ almost never — the escape-shallow-traps discipline, now derived rather than invented.

That leaves the schedule. A fixed positive $T$ never commits — it equilibrates to a Boltzmann distribution with nonzero spread and keeps jiggling near, but never onto, the minimum; an instant drop to $T=0$ descends greedily from a random hot configuration into the nearest local minimum, throwing away all the high-temperature exploration. This is precisely the metallurgical failure mode: cooling a metal suddenly — *quenching* — traps it in a defected, glassy, high-energy state, whereas approaching a clean low-energy crystal requires *annealing*, melting it and lowering the temperature slowly, spending a long time near freezing so the atoms keep rearranging toward equilibrium as $T$ drops. The mapping is exact, so the recipe writes itself: start molten at a high $T_{\max}$ where nearly all moves are accepted, then lower $T$ slowly toward a frozen $T_{\min}$, spending enough Metropolis steps at each temperature for the walk to re-equilibrate before cooling further; large-scale structure settles at high $T$, fine detail at low $T$. The cooling law is geometric, $T_{k+1} = \alpha T_k$ with $\alpha$ a little below one (around $0.9$), equivalently linear in $\log T$:

$$T(s) = T_{\max}\left(\frac{T_{\min}}{T_{\max}}\right)^{s/S} = T_{\max}\exp\!\left(-\ln\frac{T_{\max}}{T_{\min}}\cdot\frac{s}{S}\right),$$

so step $0$ gives $T_{\max}$ and step $S$ gives $T_{\min}$. Geometric beats linear because it is scale-free — it needs only a ratio, not the absolute energy scale — and makes each stage a comparable fractional change, automatically slowing in absolute $T$-units as $T$ shrinks; a logarithmic schedule $T_k \propto c/\log k$ carries an asymptotic global-convergence guarantee but is impractically slow, so geometric is the deliberate trade of that guarantee for a method that terminates. The physics also hands a diagnostic: with $k_B=1$, the specific heat $C(T) = d\langle E\rangle/dT = \mathrm{Var}(E)/T^2$, so the cost variance at fixed $T$ flags where $C(T)$ spikes — the phase-transition-like temperatures where the solution's large-scale structure is being decided and cooling should be slowest. $T_{\max}$ is calibrated so the initial acceptance rate is near one, $T_{\min}$ so the final improvement rate is near zero. Two practical points finish it. The configuration in hand at $T_{\min}$ is a sample from the near-frozen distribution, not necessarily the best ever visited, so we track the best configuration seen over the entire run separately and return that — cheap insurance. And the inner loop needs only $\Delta E$, never the absolute cost, so for a local move (a swap or segment reversal touches only a few edges) we compute the incremental $\Delta E$ from the changed part rather than re-evaluating $f$ wholesale. Four ingredients are all any problem must supply — a configuration representation, a random local move, a cost (energy), and a cooling schedule — and everything else is the same loop, which is what makes the method generic across partitioning, placement, routing, and scheduling.

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
