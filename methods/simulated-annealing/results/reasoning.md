Let me start from the thing that actually defeats me in practice, which is that the obvious way to minimize a complicated cost function over a huge discrete space — just keep making it better until you can't — reliably lands me somewhere mediocre and there is nothing I can do about it from inside the method.

Make it concrete. I have a finite but astronomically large set of configurations and a cost `f` that scores each one; I want the configuration of minimum cost. Traveling salesman is the clean case: `N` cities, a tour is a permutation, `f` is the total length, and the number of tours is `(N-1)!/2`, which is hopeless to enumerate past a handful of cities. The decision problem is NP-complete and the optimization problem is NP-hard, so exact methods cost exponential effort and die at modest sizes. For anything real I need a heuristic whose effort grows like a small power of `N`. Fine. The standard heuristic is iterative improvement: start somewhere, define a local move (swap two cities, reverse a segment), and repeatedly apply moves, accepting one only if it lowers `f`, until no move improves. It's a downhill walk.

And here's the wall. A downhill walk stops the instant it reaches a configuration with no downhill neighbor — a local minimum. For a rugged cost like a tour length, the landscape is riddled with local minima, and generically the one I stop at sits far above the global minimum. The method has no mechanism to climb out, because climbing out means, by definition, going *up* in cost first, and I've forbidden exactly that. The usual patch is to restart from many random configurations and keep the best of the runs. That helps a little — more rolls of the dice — but it doesn't change the fundamental fact: every individual run is a downhill walk that freezes in the first basin it falls into. The basins themselves are never escaped; I just hope one of the random starts happens to sit in a good basin. That's not a solution, that's resignation.

So let me name the defect precisely, because the precision is the clue. The defect is the *strictness* of "accept only if it lowers `f`." That strictness is what makes the walk monotone, and monotone means it can only ever descend into whatever basin it started in. If I want to escape a local minimum I have, at some point, to accept a move that makes `f` *worse* — to walk uphill, temporarily, to get over the ridge into a deeper basin. But I can't just accept arbitrary uphill moves, or I'm doing a random walk and never converge on anything good. I need a *disciplined* way to accept worse moves: enough uphill motion to escape shallow traps, not so much that I wander forever. What's the right discipline?

Let me think about what "escape shallow traps but settle into deep ones" even means, because that phrasing is doing a lot of work. A trap is shallow if the ridge around it is low — a small uphill step gets me out. A trap is deep if escaping it requires a large uphill excursion. So the discipline I want should make small uphill moves *easy* to accept and large uphill moves *hard*. And I want a single dial that tunes how willing I am to go uphill at all: turned all the way down, I recover plain downhill-only descent; turned up, I roam freely. That dial would let me start roaming (to find the good region) and end descending (to settle). I don't yet know the functional form of "accept a worse move of size `Δ` with what probability," but I know the shape I want: a decreasing function of `Δ`, controlled by one parameter.

I could just invent a curve — say, accept an uphill move of size `Δ` with probability `1 - Δ/(some scale)`, clipped — but that's arbitrary, and arbitrary acceptance rules will have arbitrary stationary behavior that I can't reason about. I want a *principled* rule, one where I can actually say what distribution over configurations the walk converges to. Where do I find a process that wanders a huge configuration space, sometimes uphill, sometimes downhill, with a single temperature-like dial, and whose limiting distribution is known exactly?

Physics. A physical system with many degrees of freedom at temperature `T` does precisely this. Its atoms jostle, sometimes into higher-energy arrangements, sometimes lower, and in thermal equilibrium it visits a configuration `{r_i}` of energy `E({r_i})` with the Boltzmann probability `∝ e^{-E/(k_B T)}`. That's the canonical ensemble. And look at what `T` does to that distribution. At high `T`, `e^{-E/(k_B T)}` is nearly flat in `E` — every configuration is roughly equally likely, the system roams. At low `T`, the exponential is brutally steep: configurations even slightly above the minimum energy are exponentially suppressed, so the distribution piles up on the lowest-energy states. In the limit `T → 0` the Boltzmann distribution collapses entirely onto the ground state(s). The ground states are exponentially rare among all configurations, yet at low temperature they *dominate*, purely because of the exponential weighting.

That is exactly the behavior I want, transcribed. Identify my cost `f` with the energy `E`, my configurations with the system's states, and introduce a control parameter `T` — call it a temperature, set Boltzmann's constant to one since it's just a choice of units for `T`. Then "find the minimum-cost configuration" becomes "find the ground state," and the Boltzmann distribution at low `T` concentrates on exactly the configurations I'm after. The temperature *is* my dial: high `T` roams (escapes traps), low `T` settles onto minima (descends). If I could sample configurations from `e^{-f/T}` and slide `T` down to zero, I'd get the minimum. The whole problem reduces to: how do I sample from the Boltzmann distribution over this configuration space?

I can't write down the distribution explicitly — the normalizer is the partition function `Z = Σ e^{-f/T}` summed over all configurations, the same astronomical sum I can't enumerate. I can't sample uniformly and reweight either: if I pick configurations at random and weight each by `e^{-f/T}`, then at any interesting (low) `T` almost every random configuration has huge `f`, hence weight essentially zero, and I throw away all my samples on worthless points. So direct sampling is out for the same reason brute-force optimization is out — the good configurations are a vanishing fraction of the space.

But I don't actually need independent samples from the Boltzmann distribution. I need a *process* whose long-run visiting frequencies are Boltzmann. A random walk that hops between neighboring configurations and, over time, spends time in each configuration in proportion to `e^{-f/T}`. This is exactly the problem the Metropolis algorithm solved thirty years ago for computing equilibrium averages of fluids, and the trick there was importance sampling by construction: rather than sample then weight, build a Markov chain that *visits* configurations with the Boltzmann frequency, so you can weight them evenly.

Let me reconstruct their rule from scratch, because I need to know not just what it is but *why it's the right one*, so I can trust what `T → 0` will do. I have a current configuration. I propose a local move to a neighbor, symmetrically — meaning the probability of proposing `r → s` equals the probability of proposing `s → r`. (A random small displacement, or a random swap, has this symmetry built in: I'm equally likely to propose a move and its reverse.) Now I have to decide whether to accept. I want the chain's stationary distribution `ν` to be `ν_r ∝ e^{-E_r/T}`. Let me figure out what acceptance rule forces that.

Consider a large ensemble of independent copies of the walk; let `ν_r` be the fraction in configuration `r`. Pick two configurations `r` and `s` that are neighbors, and suppose `E_r > E_s` — `s` is the lower-energy one. With a symmetric proposal, in one step the number proposing `r → s` is `ν_r P_{rs}` and the number proposing `s → r` is `ν_s P_{sr} = ν_s P_{rs}`. Now apply whatever acceptance rule I'm designing. The move `r → s` goes downhill (`E_s < E_r`); the move `s → r` goes uphill. I'll insist downhill moves are always accepted — that's the part I never wanted to give up, it's just ordinary improvement. So all `ν_r P_{rs}` of the `r → s` proposals succeed. For the uphill direction `s → r`, suppose I accept with some probability `A` that I get to choose. Then the number actually moving `s → r` is `ν_s P_{rs} A`. The net flow from `s` to `r` is

`P_{rs} (ν_s A - ν_r)`.

For the ensemble to sit still — for `ν` to be stationary, with no net drift between `r` and `s` — I need this net flow to vanish, `ν_s A = ν_r`, i.e. `A = ν_r / ν_s`. And I *want* `ν_r ∝ e^{-E_r/T}`, so the acceptance probability for the uphill move `s → r` must be

`A = ν_r/ν_s = e^{-E_r/T} / e^{-E_s/T} = e^{-(E_r - E_s)/T}`.

Write `ΔE = E_r - E_s > 0` for the cost increase of the uphill move. The rule is: **accept an uphill move of size `ΔE` with probability `e^{-ΔE/T}`.** And the downhill case I already fixed as "always accept," which I can fold into the same expression: a downhill move has `ΔE < 0`, so `e^{-ΔE/T} > 1`, and "accept with probability `min(1, e^{-ΔE/T})`" gives probability 1. One formula covers both: **accept with probability `min(1, e^{-ΔE/T})`.** Operationally: compute `ΔE`; if `ΔE ≤ 0`, accept; else draw a uniform random number `u ∈ (0,1)` and accept iff `u < e^{-ΔE/T}`, otherwise stay put and keep the current configuration.

Once I insist that downhill moves are always accepted, the detailed-balance argument pins down the uphill probability for a symmetric proposal. If I picked some other uphill curve while keeping that downhill convention, the pairwise flow would not vanish at the Boltzmann ratios and the walk would converge to some other distribution I couldn't control. Two more things make me trust it. First, I have to make sure the chain can actually reach every configuration — ergodicity — or it might sample a Boltzmann distribution restricted to a sub-region; as long as my move set connects the whole space (any tour reachable from any other by enough swaps/reversals), that holds, and combined with the zero-net-flow condition it pins the limit to `ν_r ∝ e^{-E_r/T}`. Second, a subtlety from the equilibrium bookkeeping: when an uphill move is *rejected*, I must count the current configuration *again* as the next state of the walk — not skip it. If I quietly dropped rejected steps I'd be removing from the ensemble exactly the systems that tried to leave the low-energy state `s` and failed, which would deplete `s` relative to `r` and break the balance I just engineered. So a rejected move means "stay, and that staying is a real step."

Now look at what this acceptance rule does as a function of `T`, because the temperature is the dial I came for. At high `T`, `e^{-ΔE/T} ≈ 1` for any reasonable `ΔE` — almost every proposed move, uphill or down, is accepted; the walk roams freely over the configuration space, essentially a random walk, blind to cost. At `T → 0`, `e^{-ΔE/T} → 0` for every uphill move (`ΔE > 0`), so *no* uphill move is ever accepted; only downhill moves survive, and the walk is exactly plain iterative improvement — a pure downhill descent that freezes in the first local minimum. So the strict greedy heuristic I started with is just this rule at `T = 0`. The temperature continuously interpolates between blind exploration (`T` large) and greedy exploitation (`T = 0`), and in between, the rule accepts *small* uphill moves much more readily than *large* ones — exactly the "escape shallow traps, stay out of deep climbs" discipline I sketched abstractly, now derived rather than invented. The size of uphill excursion the walk will tolerate is set by `T`: roughly, moves with `ΔE ≲ T` are easy, `ΔE ≫ T` are nearly forbidden.

So I have my engine: run the Metropolis walk at temperature `T`. The question is what `T` to use. And here's the temptation — just set `T = 0` (or tiny) and let it descend. But that's literally the greedy heuristic again; it freezes immediately in a local minimum. The whole point was to avoid that. So `T` must be positive, at least for a while. But if I hold `T` fixed at some positive value forever, the walk equilibrates to the Boltzmann distribution at *that* `T`, which still has nonzero spread — it keeps accepting uphill moves, keeps jiggling around, and never actually commits to the minimum. At fixed positive `T` I get samples concentrated *near* low cost but never the minimum itself; at `T = 0` I get a local minimum, not the global one. Neither fixed temperature works. The temperature has to *change*.

What changes it, and in which direction? I want to start high — high enough that the walk roams the whole landscape and isn't trapped by where I happened to initialize it, so it finds the broad region where good configurations live. Then I want to end low — low enough that the walk descends into and commits to a minimum. So: start `T` high, lower it toward zero over the run. The intuition is that at high `T` the Boltzmann distribution is broad and the walk samples the large-scale structure of the landscape, discovering which deep valley to be in; as `T` falls the distribution sharpens and the walk concentrates into that valley; near `T = 0` it settles to the bottom. In the ideal quasistatic limit, if I lower `T` while keeping the walk equilibrated at each temperature, the sampled distribution tracks Boltzmann toward its zero-temperature mass on the ground states. In a finite computation I cannot promise that ideal, but it gives me the design principle: lower temperature gradually enough that the walk has time to adjust.

But how gradual? Why not just slam `T` from high to zero in one step? Let me think about what goes wrong. If I drop `T` instantly to zero, the walk is suddenly only allowed to go downhill, starting from wherever it happened to be when I dropped it — which is some random high-temperature configuration, not a carefully chosen one. From there it descends greedily into the nearest local minimum. That's no better than greedy from a random start; I've thrown away the whole benefit of the high-temperature exploration. More generally, if I cool *faster* than the walk can re-equilibrate at each new temperature, the walk lags behind its own Boltzmann distribution — it gets stranded in whatever region it occupied at the higher temperature, and as `T` keeps dropping that region freezes around it before the walk has a chance to find the lower-cost region it should have moved to. The system locks into a poor, metastable configuration.

This is not a vague worry; it is the physical failure mode of cooling too fast. Think about how you actually get a metal or a crystal into a low-energy state. You don't cool it suddenly; sudden cooling, *quenching*, traps the material in a defected, glassy, high-energy state full of frozen-in disorder. To approach the true low-energy ground state — a clean single crystal — you *anneal*: melt it, then lower the temperature *slowly*, spending a long time near the freezing point, letting the atoms continually rearrange toward equilibrium as the temperature drops. Low temperature alone is not enough; the crystal grower's whole craft is the *schedule* of cooling. The mapping is exact: accepting only downhill moves (`T = 0`) or cooling too fast is quenching, and it freezes my optimizer into a metastable local minimum; cooling slowly, holding the walk near equilibrium at each temperature, is annealing, and it gives the optimizer a chance to keep following the low-energy distribution instead of committing to the first basin. So the recipe writes itself: **simulate annealing.** Start at a high temperature where the system is effectively "melted" — essentially all moves accepted — and lower `T` slowly toward zero, spending enough steps at each temperature for the Metropolis walk to re-equilibrate before cooling further.

Let me pin down "high temperature where essentially all moves accepted." At the start I want `T` large enough that `e^{-ΔE/T} ≈ 1` even for the larger uphill moves the problem presents — concretely, a `T` at which the acceptance rate is near one (say, almost every proposed move taken). Too low a starting `T` and I've effectively skipped the exploration phase; I want to begin genuinely molten. And the end: keep cooling until the system is "frozen" — until lowering `T` further produces essentially no accepted uphill moves and no further improvement, the acceptance rate near zero. Between those, I need a cooling *law*.

The simplest law that has the right qualitative shape is geometric: `T_{k+1} = α T_k` with a constant ratio `α` a little below one (something like `0.9`). Why geometric and not, say, linear in the step count? Two reasons. Linear cooling (`T` decreasing by a fixed amount each stage) spends equal absolute time per unit of temperature, so with a wide temperature range it can burn many stages where everything is still molten and then cross the low-temperature relative changes too coarsely. Geometric cooling, equivalently linear in `log T`, automatically slows down in absolute `T`-units as `T` gets small, so each stage is a comparable fractional change. And geometric cooling is *scale-free*: it doesn't require me to know the absolute energy scale of the problem in advance, only a ratio. There is a known result that a *logarithmic* schedule, `T_k ∝ c / log k`, is slow enough to guarantee convergence to the global optimum in the limit — but it is impractically, uselessly slow, cooling so gradually that you'd never finish. So I take the geometric schedule as the practical compromise: fast enough to run, slow enough to anneal rather than quench, and I accept that I'm trading the asymptotic guarantee for a method that actually terminates.

Concretely I'll often parameterize it by the endpoints and the number of steps rather than by `α` directly: pick a starting temperature `T_max` (molten) and a final temperature `T_min` (frozen) and interpolate geometrically between them over the planned number of steps. Geometric interpolation in `T` is linear interpolation in `log T`, so if I run for `S` total steps, the temperature at step `s` is `T(s) = T_max · (T_min/T_max)^{s/S} = T_max · exp( -ln(T_max/T_min) · s/S )`. At `s = 0` that's `T_max`; at `s = S` it's `T_max · (T_min/T_max) = T_min`. Smooth geometric cool from molten to frozen. I still want enough moves *per temperature* that the walk re-equilibrates before `T` drops appreciably — equivalently, `S` large enough that consecutive temperatures are close — because re-equilibration at each `T` is the whole reason slow cooling beats quenching; cooling faster than the walk can equilibrate is just quenching with extra steps.

There's a nice secondary diagnostic the physics hands me for *where* to cool slowly. With `k_B = 1`, the canonical identity is `C(T) = d⟨E⟩/dT = Var(E)/T^2`, so the variance of the cost at a fixed temperature tells me where the specific heat spikes. Those spikes mark temperatures where the system undergoes a phase-transition-like reorganization — where the large-scale structure of the solution is being decided. A peak in `C(T)` tells me which temperature ranges are critical and deserve extra-slow cooling. I won't always need it, but it's there: it says the dangerous, must-not-quench part of the schedule is exactly near the cost-variance peak.

One more practical point. As I cool, the walk wanders, and the configuration sitting in my hands at the final temperature is a *sample* from the near-frozen distribution — it is not guaranteed to be the lowest-cost configuration the walk ever visited. The walk might have passed through something better mid-anneal and then drifted slightly off it. So I should track the best configuration seen *over the entire run* separately, and return that, rather than trusting the endpoint. Cheap insurance.

Let me also make sure I'm computing `ΔE` efficiently, because the inner loop runs millions of times. For a local move I usually don't need to recompute `f` from scratch; the move touches a small part of the configuration, so I can compute the *incremental* `ΔE` directly. For a tour, swapping or reversing a segment only changes the few edges at the cut points, so `ΔE` is the difference of a handful of edge lengths, not the whole tour length. The Metropolis test only ever needs `ΔE`, never the absolute cost, which is exactly why this is cheap — and it's the same reason the algorithm is generic: it asks the problem only for a move and the resulting `ΔE` (or, if incremental bookkeeping is awkward, a fresh `f` it can difference). Four ingredients are all the method needs from any problem: a representation of a configuration, a random move that rearranges it, a cost function (energy), and a cooling schedule of temperatures and steps. Everything else is the same loop.

So the algorithm, end to end. Pick a starting configuration. Set `T = T_max`, hot enough that almost all moves are accepted. Loop: propose a local move, compute `ΔE`; accept it if `ΔE ≤ 0`, or if a uniform random `u < e^{-ΔE/T}`; otherwise restore the previous configuration. Every step lower `T` toward `T_min` geometrically. Keep the best configuration ever seen. Stop when `T` reaches `T_min` or the system is effectively frozen, and return the best. At high `T` it explores and ignores cost; as `T` falls it concentrates into a good valley; near `T_min` it descends to a minimum. If the cooling is slow enough relative to the mixing time, that minimum can be the global one; in ordinary finite runs the honest promise is a much better shot at a low-cost basin than a greedy descent gets from the same start.

Let me write it, mirroring a clean general-purpose implementation: an abstract problem exposing `move()` and `energy()`, and an anneal loop that does the geometric cooling and the Metropolis test.

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
        to signal 'recompute energy from scratch'."""
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
        # Geometric cooling: T(step) interpolates Tmax -> Tmin linearly in log T,
        # i.e. T = Tmax * exp(Tfactor * step/steps) with Tfactor = -ln(Tmax/Tmin),
        # so step 0 -> Tmax and step `steps` -> Tmin. Slow cool = anneal, not quench.
        if self.Tmin <= 0.0:
            raise ValueError("geometric cooling requires Tmin > 0")
        Tfactor = -math.log(self.Tmax / self.Tmin)
        self.start = time.time()

        T = self.Tmax
        E = self.energy()
        prev_state = self.copy_state(self.state)   # to restore on rejection
        prev_energy = E
        self.best_state = self.copy_state(self.state)   # track best ever seen
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
            T = self.Tmax * math.exp(Tfactor * step / self.steps)   # cool

            dE = self.move()
            if dE is None:                      # no incremental delta supplied
                E = self.energy()
                dE = E - prev_energy
            else:
                E += dE                         # cheap: only the touched edges

            trials += 1
            # Metropolis test: reject an uphill move with prob 1 - exp(-dE/T).
            # Equivalently accept iff dE <= 0 or random() <= exp(-dE/T).
            if dE > 0.0 and math.exp(-dE / T) < random.random():
                self.state = self.copy_state(prev_state)   # rejected: restore
                E = prev_energy
            else:                                        # accepted
                accepts += 1
                if dE < 0.0:
                    improves += 1
                prev_state = self.copy_state(self.state)
                prev_energy = E
                if E < self.best_energy:                 # remember the best
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
```

To recap the causal chain: greedy descent over a rugged cost freezes in a local minimum because it forbids any uphill step; to escape I must accept worse moves, but in a disciplined way, which I find by reading my cost as a physical energy — a system at temperature `T` visits configurations with Boltzmann probability `e^{-f/T}`, whose `T → 0` limit puts its mass on the minima; once downhill moves are always accepted, the Metropolis uphill probability `e^{-ΔE/T}` is forced by detailed balance for symmetric proposals, and its temperature dial interpolates between roaming and greedy descent; but a fixed `T` never commits and an instant drop to `T = 0` quenches into a trap, so I cool `T` slowly from molten to frozen — annealing, not quenching — geometrically, track the best configuration seen, calibrate the endpoints from acceptance and improvement rates, and get a low-cost configuration from a method that asks each problem for nothing but a configuration, a move, and a cost.
