The setting is the bleakest one in optimization: I have a black box that takes a point $x$ in a continuous, possibly high-dimensional space and returns a single scalar $f(x)$, and that is all. There is no derivative — $f$ might be a noisy simulation, the training error of a neural network as a function of its weights, or some hand-built objective with no closed form — and $f$ is allowed to be wildly multimodal, a landscape of many basins and local optima with the global optimum hiding among them. Everything that follows a gradient is therefore disqualified, and even a finite-differenced gradient would do nothing but slide me into whichever pit I happen to be standing next to. The deeper trouble is that any single trajectory — gradient descent, hill-climbing, a Nelder-Mead simplex from one start — commits to a basin the moment it begins moving, and on a multimodal surface that basin is almost never the one I want. The obvious patch is to launch many searchers from many random starts, and at least one of them probably begins in the global basin; but random restarts treat each searcher as an island, so a searcher that stumbles into a wonderful region cannot tell the others, and on a landscape where good regions are sparse and scattered the population spends almost all its evaluations re-discovering the same mediocre pits. Genetic algorithms are the reigning population-based answer and they genuinely handle multimodal black boxes, but they sit wrong for a continuous problem: crossover recombines chunks of a discrete encoding rather than moving a point through space toward a better region, and a GA individual has no persistence — it is a string that gets spliced and discarded, with no notion of *this* candidate, with a trajectory, remembering "I personally did well over there." What I actually want is a population of candidates that *fly* through the continuous space and *share* what they find, so that one member's discovery becomes the whole group's information. That sharing is precisely what random restarts throw away, and Wilson's observation that an individual in a fish school "can profit from the discoveries and previous experience of all other members of the school" exactly "whenever the resource is unpredictably distributed in patches" describes my multimodal landscape — patchy, sparse "resource" being low $f$ — and tells me cooperation should win.

The method I propose is Particle Swarm Optimization. I keep a population of candidate particles, each carrying a position $x$ and a velocity $v$, and I steer each one with two attractors built entirely out of past evaluations rather than out of any pre-known target. Reynolds' boids showed that very simple per-agent rules, each agent looking only at local information, can make a coordinated flock *emerge*; Heppner's roost variant showed that a flock plus a position-quality function actually converges onto a target — but his birds *know where the roost is*, and a real search does not. So the attractors have to be assembled from what the swarm itself discovers. Each particle remembers the best position it has personally visited, its autobiographical $\text{pbest}$ — the spot that has worked out best for me — and the swarm publishes the single best position any particle has found, the social $\text{gbest}$. Both are mere bookkeeping over evaluations I am making anyway, and neither requires knowing the true optimum. The defining update, for each particle and each dimension, is

$$v \leftarrow w\,v + c_1\,r_1\,(\text{pbest} - x) + c_2\,r_2\,(\text{gbest} - x), \qquad x \leftarrow x + v,$$

with $r_1, r_2 \sim U(0,1)$ drawn independently per particle, per dimension, per step. The first pull is nostalgia — the particle returning toward its own best spot; the second is conformity — the particle adjusting toward the group's publicized best. Because both attractors *drift* every time some particle finds a better point, the swarm is not falling into a fixed cornfield but onto a running, improving consensus about where the good region is.

Several of these design choices are load-bearing, and each beats the obvious alternative for a concrete reason. The pulls use the actual signed difference $(\text{pbest}-x)$ rather than a sign test, because a sign test discards the magnitude of how far off I am and gives the same kick whether I am one unit or a thousand units from my best, whereas a distance-proportional pull rightly accelerates a far particle harder. The two pulls are kept *separate*, with independent randoms $r_1$ and $r_2$, rather than folded into one pull toward the midpoint of pbest and gbest: collapsing them makes the swarm contract onto that midpoint and sit there even when the midpoint is a meaningless valley between two basins, while two independent stochastic kicks keep the swarm probing instead of averaging. The random weights have mean $1$ — I use $c_i \cdot \text{rand()}$ with $c_i = 2$ and $\text{rand()} \sim U(0,1)$, so the weight is uniform on $[0,2]$ — so that the expected pull just reaches the attractor while giving a coin-flip chance of *overshooting* it, and overshooting is good, because a particle that flies past the current best keeps probing the far side rather than braking exactly onto it. The constants $c_1$ and $c_2$ are equal by default because their *ratio* is the explore/exploit balance — $c_1 \gg c_2$ leaves isolated particles wandering their own histories without ever cohering, while $c_2 \gg c_1$ yanks the whole swarm onto the first decent point anyone finds, a premature collapse — and with no principled way to know the right ratio for an arbitrary black box, equal is the neutral setting. Stripping the inherited flock scaffolding confirms what is essential: dropping the "craziness" perturbation and the nearest-neighbor velocity matching changes nothing (the two stochastic pulls already supply all the needed randomness), and with neighbor-matching gone it is no longer a flock holding formation but a *swarm* of independent flying points — hence particles.

The one piece I must not drop is the carried velocity $w\,v$. Setting the velocity fresh each step from the pulls alone makes the optimizer *much* worse, and the reason is exactly the overshoot mechanism: without inertia a particle recomputes a pull from its current position and steps precisely that far, braking and reorienting every single step, hugging the line between pbest and gbest, never building up the speed to blow past the current consensus and survey the unknown territory beyond it before the pulls reel it back. Momentum *is* the exploration mechanism. But unchecked momentum detonates the swarm. Reading the noise-free one-dimensional recurrence $v \leftarrow v + a(p-x)$, $x \leftarrow x+v$ as a discrete oscillator — a mass on a spring — shows that if the pull is too strong or the carried velocity already large, the position overshoots by more than it was off, the next pull grows, the next overshoot grows, and the velocity blows up to infinity. The original form keeps $100\%$ of the previous velocity, so energy only ever accumulates; the blunt fix is to clamp $|v| \le V_{\max}$, but that adds a parameter whose right value depends on the unknown scale of the search space. The principled fix is to keep only a fraction $w$ of the previous velocity. With $w < 1$ the oscillator is *damped* and energy bleeds off instead of compounding, so the runaway is tamed at its source and $V_{\max}$ becomes optional; and $w$ is simultaneously the cleanest explore/exploit dial — large $w$ near $1$ means the particle hangs onto its speed and roams, small $w$ means the velocity dies each step and the particle is governed by the local pulls onto the consensus and refines. That double role tells me how to use it: start $w$ high to survey the landscape and decay it linearly, roughly $0.9 \to 0.4$ across the run, to settle into the best basin found, annealing exploration into exploitation through one parameter. This is no mere hope that $w<1$ suffices. Writing the damped recurrence as a linear map on $(x-a, v)$ with total pull $\varphi = c_1 + c_2$, boundedness requires the map's eigenvalues to lie within the unit circle, which constrains $w$ and $\varphi$ *together* — you cannot make both the inertia and the total pull large. Equivalently, the constriction factor $\chi = 2/\lvert 2 - \varphi - \sqrt{\varphi^2 - 4\varphi}\rvert$ for $\varphi > 4$ (e.g. $\varphi = 4.1 \Rightarrow \chi \approx 0.7298$) multiplies every term so the eigenvalues sit on the unit circle by construction, and applying $\chi$ everywhere is algebraically identical to running inertia with $w = \chi$ and effective constants $\chi\,c_i$. So inertia-with-$w<1$ and the constriction factor are the same stabilization seen from two angles, and with $w \approx 0.7\text{–}0.9$ decaying and $c_1 = c_2 \approx 2$ I am safely inside the stable region. One topology choice remains: taking $\text{gbest}$ as the single global best is a fully connected star where news reaches everyone in one step — fast, but prone to the dogpile; replacing it with the best in each particle's ring neighborhood spreads good news slowly, hop by hop, letting subpopulations keep exploring different basins on deceptive surfaces at the cost of slower convergence. I build the global-best star as the default and keep the best-of-neighborhood computation factored out so the ring is a drop-in.

```python
import numpy as np

class Swarm:
    def __init__(self, n_particles, dimensions, bounds, options):
        lo, hi = bounds
        self.position   = np.random.uniform(lo, hi, (n_particles, dimensions))
        self.velocity   = np.random.uniform(0, 1, (n_particles, dimensions))
        self.pbest_pos  = self.position.copy()          # each particle's own best position
        self.pbest_cost = np.full(n_particles, np.inf)  # ...and its value
        self.best_pos   = None                          # swarm's best position (gbest)
        self.best_cost  = np.inf
        self.options    = options                       # {'w':..., 'c1':..., 'c2':...}

def compute_pbest(swarm, current_cost):
    improved = current_cost < swarm.pbest_cost
    swarm.pbest_pos  = np.where(improved[:, None], swarm.position, swarm.pbest_pos)
    swarm.pbest_cost = np.where(improved,          current_cost,   swarm.pbest_cost)

def compute_gbest(swarm):
    i = int(np.argmin(swarm.pbest_cost))
    if swarm.pbest_cost[i] < swarm.best_cost:
        swarm.best_cost = swarm.pbest_cost[i]
        swarm.best_pos  = swarm.pbest_pos[i].copy()

def compute_velocity(swarm, vmax=None):
    w, c1, c2 = swarm.options['w'], swarm.options['c1'], swarm.options['c2']
    shape = swarm.position.shape
    cognitive = c1 * np.random.uniform(0, 1, shape) * (swarm.pbest_pos - swarm.position)
    social    = c2 * np.random.uniform(0, 1, shape) * (swarm.best_pos  - swarm.position)
    v = w * swarm.velocity + cognitive + social         # inertia + cognitive + social
    if vmax is not None:
        v = np.clip(v, -vmax, vmax)
    swarm.velocity = v

def compute_position(swarm, bounds=None):
    swarm.position = swarm.position + swarm.velocity     # x <- x + v
    if bounds is not None:
        lo, hi = bounds
        swarm.position = np.clip(swarm.position, lo, hi)

def optimize(f, n_particles, dimensions, bounds, options, iters):
    swarm = Swarm(n_particles, dimensions, bounds, options)
    w0, w1 = 0.9, 0.4                                    # linear inertia decay (explore -> exploit)
    for t in range(iters):
        swarm.options['w'] = w0 - (w0 - w1) * t / max(1, iters - 1)
        cost = f(swarm.position)                         # one batched black-box evaluation
        compute_pbest(swarm, cost)
        compute_gbest(swarm)
        compute_velocity(swarm)
        compute_position(swarm, bounds)
    return swarm.best_cost, swarm.best_pos

# Example: minimize the sphere function in 10-D
if __name__ == "__main__":
    sphere = lambda X: np.sum(X ** 2, axis=1)
    options = {'w': 0.9, 'c1': 2.0, 'c2': 2.0}
    bounds = (-5.12 * np.ones(10), 5.12 * np.ones(10))
    best_cost, best_pos = optimize(sphere, n_particles=30, dimensions=10,
                                   bounds=bounds, options=options, iters=1000)
```
