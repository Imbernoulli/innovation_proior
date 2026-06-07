# Particle Swarm Optimization (PSO)

**Domain:** Global Optimization (population-based, derivative-free black-box optimization).

## Problem

Minimize a scalar objective f(x) over a continuous, possibly high-dimensional space, given only the ability to evaluate f — no gradient, and f may be noisy, non-convex, and highly multimodal (many local optima). A single descent trajectory commits to the nearest basin; the goal is a search that explores broadly enough to find the right basin and exploits enough to refine it, using only primitive arithmetic and a few parameters.

## Key idea

Keep a **population of candidate "particles"** flying through the search space. Each particle has a position x and a velocity v. It is steered by two attractors built entirely from past evaluations: its **own best-seen position** `pbest` (cognitive / individual memory) and the **swarm's best-seen position** `gbest` (social / collective memory). Each step, the velocity keeps a fraction of itself (inertia) and is pulled toward both attractors with independent random weights; the position then moves by the velocity. The attractors drift as the swarm discovers better points, so the swarm converges on its own improving consensus rather than on a pre-known target.

## Algorithm

For each particle i and dimension j, with random scalars r1, r2 ~ U(0,1) drawn independently per particle, dimension, and step:

```
v_ij ← w · v_ij + c1 · r1 · (pbest_ij − x_ij) + c2 · r2 · (gbest_j − x_ij)
x_ij ← x_ij + v_ij
```

- `w` — **inertia weight**: fraction of the previous velocity carried forward. Large w explores (particles sail and overshoot), small w exploits (velocity is killed, particle brakes onto the consensus). Decaying w linearly from ≈0.9 to ≈0.4 across the run anneals exploration into exploitation. With w < 1 the velocity recurrence is damped, which prevents the velocity explosion that 100%-retention (the original momentum-coefficient-1 form) suffers.
- `c1`, `c2` — **cognitive** and **social** acceleration constants (commonly c1 = c2 = 2). Their ratio is the explore/exploit balance: c1 ≫ c2 → isolated wandering; c2 ≫ c1 → premature convergence onto an early good point. Equal is the neutral default.
- `r1`, `r2` — independent uniform weights. Two *separate* stochastic pulls are required: collapsing them into a single pull toward the midpoint of pbest and gbest makes the swarm converge onto that midpoint even when it is not an optimum.
- Each procedure step: evaluate all particles, update each `pbest` where the new cost improves on its record, recompute `gbest` as the best `pbest`, then update velocities and positions.

**Stability.** Treating the noise-free 1-D update as a discrete oscillator, boundedness requires keeping w below 1 and the total pull φ = c1 + c2 bounded together (the linear map's eigenvalues must stay within the unit circle). The constriction-factor form multiplies every term by χ = 2 / |2 − φ − √(φ² − 4φ)| for φ > 4 (e.g. φ = 4.1 → χ ≈ 0.7298), which is algebraically inertia with w = χ and effective constants χ·c_i; it guarantees convergence and makes the velocity clamp Vmax optional. The earlier, pre-inertia form instead clamped |v| ≤ Vmax to suppress the explosion.

**Topology.** `gbest` here is the global best over the whole swarm — a *star* network, fast information spread, faster convergence but more prone to premature collapse. Replacing it with the best in each particle's local neighborhood — a *ring* (local-best PSO) — spreads good news slowly and keeps subpopulations exploring different basins, trading convergence speed for robustness on deceptive landscapes.

## Code

A faithful, vectorized global-best (gbest) PSO over a whole swarm at once with numpy:

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

The structure mirrors the canonical numpy PSO (a `Swarm` state object; `compute_pbest` / `compute_gbest` / `compute_velocity` / `compute_position` operators; a star topology for the global best; an `optimize` loop), differing only in keeping the example self-contained.
