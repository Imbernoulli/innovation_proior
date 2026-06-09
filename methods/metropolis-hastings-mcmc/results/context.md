# Context

## Research question

I want to compute high-dimensional expectations against a probability distribution that I can write down only up to an unknown constant of proportionality. Concretely, in classical equilibrium statistical mechanics the average of any observable F over the canonical ensemble is

  F̄ = [ ∫ F(x) e^{−E(x)/kT} dx ] / [ ∫ e^{−E(x)/kT} dx ],

where x ranges over the configuration of N interacting particles — a space of several hundred dimensions once N is in the hundreds — E(x) is the potential energy, k is Boltzmann's constant and T the temperature. The momentum integrals factor out because the forces are velocity-independent, so the difficulty is purely the configuration integral. The Boltzmann weight e^{−E/kT} defines a probability density on configuration space, but its normalizer (the partition function) is itself a hundred-dimensional integral I cannot evaluate.

The pain point is sharp. Standard numerical quadrature is hopeless: a grid with even a handful of points per axis has an astronomical number of nodes in a few-hundred-dimensional space. So the goal is a procedure that produces a *sample* of configurations distributed according to the Boltzmann density — or at least lets me average an observable against it — using only quantities I can actually compute: energy *differences* E(x') − E(x), never the normalizer. A solution has to (i) work in hundreds of dimensions, (ii) never require the normalizing constant, and (iii) concentrate its effort on the configurations that actually carry probability mass rather than on the overwhelming majority that carry essentially none. More generally, the same need arises for any target distribution known only up to a constant.

## Background

**The canonical ensemble and the partition function.** For a system in thermal equilibrium at temperature T, the probability of a configuration x is the Boltzmann distribution p(x) ∝ e^{−E(x)/kT}. The proportionality constant is the partition function Z = ∫ e^{−E(x)/kT} dx, the generator of all equilibrium thermodynamics. Z is exactly the object that is intractable: every quantity of interest is a ratio of integrals against e^{−E/kT}, and although those ratios are finite and physical, computing either integral by brute force is out of reach for an interacting many-body system.

**Two routes to an ensemble average, and why one is forced.** A thermodynamic average can in principle be obtained two ways. One is a *time* average: integrate Newton's equations for the particles and average the observable along the trajectory; ergodicity of the dynamics then equates the time average with the ensemble average. This requires following the detailed kinematics of every particle through time — feasible only if the machine can integrate the coupled equations of motion for long enough. The other is an *ensemble* average computed directly as the integral above. When the available computer cannot time-integrate a many-body system adequately, the ensemble route is the one left open, and the question becomes how to do the high-dimensional integral.

**The Monte Carlo method for integration.** Von Neumann and Ulam established that high-dimensional integrals can be estimated by random sampling: draw points at random and average the integrand. The error falls like the inverse square root of the sample size independent of dimension, which is what makes it attractive where grids fail. Applied naively here, one draws configurations x uniformly at random over the accessible space and forms

  F̄ ≈ [ Σ_i F(x_i) e^{−E(x_i)/kT} ] / [ Σ_i e^{−E(x_i)/kT} ] = Σ_i F(x_i) p_i,  p_i = e^{−E(x_i)/kT} / Σ_j e^{−E(x_j)/kT}.

**The diagnostic failure of uniform sampling — a needle in a haystack.** For anything denser than a dilute gas this is catastrophically inefficient, and the reason is knowable in advance from the shape of the Boltzmann weight. In a close-packed or liquid-density system, a configuration drawn with particle centers placed uniformly at random almost certainly has at least one pair of particles overlapping or nearly so; the potential energy is then enormous and e^{−E/kT} is vanishingly small. The configurations that carry essentially all of the probability mass — the ones with no bad overlaps — occupy a fantastically small fraction of the uniform volume. So almost every uniformly-drawn point contributes a near-zero weight to both sums, and the estimate is dominated by the rare lucky draw. The sampling effort is spent searching a haystack for the few needles that matter.

**Importance sampling and where it strains.** The textbook remedy is importance sampling: to estimate J = ∫ f(x) p(x) dx = E_p(f), draw instead from a more convenient density q and use Ĵ = (1/N) Σ_i f(x_i) p(x_i)/q(x_i), correcting each draw by the weight w(x) = p(x)/q(x). With a well-chosen q this concentrates draws where they matter. But in a large number of dimensions the weights w(x) for a fixed sample size are typically either almost all extremely small or, for a few draws, extremely large; the estimate is then controlled by a handful of samples and its variance is unreliable. Finding a q close enough to the target to keep the weights tame is itself the hard part in high dimension. And in the raw unbiased form this also asks for p's absolute normalizer; a self-normalized ratio can cancel that constant, but it does not remove the high-dimensional weight collapse.

**Markov chains and stationary distributions.** A finite Markov chain on states with transition probabilities p_ij = Pr{X(t+1)=j | X(t)=i} has, when it is irreducible (every state reachable from every other in finitely many steps), a unique stationary distribution π satisfying π = πP, i.e. Σ_i π_i p_ij = π_j for all j. With aperiodicity, the distribution started anywhere converges to π, and time averages of a function along a single realization, Σ_t f(X(t))/N, converge to the stationary expectation E_π(f) as N → ∞ (and are asymptotically normal). In the classical use of such chains in simulation — for instance in following radiation transport through matter — the per-step transition law P(i→j) is given by the physics a priori, and one *computes* the resulting stationary distribution p(i). Here the distribution π = p = Boltzmann is the thing specified in advance, while no transition law is given.

**Acceptance–rejection sampling.** A standard way to draw from a density f(x) (known up to a constant) is acceptance–rejection: find a density h and a constant c with f(x) ≤ c·h(x) everywhere, draw a candidate Z from h and an independent u ∼ Uniform(0,1), and accept Z if u ≤ f(Z)/(c·h(Z)), otherwise discard it and draw a fresh independent candidate. Accepted draws are exact, independent samples from f. The catch is the blanket: one must find an h and a c that dominate f everywhere, which is hard in high dimension, and if c is large the acceptance probability c^{−1} is tiny so almost everything is rejected. The method stalls on the requirement of a global dominating envelope: each rejected candidate is discarded and a fresh independent one drawn, with no use made of where the chain already is.

## Baselines

**Naive (uniform) Monte Carlo integration.** Draw configurations uniformly over the accessible region, weight each by its Boltzmann factor, and form the ratio of weighted sums above. Core math: F̄ ≈ Σ_i F(x_i) e^{−E(x_i)/kT} / Σ_i e^{−E(x_i)/kT}. Correct in the limit, dimension-robust in its error rate, and it never needs Z explicitly because Z cancels between numerator and denominator. Limitation: at liquid/solid density essentially all uniform draws land on overlapping (near-infinite-energy) configurations with negligible weight, so the effective sample size is a tiny fraction of N and the estimate is dominated by rare draws. It samples the wrong region.

**Importance sampling.** Replace the uniform draw by a draw from a tailored density q and reweight by p/q. Core math: Ĵ = (1/N) Σ_i f(x_i) p(x_i)/q(x_i). Concentrates effort where the integrand is large and is a genuine variance-reduction tool. Limitation: in high dimension the weight distribution is extremely skewed — a few enormous weights, a sea of negligible ones — so the variance of the estimator is large and hard to assess; and constructing a q that tracks a complicated high-dimensional target is itself unsolved. The raw unbiased estimator also presupposes an absolutely-normalized p; self-normalized ratios avoid that constant but keep the same weight-degeneracy problem.

**Acceptance–rejection sampling.** Draw a candidate from a blanket density h dominating the target and accept with probability f/(c·h). Core math: accept Z ∼ h if u ≤ f(Z)/(c·h(Z)); accepted draws are iid from f. Limitation: requires a global envelope c·h ≥ f everywhere — hard to construct in high dimension and inefficient (acceptance c^{−1}) when the envelope is loose; each rejection throws the draw away entirely and starts over independently.

**Molecular dynamics / time averaging.** Integrate the equations of motion and average along the trajectory, equating time and ensemble averages by ergodicity. Core idea: F̄ = lim_{T→∞} (1/T) ∫₀ᵀ F(x(t)) dt. Limitation: requires the machine to follow the detailed coupled kinematics of all particles for long simulated times — beyond reach when the computer is not fast or large enough — and it answers a dynamical question when only the equilibrium average is wanted.

## Evaluation settings

The natural testbed is a system of N interacting particles in a box with periodic boundary conditions (each particle interacts with the nearest image of every other, so surface effects are minimized and a few hundred particles can stand in for a bulk phase). The two-dimensional system of rigid disks (hard spheres) is the canonical hard case: the interaction is zero when disks do not overlap and infinite when they do, so it isolates the packing/excluded-volume difficulty without a smooth potential to complicate matters. The observable of central interest is the equation of state — pressure as a function of density (area per particle), obtained from the virial theorem, which for the rigid-disk system reduces to the average contact density of the radial distribution function. The reference yardsticks that exist beforehand are the free-volume theory of the dense fluid and the virial-coefficient expansion of the equation of state valid at low density; the relevant diagnostic statistics are the radial distribution function N(r²) and the running average of the observable over successive sweeps through the particles. The metric of merit for the *sampler itself* is whether the configurations visited are distributed according to the Boltzmann weight and how quickly the running averages settle — controlled by how the proposed-move size trades off against the fraction of moves rejected.

## Code framework

Available pieces: an unnormalized log weight, a uniform random-number source, a candidate generator for tentative local changes, a state container with boundary handling, and an averaging loop. The open slot is the rule that turns the current configuration into the next one.

```python
import numpy as np

# --- already available: target up to a constant and tentative local changes ---
def log_weight(state):
    """Unnormalized log density, e.g. -E(state)/(kT), with any additive
    normalizing constant absent."""
    ...

def candidate_rule(state, rng):
    """Generate a tentative next state from the current state.

    If the candidate density is already available, return the forward and
    reverse log densities alongside the candidate.
    """
    ...

# --- open slot ---
def transition(state, log_weight, candidate_rule, rng):
    """Produce the next state from the current state.

    The missing rule decides how a tentative local change becomes the next
    state so that long-run averages come out against the intended weight.
    """
    # TODO: decide how the tentative change becomes the next state
    pass

def sample(state0, log_weight, candidate_rule, n_steps, rng):
    """Run the chain and average observables along it."""
    state = state0
    samples = []
    for _ in range(n_steps):
        state = transition(state, log_weight, candidate_rule, rng)
        samples.append(np.copy(state))
    return samples

def estimate(samples, observable):
    """Ensemble average of an observable over the collected configurations."""
    return np.mean([observable(c) for c in samples], axis=0)
```
