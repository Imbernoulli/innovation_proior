Let me start from what actually breaks when I push a multi-objective evolutionary algorithm past three objectives. I have `M` conflicting objectives `f_1, ..., f_M` to minimize, and the answer is never one point; it's the non-dominated set, where no feasible point is at least as good on every objective and strictly better on one. With a finite population I can only return a sample of that front, so my real target is a population that does two things at once: drives toward the front (convergence) and stays evenly spread across it (diversity). For two or three objectives the tool everyone reaches for is Pareto dominance plus a density device — exactly NSGA-II — and it works beautifully. The question that's nagging me is why it falls apart when `M` gets large, because if I understand precisely *how* it breaks, the fix should follow.

NSGA-II's machine: breed offspring `Q_t` from parents `P_t` with SBX and polynomial mutation, pool `R_t = P_t ∪ Q_t` of size `2N`, fast-non-dominated-sort into fronts `F_1, F_2, ...`, fill the next generation front by front, and truncate the overflowing front by crowding distance — sort each objective, give the two extremes infinite distance, give each interior point the summed normalized neighbour gaps, keep the most isolated. So convergence pressure comes from the dominance sort, and diversity comes from the crowding tie-break. Now turn up `M`. Pick two random candidate solutions in a high-dimensional objective space; for one to dominate the other it must be no worse on *all* `M` objectives, and as `M` grows the chance that one vector beats another on every single coordinate plummets — almost every pair comes out mutually non-dominated. So `F_1`, the first non-dominated front, swallows nearly the entire pooled population, and it does this early, often in the first handful of generations. Once that happens, the dominance sort is silent: it puts everyone in the same front, so the *only* thing deciding survival is the crowding-distance tie-break. The pressure that's supposed to push the population toward the front has evaporated, and a density heuristic — designed to break ties, not to converge — is left doing the convergence job it was never meant to do. And that density heuristic is itself in trouble: in a high-dimensional objective space the population sits very sparsely, so the per-objective neighbour gaps that crowding distance sums up become noisy and uninformative. Both halves of NSGA-II degrade at once, and they degrade *because* `M` is large. That's the diagnosis: dominance is a partial order, and a partial order goes blind exactly when everything is incomparable.

So I want a comparison that never goes blind — a *total* order, so I can always say which of two solutions is better, no matter how non-dominated the population gets. Where does a total order come from? Scalarization: aggregate the `M` objectives into a single scalar, and a scalar has a total order by construction. The classical multi-objective theory is full of this. MOEA/D took the scalarization route hard — pick `N` weight vectors spread over the simplex, attach a scalar sub-problem to each, optimize them cooperatively. The cleanest scalar for the job is PBI, penalty-based boundary intersection: fix a reference point `z*`, a weight direction `w`, normalize `u = w/||w||`, decompose the displacement `F(x) - z*` into a piece *along* the line, `d_1 = ||(F(x) - z*)^T u||`, and a piece *off* it, `d_2 = ||(F(x) - z*) - d_1 u||`, and minimize `g^pbi = d_1 + theta·d_2`. That's lovely structurally: `d_1` is a convergence term (how far down the line you've pushed toward the front) and `theta·d_2` is a diversity term (a penalty for drifting sideways off your prescribed direction), and it's a total order. This is the shape I want — convergence and diversity fused into one scalar per direction, selection within each direction. So let me see whether I can just lift PBI into the many-objective setting and be done.

Two things stop me, and they're worth being precise about because the precise failure is the clue. First, `theta`. It's a single fixed scalar that sets the convergence-vs-diversity tradeoff for the *entire run* and for *every* sub-problem and for *every* problem instance. Ishibuchi and others have observed there's no value of `theta` that works well across different problems and different numbers of objectives — too small and the population converges but bunches up, too large and it spreads but stalls short of the front. A fixed knob I have to re-tune per problem is exactly the kind of fragility I'm trying to escape, and it's worse in many objectives where I have the least intuition for where to set it. Second, and this is the deeper one: `d_2` is a *Euclidean* perpendicular distance. A Euclidean distance is unbounded and it scales with the magnitude of the objective vector — a solution far out from `z*` has a large `d_2` simply by being far out, even if it's perfectly on-direction in *angular* terms. So `d_1` and `d_2` are tangled: both grow with how far the solution sits from the ideal point, which means the "pure diversity" term isn't pure — it's contaminated by convergence. And as `M` grows and the front's scale varies, an unbounded Euclidean off-line distance is a nightmare to normalize across sub-problems. Wall. PBI has the right skeleton but the wrong diversity measure.

Let me stare at `d_2` and ask what it's really trying to capture. It wants to say: how far is this solution from its prescribed direction? But it answers in absolute length, which couples to distance-from-ideal. What I actually want is the *direction mismatch*, independent of how far out the solution is. That's an angle. Take the translated objective vector and the reference direction, and ask for the acute angle between them. An angle has exactly the properties `d_2` lacks. It's bounded — for first-quadrant vectors it lives in `[0, π/2]` — and it's *scale-invariant*: scale the objective vector by any positive constant and its angle to the reference direction doesn't budge, because `cos θ = (a·b)/(||a|| ||b||)` is homogeneous of degree zero in each argument. So no matter how far a solution is from the ideal point, its angle to a reference vector is constant. The angle is a clean diversity signal, completely decoupled from convergence. And convergence I can measure separately and just as cleanly: after I translate the objectives so the ideal point sits at the origin, the *length* of the translated objective vector *is* its distance to the ideal point. So I have two genuinely orthogonal readings — length for convergence, angle for diversity — instead of PBI's two entangled distances.

I want a set of `N` directions spread evenly over the objective space, one per sub-problem, and the natural construction is already on the table — Das & Dennis's simplex-lattice: every vector `u_i` with entries in `{0/H, ..., H/H}` summing to one, giving `N = C(H+M-1, M-1)` points evenly placed on the unit hyperplane. But those points live on the simplex, and my criterion is *angular*, so the natural home for them is the unit hypersphere, not the simplex. Map each one out to unit length, `v_i = u_i / ||u_i||`, and now I have `N` unit reference vectors fanning out from the origin into the first quadrant. Because they're unit, any cosine I compute against them needs no renormalization on their side. Good — these are my reference vectors, and each one owns a sub-problem.

Now the selection itself. Each generation, after the usual μ+λ shuffle — combine the parent population with the offspring into a pool — I want to pick one survivor per reference vector. Step one, translation: compute the current per-objective minimum `z^min` over the pool (the cheap substitute for the true ideal point, which I can't afford to compute exactly), and translate every objective vector `f' = f - z^min`. This pins the ideal to the origin, guarantees every `f'` sits in the first quadrant, and — the payoff — makes `||f'||` the distance to the ideal, my convergence reading, for free. Step two, partition: associate each individual with the reference vector it points most nearly along. For individual `i` and reference vector `j`, `cos θ_{i,j} = (f'_i · v_j) / ||f'_i||` (the `v_j` is unit, so it drops out of the denominator), and I assign `i` to the vector of *maximum* cosine, i.e. *minimum* acute angle. That carves the pool into `N` subpopulations, one per reference vector — exactly the subspace-partition-then-select-within structure that the M2M and inverse-model lines showed was workable; what they left open was the scalar criterion *inside* a subspace, and that's what I'm building now. Step three: from each subpopulation, keep the single best individual by my scalar. Step four: that roster of one-survivor-per-vector is the next generation. No dominance comparison appears anywhere — selection is now a total order within each subspace, so it can't go blind.

So what's the scalar that picks the survivor inside a subpopulation? I have two readings: convergence `||f'_i||` (smaller is better) and diversity, the angle `θ_{i,j}` to the associated vector (smaller is better — closer to your prescribed direction). PBI added them; I argued that addition is what entangles them. Let me try *scaling* instead: take the convergence distance and inflate it by a factor that grows with the angle. Something like `d = (1 + penalty(θ)) · ||f'_i||`. Watch what this gives me. If a solution is perfectly on-direction, `θ = 0`, the penalty is zero, and `d = ||f'_i||` — pure convergence distance, so among on-direction solutions I simply keep the one closest to the front. If a solution is off-direction, its convergence distance gets inflated in proportion to how far out it already is, so a far-out, badly-aligned solution is penalized hard while a far-out but well-aligned one is barely touched. The multiplicative form makes the penalty's *effect* scale with `||f'||`, which is right: pushing a solution off its niche matters more the further out it is. This is the angle-penalized distance — a scaled convergence distance, where the scaling is the diversity penalty. Call it the APD.

Now I have to actually design `penalty(θ)`, and I want to derive its shape from the search dynamics rather than guess. Think about what the right convergence-vs-diversity balance is at different stages. Early in the run the population is far from the front and scattered; the urgent thing is to *get to the front* — diversity is premature, because spreading out a population that hasn't converged just spreads it out in the wrong place. So early, I want almost pure convergence pressure: `penalty ≈ 0`, `d ≈ ||f'||`, and within each niche I just race toward the front. Late in the run the population is near the front; now convergence is mostly done and the urgent thing is to *spread evenly* along it — so the angle penalty should bite, pushing each niche's survivor toward its own reference direction and away from crowding its neighbours. So the penalty should start near zero and grow as the search proceeds. The cleanest dial for "how far along the search am I" is the generation ratio `t/t_max`, which runs from 0 to 1. A penalty that's `(t/t_max)` raised to a power gives me exactly the profile I want: small for most of the run, ramping up near the end. Raise it to `α`: with `α > 1` the ramp is convex, staying nearly flat for a long while and then climbing steeply near `t_max`, which matches "converge for most of the run, spread hard at the end." `α = 2`, a quadratic ramp, is a natural default — gentle enough that convergence dominates the early and middle game, sharp enough that diversity takes over at the close. So a factor `(t/t_max)^α` belongs in the penalty.

But `(t/t_max)^α · θ` alone isn't right yet, and the reason exposes the next subtlety. My reference vectors are *not* perfectly uniform in angular spacing. The simplex-lattice puts them evenly on the hyperplane, but the map to the sphere distorts spacing, and once I start adapting the vectors (which I'll need to, in a moment) the spacing gets genuinely irregular — some vectors end up with close neighbours, some isolated. Now suppose two niches: niche A whose vector has a neighbour only a tiny angle away, niche B whose nearest neighbour is far. A solution sitting at angle `θ` from its vector means very different things in the two niches. In A, `θ` might already overlap the neighbouring niche's territory — that solution is poaching, it should be penalized. In B, the same `θ` is still comfortably inside its own wide niche — no poaching, leave it alone. A raw `θ` penalty can't tell these apart; it would over-penalize wide niches and under-penalize narrow ones. What I need is `θ` measured *relative to the niche's own width*. So define, for each reference vector `v_j`, the angle to its nearest neighbour among the other reference vectors — call it `γ_{v_j} = min_{i≠j} angle(v_i, v_j)` — and normalize: `θ / γ_{v_j}`. Now the penalty reads "how big is your angular deviation relative to your niche's spacing," which is the scale-free quantity I actually want, and it's comparable across dense and sparse regions of the reference set. Crucially I'm normalizing the *angle*, not the objectives — so the actual objective values, which carry the convergence information `||f'||` reads off, are left untouched, and the normalization is done independently inside each subspace, so adjusting one niche's penalty doesn't perturb another's.

One more factor. The sparsity of solutions in objective space grows with the number of objectives — the same population spreads thinner as `M` rises — so the angular scales I'm normalizing against grow with `M` too. To keep the penalty's magnitude meaningful as the objective dimension climbs, I scale it by `M`. Putting the pieces together, the penalty is

  P(θ_{i,j}) = M · (t/t_max)^α · θ_{i,j} / γ_{v_j},

and the angle-penalized distance is

  d_{i,j} = (1 + P(θ_{i,j})) · ||f'_i||.

Within each subpopulation I keep the individual of minimum APD; that's my survivor, one per reference vector. Let me sanity-check the two regimes against the formula. At `t ≪ t_max`, `(t/t_max)^α ≈ 0`, so `P ≈ 0` and `d ≈ ||f'_i||` — pure convergence, exactly as I argued I want early. At `t → t_max`, `(t/t_max)^α → 1`, `P` is at full strength, and an off-direction solution's distance is inflated by `M·θ/γ`, so a well-aligned solution beats a poorly-aligned one even if the poorly-aligned one is slightly closer to the front — diversity takes over, as I want late. And contrast it against PBI one more time to be sure I've actually fixed the two complaints: PBI's `theta` was a fixed scalar with no good cross-problem setting, whereas my penalty *self-adapts* — it's normalized to a comparable range by `γ` no matter how dense or sparse the vectors are, and it shifts with `t` over the run, so the single `α` doesn't need re-tuning per problem the way PBI's `theta` did; and PBI's diversity used an unbounded Euclidean off-line distance entangled with convergence, whereas mine uses a bounded, scale-invariant angle that's decoupled from `||f'||`. Both complaints answered, and the answer fell out of insisting the diversity reading be an angle.

Now the scaling problem I deferred. I assumed I could compare angles across reference vectors as if the objectives were all on a common `[0,1]` scale. But real many-objective problems have objectives scaled to wildly different ranges — one objective in the hundreds, another in tenths (the WFG problems, the scaled DTLZ problems). When the objectives are scaled like that, a uniform fan of reference vectors does *not* produce a uniform spread of solutions: the vectors point into the cube `[0,1]^M`, but the front lives in a stretched box, so most vectors miss the front's bulk and the solutions clump. I need to fit the reference vectors to the actual scale of the front. The obvious move is to normalize the objectives — that's what NSGA-III does, estimating extreme points and intercepts each generation and rescaling everything onto the unit hyperplane. But I can't do that, and the reason is exactly my own design. Normalization *changes the actual objective values*, and my selection criterion *reads* the actual objective values — both the convergence length `||f'||` and the angle depend on where the translated vectors really sit. Watch what normalization does to the geometry: take two translated vectors `f'_1 = (0.1, 2)` and `f'_2 = (1, 10)`; their separation is `||f'_2 - f'_1|| = ||(0.9, 8)|| = 8.06`. Normalize each objective by its range and they become `(0.1, 0.2)` and `(1, 1)`; the dominance relation is preserved — `f'_1` still dominates `f'_2` — which is exactly why NSGA-III can do this safely, since dominance is invariant to monotone per-objective rescaling. But the *separation* collapses to `||(0.9, 0.8)|| = 1.2`. The geometry my angle-and-length criterion lives on is mangled, and worse, it's mangled *differently every generation* as the running scales drift, so the criterion would never sit still. Wall. Objective normalization is fine for a dominance-based method and poison for a geometry-based one.

So I have to fit the front's scale without touching the objectives — which means the thing I move can't be the objectives, it has to be the reference vectors. If the front is stretched by some per-objective range, stretch each reference vector by the same range and it will once again point into the front's bulk, while every objective value stays exactly where it is. Concretely: estimate the per-objective range from the current population, `z^max - z^min` over the survivors, take each *original* unit reference vector `v_{0,i}` (the pristine simplex-lattice direction, kept around from initialization), scale it component-wise by that range, and renormalize back to unit length:

  v_{t+1,i} = (v_{0,i} ∘ (z^max - z^min)) / ||v_{0,i} ∘ (z^max - z^min)||,

where `∘` is the Hadamard (element-wise) product. Stretching then renormalizing tilts each vector toward the long axes of the front so the fan re-spreads across the actual front shape, and the objectives are untouched, so my angle and length readings stay coherent. Always rebuild from the original `v_0` rather than from the last adapted set, so errors don't accumulate across adaptations. And one thing I must not forget: when the vectors move, their neighbour spacings `γ_{v_j}` change, so I have to recompute every `γ` right after an adaptation — otherwise the APD penalty would be normalizing against stale niche widths.

How often should I adapt? My instinct is "every generation, to track the scale as tightly as possible," but I think that's wrong, and Giagkiozis and colleagues observed why: if the reference vectors keep moving every generation, the niches keep shifting under the population's feet, the association of solutions to vectors churns, and convergence destabilizes — the algorithm spends its effort chasing a moving target instead of settling into the front. Unlike objective normalization, which a dominance method *must* redo every generation, reference-vector adaptation can be occasional: I only need to correct the scale a handful of times over the run. So introduce a frequency `f_r` and adapt only every `⌈f_r · t_max⌉` generations. With `f_r = 0.1`, that's ten adaptations across the run — frequent enough to track a drifting scale, rare enough to keep convergence stable. And it's nearly free: `O(MN / (f_r · t_max))` amortized.

Let me settle the remaining knobs and the offspring side, then check the cost. Offspring: I deliberately use *no fitness-biased mating selection*. Normally you'd select good parents to breed, but here every individual in the population is already the elitist of one subspace, so each survivor represents its niche; there's no reason to bias mating toward some niches over others, and a fancy selection would only cost time. So I sample parents uniformly and pair them at random — every survivor has equal probability of breeding — and apply the standard real-coded operators: SBX with distribution index `η_c` and polynomial mutation with `η_m`, `p_c = 1.0` (always cross), `p_m = 1/n` (about one decision variable mutated per child on average). The reference-vector selection downstream does all the convergence-and-diversity balancing, which is what lets me get away with such a plain offspring stage. For the indices I take `η_c = 30` (a touch more local than the usual 20, keeping children near parents so the well-placed niche survivors aren't disrupted too much) and `η_m = 20`. And `α = 2`, `f_r = 0.1` as derived above.

The cost, per generation: objective translation is `O(MN)`; the population partition compares `N` solutions against `N` reference vectors, `O(MN^2)`; the APD computation and the per-niche elitism are `O(MN^2)` and `O(N^2)`; the occasional adaptation is `O(MN/(f_r t_max))` amortized. So the whole generation is `O(MN^2)`, dominated by the partition — and notably there's no non-dominated sort and no dominance comparison anywhere, which is the same `O(MN^2)` order as NSGA-II's sort but without the partial-order pathology that made that sort useless in many objectives. The memory is one population plus `N` reference vectors and their `γ` spacings, linear in the population.

The harness hooks are `select` (who mates), `vary` (produce offspring), `survive` (who carries forward), and `on_generation` (adaptive state). The mapping is natural: `select` just hands back the roster (no fitness-biased mating selection), `vary` does the random-pair SBX+PM, `survive` is the whole APD machine — translate, partition by angle, score by APD, keep the min-APD survivor per non-empty reference vector — and `on_generation` fires the reference-vector adaptation on the `f_r·t_max` schedule and recomputes `γ`. The reference vectors, their pristine copies, and the running ideal are state set up once. Two details to handle: if a reference vector's subpopulation is empty I simply skip it (no survivor from an empty niche), and because empty niches mean I may return fewer than `pop_size` survivors, I top up from the remaining individuals by smallest distance-to-ideal so the population stays at size `N`. The numpy core is direct — angles via `arccos` of the normalized-dot-product matrix, `γ` via the second-largest cosine per vector (the largest is the vector with itself), APD as the scaled distance.

```python
import random
from copy import deepcopy

import numpy as np
from deap import tools   # cxSimulatedBinaryBounded, mutPolynomialBounded, uniform_reference_points


class CustomMOEA:
    """RVEA: a reference-vector guided EA for many-objective optimization.
    Selection is a total order per reference vector via the angle-penalized
    distance (APD = scaled distance-to-ideal), so it never goes blind the way
    Pareto dominance does once everything is mutually non-dominated; reference
    vectors are adapted to the objective scales rather than normalizing the
    objectives (which would corrupt the angle/length geometry)."""

    def __init__(self, pop_size, n_obj, n_var, bounds,
                 cx_eta=30.0, mut_eta=20.0, mut_prob=None):
        self.n_obj = n_obj
        self.n_var = n_var
        self.bounds = bounds                       # (low, up) of the box decision space
        self.cx_eta = cx_eta                       # SBX distribution index (RVEA uses 30)
        self.mut_eta = mut_eta                     # polynomial-mutation index (20)
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var
        self.alpha = 2.0                           # APD penalty ramp exponent
        self.fr = 0.1                              # reference-vector adaptation frequency
        self.max_gen = 400                         # horizon for the (t/t_max)^alpha ramp;
                                                   #   the driver can overwrite it per problem

        # N evenly spread reference points (Das-Dennis simplex-lattice), mapped to
        # the unit sphere: directions, not simplex weights, since the criterion is angular.
        p = pop_size - 1 if n_obj == 2 else 12
        V0 = np.asarray(tools.uniform_reference_points(n_obj, p=p), dtype=float)
        self.V0 = self._unit(V0)                   # pristine unit reference vectors
        self.V = self.V0.copy()                    # current (adaptable) reference vectors
        self.pop_size = len(self.V0)
        self.gamma = self._gamma(self.V)           # per-vector nearest-neighbour angle
        self.z_min = None                          # running ideal (per-objective min)
        self._gen = 0

    @staticmethod
    def _unit(V):
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1e-12
        return V / norms

    def _gamma(self, V):
        # smallest acute angle from each v_j to any OTHER reference vector:
        # arccos of the second-largest cosine in each row (largest = v_j with itself).
        cos = np.clip(V @ V.T, -1.0, 1.0)
        cos_sorted = np.sort(cos, axis=1)          # ascending; last col is self (cos=1)
        gamma = np.arccos(cos_sorted[:, -2])       # second-largest cosine -> nearest neighbour
        return np.maximum(gamma, 1e-12)

    def select(self, population, k):
        # No fitness-biased mating selection: each individual already elitist of its
        # own subspace, so all represent their niche equally. Hand back the roster.
        return [deepcopy(ind) for ind in population]

    def vary(self, parents):
        # Random pairing + standard real-coded operators; the APD selection in
        # survive() does all the convergence/diversity balancing.
        offspring = [deepcopy(ind) for ind in parents]
        random.shuffle(offspring)
        lo, hi = self.bounds
        for i in range(0, len(offspring) - 1, 2):
            tools.cxSimulatedBinaryBounded(offspring[i], offspring[i + 1],
                                           eta=self.cx_eta, low=lo, up=hi)  # p_c = 1
            del offspring[i].fitness.values
            del offspring[i + 1].fitness.values
        for ind in offspring:
            tools.mutPolynomialBounded(ind, eta=self.mut_eta, low=lo, up=hi,
                                       indpb=self.mut_prob)
            if ind.fitness.valid:
                del ind.fitness.values
        return offspring

    def _apd_select(self, valid, gen):
        F = np.array([ind.fitness.values for ind in valid], dtype=float)
        # Step 1: translate so the ideal point is the origin; ||f'|| = distance to ideal.
        z = np.min(F, axis=0)
        self.z_min = z if self.z_min is None else np.minimum(self.z_min, z)
        Fp = F - self.z_min
        dist = np.linalg.norm(Fp, axis=1)          # convergence reading
        dist[dist < 1e-12] = 1e-12
        # Step 2: partition by acute angle to the reference vectors (v unit -> cos = Fp_hat . v).
        cos = np.clip((Fp / dist[:, None]) @ self.V.T, -1.0, 1.0)
        angle = np.arccos(cos)                      # (n, N) acute angles; diversity reading
        assoc = np.argmin(angle, axis=1)           # nearest reference vector (min angle)
        # Step 3: APD per non-empty subpopulation, keep the min-APD survivor.
        ramp = (gen / max(self.max_gen, 1)) ** self.alpha   # (t/t_max)^alpha
        survivors = []
        for j in range(self.pop_size):
            members = np.where(assoc == j)[0]
            if len(members) == 0:
                continue
            theta = angle[members, j]
            penalty = self.n_obj * ramp * (theta / self.gamma[j])
            apd = (1.0 + penalty) * dist[members]  # scaled distance-to-ideal
            survivors.append(int(members[int(np.argmin(apd))]))
        # Top up to pop_size from the leftover individuals by smallest distance-to-ideal.
        if len(survivors) < self.pop_size:
            chosen = set(survivors)
            leftover = [i for i in range(len(valid)) if i not in chosen]
            leftover.sort(key=lambda i: dist[i])
            for i in leftover:
                survivors.append(i)
                if len(survivors) >= self.pop_size:
                    break
        return [valid[i] for i in survivors[:self.pop_size]]

    def survive(self, population, offspring):
        combined = population + offspring          # mu + lambda elitism shell
        valid = [ind for ind in combined if ind.fitness.valid]
        if len(valid) <= self.pop_size:
            return valid
        return self._apd_select(valid, self._gen)

    def on_generation(self, gen, population):
        # Remember the generation index for the APD ramp, and adapt the reference
        # vectors to the objective scales every ceil(fr * max_gen) generations.
        self._gen = gen
        period = max(1, int(np.ceil(self.fr * self.max_gen)))
        if gen % period == 0 and population:
            F = np.array([ind.fitness.values for ind in population
                          if ind.fitness.valid], dtype=float)
            if len(F) == 0:
                return
            z_min, z_max = np.min(F, axis=0), np.max(F, axis=0)
            rng = z_max - z_min
            rng[rng < 1e-12] = 1e-12
            # v = v0 ◦ (z_max - z_min), renormalized; rebuild from the pristine V0
            # so scaling errors don't accumulate. Objectives are left untouched.
            self.V = self._unit(self.V0 * rng)
            self.gamma = self._gamma(self.V)       # niche widths moved -> recompute gamma
```

So the causal chain is this. Dominance gives only a partial order, and as the number of objectives grows nearly every pair of solutions becomes mutually non-dominated almost immediately, so the dominance sort can no longer separate individuals and the convergence pressure collapses, leaving a density tie-break — itself degraded in high dimensions — to do a job it was never meant for. I wanted a total order, so I turned to scalarization, and PBI had the right skeleton (one scalar per direction, convergence plus diversity, selection within each direction) but the wrong diversity term: a fixed, un-tunable penalty `theta` and an unbounded Euclidean off-line distance entangled with how far the solution sits from the ideal. Insisting the diversity reading be an *angle* fixed both problems at once — bounded, scale-invariant, decoupled from convergence — and once I translate the objectives so the ideal is the origin, the length of the translated vector gives convergence for free, so I have two orthogonal readings, length and angle. Scaling the convergence distance by `(1 + penalty(angle))` rather than adding the penalty keeps the two readings from tangling and makes the penalty's effect proportional to distance-from-front. The penalty's shape I derived from the dynamics: nearly zero early (converge first), growing as `(t/t_max)^α` so diversity takes over late, normalized per reference vector by the nearest-neighbour angle `γ` so it's comparable across dense and sparse niches without touching the objectives, and scaled by `M` for the growing sparsity. Badly-scaled fronts I couldn't fix by normalizing objectives — that would corrupt the very angle-and-length geometry my criterion reads, and re-corrupt it every generation — so I adapted the reference vectors to the objective ranges instead, stretching the pristine vectors by `z^max - z^min` and renormalizing, leaving the objectives untouched, and only occasionally (every `f_r·t_max` generations) so the niches don't churn, recomputing `γ` each time the vectors move. With no fitness-biased mating selection needed and the standard SBX/PM operators on top of the μ+λ elitism shell, the result is one population-based run that keeps full total-order selection pressure in many objectives, spreads through reference directions instead of a density device, shifts from convergence-first to spread-later on its own, and tolerates differently scaled fronts — all at `O(MN^2)` per generation with no dominance comparison anywhere.
