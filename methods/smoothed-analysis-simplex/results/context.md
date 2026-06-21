# Context: explaining why the simplex method is fast in practice

## Research question

The simplex method has been the workhorse of linear programming since 1947, and for fifty
years practitioners have observed it solve real linear programs in a number of pivot steps
that grows roughly linearly in the problem size. Yet the theory says the opposite: on
specially built inputs the simplex method takes a number of steps exponential in the
dimension. The two standard ways of analyzing an algorithm produce conflicting conclusions.
Worst-case analysis reports the exponential blow-up and so brands the method intractable.
Average-case analysis reports expected-polynomial running time, but only on inputs drawn from
idealized random distributions.

The question is therefore: **is there a mode of analysis that can reconcile the exponential
worst case with the everyday polynomial speed observed on practical linear programs?**

## Background

**Linear programming and the simplex method.** A linear program asks to maximize a linear
objective zᵀx subject to linear inequality constraints Ax ≤ y, with A an n×d matrix (n
constraints, d variables). The feasible region is a polyhedron, and the optimum, when it
exists, is attained at a vertex. Dantzig's simplex method (1947) walks from vertex to adjacent
vertex along edges of the polyhedron, each step improving the objective, until it reaches a
vertex with no improving neighbor — the optimum. A simplex method is not fully specified until
one fixes a *pivot rule*: the policy for choosing the next vertex when several improve the
objective.

**The worst case is exponential.** Klee and Minty (1972) built a "squashed cube" — a
combinatorial d-cube with slightly perturbed facets — on which Dantzig's largest-coefficient
rule visits all 2ᵈ vertices before reaching the optimum. This was not an isolated curiosity:
for essentially every deterministic pivot rule a family of polytopes is known on which that
rule takes exponentially many steps (Jeroslow, Goldfarb, Goldfarb–Sit, Avis–Chvátal, Murty;
Amenta–Ziegler later gave a unified construction). The best known randomized pivot rules
(Kalai; Matoušek–Sharir–Welzl) take an expected nᴼ⁽√ᵈ⁾ steps — sub-exponential, but still far
from the polynomial behavior seen in practice. A separate line of thought asks not whether the
algorithm is fast but whether a short path even *exists*: the Hirsch conjecture posits a path of
length at most n−d between any two vertices, and Kalai–Kleitman proved a path of length at most
n^(log₂ d + 2) always exists. But the existence of a short walk does not mean the simplex method
will find it, so this route does not by itself explain the algorithm's speed.

**Other LP algorithms exist but did not displace simplex.** Khachiyan (1979) applied the
ellipsoid algorithm and proved linear programming solvable in time polynomial in d, n, and L
(the bit-length of the input) — a landmark for complexity theory, but uncompetitive with
simplex in practice. Karmarkar (1984) introduced interior-point methods, also polynomial in d,
n, L, and these *are* competitive with simplex. Still, simplex remained the most popular method.

**Average-case analyses.** Beginning in the late 1970s a body of work proved the simplex method
fast on random inputs. Borgwardt (1977, 1982) proved that the shadow-vertex pivot rule runs in
expected polynomial time when the constraint vectors are drawn independently from a spherically
symmetric distribution (e.g. Gaussians centered at the origin). Smale, and then Megiddo with a
much improved bound, analyzed Lemke's self-dual parametric simplex algorithm under spherically
symmetric inputs. A second model, initiated independently by Haimovich (1983) and Adler (1983)
and developed by Todd, Adler–Megiddo, and Adler–Karp–Shamir, took the *maximum* over constraint
matrices A of the expected number of steps when the *directions* of the inequalities are flipped
at random; they proved an expected linear (and later quadratic) number of steps.

**The shadow-vertex method.** Among pivot rules, one has a uniquely clean geometric description,
due originally to Gass and Saaty and developed by Borgwardt. Given the objective z and a second
objective t that is optimized by a known starting vertex, project the polyhedron orthogonally
onto the two-dimensional plane spanned by t and z. The projection — the *shadow* — is a convex
polygon, and (this is the key structural fact) every vertex and every edge of the shadow polygon
is the image of a vertex or edge of the original polyhedron; moreover the starting vertex and the
z-optimizing vertex both map onto the polygon. Consequently, if one rotates the objective from t
toward z and follows the corresponding vertex of the polygon, one walks along the boundary of the
shadow from the image of the start vertex to the image of the optimum. The number of pivot steps
is then at most the number of edges of the shadow polygon. The shadow-vertex rule reduces the
running time to a single geometric quantity — the size of a two-dimensional shadow of the
polyhedron.

**A discrete model of randomly modifying an input.** The semi-random source model of Santha and
Vazirani fixes an input adversarially and then independently flips each bit with some probability.
Blum–Spencer, Feige–Krauthgamer, and Feige–Kilian used such models to give algorithms that succeed
on semi-random graph inputs. These models live in the discrete world of bits rather than
continuous real data.

## Baselines

- **Worst-case analysis of pivot rules.** For a pivot rule, report max over inputs of the step
  count. Klee–Minty and its successors show this is exponential for every standard deterministic
  rule and nᴼ⁽√ᵈ⁾ for the best randomized rules.

- **Borgwardt's average-case shadow-vertex bound.** Draw constraints from a spherically
  symmetric distribution; prove the expected shadow size, hence expected pivot count, is
  polynomial. Concretely the expectation is taken over rotationally invariant random a_i, and the
  analysis is essentially a computation of the expected number of facets of a random polytope's
  two-dimensional shadow.

- **Smale / Megiddo, self-dual parametric simplex, spherically symmetric inputs.** Expected
  step bounds for Lemke's algorithm under rotationally invariant inputs, with Megiddo
  sharpening Smale's bound dramatically.

- **Haimovich–Adler random-sign model.** Fix the constraint matrix adversarially, randomize the
  inequality directions, and bound the expected number of parametric steps (linear, later
  quadratic).

- **Polynomial-time LP algorithms (ellipsoid, interior-point).** Khachiyan and Karmarkar give
  genuine worst-case polynomial bounds in d, n, L.

## Evaluation settings

The object of study is a linear program max zᵀx s.t. Ax ≤ y with A ∈ ℝ^{n×d}, y ∈ ℝⁿ, z ∈ ℝᵈ,
n > d ≥ 3. The natural complexity measure is the number of pivot steps (vertices visited) taken
by a given simplex pivot rule, since per-step linear-algebra cost is a fixed polynomial. The
geometric proxy for the shadow-vertex rule is the number of edges of the shadow — the projection
of the feasible polyhedron onto the plane spanned by the two objective vectors. To model the noise
in continuous real data, the available perturbation tool adds independent Gaussian noise to the LP
data: every entry of A and y is perturbed by independent Gaussians of standard deviation
σ·max_i‖(y_i, a_i)‖, so the model is scale-free. The regime of interest is σ small relative to the
data — corresponding to imprecision in the low-order digits. The canonical adversarial stress test
is the Klee–Minty cube and its relatives, on which the unperturbed simplex method visits
exponentially many vertices.
