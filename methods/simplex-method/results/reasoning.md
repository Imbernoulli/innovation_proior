OK, let me think this through from where I actually am, which is the Air Force Comptroller's office with a deployment-and-supply program to build and no machine that will build it for me. The thing I do all day is *programming* in the military sense — laying out a time-staged schedule of training, supply, and deployment so that everything balances. Hitchcock and Wood have thrown down a challenge: mechanize this. Find a way to compute a time-staged program faster than a roomful of people on desk calculators. So the first thing I owe is a model — the relations a planner actually juggles, written down as mathematics.

I keep coming back to Leontief's input-output picture, because it's the cleanest thing I know that talks about a whole economy of interlocking activities. He has a matrix: one production process per commodity, each process eating some commodities and making one, and a system of linear equations saying total output of each item equals total use. It's simple and it computes. But it's the wrong shape for planning in three specific ways, and I can feel each one. It's steady-state — nothing changes over time — and my problem is nothing *but* time stages. It ties each item to exactly one process, so there's no notion of *choosing* among alternative ways to make the same thing — and choice is the whole point of planning. And it has no objective; it's a descriptive equilibrium, it tells you *the* solution of a determined system, not the *best* of many. So I generalize: let an activity be any process that consumes and produces items in fixed non-negative proportions, allow many alternative activities competing to supply each item, run it over time stages — that gives a staircase-structured matrix — and keep everything linear because the relations genuinely are. Levels of activities are non-negative; item balances are equations or inequalities. Good. That's a model: a big system of linear equalities and inequalities in non-negative variables.

Now the trap I've watched every planner fall into, including myself. Once I have this model, how do I pick *which* feasible program to run? In practice the answer is a thick book of ground rules — edicts from on high, "build battleships," "build bombers." And I finally see clearly what's wrong with that: the means has been promoted to the goal. Ask a commander the objective and he says "win the war"; press him and he says "build bombers" — but bombers are a *means*, and now that means spawns sub-rules about how to build bombers, and those get confused with goals, all the way down. The ground rules aren't the objective; they're at best a clumsy way of groping toward one. What I want is to separate the two: the model says which programs are *feasible*, and something else, stated apart from the rules, has to say which feasible program is *best*. The only thing I can write down that ranks programs and is consistent with the linear bookkeeping is itself a linear form — a linear function of the activity levels — and I extremize *that*, subject to the model's constraints. So the model carries the constraints; a separate linear objective carries the goal.

Let me be honest about *why* this is forced and not just tidy. Suppose I don't make the objective explicit and instead try to find the best plan by listing feasible plans and comparing them. Take the cleanest sub-problem: assign 70 men to 70 jobs, each man to one job, each job to one man. Two-and-seventy restrictions, but 70×70 = 4900 zero-one activity variables, and the number of feasible assignments is 70!. That's bigger than 10^100. No plausible calculating machinery can compare that many plans one by one, and the real planning programs are larger. Enumeration isn't a slow method, it's not a method at all. So I need *both* an explicit objective to optimize *and* an algorithm that finds the optimum without ever enumerating the feasible set. Stating the objective is half; the algorithm is the half that's about to give me trouble.

So I write the problem in its clean form. Collect all the constraints — add a non-negative slack to every inequality so it becomes an equation — and the whole thing is

```
minimize  z = c·x   subject to   A x = b,   x ≥ 0,
```

with A an m×n matrix. Now: can one actually solve this? Let me first ask the economists, since this *looks* like their territory — I go to Koopmans in Chicago in June. He gets excited, he sees the implications, but the economists don't have a *method of solution*. Fine. So I try my own luck at an algorithm.

Let me think about the geometry, because that's where an algorithm usually hides. The feasible set — all x with Ax = b, x ≥ 0 — is a convex polyhedron, and because x is constrained to be non-negative it is pointed once I remove redundant equality rows. The objective c·x is a linear form, constant on parallel hyperplanes; if a finite minimum is attained, the minimizing hyperplane exposes a face of the polyhedron. If the point I am staring at inside that face is not an extreme point, it is a convex combination of two other feasible points; linearity says their objective values average to the same minimum, so neither can be better and both are optimal too. I can keep moving inside the same optimal face until no convex splitting remains. So there is an optimal vertex. And vertices aren't a vague geometric notion — they're algebraic. Pick m linearly independent columns of A to be the *basic* columns, set the other n−m variables to zero, solve the square system for the m basic variables; if they come out non-negative, that's a *basic feasible solution*, and it *is* a vertex of the polyhedron. Conversely, every vertex has at least one such basis; if a basic value is zero, the same vertex may have several bases. Vertices ↔ basic feasible solutions. There are finitely many — at most "n choose m" bases. That already kills the continuum; I only have to search basic feasible solutions.

And here's the first idea, the one that would occur to any trained mathematician in about ten seconds: start at some vertex, look at the edges leaving it, and if an edge goes downhill in the objective, walk along it to the adjacent vertex. Repeat. Step by step descent along the edges of the polyhedron, vertex to adjacent vertex, until no leaving edge goes down — and then you're at the optimum. It's obvious, and that is exactly why I distrust it.

And my gut reaction is to *throw it away*. Look at what it actually proposes: it solves the problem by wandering along a path of *outside* edges of the polyhedron until it stumbles onto the optimal vertex. The boundary of a polyhedron in high dimensions is an enormous, convoluted thing. A path along its outside could meander astronomically far before it gets to the right corner. It has to be inefficient. Any well-trained mathematician would propose this in the row geometry of the variables — and anyone who then had to consider it as a *practical* method, as I do, would reject it on intuitive grounds as a very stupid idea without merit. So I reject it, and I go looking for something that cuts *through the interior* instead, straight toward the answer, not crawling around the rim.

So that's where I am: I've discarded the obvious algorithm and I'm hunting for an interior method. Let me leave that and come at the problem from the side it actually cracked open from — my thesis.

My thesis at Berkeley grew out of two problems Neyman wrote on the board that I thought were a homework assignment and went home and solved; one of them, the Neyman–Pearson one, I wrote up with Wald. At its core it's about the existence of Lagrange multipliers — the prices dual to the constraints — for an optimization that's linear but strange: a *continuum* of variables, each bounded in [0,1], with constraints written as Lebesgue–Stieltjes integrals and a linear objective to be extremized. Concretely: a sample space Ω with probability dP(u), find a characteristic function φ(u) ∈ {0,1} (relax to 0 ≤ φ ≤ 1) so that

```
∫ φ(u) dP(u) = α,
α⁻¹ ∫ φ(u) f(u) dP(u) = b,
α⁻¹ ∫ φ(u) g(u) dP(u) = z (minimize),
```

with f an (m−1)-vector function and g a scalar. Discretize: pick representative sample points u₁,…,uₙ with point masses pⱼ, set xⱼ = (pⱼ/α)φ(uⱼ), and the thing becomes a bounded-variable linear program — find min z, 0 ≤ xⱼ ≤ pⱼ/α, with Σxⱼ = 1 (the convexity constraint), Σ f(uⱼ)xⱼ = b, Σ g(uⱼ)xⱼ = z. The first coefficient in every column is that 1 from the convexity row.

Now the key accident, the thing that changes everything. Because n — the number of discrete sample points — could be enormous, even infinite, I do *not* want to work in the geometry of the variables, the row geometry, where the dimension is n. I want a geometry whose dimension is fixed and small. So I work in the space of a *column*. Each column is (Aⱼ, cⱼ) together with that leading 1; throw away the leading-1 coordinate since it's always 1, and each column becomes a point (yⱼ, zⱼ) in (m)-dimensional space — m−1 coordinates for y, one for z. The number of these points is n, possibly infinite, but they all live in a space whose dimension is m, set by the number of *constraints*, not the number of variables. That's the move. Look at the problem in the dimension of the columns.

Restate the whole program in that picture. I'm assigning non-negative weights xⱼ to the n points (yⱼ, zⱼ). The convexity row says the weights sum to 1, so I'm taking a *convex combination* — a center of gravity of the points, with masses xⱼ. The constraint Σ Aⱼ xⱼ = b says the y-coordinate of that center of gravity must equal b. And I want its z-coordinate as small as possible. So geometrically: put weights on a cloud of points so their center of gravity sits on the vertical "requirement line" through y = b, and slide that center of gravity as far *down* in z as it will go. The whole linear program is "lowest center of gravity over y = b."

Let me see what an algorithm looks like in *this* geometry, because it's going to look completely different from edge-walking. To have a center of gravity at a specified y = b in an m-dimensional picture using convex weights, I need, generically, m points whose convex hull contains the requirement line's crossing — m points spanning an (m−1)-dimensional simplex, with the requirement line piercing the *interior* of that simplex. So a "vertex" of my problem, in column language, is a *simplex of m points*. Say m = 3 for the picture: three points making a triangle, and the vertical requirement line through b pokes through the triangle at some height (b, zₜ). That zₜ is my current objective value — the height at which the line crosses the *plane* of the triangle.

How do I improve it? The plane through my m basic points has an equation z = π·y + π₀, and I find π, π₀ by solving the m equations π·Aⱼᵢ + π₀ = cⱼᵢ over the m basic columns — that's just fitting the plane to the m points. Now every *other* point (Aⱼ, cⱼ) in the cloud sits at some height cⱼ, and the plane predicts height π·Aⱼ + π₀ at that same y. The gap is cⱼ − (π·Aⱼ + π₀). If some point lies *below* the plane — negative gap — then bringing that point into the simplex lets me tilt the supporting plane downward at y = b, lowering zₜ. So I look for the point most below the plane:

```
s = argmin_j [ cⱼ − (π·Aⱼ + π₀) ].
```

Let me run one pricing pass by hand to make sure this rule actually points somewhere sensible, because I don't trust it until I've watched it move on numbers. Take m = 2 — one convexity row, one y-row — and five points (yⱼ, zⱼ): (0,3), (1,1), (2,4), (3,0), (4,5). The requirement is y = 2. Start with the two outer points 0 and 4 as my basic simplex; their y-interval [0,4] straddles 2, so the requirement line does pierce the segment between them. Fitting the plane (here a line) through them: π·0 + π₀ = 3 and π·4 + π₀ = 5 give π₀ = 3, π = 0.5. The current crossing height at y = 2 is 0.5·2 + 3 = 4. Now the gaps cⱼ − (π·yⱼ + π₀) for the five points: point 0 gives 3 − 3 = 0, point 1 gives 1 − 3.5 = −2.5, point 2 gives 4 − 4 = 0, point 3 gives 0 − 4.5 = −4.5, point 4 gives 5 − 5 = 0. The two basic points read 0, as they must — they're *on* the plane. The most negative is point 3 at −4.5, so point 3 enters. Geometrically point 3 = (3,0) is the lowest point sitting under my tilted line, exactly the one I'd want to pull the line down toward. Good — the rule moves, and it moves toward the floor of the cloud, not at random.

If that minimum is ≥ 0 — no point lies below the current plane — then the plane already supports the whole cloud from below. Any feasible center of gravity is a convex combination of the columns; the gaps at all columns are then non-negative, and the gap at the center of gravity is the same convex combination of the column gaps (everything here is linear in the weights), so it too is non-negative. At y = b, then, no feasible center of gravity can sit below the plane's height there, which is the current value — so I can't do better, and I stop. That gives me an optimality test, and the thing I like about it is that it's *purely local*: I read it off the m basic points and one pass over the columns; I never enumerate vertices.

If the minimum is negative, point s should enter. Geometrically I now have m+1 points — the old simplex plus s — forming an m-dimensional body, a tetrahedron in the m = 3 picture. The requirement line pierces this tetrahedron along a segment between two heights, (b, zₜ) and (b, zₜ₊₁), and because s was below the old plane, zₜ₊₁ < zₜ — the new crossing is strictly lower. The lower end of that segment lies on one *face* of the tetrahedron, an (m−1)-simplex of m points, and *that* face is my new vertex. Operationally: point s came in, and exactly one of the old points — call it r — got dropped to make room, because the new lower crossing lies on the face *opposite* r. So one column enters the basis, one leaves. And the index r isn't hard to pin down algebraically: as I bring s in with increasing weight and rebalance the others to keep the center of gravity on y = b, the weights of the old points change linearly, and r is the first one whose weight hits zero. Push past that and its weight would go negative — leave the simplex — which I can't allow. So r is the first basic weight driven to zero. Entering by the most-below-the-plane rule, leaving by the first-weight-to-zero rule.

Let me look at the two-dimensional version to *feel* why this is fast, because this is the picture that flips my intuition. Plot all the points (yⱼ, zⱼ) and take the *underbelly* of their convex hull — the lower boundary, a piecewise-linear convex function y ↦ f(y). The optimal z at y = b is f(b). My algorithm, given a current pair of points straddling the requirement line with a connecting line of some slope σ, finds the point most below that line — substitute each (yⱼ, zⱼ) into the line and maximize σ(yⱼ − y₆) − (zⱼ − z₆) — and swaps it in. It's a kind of secant method on the underbelly: given a slope it's cheap to find the hull point that supports that slope, even though I can't directly compute f(b). And here's the thing that turns my gut around. In any realistic problem, *most* of the points lie well *above* the underbelly of their hull. Only a few are extreme points of the lower hull at all. My algorithm only ever moves to points on the underbelly — it only ever picks an entering point that's a lower-hull vertex — and those are *rare*. So the number of choices it actually faces is small. I conjecture it takes about m steps. Not a wandering crawl around the whole rim — a short walk across a handful of lower-hull corners.

Let me make that quantitative, because "I have a feeling it's fast" isn't worth much and I want to know what governs the rate when it *does* take more than m steps. Track Δₜ = zₜ − z*, the gap from the current value to the optimum. Suppose at every iteration the entering variable comes in at value θₜ, and suppose all basic variable values stay at least θ > 0 (a non-degeneracy floor). Eliminate the basic variables from the objective row, so the current reduced costs are c̄ⱼ, with c̄ⱼ = 0 on basic j and c̄ₛ < 0 the entering one. At the optimum the optimal weights xⱼ* are non-negative and reproduce the gap:

```
Δₜ₋₁ = zₜ₋₁ − z* = Σⱼ (−c̄ⱼ) xⱼ*  ≤  (−c̄ₛ) Σⱼ xⱼ*  =  (−c̄ₛ),
```

using that −c̄ₛ is the *most* negative reduced cost so it dominates each term, and Σ xⱼ* = 1 from convexity. So the most-negative reduced cost is at least the whole remaining gap, −c̄ₛ ≥ Δₜ₋₁. Now the actual one-step decrease is that reduced cost times how far I moved:

```
Δₜ₋₁ − Δₜ = zₜ₋₁ − zₜ = (−c̄ₛ) θₜ  ≥  Δₜ₋₁ θₜ,
```

the inequality by multiplying −c̄ₛ ≥ Δₜ₋₁ by θₜ > 0. Rearrange:

```
Δₜ ≤ (1 − θₜ) Δₜ₋₁ ≤ e^{−θₜ} Δₜ₋₁,
```

using 1 − θ ≤ e^{−θ}. Let me check that last inequality isn't wishful, since the whole bound rests on it: at θ = 0.1, 1 − θ = 0.900 against e^{−0.1} = 0.905; at θ = 0.5, 0.500 against 0.607; at θ = 0.9, 0.100 against 0.407. The right side is larger every time — and it has to be, since e^{−θ} = 1 − θ + θ²/2 − … sits above 1 − θ by a positive θ²/2 to leading order. Good, the step is honest. Chain it over the iterations:

```
Δₜ / Δ₀ ≤ (1−θ₁)(1−θ₂)⋯(1−θₜ) ≤ e^{−(θ₁+⋯+θₜ)} = e^{−Σθᵢ}.
```

Independent of n, the number of variables. The gap decays geometrically in the *sum of the entering values*. Now let me put numbers on what that buys. Suppose θ is, on average, about 1/m — the entering weight is a typical share among m basic variables. To drive the gap down by e^{−7} (which is 0.00091, under a thousandth) I need Σθᵢ ≥ 7, and at θ ≈ 1/m that's about 7m iterations. Let me sanity-check the geometric product directly rather than trust the exponential bound: fourteen steps at θ = 0.5 each (Σθ = 7) give ∏(1 − 0.5) = 0.5¹⁴ = 6.1×10⁻⁵, comfortably under 10⁻³, and indeed below the e⁻⁷ = 9.1×10⁻⁴ the bound promises. So for m = 5, roughly 7m = 35 pivots cut the gap a thousand-fold. *That's* the calculation that convinces me, back here in 1947, that this could be genuinely efficient — a handful of m-sized batches of steps, not a combinatorial crawl. (It leans on the non-degeneracy floor θ > 0; I'll have to come back to what happens when that floor fails.)

Before I trust any of this, I want to run my small example all the way to its end in the column picture and see where it lands, then check that against the answer computed a completely different way. From the basic pair {0,4} the rule brought in point 3 = (3,0). The requirement line y = 2 now has to be supported by a pair straddling it, and from the points actually on the lower boundary the straddling pair is {1,3} = (1,1) and (3,0): the line through them has slope (0−1)/(3−1) = −0.5, so at y = 2 it gives z = 1 + (−0.5)(2−1) = 0.5. A fresh pricing pass against that line finds no point below it — the remaining points 0, 2, 4 all sit at or above the lower hull — so 0.5 is where it stops. Independently: the lowest z achievable at y = 2 is just f(2), the underbelly of the hull of those five points, and the underbelly near y = 2 is exactly the segment from (1,1) to (3,0), giving f(2) = 0.5 with weights ½ on point 1 and ½ on point 3 (so that ½·1 + ½·3 = 2 lands the center of gravity on the line). The two routes agree: 0.5, weights (0, ½, 0, ½, 0). The algorithm reaches the geometric answer. That's the first thing that makes me believe the column picture isn't just a pretty story.

Now the punch that I don't see coming until I build it. I strip the convexity constraint Σxⱼ = 1 — I don't always have that leading-1 row; the general planning LP is just Ax = b, x ≥ 0 — and I work out the same column-geometry algorithm without it, and arrange in the fall of '47 to have the Bureau of Standards run it on Stigler's nutrition problem. And as I watch what it does, step by step, choosing an entering column, dropping a leaving column, moving to an adjacent basis — I look at where each basis sits in the *row* geometry, the geometry of the variables, and each one is a vertex of the feasible polyhedron, each step crossing to a vertex that differs in exactly one basic column — an adjacent vertex, along an edge. This is descent from vertex to adjacent vertex along the edges of the polyhedron. It is the *same algorithm I rejected at the start as a stupid boundary-wandering idea*. The column geometry didn't give me a new algorithm. It gave me a new pair of eyes on the old one — and through those eyes the same edge-walk is fast for a concrete reason I can now point to: the only points it ever moves to are extreme points of the underbelly, and in any cloud most points lie well above the underbelly, so those are rare. The intuition that said "wandering the outside boundary must be slow" was counting all boundary vertices; the column picture shows the walk only ever visits the few that matter.

I can double-check that the row-geometry machine and the column-geometry reasoning give the same number, because by now I have the tableau coded. On a tiny case I can solve by hand — maximize 3x₁ + 2x₂ subject to x₁ + x₂ ≤ 4, x₁ + 3x₂ ≤ 6, x ≥ 0, i.e. minimize −3x₁ − 2x₂ — the feasible vertices are (0,0) with value 0, (4,0) with 12, (0,2) with 4, and the intersection (3,1) with 11; the max is 12 at (4,0). Running the tableau code on the standard form returns x = (4, 0) and objective −12, i.e. max 12 — the right vertex, found by edge-descent. And feeding it the convexity example above (convexity row, y-row = 2, costs = the zⱼ) returns weights (0, ½, 0, ½, 0) and z = 0.5, matching the underbelly computation I did by hand. The edge-walk lands on the optima I can check independently.

So I stop hunting for an interior method. The boundary walk *is* the method; I just had to see it in the column geometry to trust it. What it's called — moving from one simplex of points to a neighboring simplex — is why Motzkin calls it "simplex," and the name fits.

Let me now nail the algebra so it actually runs on a machine, because the geometry has to become arithmetic. Drop the column picture and go back to z = c·x, Ax = b, x ≥ 0. A basis is a choice of m columns B forming an invertible matrix; the basic variables are x_B = B⁻¹b, the rest are zero. I keep everything in a *tableau*: the constraint rows reduced so the basic columns form an identity, an extra row holding the reduced costs, and the basis recorded as which column sits in each row.

The reduced cost of column j is c̄ⱼ = cⱼ − c_B·(B⁻¹Aⱼ) = cⱼ − π·Aⱼ where π = c_B B⁻¹ is the vector of dual prices — the same π·Aⱼ + π₀ comparison from the plane, now as "pricing out" column j. The sign has to be right: with the current basis fixed, every feasible x can be priced as

```
c·x = c_B B⁻¹b + Σⱼ c̄ⱼ xⱼ.
```

The first term is the value of the current basic solution. If every c̄ⱼ ≥ 0, the second term is non-negative for every feasible x ≥ 0, so no feasible point can have smaller objective value. That is the global optimality certificate, not just a local warning that the adjacent edges look flat. If some c̄ⱼ < 0, increasing that non-basic variable lowers z at first. Entering rule: pick the column with the *most negative* reduced cost — steepest decrease in z per unit of the entering variable, my argmin-most-below-the-plane. That's the pricing step, and it's cheap because the columns Aⱼ are sparse — four or five non-zeros each — so π·Aⱼ is a tiny dot product.

Leaving rule — the ratio test, which is "first basic weight to hit zero" turned into arithmetic. I raise the entering variable from zero; each basic variable changes as x_B − t·(B⁻¹A_enter) where t is the entering value. The first basic variable to reach zero is the one with the smallest ratio of its current value to its *positive* coefficient in the entering column:

```
t* = min over rows i with a_i > 0 of  b_i / a_i,
```

and that row's basic variable leaves. Only positive coefficients can block me — a zero or negative coefficient means that basic variable doesn't decrease as the entering variable rises, so it imposes no limit. If *no* coefficient is positive, nothing ever blocks: I can push the entering variable to infinity, z drops without bound, the problem is unbounded. That's the unboundedness flag, and it falls right out of the ratio test having no eligible row.

Then pivot: with pivot row r and entering column q, divide row r by the pivot entry to make it 1, and subtract multiples of it from every other row — *including the objective row* — to clear column q to zero elsewhere. That's exactly Gaussian elimination on the tableau, and it carries the whole thing to the adjacent basic feasible solution: the new basis has q where r used to be, the new x_B reads straight off the b-column, and the new reduced costs are sitting in the objective row ready for the next pricing pass. `take_step` is one pivot.

Two more things or it won't run on real problems. First, where do I start? I need *some* basic feasible solution. If every constraint was a ≤ inequality with a non-negative right-hand side, the slack variables I added form a ready-made non-negative basis — start there. But equality and ≥ rows have no such free slack, and a general equality-form routine should not assume the caller has handed me a basis. So I run a Phase 1 uniformly: add an artificial variable to each row, minimize the *sum of the artificials*; if I can drive it to zero, any remaining artificial basic rows are either pivoted out or recognized as redundant, and I'm sitting on a genuine basic feasible solution of the original problem, which I hand to Phase 2 to optimize the real objective. If the minimum sum of artificials is positive, the original constraints are infeasible — no plan exists.

Second, the thing Koopmans makes me confront. I assumed non-degeneracy — that no basic variable is ever zero — when I argued convergence, and I figured degeneracy has probability zero: what are the odds of four planes in three-space meeting exactly at a point? But practical Air Force programs keep producing zero basic variables. Degeneracy is not a curiosity; it is routine. Degeneracy means the ratio test can give t* = 0 — a pivot that swaps the basis but doesn't move the vertex or lower z — and a sequence of those can *cycle*, returning to a basis it already visited and looping forever. The proof-level fix is to perturb the right-hand side infinitesimally, or keep lexicographic bookkeeping that behaves as if such a perturbation had been made, so tied zero steps are ordered consistently and the basis cannot repeat. In code I also want a Bland-style smallest-index option, because a full tableau routine needs a deterministic anti-cycling path when degeneracy appears.

Let me write it the way it actually runs — a full tableau, an artificial-variable first phase for feasibility, pricing for the entering column, ratio test for the leaving row, pivot, repeat.

```python
import numpy as np

def to_standard_form(A_ub, b_ub, c):
    """Convert A_ub @ x <= b_ub, x >= 0 into A @ x == b, x >= 0."""
    c = np.asarray(c, dtype=float).reshape(-1)
    A_ub = np.asarray(A_ub, dtype=float)
    b = np.asarray(b_ub, dtype=float).reshape(-1)
    if A_ub.ndim != 2:
        raise ValueError("A_ub must be a two-dimensional array")
    m, n = A_ub.shape
    if c.size != n or b.size != m:
        raise ValueError("incompatible dimensions for c, A_ub, and b_ub")

    A = np.hstack((A_ub, np.eye(m)))
    c_ext = np.concatenate((c, np.zeros(m)))
    negative = b < 0
    A[negative, :] *= -1
    b[negative] *= -1
    return c_ext, A, b


def choose_improving_column(tableau, tol=1e-9, smallest_index_ties=False):
    """Return an entering column, or None when all reduced costs are non-negative."""
    objective = tableau[-1, :-1]
    candidates = np.where(objective < -tol)[0]
    if candidates.size == 0:
        return None
    if smallest_index_ties:
        return int(candidates[0])
    return int(candidates[np.argmin(objective[candidates])])


def choose_blocking_row(tableau, basis, column, n_rows,
                        tol=1e-9, smallest_index_ties=False):
    """Minimum-ratio test over positive pivot-column entries."""
    pivot_column = tableau[:n_rows, column]
    rhs = tableau[:n_rows, -1]
    rows = np.where(pivot_column > tol)[0]
    if rows.size == 0:
        return None
    ratios = rhs[rows] / pivot_column[rows]
    best = ratios.min()
    tied = rows[ratios <= best + tol]
    if smallest_index_ties:
        basis = np.asarray(basis)
        return int(tied[np.argmin(basis[tied])])
    return int(tied[0])


def apply_row_operation(tableau, basis, row, column):
    """Gaussian elimination pivot: make the pivot 1 and clear its column."""
    tableau[row, :] = tableau[row, :] / tableau[row, column]
    for other in range(tableau.shape[0]):
        if other != row:
            tableau[other, :] -= tableau[other, column] * tableau[row, :]
    basis[row] = column


def solve_tableau(tableau, basis, n_rows, tol=1e-9,
                  smallest_index_ties=False, maxiter=1000, nit0=0):
    """Run tableau pivots against the last row as the active objective."""
    nit = nit0
    while True:
        column = choose_improving_column(tableau, tol, smallest_index_ties)
        if column is None:
            return "optimal", nit

        row = choose_blocking_row(
            tableau, basis, column, n_rows, tol, smallest_index_ties
        )
        if row is None:
            return "unbounded", nit
        if nit >= maxiter:
            return "iteration_limit", nit

        apply_row_operation(tableau, basis, row, column)
        nit += 1


def solve_linear_program(c, A, b, tol=1e-9, maxiter=1000,
                         smallest_index_ties=False):
    """Minimize c @ x subject to A @ x == b and x >= 0.

    Returns (x, objective_value, status), where status is one of
    "optimal", "infeasible", "unbounded", or "iteration_limit".
    """
    c = np.asarray(c, dtype=float).reshape(-1)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    if A.ndim != 2:
        raise ValueError("A must be a two-dimensional array")
    m, n = A.shape
    if c.size != n or b.size != m:
        raise ValueError("incompatible dimensions for c, A, and b")

    # Standard-form rows need b >= 0 so artificial variables start feasible.
    A = A.copy()
    b = b.copy()
    negative = b < 0
    A[negative, :] *= -1
    b[negative] *= -1

    # Phase 1 tableau: constraints, true objective, and feasibility objective.
    tableau = np.zeros((m + 2, n + m + 1))
    tableau[:m, :n] = A
    tableau[:m, n:n + m] = np.eye(m)
    tableau[:m, -1] = b
    tableau[m, :n] = c
    basis = list(range(n, n + m))
    tableau[-1, :] = -tableau[:m, :].sum(axis=0)
    tableau[-1, n:n + m] = 0

    status, nit = solve_tableau(
        tableau, basis, m, tol, smallest_index_ties, maxiter
    )
    if status == "iteration_limit":
        return None, None, status
    if abs(tableau[-1, -1]) > tol:
        return None, np.inf, "infeasible"

    # Remove artificial columns; pivot any zero-valued artificial basic out first.
    keep_rows = []
    for row in range(m):
        if basis[row] >= n:
            choices = np.where(np.abs(tableau[row, :n]) > tol)[0]
            if choices.size:
                apply_row_operation(tableau, basis, row, int(choices[0]))
                keep_rows.append(row)
        else:
            keep_rows.append(row)

    phase2 = np.vstack((tableau[keep_rows, :], tableau[[m], :]))
    phase2 = np.delete(phase2, np.s_[n:n + m], axis=1)
    basis = [basis[row] for row in keep_rows]

    status, nit = solve_tableau(
        phase2, basis, len(keep_rows), tol, smallest_index_ties, maxiter, nit
    )
    if status == "iteration_limit":
        return None, None, status
    if status == "unbounded":
        return None, -np.inf, "unbounded"

    x = np.zeros(n)
    for row, variable in enumerate(basis):
        value = phase2[row, -1]
        x[variable] = 0.0 if abs(value) < tol else value
    return x, float(c @ x), "optimal"
```

The causal chain, start to finish: I need to mechanize military program planning, so I generalize Leontief's input-output model into a dynamic, many-activity linear system — and the genuinely new step is pulling the goal out of the ground rules and stating it as an explicit *linear objective*, because comparing plans by enumeration is hopeless (70! > 10^100). The optimum of a linear objective over the constraint polyhedron sits at a vertex, i.e. a basic feasible solution, so I only have to search vertices; the obvious way is to walk edge to edge improving the objective, but I reject that on sight as a slow boundary-wander and go looking for an interior method. The column geometry from my thesis — weighting points so their center of gravity sits on the requirement line at lowest height — recasts the problem so the same walk becomes a short hop across the rare extreme points of the convex hull's underbelly, with a clean e^{−Σθ} convergence bound saying about m-sized batches of steps suffice; and when I strip the convexity constraint and run it on the diet problem I see that this "new" column-geometry algorithm *is* the edge-descent I'd discarded, now visibly efficient. Made arithmetic, it's the tableau simplex: price out columns to pick the most-negative reduced cost as the entering variable, ratio-test to find the first basic variable forced to zero as the leaving one, pivot by Gaussian elimination to the adjacent vertex, use an artificial-variable first phase when no starting basis is available, and guard degenerate zero steps with consistent tie handling or perturbation.
