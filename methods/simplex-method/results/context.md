# Context: optimizing a linear objective under linear constraints

## Research question

It is 1947. Large enterprises — above all the U.S. Air Force, whose wartime "programs" are time-staged schedules of training, logistical supply, and deployment — need to *plan*: to choose levels for thousands of interdependent activities so that the resources balance. The planning is done by hand on desk calculators, and the way a "best" plan is selected is by a thick book of ad-hoc ground rules and edicts handed down by those in authority. There is no explicit statement of what makes one plan better than another; there is only a procedure for producing *some* feasible plan.

The question is whether this can be **mechanized** — whether the relations a planner juggles can be written as a mathematical model, an explicit notion of "best" can be attached, and a *computable* procedure can find the best plan rather than merely a feasible one. A toy version — assign 70 men to 70 jobs, each man to exactly one job and each job to exactly one man — already has 70! ≈ 10^100 feasible assignments. Exhaustive comparison is not viable for realistic planning problems that reach hundreds or thousands of activities.

## Background

**Planning as "programming."** In military usage a "program" is a proposed schedule of activities over time. A planner who has spent five wartime years building such programs on desk calculators knows the relations involved intimately: each activity consumes some items and produces others; supplies of items must balance demands; activities run at non-negative levels and over successive time stages. These relations are essentially linear, there are hundreds of items and activities, and the problem is dynamic (it changes stage to stage).

**Leontief's interindustry input-output model (1936).** Wassily Leontief proposed a simple matrix structure for an economy: one production process per commodity, each process consuming commodities as inputs to make its one output, with a system of linear equations — one per commodity — expressing that total output equals total use across the economy. It is simple in concept and implementable in enough detail to be useful.

**Linear inequalities are a thin, neglected theory.** By 1947 the theory of linear *equations* — linear algebra, approximation — has an enormous literature, while linear *inequalities* have perhaps forty papers in total (Motzkin's 1936 thesis lists 42 before 1936, names like Stokes, Dines, McCoy, Farkas). The folk belief is that inequality systems are impractical to solve unless they have three or fewer variables. The tools that do exist:

- **Fourier–Motzkin elimination** (Fourier, 1826/27; rediscovered by Motzkin, 1936): to test feasibility of a system of linear inequalities, eliminate one variable at a time by pairing each inequality that bounds it from above with each that bounds it from below, producing a smaller system on the remaining variables — geometrically, projecting the feasible polyhedron onto a lower-dimensional subspace. It decides solvability and can even yield duality.
- **Special-case optimization papers**, each isolated and without influence on the others: least-sum-of-absolute-deviations and minimize-the-maximum-deviation fits, and the transportation problem (Hitchcock, 1941). Several of these, going back to Fourier (1824) and de la Vallée Poussin (1911), propose solving by *descending along the outside edges* of the feasible polyhedron from corner to corner, yet each sparked essentially zero interest and they were unknown to working planners.

**von Neumann's game theory.** In 1928 von Neumann proved the Minimax Theorem for zero-sum two-person games (every such game has a value, attained by mixed strategies), and with Morgenstern produced the *Theory of Games and Economic Behavior* (1944). Bound up in this is Farkas' Lemma — a theorem of the alternative for linear inequalities — and a dual structure pairing a maximization with a minimization.

**The convex geometry of linear constraints.** The set of points satisfying a finite system of linear inequalities (and equalities) is a convex polyhedron. A linear objective is constant along hyperplanes, so an optimum, when finite and attained, lies on an exposed face. In standard form, after redundant equality rows are removed, the non-negativity constraints make the feasible polyhedron pointed; any nonempty optimal face contains at least one **vertex** (extreme point). Vertices correspond algebraically to **basic** solutions: choose `m` linearly independent columns of `A` as a basis, set the other variables to zero, and solve the square system for the basic variables; if the result is non-negative it is a *basic feasible solution* and geometrically a vertex. Conversely, every vertex has at least one such basis, with degeneracy allowing more than one basis for the same vertex. This correspondence — vertices ↔ basic feasible solutions — is the bridge between the geometry and the algebra.

## Baselines

A method for this problem would be measured against the alternatives a 1947 planner actually had:

- **Ground rules / human judgment.** The status quo. Produces a feasible program by applying ad-hoc edicts and ground rules passed down by those in authority.
- **Brute-force enumeration.** Conceptually find the best plan by listing all feasible plans and comparing.
- **Fourier–Motzkin elimination.** A genuine algorithm on linear inequalities that decides feasibility by projecting the polyhedron onto lower-dimensional subspaces.
- **Leontief input-output computation.** Solve the linear *equation* system for the equilibrium activity levels, yielding a unique solution of a determined system.
- **Edge-descent on special cases** (Fourier; de la Vallée Poussin; Hitchcock's transportation method). The geometric idea of walking corner to corner along the polyhedron's outside edges, improving the objective — stated only for narrow special cases and never as a general method.

## Evaluation settings

The natural yardsticks at the time are small but real linear programs solvable on the emerging digital computers and on desk calculators. The canonical test is Stigler's **nutrition (diet) problem** — choose quantities of foods to meet a set of nutritional requirements at least cost — a linear program with a handful of nutrient constraints and dozens of food activities, which the National Bureau of Standards (with Pentagon funding, under Curtiss, Hoffman, Motzkin) was equipped to run on early computing machinery. More broadly, the **Air Force time-staged deployment/training/supply programs** (the staircase-structured dynamic models) and the **transportation / assignment problems** are the application classes a general method must handle. The metric of interest is the number of iterations (steps from vertex to vertex) needed to reach the optimum, and whether that count stays small as the number of variables grows into the hundreds and thousands.

## Code framework

What already exists is dense linear algebra on a matrix of numbers: building a system from coefficients, and Gaussian elimination / row operations to solve a square system. Around that, a scaffold for "find the best feasible plan" needs a way to state the model and an empty slot for whatever procedure carries the stated objective to its optimum.

```python
import numpy as np

# --- stating the model: minimize c·x subject to A x = b, x >= 0 ---
# inequalities are put in equality form by adding non-negative slack
# variables, so the model is always A x = b with x >= 0.

def to_standard_form(A_ub, b_ub, c):
    """Add slack variables so each '<=' row becomes an equation.
    Returns the extended cost vector, A with slack columns, and b."""
    pass  # TODO: append identity columns for slacks; flip signs so b >= 0


def solve_linear_program(c, A, b, tol=1e-9, maxiter=1000):
    """Find the best non-negative solution of A x = b for the cost vector c."""
    pass  # TODO: an as-yet-undetermined procedure for reaching the optimum
```

The empty slot is the part ordinary row operations do not supply by themselves: a procedure that finds the best feasible plan rather than merely solving a square system.
