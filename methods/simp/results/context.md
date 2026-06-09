## Research question

Given a fixed rectangular design domain, a set of supports, and applied loads, decide at *every point* of the domain whether to place solid material or leave void, so that the resulting linearly elastic structure is as stiff as possible (minimum compliance) while using no more than a prescribed amount of material. This is *topology* design, not sizing or shape design: the number, position, and connectivity of holes are not fixed in advance — bars may appear, thicken, merge, or vanish during optimization.

What makes this hard is the nature of the design variable. The faithful description is a 0/1 indicator field — solid or void at each point. Discretized on a finite element mesh with N elements (N in the thousands to hundreds of thousands), that is an integer program in N binary variables, with the elasticity equilibrium equation as an expensive implicit constraint coupling all of them. A usable method has to (i) escape the combinatorial blow-up of the 0/1 variables, (ii) end up with designs that are essentially solid-or-void (a grey, half-dense structure cannot be manufactured and is not what "topology" means), and (iii) produce designs that are physically meaningful and stable — they must not depend on how finely the domain happens to be meshed, and they must not be dominated by numerical artifacts of the finite element discretization.

## Background

**Elasticity and compliance.** For a discretized linear elastic structure, the displacement vector U solves the equilibrium equation KU = F, where K is the global stiffness matrix (assembled from element matrices) and F is the load vector. *Compliance* c = F^T U = U^T K U is the work done by the load; minimizing it maximizes overall stiffness. The element stiffness matrix for a four-node bilinear (Q4) plane element with two displacement degrees of freedom per node is a standard 8×8 matrix; for a unit Young's modulus and Poisson ratio ν it depends only on element geometry, so on a regular square mesh a single element matrix k0 serves every element.

**The integer/continuum difficulty.** The basic continuum topology problem — choose a solid/void indicator field minimizing compliance under a volume bound — is *ill-posed*: it has no optimal solution. Minimizing sequences develop finer and finer alternations of solid and void (microstructure), and the infimum is only approached in the limit of infinitely fine perforation. On a fixed mesh this shows up as two pathologies. *Mesh-dependence*: refining the mesh does not merely sharpen a fixed design, it produces a different, finer topology with more, thinner members — there is no intrinsic length scale. *Checkerboarding*: the optimizer favours regions of alternating full/empty elements in a checkerboard pattern, because the low-order Q4 element over-estimates the stiffness of such a pattern; the checkerboard is a finite element artifact, not a good structure.

**The homogenization route and what it taught.** Bendsøe and Kikuchi (1988, "Generating optimal topologies in structural design using a homogenization method") made the problem well-posed by *enlarging* the design space: each point is allowed to be a periodic composite (e.g. a square cell with a rectangular hole), characterized by a few microstructural parameters, and the macroscopic stiffness of that composite is computed by homogenization. Optimizing over composites has a solution, and that solution is what the bare 0/1 problem was trying to approximate. The lesson and the limitation are the same fact: the optimal homogenized design is full of *grey* — regions of intermediate density realized by fine microstructure. That is physically correct but practically awkward: the result is not a clean solid/void blueprint, and the microstructural parametrization and its orientation variables are cumbersome.

**Density as a single design field.** A continuous relative-density field ρ(x) ∈ [0,1] per element, with ρ=1 solid and ρ=0 void, replaces the integer field. The combinatorial problem becomes a smooth nonlinear program with box constraints, accessible to gradient-based optimization. This is the move that makes large-scale topology optimization tractable. By itself, though, the relaxed problem need not return designs that are close to solid/void.

**Variational bounds on composites.** The Hashin–Shtrikman bounds give, for an isotropic two-phase composite (here: solid of Young's modulus E0 and void) at a given volume fraction ρ, the tightest possible upper bound on the effective isotropic stiffness achievable by *any* microstructure.

**Numerical instabilities as a recognized class.** Sigmund and Petersson (1998, "Numerical instabilities in topology optimization: a survey on procedures dealing with checkerboards, mesh-dependencies and local minima") catalogued the checkerboard, mesh-dependence, and local-minimum phenomena and the restriction methods that suppress them — perimeter control, gradient constraints, and spatial filtering — establishing that some restriction on the design's spatial variation is mandatory for a well-posed, mesh-independent problem.

## Baselines

**Homogenization-based topology optimization (Bendsøe & Kikuchi 1988).** Core idea: admit microstructured composites at every point; compute macroscopic stiffness by homogenization of a parametrized unit cell; optimize the microstructural parameters (and orientations) over the domain. Math: at each point the stiffness tensor is a known function of cell parameters; equilibrium and a volume constraint close the problem; sensitivities follow from the homogenized constitutive law. Gap it leaves: solutions are grey/microstructured rather than solid/void, the cell parametrization and orientation fields are heavy to carry, and turning the result into a manufacturable black-and-white layout needs a separate post-processing step.

**Linear density-stiffness interpolation.** The most naive way to use a continuous density: make stiffness proportional to density, E(ρ) = ρ E0. Core idea/math: K(ρ) = Σ_e ρ_e k0 assembled element-wise; sensitivities are trivial. Gap it leaves: intermediate density is not penalized by the interpolation itself — a half-dense element costs half the material and is modeled as returning half the stiffness — while homogenization already shows that grey fields can represent fine microstructure rather than mere numerical noise. The minimizer can therefore remain smeared and grey: the relaxation has lost the very 0/1 character that "topology" requires.

**General nonlinear programming for the update — the Method of Moving Asymptotes (Svanberg 1987).** Core idea: at each iteration build a convex, separable approximation of objective and constraints using moving asymptotes, and solve that subproblem; handles many constraints and arbitrary sign sensitivities. Gap it leaves for this specific problem: it is general-purpose and comparatively heavy, carrying machinery for many constraints of arbitrary sign that a single volume constraint does not need.

**Restriction methods for the instabilities (Sigmund & Petersson 1998; Sigmund 1994/1997 sensitivity filter; Bourdin 2001 and Bruns & Tortorelli 2001 density filter).** Core idea: forbid arbitrarily fine spatial variation by imposing a length scale r_min. Two families exist in prior art: sensitivity-based restriction and density-based restriction, both organized around a neighbourhood radius r_min. Gap each addresses: without such a restriction the relaxed-and-penalized problem reinstates the ill-posedness; with one, a minimum feature size is enforced and solutions converge under mesh refinement.

## Evaluation settings

The canonical test problem is the *MBB beam* (Messerschmitt–Bölkow–Blohm beam): a long simply-supported beam carrying a central point load, of which, by symmetry, only the left half is modelled. Concretely, the half-beam is a rectangular domain discretized by square Q4 elements; the left edge carries a symmetry condition (horizontal displacements fixed — a roller), the bottom-right corner carries a vertical support, and a unit downward point load is applied at the top-left corner. The free parameters of an experiment are the mesh resolution (e.g. 60×20, 150×50, 300×100 elements), the prescribed volume fraction f (e.g. 0.4–0.5), any parameters of the chosen material-interpolation law, the restriction length scale r_min (e.g. a few element widths), and its type. The metric is the converged compliance c, with the design's solid/void sharpness and its stability under mesh refinement as qualitative checks. Convergence is declared when the maximum change in any design variable between consecutive iterations drops below a small tolerance (e.g. 0.01).

## Code framework

Available pieces: a finite element solver for linear plane elasticity, a routine for the element stiffness matrix, sparse assembly of the global matrix, and an iteration loop with a plotting/printing tail. What goes inside the loop to turn the densities into a converged design is left open.

```python
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

def lk(nu=0.3):
    # 8x8 stiffness for a unit-modulus bilinear Q4 plane-stress element.
    k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8,
                  -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
    KE = 1/(1-nu**2) * np.array([
        [k[0],k[1],k[2],k[3],k[4],k[5],k[6],k[7]],
        [k[1],k[0],k[7],k[6],k[5],k[4],k[3],k[2]],
        [k[2],k[7],k[0],k[5],k[6],k[3],k[4],k[1]],
        [k[3],k[6],k[5],k[0],k[7],k[2],k[1],k[4]],
        [k[4],k[5],k[6],k[7],k[0],k[1],k[2],k[3]],
        [k[5],k[4],k[3],k[2],k[1],k[0],k[7],k[6]],
        [k[6],k[3],k[4],k[1],k[2],k[7],k[0],k[5]],
        [k[7],k[2],k[1],k[4],k[3],k[6],k[5],k[0]]])
    return KE

def main(nelx, nely, volfrac):
    KE = lk()
    ndof = 2*(nelx+1)*(nely+1)
    # ... build edofMat, iK, jK for sparse assembly ...
    # ... build load vector f and fixed/free degree-of-freedom sets ...
    x = volfrac * np.ones(nelx * nely)
    u = np.zeros((ndof, 1))
    change = 1.0
    while change > 0.01:
        # TODO: given the current densities, solve KU = F and advance the design
        #       one iteration toward a feasible, converged layout.
        change = ...
        # print / plot density field
```
