Let me start from what I actually want and what's blocking me.

I have a rectangular chunk of space, some supports, and a load pushing on it. I get to decide, point by point, whether there's material there or a hole. I want the stiffest possible structure for a fixed amount of material — equivalently, the smallest compliance, the least the thing deflects under the load. That's it. The honest variable is: at each point, solid or void. One bit per point.

Discretize the domain into a grid of little square finite elements, N of them, and the honest variable is a vector of N bits: element e is either full (1) or empty (0). And immediately I'm in trouble. N is in the thousands, easily tens or hundreds of thousands. A 0/1 program in that many variables, with the elasticity equilibrium KU = F as a coupling constraint that I have to re-solve every time I touch a variable — there is no hope of branch-and-bound or any combinatorial search here. The number of layouts is 2^N. I need gradients, and bits don't have gradients.

So the first move almost forces itself: relax the bit to a number. Let each element carry a density ρ_e ∈ [0,1], with ρ_e = 1 meaning solid, ρ_e = 0 meaning void, and everything in between allowed *for now*. The combinatorial problem becomes a smooth nonlinear program: minimize compliance over a box [0,1]^N subject to the volume budget and the equilibrium equation. Now I can differentiate, now I can do gradient descent / KKT. Good.

But I've bought tractability with a debt I have to pay back. The relaxed solution will generally be *grey* — densities sitting at 0.4, 0.6, all over the domain — and a half-dense element is not a structure I can build. "Topology" means I eventually need the densities to settle at 0 or 1. Relaxing was necessary; now I need a force that pushes the relaxed answer back to the corners of the box.

Let me see why the naive relaxation gives grey, because the reason tells me how to fix it. I have to say how a partially-dense element behaves mechanically — what stiffness does ρ_e = 0.5 give? The obvious choice is linear: the effective Young's modulus is E(ρ) = ρ E0, so a half-dense element is half as stiff. Assemble K(ρ) = Σ_e ρ_e k0, where k0 is the unit-modulus element stiffness matrix. Clean, differentiable.

Now stare at the economics the optimizer sees. The material budget I spend on element e is proportional to ρ_e — volume is linear in density, no way around that, a half-full element uses half a unit of material. And under linear interpolation its stiffness contribution is *also* proportional to ρ_e. So the "exchange rate" — stiffness bought per unit of material spent — is the same at ρ = 0.5 as at ρ = 1. Grey is exactly as economical as black. The optimizer has zero incentive to commit to solid or void; it will happily smear material around as a continuous grey field. And worse, that grey isn't even cheating physically: linear interpolation is essentially what the homogenization people found — intermediate density *is* achievable by a fine composite microstructure (rank laminates and the like reach the lower edge of what composites can do), so grey is a genuine, physical, optimal answer to the relaxed problem. The relaxation didn't approximate the 0/1 problem; it dissolved it.

So I need to change the exchange rate. I want intermediate density to be a *bad deal* — to cost its full share of material but deliver less than its share of stiffness. The volume term I can't touch; volume is honestly linear in ρ. So I have to bend the *stiffness* law to grow more slowly than linearly at low density. The simplest knob: a power.

E(ρ) = ρ^p E0,  with p > 1.

Look at what this does. Volume per element still ∝ ρ. Stiffness per element now ∝ ρ^p, and for p > 1 and ρ ∈ (0,1), ρ^p < ρ. A half-dense element costs 1/2 of the budget but yields only (1/2)^p of the stiffness — for p = 3 that's 1/8. Intermediate density is now strictly uneconomical: any grey element would do better to either fully commit (get the full stiffness for the material) or fully evacuate (free the material for somewhere it pays off). The optimizer, chasing the best stiffness-per-volume, is pushed toward ρ ∈ {0,1}. The beauty of it is that there's no explicit penalty term sitting in the objective with a weight to tune — the penalization is folded entirely into the interpolation law. The objective is still just compliance. This is the whole idea: solid-isotropic-material with the penalization built into the density-stiffness map.

Now the question that decides whether this is a principled method or a hack: *which p?* If I just crank p up, I do get sharper 0/1 designs — but am I lying about physics? The intermediate-density material is fictitious, so at minimum it must not claim stiffness that no void/solid composite at the same density could have. If my law sits within the composite bounds, then a grey element can be read as a deliberately inefficient isotropic composite; if my law rises above the bound, then the optimizer is being offered a material that physics does not allow.

So I should compare ρ^p E0 against the best stiffness any isotropic two-phase mixture of solid (E0, Poisson ratio ν) and void can actually reach at volume fraction ρ. That best-possible stiffness is what the Hashin–Shtrikman upper bound gives — the tightest variational upper bound on the effective isotropic modulus at that density. The condition I want is: the power-law stiffness ρ^p E0 should sit on or below the H-S upper bound for every ρ ∈ [0,1]. Then the fictitious isotropic material is not claiming impossible stiffness, and material-design arguments can realize the modeled intermediate stiffness with a void/solid composite. The interpolation is still intentionally unattractive for a macro-scale layout, because volume is paid linearly while stiffness is discounted by ρ^p.

Working out where ρ^p crosses the H-S upper bound (expressing the bound through the Young's modulus for an isotropic base material) gives, in two dimensions, a lower limit on the power:

p ≥ max{ 4/(1+ν),  2/(1−ν) }.

Plug in the usual ν = 1/3: 4/(1+1/3) = 4/(4/3) = 3, and 2/(1−1/3) = 2/(2/3) = 3. Both terms land on exactly 3. So p = 3, with ν = 1/3, is the *smallest* exponent that makes the power law consistent with the physics of composites — small enough to avoid unnecessary extra nonconvexity, large enough that intermediate densities are penalized without claiming impossible stiffness. That's why p = 3 is the default, and it's not a fudge: it's a necessary condition for the interpolated material to be representable by a real composite. If I went below 3 the law would float above the H-S bound at intermediate densities (claiming impossible stiffness), and if I went well above 3 the penalization gets stronger and the problem more nonconvex — more local minima — which I'd have to manage by slowly ramping p up from 1 toward 3 (a continuation), but for a single solve p = 3 is the principled choice. The small educational MBB code often uses ν = 0.3 in the element matrix while keeping penal = 3; the exact equality I just used is the ν = 1/3 composite-bound calculation.

One more thing about the interpolation before I move on. If I literally use E(ρ) = ρ^p E0, then a void element (ρ = 0) has *exactly zero* stiffness, and the global K develops a zero block — it goes singular, so the FE solve can break. There is also a separate locking fact: for p > 1 the derivative at exactly ρ = 0 vanishes. The stiffness singularity is the one I can fix cleanly by giving void a tiny but nonzero stiffness:

E(ρ) = Emin + ρ^p (E0 − Emin),  Emin ≈ 10^{-9} E0.

Now ρ = 0 gives Emin, not 0, so K stays non-singular and the density variable is allowed to reach exactly zero instead of being artificially bounded below. I have to be careful about the locking fact: pρ^{p−1}(E0−Emin) is still zero at exactly ρ = 0, so Emin does not give a dead element a local gradient. The practical loop starts from a positive uniform density and moves through finite changes; Emin's job is the finite-element solve and the clean zero-density limit. At ρ=1 the element still has E0, so the solid limit is unchanged.

Now the actual optimization. Fix the discretization: square bilinear Q4 elements, two displacement degrees of freedom per node. The unit-modulus element stiffness k0 for plane stress with ν = 0.3 is a fixed 8×8 matrix — and because the mesh is regular, it's the *same* matrix for every element, so I compute it once. The global stiffness is

K(ρ) = Σ_e (Emin + ρ_e^p (E0 − Emin)) k0   (placed into the element's dofs),

and the state is U solving K(ρ) U = F. Compliance is c(ρ) = F^T U = U^T K U. That last equality matters: since KU = F, c = U^T K U = U^T F = F^T U, all the same number.

To run gradient-based optimization I need dc/dρ_e. Differentiating c = U^T K U directly is annoying because U depends on ρ implicitly through KU = F, and ∂U/∂ρ_e would need a linear solve per element — N solves, unaffordable. The standard escape is the adjoint trick: append a multiple of the (identically zero) residual to the objective without changing its value. Write

c = F^T U − λ^T (K U − F),

for any vector λ I like (the added term is zero at equilibrium). Differentiate with respect to ρ_e, remembering F is independent of ρ:

dc/dρ_e = F^T ∂U/∂ρ_e − λ^T ( (∂K/∂ρ_e) U + K ∂U/∂ρ_e )
        = (F^T − λ^T K) ∂U/∂ρ_e − λ^T (∂K/∂ρ_e) U.

The ∂U/∂ρ_e term is the expensive, unknown one. I can annihilate it by choosing λ to make its coefficient vanish: F^T − λ^T K = 0, i.e. K λ = F (using K symmetric). But that's the *same* system the state solves: K U = F. So λ = U. The adjoint problem is the state problem — compliance is self-adjoint. No extra solve at all. With that choice,

dc/dρ_e = − U^T (∂K/∂ρ_e) U = − u_e^T (dE/dρ_e) k0 u_e,

where u_e is the local element displacement. And dE/dρ_e = p ρ_e^{p−1}(E0 − Emin), so

dc/dρ_e = − p ρ_e^{p−1} (E0 − Emin) u_e^T k0 u_e.

Note the sign: u_e^T k0 u_e ≥ 0 (k0 is positive semidefinite, it's a strain energy), and the leading factor is positive, so dc/dρ_e ≤ 0 always. Adding material to *any* element never increases compliance — it always helps, or at worst does nothing. That's intuitive (more stuff = stiffer) and it has a consequence: the volume constraint is active in the compliance problem. The optimizer wants as much useful material as the budget allows, and the design is decided by *where* the material goes, governed by the relative magnitudes of these non-positive sensitivities.

For the volume side, V(ρ) = Σ_e ρ_e v_e; with unit element volume, ∂V/∂ρ_e = 1.

Now I need an update rule. I could reach for a general nonlinear-programming method — Svanberg's Method of Moving Asymptotes (1987) builds a convex separable approximation with moving asymptotes each iteration and solves that subproblem; it handles many constraints and sensitivities of either sign. But here I have exactly one constraint (volume) and compliance sensitivities that all have the same non-positive sign. That's a very special, easy structure, and it admits a much cheaper purpose-built update — the optimality-criteria method. Let me derive it.

Lagrangian for min c s.t. Σ ρ_e v_e = V*, with box 0 ≤ ρ ≤ 1:

L = c + λ (Σ_e ρ_e v_e − V*).

For an element strictly interior to the box, stationarity is ∂c/∂ρ_e + λ v_e = 0, i.e.

(− ∂c/∂ρ_e) / (λ ∂V/∂ρ_e) = 1.

Define that ratio as B_e := (− ∂c/∂ρ_e) / (λ ∂V/∂ρ_e). At the optimum every interior element has B_e = 1. Read B_e as a price signal: the numerator −∂c/∂ρ_e is the stiffness *gained* by adding a unit of material to element e (positive, since the sensitivity is negative), and the denominator λ ∂V/∂ρ_e is the "cost" of that material at the going price λ. If B_e > 1, this element buys more stiffness than the material costs → push ρ_e up. If B_e < 1, it's overpriced → push ρ_e down. So a natural fixed-point iteration multiplies the current density by B_e:

ρ_e ← ρ_e · B_e^η,

with the exponent η a damping factor. Why the exponent, and why η < 1? If I used η = 1 the multiplicative step is aggressive and the iteration oscillates — densities overshoot above and below their targets and the scheme rings. Taking η = 1/2 damps each multiplicative correction (it's the geometric mean pull toward B_e = 1), which I find is what makes the fixed point converge smoothly. And I cap how far any element can move in one step with a move limit m (say 0.2): the whole update is built on the *current* sensitivities, which are only locally valid, so I clamp the change to [ρ_e − m, ρ_e + m] and then to the box [0,1]. Writing it out as the clipped update:

ρ_e^new = clip to [max(0, ρ_e − m), min(1, ρ_e + m)] of ( ρ_e (B_e)^η ),
        = max(0, ρ_e − m)        if ρ_e B_e^η ≤ max(0, ρ_e − m),
          min(1, ρ_e + m)        if ρ_e B_e^η ≥ min(1, ρ_e + m),
          ρ_e B_e^η               otherwise.

The one thing left dangling is λ, the material price — it has to be set so the *new* densities exactly use the volume budget. As I raise λ, material gets "more expensive", every B_e shrinks, every ρ_e^new shrinks, so the total volume Σ ρ_e^new is monotonically decreasing in λ. A monotone scalar equation in one unknown — solve it by bisection: bracket λ between a small and a large value (l1 = 0, l2 = 10^9), repeatedly halve, picking the side that brings the total volume to the target. Each bisection step just re-evaluates the closed-form update; no FE solve inside the bisection. Cheap.

So the loop so far: solve KU = F; compute c and the sensitivities; bisect for λ and apply the OC update; check convergence (max change in any ρ below 1%); repeat. Let me run it in my head on the MBB beam and… it doesn't behave. Two things go wrong, and they're the debt I deferred when I relaxed the integer problem.

First, the design comes out riddled with *checkerboards* — patches of alternating fully-solid and fully-empty elements, like a chessboard. They look like microstructure but they're a lie: the low-order Q4 element badly over-estimates the stiffness of a checkerboard arrangement, so the optimizer, hunting stiffness-per-volume, is thrilled to make them. They're a finite-element artifact, not good structure. And the penalization I added makes it *worse*, because penalization rewards committing each element to 0 or 1 at the finest available scale, which is exactly the checkerboard scale.

Second, *mesh-dependence*: if I refine the mesh, I don't get a sharper version of the same beam — I get a different beam, with more and thinner members. There's no intrinsic length scale in the formulation, so finer mesh = finer features, without limit. This is the old ill-posedness of the continuum 0/1 problem coming back to bite: minimizing sequences develop ever-finer perforation, and my discrete optimizer is faithfully chasing that non-existent limit.

Both pathologies are the same root cause — nothing in the problem forbids arbitrarily fine spatial variation — and so they have the same cure: impose a length scale. Forbid the design from varying faster than some radius r_min. The cleanest way to graft that on is to filter.

The first filter to try acts on the *sensitivities*. Instead of letting each element's compliance sensitivity be its raw local value, replace it with a distance-weighted average over the neighbours within r_min:

(dc/dρ_e)_filtered = [ Σ_{i∈N_e} H_ei ρ_i (dc/dρ_i) ] / [ ρ_e Σ_{i∈N_e} H_ei ],

with the conical weight H_ei = max(0, r_min − dist(e,i)) that falls linearly to zero at the filter radius. (When void densities can be exactly zero — which the modified-SIMP Emin allows — guard the ρ_e in the denominator with max(γ, ρ_e), γ ≈ 10^{-3}, to avoid dividing by zero.) Neighbouring elements now share gradient information, so a lone solid element in a sea of void no longer reads as a great deal — its neighbours drag its sensitivity toward theirs. Checkerboards are smoothed out, and because the averaging radius r_min is a fixed physical length, refining the mesh no longer refines the topology: you just resolve the same r_min-scale members more finely. It's heuristic — it modifies the gradient rather than the objective — but it works robustly, and it costs almost nothing because the weights H_ei depend only on geometry and can be assembled once.

There's a more principled variant that filters the *densities* themselves rather than the sensitivities. Introduce a "physical" density field as the filtered design field:

ρ̃_e = [ Σ_{i∈N_e} H_ei ρ_i ] / [ Σ_{i∈N_e} H_ei ],

and do *everything* — assemble K, compute compliance, compute the volume — using ρ̃ instead of the raw design variable ρ. The raw ρ_e are the design variables the optimizer moves; the ρ̃_e are what physically exists and what you should plot as the answer. Because the analysis sees ρ̃, the design can't have features finer than r_min: a single isolated design spike gets blurred into a r_min-wide blob before it ever enters the physics. This restores a genuine minimum length scale and makes the problem well-posed (a minimizer exists). The cost is a chain rule: a sensitivity with respect to the physical density must be pushed back to the design variables through the filter,

∂ψ/∂ρ_j = Σ_{e∈N_j} (∂ψ/∂ρ̃_e)(∂ρ̃_e/∂ρ_j) = Σ_{e∈N_j} (∂ψ/∂ρ̃_e) · H_je / (Σ_{i∈N_e} H_ei),

for ψ either the compliance c or the volume V. With the conical averaging filter the total volume is preserved, so I could check the volume on either ρ or ρ̃, but the chain rule still changes the design-variable volume sensitivities. I should carry dv explicitly through the OC update; with the pure sensitivity filter those sensitivities are all equal and drop out. Same conical weights, same precomputed normalization Σ H_ei; just applied to densities and chained back.

Either way, the wall is cleared: with a filter in place, the loop converges to clean, checkerboard-free, mesh-independent designs.

Let me assemble the whole thing into code, mirroring the standard compact implementation. Initialize every density to the volume fraction (a uniform feasible start), then loop: assemble K with the SIMP modulus, solve for U on the free dofs, form the elementwise strain energy and from it the compliance and the −p ρ^{p−1}(E0−Emin) u^T k0 u sensitivity, filter (sensitivities, or densities + chain rule), run the OC bisection update, measure the max density change, and stop when it's below 1%. Here is the structure, as a direct Python translation of the canonical compact MATLAB code.

```python
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

def lk(nu=0.3):
    # unit-modulus Q4 plane-stress element stiffness (one matrix, regular mesh)
    k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8,
                  -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
    KE = 1/(1-nu**2)*np.array([
        [k[0],k[1],k[2],k[3],k[4],k[5],k[6],k[7]],
        [k[1],k[0],k[7],k[6],k[5],k[4],k[3],k[2]],
        [k[2],k[7],k[0],k[5],k[6],k[3],k[4],k[1]],
        [k[3],k[6],k[5],k[0],k[7],k[2],k[1],k[4]],
        [k[4],k[5],k[6],k[7],k[0],k[1],k[2],k[3]],
        [k[5],k[4],k[3],k[2],k[1],k[0],k[7],k[6]],
        [k[6],k[3],k[4],k[1],k[2],k[7],k[0],k[5]],
        [k[7],k[2],k[1],k[4],k[3],k[6],k[5],k[0]]])
    return KE

def oc(nelx, nely, x, volfrac, dc, dv, g):
    # bisect the Lagrange multiplier; volume is monotone decreasing in lmid
    l1, l2, move = 0.0, 1e9, 0.2
    xnew = np.zeros(nelx*nely)
    while (l2-l1)/(l1+l2) > 1e-3:
        lmid = 0.5*(l2+l1)
        # rho * sqrt(B_e)  with eta = 1/2;  B_e = -dc/(lmid*dv)
        xnew[:] = np.maximum(0.0, np.maximum(x-move,
                  np.minimum(1.0, np.minimum(x+move, x*np.sqrt(-dc/dv/lmid)))))
        gt = g + np.sum(dv*(xnew-x))          # linearized volume residual
        l1, l2 = (lmid, l2) if gt > 0 else (l1, lmid)
    return xnew, gt

def main(nelx, nely, volfrac, penal, rmin, ft):
    Emin, Emax = 1e-9, 1.0
    ndof = 2*(nelx+1)*(nely+1)
    x = volfrac*np.ones(nely*nelx)
    xPhys = x.copy()
    g = 0
    KE = lk()

    # element -> global dof map
    edofMat = np.zeros((nelx*nely, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx*nely
            n1 = (nely+1)*elx + ely
            n2 = (nely+1)*(elx+1) + ely
            edofMat[el, :] = [2*n1+2,2*n1+3, 2*n2+2,2*n2+3, 2*n2,2*n2+1, 2*n1,2*n1+1]
    iK = np.kron(edofMat, np.ones((8,1))).flatten()
    jK = np.kron(edofMat, np.ones((1,8))).flatten()

    # filter weights H_ei = max(0, rmin - dist), assembled once
    nfilter = int(nelx*nely*((2*(np.ceil(rmin)-1)+1)**2))
    iH = np.zeros(nfilter); jH = np.zeros(nfilter); sH = np.zeros(nfilter); cc = 0
    for i in range(nelx):
        for j in range(nely):
            row = i*nely + j
            for k in range(int(max(i-(np.ceil(rmin)-1),0)), int(min(i+np.ceil(rmin),nelx))):
                for l in range(int(max(j-(np.ceil(rmin)-1),0)), int(min(j+np.ceil(rmin),nely))):
                    col = k*nely + l
                    iH[cc], jH[cc] = row, col
                    sH[cc] = max(0.0, rmin - np.sqrt((i-k)**2 + (j-l)**2))
                    cc += 1
    H = coo_matrix((sH, (iH, jH)), shape=(nelx*nely, nelx*nely)).tocsc()
    Hs = H.sum(1)

    # supports + load: half MBB beam (symmetry roller on left, support at bottom-right)
    dofs = np.arange(ndof)
    fixed = np.union1d(dofs[0:2*(nely+1):2], np.array([ndof-1]))
    free = np.setdiff1d(dofs, fixed)
    f = np.zeros((ndof,1)); u = np.zeros((ndof,1))
    f[1,0] = -1                                    # unit downward load, top-left

    change, loop = 1.0, 0
    dv = np.ones(nely*nelx); dc = np.ones(nely*nelx); ce = np.ones(nely*nelx)
    while change > 0.01 and loop < 2000:
        loop += 1
        # --- assemble SIMP-interpolated K and solve KU = F ---
        sK = ((KE.flatten()[None]).T*(Emin + xPhys**penal*(Emax-Emin))).flatten(order='F')
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        u[free,0] = spsolve(K[free,:][:,free], f[free,0])
        # --- compliance and self-adjoint sensitivity ---
        ce[:] = (np.dot(u[edofMat].reshape(nelx*nely,8), KE)
                 * u[edofMat].reshape(nelx*nely,8)).sum(1)        # u_e^T k0 u_e
        obj = ((Emin + xPhys**penal*(Emax-Emin))*ce).sum()        # c = sum E(rho) u^T k0 u
        dc[:] = (-penal*xPhys**(penal-1)*(Emax-Emin))*ce          # dc/drho = -p rho^{p-1}(E0-Emin) u^T k0 u
        dv[:] = np.ones(nely*nelx)
        # --- filtering: ft=0 sensitivity filter, ft=1 density filter (+ chain rule) ---
        if ft == 0:
            dc[:] = np.asarray((H*(x*dc))[None].T/Hs)[:,0] / np.maximum(0.001, x)
        elif ft == 1:
            dc[:] = np.asarray(H*(dc[None].T/Hs))[:,0]
            dv[:] = np.asarray(H*(dv[None].T/Hs))[:,0]
        # --- OC update, then map design variables to physical densities ---
        xold = x.copy()
        x, g = oc(nelx, nely, x, volfrac, dc, dv, g)
        xPhys = x.copy() if ft == 0 else np.asarray(H*x[None].T/Hs)[:,0]
        change = np.linalg.norm(x - xold, np.inf)
```

So the causal chain, start to finish: I wanted the stiffest layout under a material budget, but the honest 0/1 design is a hopeless integer program, so I relaxed densities to [0,1] to get gradients; that left grey, because linear stiffness prices intermediate density honestly and grey is a real composite; so I bent the stiffness law to a power ρ^p that pays sublinearly, making grey uneconomical and pushing the optimizer to the corners; I pinned p = 3 (with ν = 1/3) as the smallest power keeping that interpolation within the Hashin–Shtrikman composite bound; I added Emin to keep the void from making K singular while preserving the zero-density limit; compliance turned out self-adjoint so its sensitivity is a single cheap non-positive formula, which makes the volume constraint active; that special structure let me replace a general optimizer with the optimality-criteria multiplicative update, damped and move-limited, with the material price λ found by bisection; and finally, because relaxation-plus-penalization reinstated the checkerboards and mesh-dependence of the original ill-posed problem, I imposed a length scale by filtering — sensitivities or densities — to land on clean, mesh-independent, black-and-white designs.
