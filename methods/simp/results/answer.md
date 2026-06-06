# SIMP — density-based topology optimization for minimum compliance

## Problem

Distribute a fixed amount of material in a design domain to make a linearly elastic structure as stiff as possible (minimum compliance) under given loads and supports. The faithful design variable is a per-point solid/void indicator — an intractable 0/1 program. SIMP (Solid Isotropic Material with Penalization) makes it tractable by relaxing the indicator to a continuous element density and then *penalizing* intermediate densities through the material law, so the optimizer is driven back to solid/void.

## Key idea

1. **Relax** the 0/1 indicator to a density ρ_e ∈ [0,1] per element (gives gradients).
2. **Penalize grey via a power law.** Interpolate the element Young's modulus as
   E(ρ_e) = Emin + ρ_e^p (E0 − Emin),  Emin ≈ 10^{-9} E0,  p > 1.
   Volume scales linearly with ρ but stiffness scales as ρ^p < ρ, so intermediate density costs its full share of material yet returns less than its share of stiffness — uneconomical, hence the optimizer pushes ρ → 0 or 1. Emin keeps K non-singular while allowing the density variable to reach exactly 0; for p > 1 the local derivative at exactly ρ = 0 is still zero.
3. **Choose p physically.** Require the power law to lie on/below the Hashin–Shtrikman upper bound for an isotropic two-phase (solid/void) composite, so the fictitious intermediate material does not claim impossible stiffness and can be represented by a composite. In 2-D this gives p ≥ max{4/(1+ν), 2/(1−ν)}; for ν = 1/3 both terms equal 3, so p = 3. The compact benchmark code uses ν = 0.3 in the Q4 element matrix while keeping the standard `penal=3.0`.

## Final formulation

min_ρ  c(ρ) = U^T K U = Σ_e E(ρ_e) u_e^T k0 u_e
s.t.   K(ρ) U = F,   V(ρ)/V0 = f,   0 ≤ ρ_e ≤ 1,

with k0 the unit-modulus element stiffness and K(ρ) = Σ_e E(ρ_e) k0.

**Self-adjoint sensitivity.** Compliance is self-adjoint (adjoint field = U), so with F independent of ρ:
  ∂c/∂ρ_e = − p ρ_e^{p−1} (E0 − Emin) u_e^T k0 u_e ≤ 0,   ∂V/∂ρ_e = 1.
The sensitivity is non-positive (more material never hurts), so the volume constraint is active in the compliance problem.

**Optimality-criteria (OC) update.** From the KKT condition the optimal interior element has
  B_e = (−∂c/∂ρ_e) / (λ ∂V/∂ρ_e) = 1.
Fixed-point update with damping η = 1/2, move limit m, and box clipping:
  ρ_e^new = max(0, ρ_e − m)   if ρ_e B_e^η ≤ max(0, ρ_e − m),
            min(1, ρ_e + m)   if ρ_e B_e^η ≥ min(1, ρ_e + m),
            ρ_e B_e^η          otherwise.
The Lagrange multiplier λ (the material "price") is found by **bisection**: total volume is monotone decreasing in λ. (For multiple/non-monotone constraints, the general-purpose MMA replaces OC.)

**Regularization (mandatory).** Relaxation + penalization reinstates checkerboarding (a Q4 FE artifact) and mesh-dependence (no intrinsic length scale). Impose a length scale r_min by filtering with conical weights H_ei = max(0, r_min − dist(e,i)):
- *Sensitivity filter*:  (∂c/∂ρ_e)_filt = [Σ_i H_ei ρ_i ∂c/∂ρ_i] / [max(γ,ρ_e) Σ_i H_ei].
- *Density filter*:  physical density ρ̃_e = (Σ_i H_ei ρ_i)/(Σ_i H_ei); analyze with ρ̃; chain-rule the sensitivities: ∂ψ/∂ρ_j = Σ_e (∂ψ/∂ρ̃_e) H_je / (Σ_i H_ei), for ψ = c and V.

Optional: ramp p from 1 → 3 (continuation) to avoid poor local minima for strong penalization.

## Code

Compact NumPy/SciPy implementation following the 88-line MATLAB / 165-line Python structure. `ft=0` sensitivity filter, `ft=1` density filter.

```python
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

def lk(nu=0.3):
    # unit-modulus bilinear Q4 plane-stress element stiffness (same for every element)
    k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8,
                  -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
    return 1/(1-nu**2)*np.array([
        [k[0],k[1],k[2],k[3],k[4],k[5],k[6],k[7]],
        [k[1],k[0],k[7],k[6],k[5],k[4],k[3],k[2]],
        [k[2],k[7],k[0],k[5],k[6],k[3],k[4],k[1]],
        [k[3],k[6],k[5],k[0],k[7],k[2],k[1],k[4]],
        [k[4],k[5],k[6],k[7],k[0],k[1],k[2],k[3]],
        [k[5],k[4],k[3],k[2],k[1],k[0],k[7],k[6]],
        [k[6],k[3],k[4],k[1],k[2],k[7],k[0],k[5]],
        [k[7],k[2],k[1],k[4],k[3],k[6],k[5],k[0]]])

def oc(nelx, nely, x, volfrac, dc, dv, g):
    # bisection on the Lagrange multiplier; volume is monotone decreasing in lmid
    l1, l2, move = 0.0, 1e9, 0.2
    xnew = np.zeros(nelx*nely)
    while (l2-l1)/(l1+l2) > 1e-3:
        lmid = 0.5*(l2+l1)
        # x * sqrt(B_e), eta = 1/2, B_e = -dc/(lmid*dv); clip to move limit and [0,1]
        xnew[:] = np.maximum(0.0, np.maximum(x-move,
                  np.minimum(1.0, np.minimum(x+move, x*np.sqrt(-dc/dv/lmid)))))
        gt = g + np.sum(dv*(xnew-x))
        l1, l2 = (lmid, l2) if gt > 0 else (l1, lmid)
    return xnew, gt

def main(nelx, nely, volfrac, penal, rmin, ft):
    Emin, Emax = 1e-9, 1.0
    ndof = 2*(nelx+1)*(nely+1)
    x = volfrac*np.ones(nely*nelx); xPhys = x.copy(); g = 0
    KE = lk()
    # element -> global dof map
    edofMat = np.zeros((nelx*nely, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx*nely
            n1 = (nely+1)*elx + ely; n2 = (nely+1)*(elx+1) + ely
            edofMat[el, :] = [2*n1+2,2*n1+3, 2*n2+2,2*n2+3, 2*n2,2*n2+1, 2*n1,2*n1+1]
    iK = np.kron(edofMat, np.ones((8,1))).flatten()
    jK = np.kron(edofMat, np.ones((1,8))).flatten()
    # filter weights, assembled once
    nfilter = int(nelx*nely*((2*(np.ceil(rmin)-1)+1)**2))
    iH = np.zeros(nfilter); jH = np.zeros(nfilter); sH = np.zeros(nfilter); cc = 0
    for i in range(nelx):
        for j in range(nely):
            row = i*nely + j
            for k in range(int(max(i-(np.ceil(rmin)-1),0)), int(min(i+np.ceil(rmin),nelx))):
                for l in range(int(max(j-(np.ceil(rmin)-1),0)), int(min(j+np.ceil(rmin),nely))):
                    col = k*nely + l
                    iH[cc], jH[cc] = row, col
                    sH[cc] = max(0.0, rmin - np.sqrt((i-k)**2 + (j-l)**2)); cc += 1
    H = coo_matrix((sH, (iH, jH)), shape=(nelx*nely, nelx*nely)).tocsc()
    Hs = H.sum(1)
    # supports + load: half MBB beam
    dofs = np.arange(ndof)
    fixed = np.union1d(dofs[0:2*(nely+1):2], np.array([ndof-1]))
    free = np.setdiff1d(dofs, fixed)
    f = np.zeros((ndof,1)); u = np.zeros((ndof,1)); f[1,0] = -1
    change, loop = 1.0, 0
    dv = np.ones(nely*nelx); dc = np.ones(nely*nelx); ce = np.ones(nely*nelx)
    while change > 0.01 and loop < 2000:
        loop += 1
        # SIMP-interpolated stiffness, assemble and solve
        sK = ((KE.flatten()[None]).T*(Emin + xPhys**penal*(Emax-Emin))).flatten(order='F')
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        u[free,0] = spsolve(K[free,:][:,free], f[free,0])
        # compliance + self-adjoint sensitivity
        ce[:] = (np.dot(u[edofMat].reshape(nelx*nely,8), KE)
                 * u[edofMat].reshape(nelx*nely,8)).sum(1)
        obj = ((Emin + xPhys**penal*(Emax-Emin))*ce).sum()
        dc[:] = (-penal*xPhys**(penal-1)*(Emax-Emin))*ce
        dv[:] = np.ones(nely*nelx)
        # filtering
        if ft == 0:
            dc[:] = np.asarray((H*(x*dc))[None].T/Hs)[:,0] / np.maximum(0.001, x)
        elif ft == 1:
            dc[:] = np.asarray(H*(dc[None].T/Hs))[:,0]
            dv[:] = np.asarray(H*(dv[None].T/Hs))[:,0]
        # OC update + physical densities
        xold = x.copy()
        x, g = oc(nelx, nely, x, volfrac, dc, dv, g)
        xPhys = x.copy() if ft == 0 else np.asarray(H*x[None].T/Hs)[:,0]
        change = np.linalg.norm(x - xold, np.inf)

if __name__ == "__main__":
    main(nelx=180, nely=60, volfrac=0.4, penal=3.0, rmin=5.4, ft=1)
```

## One-line summary

Relax solid/void to a continuous density, make stiffness a penalized power of density (E = Emin + ρ^p(E0−Emin), with p=3 at ν=1/3 from the Hashin–Shtrikman condition) so grey becomes uneconomical, minimize the self-adjoint compliance with a damped optimality-criteria update whose Lagrange multiplier is bisected against the volume budget, and filter (sensitivities or densities) over a radius r_min to kill checkerboards and make the result mesh-independent.
