We are given a rectangular design domain, a set of supports, and applied loads, and we have to decide at every point whether to place solid material or leave void, so that the resulting linearly elastic structure is as stiff as possible — minimum compliance — while using no more than a prescribed amount of material. The honest design variable is a 0/1 indicator field: solid or void at each point. Discretized on a finite element mesh with $N$ elements, that is an integer program in $N$ binary variables, with the elasticity equilibrium $K U = F$ as an expensive implicit constraint coupling all of them; with $N$ in the thousands to hundreds of thousands and a space of $2^N$ layouts, there is no hope of branch-and-bound. Bits have no gradients, so we must relax. But relaxation only trades one problem for another. The homogenization route of Bendsøe and Kikuchi makes the continuum problem well-posed by admitting microstructured composites at every point and computing their macroscopic stiffness, but its solutions are grey — fields of intermediate density realized by fine microstructure — and the cell parametrization and orientation fields are heavy to carry and need a separate step to become a manufacturable black-and-white blueprint. The naive remedy, a continuous density $\rho_e \in [0,1]$ with stiffness linear in density, $E(\rho) = \rho E_0$, is differentiable and cheap but does nothing to drive the answer back to solid/void: volume is linear in $\rho$ and so is stiffness, so a half-dense element costs half the material and returns half the stiffness, the exchange rate is identical at $\rho = 0.5$ and $\rho = 1$, and the optimizer has zero incentive to commit — it smears material into a grey field that, as homogenization warns, cannot be dismissed as a mere plotting artifact. We need the tractability of the continuous relaxation without dissolving the very 0/1 character that "topology" requires.

The method is SIMP — Solid Isotropic Material with Penalization. The idea is to keep the continuous density $\rho_e \in [0,1]$ for its gradients but to bend the density-to-stiffness map so that intermediate density becomes a bad bargain. Volume is honestly linear in $\rho$ and we cannot touch it; so we make the stiffness law grow more slowly than linearly at low density, using a power, $E(\rho) = \rho^p E_0$ with $p > 1$. Now volume per element is still $\propto \rho$ but stiffness per element is $\propto \rho^p$, and since $\rho^p < \rho$ for $p > 1$ and $\rho \in (0,1)$, a half-dense element spends $1/2$ of the budget yet earns only $(1/2)^p$ of the stiffness — for $p = 3$ that is $1/8$. Grey is now uneconomical, and the optimizer is pushed toward the corners $\rho \in \{0,1\}$. The elegance is that there is no explicit penalty term with a weight to tune sitting in the objective; the penalization is folded entirely into the interpolation law and the objective remains plain compliance. The choice of $p$ is what separates a principled method from a hack. If we just crank $p$ up we sharpen the design but risk modeling a fictitious material that claims more stiffness than any real solid/void composite could deliver at that density. So we require the power law to sit on or below the Hashin–Shtrikman upper bound — the tightest variational bound on the effective isotropic modulus of an isotropic two-phase mixture of solid ($E_0$, Poisson ratio $\nu$) and void at volume fraction $\rho$. Working out where $\rho^p$ crosses that bound in two dimensions gives a lower limit on the exponent,
$$p \ge \max\!\left\{ \frac{4}{1+\nu},\ \frac{2}{1-\nu} \right\},$$
and for $\nu = 1/3$ both terms land on exactly $3$, so $p = 3$ is the smallest power consistent with the physics of composites — large enough to penalize grey, small enough to avoid unnecessary extra nonconvexity. Below $3$ the law floats above the bound (claiming impossible stiffness); well above $3$ the problem grows more nonconvex and one must ramp $p$ from $1$ toward $3$ as a continuation, but for a single solve $p = 3$ is the principled default. One repair the bare power law needs: at $\rho = 0$ it gives exactly zero stiffness, so the global $K$ develops a zero block and the FE solve goes singular. We give void a tiny nonzero stiffness,
$$E(\rho) = E_{\min} + \rho^p\,(E_0 - E_{\min}), \qquad E_{\min} \approx 10^{-9} E_0,$$
which keeps $K$ non-singular and lets the density variable reach exactly zero, while at $\rho = 1$ the element still has $E_0$ so the solid limit is unchanged.

With the material law fixed on a regular mesh of bilinear Q4 plane-stress elements, the unit-modulus element stiffness $k_0$ is one fixed $8\times8$ matrix shared by every element, the global stiffness is $K(\rho) = \sum_e E(\rho_e)\,k_0$, and the state $U$ solves $K(\rho)U = F$ with compliance $c(\rho) = F^T U = U^T K U$. To optimize we need $\partial c/\partial \rho_e$, and the direct route is poisoned by $\partial U/\partial \rho_e$, which would cost a linear solve per element. The escape is the adjoint: append $-\lambda^T(KU - F)$, which is identically zero at equilibrium, to $c$, differentiate, and choose $\lambda$ to annihilate the $\partial U/\partial \rho_e$ term by demanding $F^T - \lambda^T K = 0$, i.e. $K\lambda = F$. That is the same system the state solves, so $\lambda = U$ — compliance is self-adjoint and there is no extra solve. The sensitivity is then
$$\frac{\partial c}{\partial \rho_e} = -\,p\,\rho_e^{\,p-1}\,(E_0 - E_{\min})\,u_e^T k_0\, u_e \le 0,$$
non-positive because $u_e^T k_0 u_e$ is a strain energy: adding material never increases compliance, so the volume constraint is always active and the design is decided by where material goes, governed by the relative magnitudes of these sensitivities. The volume sensitivity is simply $\partial V/\partial \rho_e = 1$.

For the update I deliberately avoid a general-purpose nonlinear programmer like the Method of Moving Asymptotes, which carries machinery for many constraints of arbitrary sign; here there is one constraint and all sensitivities share the same non-positive sign, a special structure that admits a cheaper optimality-criteria update. From the Lagrangian $L = c + \lambda(\sum_e \rho_e v_e - V^*)$, interior stationarity reads $\partial c/\partial \rho_e + \lambda v_e = 0$, so defining the price signal $B_e = (-\partial c/\partial \rho_e)/(\lambda\,\partial V/\partial \rho_e)$ the optimum has $B_e = 1$ at every interior element: $B_e > 1$ means the element buys more stiffness than its material costs and should grow, $B_e < 1$ means it is overpriced and should shrink. The fixed-point iteration multiplies the density by $B_e^{\eta}$. The damping exponent $\eta = 1/2$ is essential — with $\eta = 1$ the multiplicative step is too aggressive and the iteration rings, while the geometric-mean pull of $\eta = 1/2$ converges smoothly — and a move limit $m$ (here $0.2$) clamps each step because the sensitivities are only locally valid, after which the density is clipped to $[0,1]$. The material price $\lambda$ is the one free scalar: total volume $\sum_e \rho_e^{\text{new}}$ is monotone decreasing in $\lambda$, so $\lambda$ is found by bisection against the volume target, bracketed between $0$ and $10^9$, each step re-evaluating only the closed-form update with no FE solve inside.

The last debt comes due once relaxation and penalization are in place: they reinstate the two pathologies of the ill-posed continuum problem. Checkerboards — patches of alternating solid and void — look like microstructure but are a lie, because the low-order Q4 element over-estimates their stiffness, and penalization sharpens the incentive to commit elements to $0/1$ at exactly the checkerboard scale. Mesh-dependence — finer mesh giving a different, finer beam rather than a sharper version of the same one — is the same root cause, the absence of any intrinsic length scale. Both are cured by imposing a length scale $r_{\min}$ through filtering with conical weights $H_{ei} = \max(0,\, r_{\min} - \mathrm{dist}(e,i))$ assembled once. The sensitivity filter replaces each raw sensitivity by a density-weighted neighborhood average,
$$\Big(\frac{\partial c}{\partial \rho_e}\Big)_{\text{filt}} = \frac{\sum_{i} H_{ei}\,\rho_i\,(\partial c/\partial \rho_i)}{\max(\gamma,\rho_e)\,\sum_{i} H_{ei}},$$
so neighbors share gradient information and a lone solid element in a sea of void no longer reads as a great deal, with $\gamma \approx 10^{-3}$ guarding the denominator when $E_{\min}$ permits exactly-zero densities. The more principled density filter instead introduces a physical field $\tilde{\rho}_e = (\sum_i H_{ei}\rho_i)/(\sum_i H_{ei})$ and performs all of the analysis — assembling $K$, computing compliance and volume — on $\tilde{\rho}$ rather than the raw design variable $\rho$, so the design literally cannot carry features finer than $r_{\min}$ and a minimizer provably exists; the cost is a chain rule $\partial \psi/\partial \rho_j = \sum_e (\partial \psi/\partial \tilde{\rho}_e)\,H_{je}/(\sum_i H_{ei})$ pushing the sensitivities of both $\psi = c$ and $\psi = V$ back through the filter. Either way the wall is cleared and the loop converges to clean, checkerboard-free, mesh-independent black-and-white designs. The whole chain, then: relax to densities for gradients; bend stiffness to $\rho^p$ so grey is uneconomical; pin $p = 3$ as the smallest power within the Hashin–Shtrikman bound; add $E_{\min}$ to keep $K$ non-singular; exploit the self-adjoint, non-positive sensitivity to drive a damped, move-limited optimality-criteria update with $\lambda$ bisected against the budget; and filter over $r_{\min}$ to land on well-posed designs.

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
