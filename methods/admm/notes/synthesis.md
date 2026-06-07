# ADMM synthesis

## Problem
minimize f(x)+g(z) s.t. Ax+Bz=c, f,g closed proper convex. Goal: solve large-scale convex
problems where the objective splits into two blocks, in a way that (i) is robust (no strict
convexity / finiteness needed) and (ii) decomposes (each block updated separately, possibly in
parallel). The two existing tools each give one but not both.

## Lineage / ancestors (load-bearing)
1. **Dual ascent** (Lagrangian L = f(x)+yᵀ(Ax−b); x⁺=argmin L(x,y); y⁺=y+α(Ax⁺−b)). Dual function
   g(y)=−f*(−Aᵀy)−bᵀy; ∇g(y)=Ax⁺−b. Fragile: needs strict convexity/finiteness for x-update to be
   well-defined; e.g. if f is affine in any component, L unbounded below → x-update fails. Step size
   αₖ delicate; with nondifferentiable g must use dual *subgradient*, nonmonotone.
2. **Dual decomposition** (Everett ~1963; Dantzig–Wolfe, Benders). If f(x)=Σfᵢ(xᵢ) and A=[A₁…A_N],
   Lagrangian separable → x-update splits into N parallel subproblems xᵢ⁺=argmin Lᵢ(xᵢ,y); gather
   residual, broadcast price y. The decomposition virtue. But inherits dual-ascent fragility.
3. **Augmented Lagrangian / method of multipliers** (Hestenes 1969, Powell 1969; Rockafellar; Bertsekas
   monograph). Lρ(x,y)=f(x)+yᵀ(Ax−b)+(ρ/2)‖Ax−b‖². = ordinary Lagrangian of equivalent problem
   min f(x)+(ρ/2)‖Ax−b‖² s.t. Ax=b. Dual gρ differentiable under mild conditions. Method:
   x⁺=argmin Lρ(x,yᵏ); yᵏ⁺¹=yᵏ+ρ(Ax⁺−b). Why step=ρ: stationarity 0=∇f(x⁺)+Aᵀ(yᵏ+ρ(Ax⁺−b))=
   ∇f(x⁺)+Aᵀyᵏ⁺¹ → each iterate is dual feasible; as r→0, optimality. Robust: converges without
   strict convexity / finiteness. COST: the quadratic (ρ/2)‖Σ Aᵢxᵢ −b‖² couples the xᵢ (cross terms
   Aᵢxᵢ·Aⱼxⱼ) → Lρ NOT separable → joint x-update can't be parallelized → decomposition LOST.

## The ADMM move
Split the variable: f(x)+g(z) s.t. Ax+Bz=c. Form Lρ(x,z,y)=f(x)+g(z)+yᵀ(Ax+Bz−c)+(ρ/2)‖Ax+Bz−c‖².
Method of multipliers would do JOINT (x,z)=argmin Lρ — that re-couples. ADMM instead does ONE
Gauss–Seidel sweep: minimize over x, then over z, then dual update:
  xᵏ⁺¹ = argmin_x Lρ(x, zᵏ, yᵏ)
  zᵏ⁺¹ = argmin_z Lρ(xᵏ⁺¹, z, yᵏ)
  yᵏ⁺¹ = yᵏ + ρ(Axᵏ⁺¹+Bzᵏ⁺¹−c)
The alternation is what restores per-block decomposition (when fixing z, the quadratic in x is just a
prox-type problem in x alone; vice versa), while keeping the augmented-Lagrangian robustness.
State is (zᵏ,yᵏ); xᵏ is intermediate. Originally Glowinski–Marroco 1975 / Gabay–Mercier 1976.

## Scaled form (derive)
r=Ax+Bz−c. yᵀr+(ρ/2)‖r‖² = (ρ/2)‖r+(1/ρ)y‖² − (1/2ρ)‖y‖² = (ρ/2)‖r+u‖² −(ρ/2)‖u‖², u=(1/ρ)y.
Drop const −(ρ/2)‖u‖² (no x,z):
  xᵏ⁺¹=argmin f(x)+(ρ/2)‖Ax+Bzᵏ−c+uᵏ‖²
  zᵏ⁺¹=argmin g(z)+(ρ/2)‖Axᵏ⁺¹+Bz−c+uᵏ‖²
  uᵏ⁺¹=uᵏ+Axᵏ⁺¹+Bzᵏ⁺¹−c     (running sum of residuals: uᵏ=u⁰+Σrʲ)

## Optimality / stopping (derive)
Optimality: primal feas Ax*+Bz*−c=0; dual feas 0∈∂f(x*)+Aᵀy*, 0∈∂g(z*)+Bᵀy*.
z-update ⇒ 0∈∂g(zᵏ⁺¹)+Bᵀyᵏ+ρBᵀrᵏ⁺¹=∂g(zᵏ⁺¹)+Bᵀyᵏ⁺¹ ⇒ (3.10) ALWAYS holds.
x-update ⇒ 0∈∂f(xᵏ⁺¹)+Aᵀyᵏ+ρAᵀ(Axᵏ⁺¹+Bzᵏ−c). Add/subtract ρAᵀBzᵏ⁺¹:
  =∂f(xᵏ⁺¹)+Aᵀ(yᵏ+ρrᵏ⁺¹+ρB(zᵏ−zᵏ⁺¹))=∂f(xᵏ⁺¹)+Aᵀyᵏ⁺¹+ρAᵀB(zᵏ−zᵏ⁺¹)
  ⇒ ρAᵀB(zᵏ⁺¹−zᵏ)∈∂f(xᵏ⁺¹)+Aᵀyᵏ⁺¹. So define
  primal residual rᵏ⁺¹=Axᵏ⁺¹+Bzᵏ⁺¹−c ; dual residual sᵏ⁺¹=ρAᵀB(zᵏ⁺¹−zᵏ).
Both →0. Suboptimality bound: f(xᵏ)+g(zᵏ)−p* ≤ −(yᵏ)ᵀrᵏ+(xᵏ−x*)ᵀsᵏ. Stop when ‖rᵏ‖≤εᵖʳⁱ and
‖sᵏ‖≤εᵈᵘᵃˡ, with εᵖʳⁱ=√p·εᵃᵇˢ+εʳᵉˡ max(‖Axᵏ‖,‖Bzᵏ‖,‖c‖), εᵈᵘᵃˡ=√n·εᵃᵇˢ+εʳᵉˡ‖Aᵀyᵏ‖.

## Soft-threshold (LASSO z-update). g(z)=λ‖z‖₁, A=I scalar prox:
argmin λ|z|+(ρ/2)(z−v)² = S_{λ/ρ}(v), S_κ(a)=(a−κ)₊−(−a−κ)₊=sign(a)max(|a|−κ,0)=(1−κ/|a|)₊a.
LASSO min ½‖Cx−b‖²+λ‖x‖₁ in ADMM: f(x)=½‖Cx−b‖², g(z)=λ‖z‖₁, x−z=0:
  xᵏ⁺¹=(CᵀC+ρI)⁻¹(Cᵀb+ρ(zᵏ−uᵏ))   [ridge; cache Cholesky]
  zᵏ⁺¹=S_{λ/ρ}(xᵏ⁺¹+uᵏ)
  uᵏ⁺¹=uᵏ+xᵏ⁺¹−zᵏ⁺¹
Here A=I,B=−I,c=0: r=x−z, s=−ρ(zᵏ⁺¹−zᵏ) (since AᵀB=−I).

## Consensus
min Σfᵢ(x). Copy: min Σfᵢ(xᵢ) s.t. xᵢ−z=0. Scaled ADMM:
  xᵢᵏ⁺¹=argmin fᵢ(xᵢ)+(ρ/2)‖xᵢ−zᵏ+uᵢᵏ‖²
  zᵏ⁺¹=(1/N)Σ(xᵢᵏ⁺¹+uᵢᵏ) = x̄ᵏ⁺¹+ū   (and with y-avg 0 after iter 1, z=x̄)
  uᵢᵏ⁺¹=uᵢᵏ+xᵢᵏ⁺¹−zᵏ⁺¹
x-updates fully parallel; only a gather (average) couples them. rᵏ=(x₁−x̄,…), ‖r‖²=Σ‖xᵢ−x̄‖²,
sᵏ=−ρ(x̄ᵏ−x̄ᵏ⁻¹) per block.

## Sharing
min Σfᵢ(xᵢ)+g(Σxᵢ). Copy z. z-update reduces from Nn to n vars by averaging. Dual to consensus.

## Douglas–Rachford / proximal-point connection (in-frame: DR is prior art, ADMM is the target)
DR splitting (Douglas–Rachford 1956 for heat eq.; Lions–Mercier 1979 for monotone operators) finds a
zero of A+B (maximal monotone) by alternating resolvents (prox). Gabay (1983) showed ADMM is DR
splitting applied to the dual; Eckstein–Bertsekas (1992) showed DR is an instance of Rockafellar's
proximal point algorithm; method of multipliers is also a proximal-point method (Rockafellar 1976).
Peaceman–Rachford ↔ ADMM-with-extra-y-update. These give ADMM's convergence theory.
Resolvent / prox: prox_{f,ρ}(v)=argmin f(x)+(ρ/2)‖x−v‖²; for indicator of C it is projection Π_C.

## Convergence proof (App A), Lyapunov Vᵏ=(1/ρ)‖yᵏ−y*‖²+ρ‖B(zᵏ−z*)‖²
Three inequalities:
(A1) Vᵏ⁺¹≤Vᵏ−ρ‖rᵏ⁺¹‖²−ρ‖B(zᵏ⁺¹−zᵏ)‖² ⇒ Σ summable ⇒ rᵏ→0, B(zᵏ⁺¹−zᵏ)→0 ⇒ sᵏ→0.
(A2) pᵏ⁺¹−p*≤−(yᵏ⁺¹)ᵀrᵏ⁺¹−ρ(B(zᵏ⁺¹−zᵏ))ᵀ(−rᵏ⁺¹+B(zᵏ⁺¹−z*)).
(A3) p*−pᵏ⁺¹≤(y*)ᵀrᵏ⁺¹.  ⇒ objective →p*.
(A2)+(A3) give (3.11) via −rᵏ⁺¹+B(zᵏ⁺¹−zᵏ)=−A(xᵏ⁺¹−x*).
Proof of A2: xᵏ⁺¹ minimizes Lρ(x,zᵏ,yᵏ): 0∈∂f(xᵏ⁺¹)+Aᵀyᵏ+ρAᵀ(Axᵏ⁺¹+Bzᵏ−c); sub yᵏ=yᵏ⁺¹−ρrᵏ⁺¹ ⇒
xᵏ⁺¹ minimizes f(x)+(yᵏ⁺¹−ρB(zᵏ⁺¹−zᵏ))ᵀAx; zᵏ⁺¹ minimizes g(z)+(yᵏ⁺¹)ᵀBz. Compare to x*,z*, add,
use Ax*+Bz*=c.
Proof A3: saddle point L₀(x*,z*,y*)≤L₀(xᵏ⁺¹,zᵏ⁺¹,y*) ⇒ p*≤pᵏ⁺¹+(y*)ᵀrᵏ⁺¹.
A1 from 2·(A2+A3), telescoping y-terms into (1/ρ)(‖yᵏ⁺¹−y*‖²−‖yᵏ−y*‖²), and using that zᵏ⁺¹,zᵏ each
minimize g(z)+yᵀBz at respective y to get (yᵏ⁺¹−yᵏ)ᵀB(zᵏ⁺¹−zᵏ)≤0 i.e. ρ(rᵏ⁺¹)ᵀB(zᵏ⁺¹−zᵏ)... cross
term sign works out.

## Code grounding (canonical Boyd MATLAB lasso → Python)
x = U\(L\q) with q=Atb+ρ(z−u); cached Cholesky of (CᵀC+ρI) (or matrix-inversion-lemma form if m<n);
z=shrinkage(x+u, λ/ρ); u=u+(x−z); shrinkage(a,κ)=max(0,a−κ)−max(0,−a−κ);
r_norm=‖x−z‖; s_norm=‖−ρ(z−zold)‖; eps_pri=√n·ABSTOL+RELTOL·max(‖x‖,‖−z‖);
eps_dual=√n·ABSTOL+RELTOL·‖ρu‖. ABSTOL~1e-4, RELTOL~1e-2.

## Design decisions → why
- split variable into x,z: so each half of objective gets its own update → decomposition.
- augmented (quadratic) penalty: robustness (no strict convexity), differentiable dual.
- alternate instead of joint min: joint re-couples (that's method of multipliers); alternation keeps
  per-block separability. One Gauss–Seidel sweep is enough for convergence (proven), don't need to
  iterate the sweep to convergence.
- dual step size = ρ (not free α): makes iterate dual feasible automatically (z-update ⇒ (3.10)).
- scaled dual u=y/ρ: collapses linear+quadratic into one squared term; updates shorter; u = running
  sum of residuals.
- two residuals r (primal) and s=ρAᵀB(zᵏ⁺¹−zᵏ) (dual): both come from the optimality conditions;
  stopping when both small bounds suboptimality.
- ρ varying scheme (residual balancing): keep ‖r‖,‖s‖ within factor μ; rescale u when ρ changes.
- modest-accuracy tool: converges fast to low accuracy, slow to high — fine for ML.
