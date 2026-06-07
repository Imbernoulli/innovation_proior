# Synthesis — Force-field torsion (dihedral) parameter fitting

## Pain point / research question
A classical MM force field decomposes energy into bonds + angles + torsions + nonbonded. The
bonds/angles/nonbonded terms are physically transferable, but the rotational barrier about a
rotatable bond (the torsion profile) is NOT captured correctly by them alone — there is residual
energy from conjugation, hyperconjugation, 1-4 electrostatics imperfection, steric clash that the
other terms miss. A QM relaxed torsion scan E_QM(phi) shows the true barrier. We need a torsion
term that, ADDED to the rest of the MM energy, reproduces E_QM(phi). Goal: given a QM scan, find
torsion parameters.

## Functional forms (all sourced)
- AMBER (Cornell 1995): E_tors = sum_{dihedrals} sum_n (V_n/2)[1 + cos(n*omega - gamma_n)].
  Full energy = sum_bonds k_b(l-l0)^2 + sum_angles k_a(theta-theta0)^2
                + sum_torsions sum_n (V_n/2)[1+cos(n w - gamma)]
                + sum_{i<j} { eps_ij[(r0/r)^12 - 2(r0/r)^6] + q_i q_j/(4 pi eps0 r_ij) }.
  Source: Wikipedia AMBER + search of Cornell 1995 JACS 117:5179.
- OPLS-AA (Jorgensen, Maxwell, Tirado-Rives 1996): same nonbonded/bond/angle skeleton, torsion
  V(phi) = (1/2)[ V1(1+cos phi) + V2(1+cos 2phi) + V3(1+cos 3phi) + V4(1+cos 4phi) ].
  Implicit phases 0/180 absorbed into sign of V_n (phases set 0 for the studied systems).
  V1..V4 fit to RHF/6-31G* rotational profiles, >50 molecules. 1-4 nonbonded scaled by 0.5.
  Source: OPLS 1996 + PMC4504185 (2015 improved torsional energetics, open access).

## Why a torsion term must exist (derive, don't measure)
Take a QM relaxed scan E_QM(phi). Compute MM energy with torsion term ZEROED, E_MM\tors(phi)
(all other terms active, geometry MM-relaxed at each fixed phi). The residual
DeltaE(phi) = E_QM(phi) - E_MM\tors(phi) is exactly what the torsion term must supply. It is
periodic in phi with period 2pi and (for a symmetric bond) even -> natural to expand in a cosine
Fourier series. Periodicities n carry physical meaning: n=3 for sp3-sp3 (3 staggered minima),
n=2 for conjugation/planarity (cis/trans), n=1 for gauche/anti asymmetry.

## The fit is LINEAR least squares (key insight)
E_tors(phi) = sum_n K_n (1 + cos(n phi)) is LINEAR in the coefficients K_n (the cosines are a
fixed basis evaluated at known angles). So fitting K to reproduce DeltaE over a scan of M
conformations is ordinary linear least squares:
  minimize F(K) = sum_k ( sum_n K_n (1+cos(n phi_k)) - DeltaE_k )^2.
Stationarity dF/dK_m = 0 gives the normal equations M K = B with
  M_{mn} = sum_k (1+cos(m phi_k))(1+cos(n phi_k))   [Gram matrix of basis]
  B_m    = sum_k DeltaE_k (1+cos(m phi_k)).
Solve K = M^{-1} B. No iterative optimizer needed; the global optimum is analytic (K_fit paper:
Kania, Sarapata, Gucwa, Wojcik-Augustyn, JPCA 2021; PMC8041298). Contrast: Monte-Carlo /
nonlinear optimizers (ForceBalance steepest descent) used historically because people kept the
phase gamma as a free nonlinear parameter — but absorbing gamma in {0,180} via the SIGN of K_n
makes it linear.

## Type-grouping (parameter sharing)
A molecule has many dihedrals of the SAME atom-type quadruplet that must share one set of K_n
(transferability). So the basis for type i at conformation k is summed over all dihedrals l of
that type: A_{ikj} = sum_l (1 + cos(j * phi_{ikl})). Model_k = sum_i sum_j K_{ij} A_{ikj}. Same
normal equations, larger system (4 coeffs x #types).

## Regularization / overfitting (sourced, OpenFF BespokeFit, PMC9709916)
Expanding all n=1..4 with no constraint can overfit / ring. OpenFF: include n=1..4, then L1
(Lasso) regularize to shrink redundant K_n to zero; prior widths widen (1.0->6.0). K_fit offers
a bounded variant lsq_linear(bounds=(fr,to)) to keep |K_n| physical. Weighting: OpenFF uses a
flat-then-attenuating weight (constant to 1 kcal/mol, off by 10 kcal/mol); OPLS-2015 uses a
Boltzmann weight exp(-DeltaE/kT) so low-energy (populated) regions dominate.

## QM scan details (sourced)
TorsionDrive: constrained QM optimization on a grid -180..180 (e.g. 15 deg), all other DOF relax
(RELAXED scan). Then re-relax under MM with the same dihedral fixed -> consistent MM\tors energy.
Levels: OPLS-1996 RHF/6-31G*; OPLS-2015 wB97X-D/6-311++G(d,p) geom + B2PLYP-D3BJ/aug-cc-pVTZ SP.

## Fourier -> Ryckaert-Bellemans conversion (VERIFIED symbolically)
With RB psi = phi - 180 (cos psi = -cos phi), matching E=sum_n k_n(1+cos n phi):
  C0 = k1+k3+2k4, C1 = -k1+3k3, C2 = 2k2-8k4, C3 = -4k3, C4 = 8k4, C5 = 0.
Matches K_fit fourier2RM exactly (verified with sympy/Chebyshev).

## Canonical code: K_fit.py (mysarapa/K_fit, GPL)
- reads QM energy column, MM(torsion=0) energy column, *.dih files (one per dihedral, first line
  = atom types), #coeffs (3 or 4), Fourier/RB.
- B_t = qm - zero_rb (residual).
- builds basis (1+cos(n*phi)) per dihedral, groups by type (forward==reverse atom order),
  accumulates M via outer products (forDerivative), B via B_t*basis.
- K = inv(M) @ B. Optional bounded fit via scipy lsq_linear.
- Verified normal-equations math myself above.

## Scaffold pieces (pre-method, generic)
energy decomposition with a torsion stub; a generic basis-eval stub; a generic linear-least-
squares solve. Final code fills: residual = qm - mm_no_tors; design matrix from cos basis;
grouping; normal equations solve; optional bounds; Fourier->RB.
