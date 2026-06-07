# Synthesis — muon scattering tomography (PoCA)

## Pain point / goal
Inspect dense/large objects (cargo containers, shielded SNM) for high-Z material non-destructively,
no artificial radiation. X-rays don't penetrate; absorption muon radiography (Alvarez pyramid,
George 1955) gives only line-integral attenuation, poor high-Z contrast, slow, no 3D localization.
Want a passive probe that (a) penetrates meters of steel, (b) is strongly Z-sensitive, (c) localizes
in 3D where the dense thing is.

## Pre-method physics facts (context, sourced)
- Cosmic-ray muons: secondary from atmospheric cascade (pi -> mu nu). Mean energy 3-4 GeV; flat
  spectrum below 1 GeV, falls as p^-2.7 above ~10 GeV. Flux ~10000 /m^2/min horizontal, ~1
  /cm^2/min ("through a fingernail per minute"), ~160 Hz/m^2. cos^2(theta) zenith dependence.
  Penetrating: GeV muons traverse meters of rock/steel. Low rate => single-particle tracking. [Schultz thesis 2.1; LANL 1306.0523]
- Muon-matter interaction: (i) ionization energy loss ~2 MeV/(g/cm^2) (basis of absorption
  radiography), (ii) multiple Coulomb scattering off nuclei. [thesis 2.2]
- Highland / Lynch-Dahl MCS: projected angle ~ Gaussian (central 98%), zero mean,
  theta0 = (13.6 MeV/(beta c p)) z sqrt(x/X0) [1 + 0.038 ln(x z^2 / X0 beta^2)].
  Schultz drops the log and uses constant 15: sigma_theta = (15/(beta c p)) sqrt(L/X0), beta~1 for
  muons => sigma_theta = (15/p[MeV]) sqrt(L/Lrad). LANL uses 14.1. PDG 13.6. [PDG; thesis 2.2,3.2; LANL]
- Radiation length X0: decreases with Z (areal density monotone in Z; depth form strongly Z).
  Water 36.1cm, concrete 10.7, iron 1.76, lead 0.56, uranium 0.32 cm. So per cm, high-Z scatters
  far more: RMS for 10cm @ 3 GeV: water 2.6 mrad, iron 11.9, lead 21.1, U 28.0 mrad. [thesis Table 2.3]
- Why MCS over absorption: scattering ~ as Z-sensitive as energy loss but milliradian angle is
  measured from mm position (two tracking planes), no spectrometer needed. [thesis 2.2.3]
- Lateral displacement also Gaussian, jointly with angle. For thickness L (PDG/standard):
  Var(angle)=L lambda; Var(disp)=L^3/3 lambda; Cov(angle,disp)=L^2/2 lambda.
  correlation = (L^2/2)/sqrt(L * L^3/3) = sqrt(3)/2 ~= 0.866. [thesis 6.16-6.17; MGH proton notes]

## Key derivations to reproduce
1. Scattering density definition. sigma_theta = (15/p0) sqrt(L/Lrad). Square:
   sigma_theta^2 = (15/p0)^2 (1/Lrad) L. Define lambda := (15/p0)^2 (1/Lrad) = sigma_theta0^2 / L.
   => "scattering density" = mean-square scattering per unit length for nominal momentum p0.
   Normalizes out momentum (fix p0) and thickness. Strongly separates low/med/high Z. [3.3-3.4]
2. Single homogeneous slab estimate: signal s = theta (projected scatter). Var(s)=lambda L.
   lambda_hat = (1/L)(1/M) sum s_i^2  (mean square of path-normalized signals). [4.6]
   With varying path lengths: lambda_hat = sum_i s_i^2 / (M L_i). [4.7]
3. Multi-cell raysum for variance: theta = sum_j theta_j (independent), so
   Var(s) = sum_j L_j lambda_j. Linear system v = L lambda (variance side). [4.8-4.10]
4. PoCA: muon path unknown, but in+out tracks measured. Assume ALL scattering at one point.
   Estimate that point = point of closest approach of in/out lines (2D: they cross; 3D: skew,
   take midpoint of common perpendicular). Signal s = theta^2, theta = theta_out - theta_in.
   Deposit s into PoCA pixel; accumulate. lambda_hat(j) = mean of deposited s over muons in
   pixel j, / L. (Algorithm 4.2.1: S accumulates signal, I counts, lambda = S/(I*L).) [4.5,4.11]
   Why cheap first cut: one point per muon, straight-line tracing, no iteration. Why fails:
   single-scatter assumption breaks for extended/multiple objects => mislocalized signal,
   blurring; ignores displacement info; momentum unknown adds variance.
5. 3D PoCA geometry (derive): lines r1=p1+t v1, r2=p2+s v2. Minimize |r1-r2|^2.
   Let w0=p1-p2, a=v1.v1,b=v1.v2,c=v2.v2,d=v1.w0,e=v2.w0. denom=ac-b^2.
   t=(b e - c d)/denom, s=(a e - b d)/denom. PoCA = midpoint (r1(t)+r2(s))/2. Parallel: denom=0.
6. MLS (Schultz 2007): per ray Gaussian P(s_i|v_i)=N(0,v_i), v_i = sum_j L_ij lambda_j.
   NLL: minimize sum_i [ ln(v_i) + s_i^2 / v_i ], v_i = (L lambda)_i, subject to lambda_j>=lambda_air.
   This is the tomographic upgrade: every cell on the path shares the signal via its path length,
   so multiple-scatter ambiguity is resolved by pooling many rays (rays from many angles
   constrain overlapping cell sets). [6.1]
7. MLSD: add displacement. Data d_i=[Delta_theta; Delta_x]. 2x2 covariance Sigma_i with
   v_theta=W_theta lambda, v_x=W_x lambda, s_thetax=W_thetax lambda, weights per 6.34-6.36:
   W_theta_j=L_j; W_thetax_j=L_j^2/2 + L_j T_j; W_x_j=L_j^3/3 + T_j L_j^2 + T_j^2 L_j, where
   T_j = downstream path length after cell j. NLL: minimize sum_i [ ln|Sigma_i| + d_i^T Sigma_i^-1 d_i ],
   lambda>=lambda_air. Displacement breaks the angle-only degeneracy (a thin dense layer high vs low
   along path gives same angle but different displacement). [6.2]
   Delta_x measured: project in-track to detector (x_proj), Delta_x=(x_out - x_proj) cos(theta_avg). [6.42]

## Design decisions -> why
- Use scattering not absorption: angle measurable to mrad from mm tracking; absorption needs huge
  flux/overburden and is weakly Z-sensitive; gives 3D not just line integral.
- Square the angle as signal (s=theta^2): lambda is a *variance* per unit length, and the unbiased
  estimator of a zero-mean Gaussian's variance is the mean square; one sample's contribution is theta^2.
- Single-scatter / PoCA assumption: the true bent path is unknown and integrating the full stochastic
  path is hard; mrad scattering is geometrically negligible vs cm pixels, so a single kink + straight
  legs is a fine geometric approximation for *localization*; gives an O(M) non-iterative reconstruction.
- Midpoint of common perpendicular in 3D: in-/out-lines generally skew (don't intersect), closest-
  approach point is the unique least-distance estimate of the single scatter vertex.
- Fix nominal p0 (ignore per-muon momentum) for plain PoCA: momentum is not measured by tracking
  alone; treating all muons at p0 keeps it simple at the cost of extra variance (spectrum spread).
- MLS over PoCA: PoCA dumps all signal in one cell -> wrong when scattering is distributed; the
  raysum variance model v_i=sum L_ij lambda_j lets each cell on the path carry its share and many
  rays jointly constrain cells -> true tomography, resolves overlap/ambiguity.
- ln(v)+s^2/v cost: this is exactly -2 log-likelihood of a zero-mean Gaussian with unknown variance;
  the ln(v) term penalizes inflating variance, s^2/v rewards fit. Not least squares because the
  unknown is the variance, not the mean.
- lambda >= lambda_air constraint: negative scattering density is unphysical; air floor regularizes.
- MLSD adds displacement: angle-only is degenerate along the ray (can't tell where on the path the
  dense cell is); displacement + the L^3/3, L^2/2 moments encode position-along-path, lifting degeneracy.

## Canonical code
PoCA is short, self-contained: 3D skew-line closest approach + voxel accumulation, grounded in
thesis Algorithm 4.2.1 and the standard TomOpt/MuonTomographySimulation geometry. MLS/MLSD shown as
the ML upgrade (NLL + nonneg constraint, scipy optimize), grounded in thesis 6.13/6.45-6.46.

## Sources (three-source)
1. Primary: Schultz PhD thesis "Cosmic Ray Muon Radiography" (Portland State, 2006) — full PoCA Ch.4,
   MLS/MLSD Ch.6; this is the detailed form of the 2004 PoCA and IEEE-TIP 2007 MLSD papers.
   https://www.hep.phy.cam.ac.uk/~hommels/CosmicConcrete_dir/MuonPapersForBart/Schultz_thesis.pdf
   Borozdin 2003 Nature 422:277 proof-of-principle (abstract + thesis 2.3).
2. Background: PDG passage-of-particles (Highland/Lynch-Dahl); cosmic-ray flux (thesis 2.1); MGH
   proton MCS notes (displacement/correlation). LANL 1306.0523 review.
3. Explainer: LANL "A new method for imaging nuclear threats using cosmic ray muons" 1306.0523
   https://arxiv.org/pdf/1306.0523 ; MGH "Techniques of Proton Radiotherapy (06) Multiple Scattering".
