# Synthesis — Automatic lens design by damped least squares (DLS / Levenberg–Marquardt)

## Three sources (retrieved this run)
1. PRIMARY: Meiron 1965, "Damped Least-Squares Method for Automatic Lens Design", JOSA 55(9):1105–1109. Paywalled (opg.optica.org) — abstract retrieved: per-parameter damping factor that compensates for relative variable sensitivities; Lagrangian multipliers for boundary conditions; 16 eqs. Equations sourced from the clean secondary (Bociort) which reproduces the DLS normal-equation + damping derivation.
2. BACKGROUND/explainer: Bociort, "Optical System Optimization", Encyclopedia of Optical Engineering (2003), TU Delft PDF — full DLS derivation: error function f = Σ wᵢ(aᵢ(x)−ãᵢ)²/Σwᵢ; operands = transverse ray aberrations / wavefront / aberration coefficients / lens params; linearize operands (Taylor, keep first order, ∂aᵢ/∂xⱼ by finite diff); normal equations of linear LS give Δx; Δx too large → linearization invalid → iteration diverges → replace f by f_D = f + p(Δx)² (damping factor p discourages large |Δx|); trust-region interpretation: f_D is the Lagrange function for the constraint (Δx)²=h², p is the Lagrange multiplier → DLS gives best Δx with |Δx| restricted to h. Also: steepest descent was first, slow on elongated contours; local minima; Lagrange-multiplier constraints; solves; escape operand.
3. THIRD-PARTY (additional): Wang, Chen, Heidrich 2021, "Lens design optimization by back-propagation" (KAUST/VCC) — geometric sequential ray tracing through parameterized surfaces; θ = curvatures/air spacings/freeform coeffs; spot diagram p(θ); merit ε(θ)=Σf(pᵢ(θ)); DLS [cites Meiron 1965] is the custom technique; Jacobian via finite differences in most ray engines; local sensitivity via Jacobian. Confirms Meiron lineage.
   Also Kidger 1993 (Opt. Eng. 32:1731), "Use of the Levenberg–Marquardt (damped least-squares) optimization method in lens design" — title/abstract only (paywall); confirms DLS = LM, special properties of lens design as an optimization problem, m≫n overdetermined, ill-conditioning.

## CODE (canonical impl, fetched into code/): optiland (HarrisonKramer/optiland)
- OptimizationProblem: operands (residuals) + variables; merit = sum_squared = Σ (wᵢ·deltaᵢ)²; residual_vector() = stack of fun() = wᵢ·deltaᵢ (UNsquared) — exactly the f vector LM needs.
- Operand: value() from ray trace / paraxial / Seidel; delta() = value − target; fun() = weight·delta().
- LeastSquares optimizer: calls scipy.optimize.least_squares(method="lm") = Levenberg–Marquardt = DLS. Residual function = _compute_residuals_vector (updates vars, traces, returns f vector); NaN ray-failure → big penalty. lm requires m≥n (overdetermined), no bounds (→ trf otherwise).
- OptimizerGeneric: scipy.minimize on scalar merit (steepest-descent-class baseline).
- Variable scaling: RadiusVariable LinearScaler factor=1/100, offset=−1 → per-variable normalization (connects to Meiron per-parameter damping / variable sensitivity).
- Worked examples: singlet (OPD_difference + f2 operands; vary thickness/radii), Cooke triplet (f2 + 5 Seidel operands; vary radii; symmetry pickups).
- Ray model: refract(n1,n2) Snell; paraxial n2·u' = n1·u − y·(n2−n1)/R.

## Derivation chain for reasoning.md
- lens = ordered surfaces with curvature c=1/R, thickness t, glass n(λ); trace rays by Snell at each surface.
- image quality = many ray-error / aberration targets fᵢ(x): transverse ray aberration of ray i = where it hits image vs chief/ideal; Seidel coeffs; focal length constraint. m operands, n variables, m≫n → overdetermined, no exact zero → minimize Σ fᵢ(x)² (NLLS).
- linearize: f(x+Δx)≈f+JΔx, J = ∂fᵢ/∂xⱼ (finite differences, one extra trace per variable). Minimize ‖f+JΔx‖² → normal equations JᵀJ Δx = −Jᵀf → Gauss–Newton Δx = −(JᵀJ)⁻¹Jᵀf.
- WALL: optical J is ill-conditioned / near rank-deficient — variables highly correlated (two curvatures of a thick element, bending, conic vs 4th-order asphere have near-identical aberration effect → Bociort's redundancy note). JᵀJ near-singular → (JᵀJ)⁻¹ blows up along small-singular-value directions → Δx huge in those directions → leaves linearization-valid region → merit goes UP, ray failure (NaN). Diverges.
- FIX: damping. Replace f by f_D=f+p|Δx|² ⇒ minimize ‖f+JΔx‖²+λ‖Δx‖² ⇒ (JᵀJ+λI)Δx=−Jᵀf ⇒ Δx=−(JᵀJ+λI)⁻¹Jᵀf (Levenberg). SVD view: step in direction vₖ scaled by σₖ/(σₖ²+λ) — for σₖ≫√λ ≈ 1/σₖ (Gauss–Newton); for σₖ≪√λ ≈ σₖ/λ → 0 (damped, no blow-up). λ→∞: Δx→−(1/λ)Jᵀf = scaled steepest descent (safe). λ→0: Gauss–Newton (fast). 
- adapt λ by trust: try step, if merit decreases accept and shrink λ; if increases reject and grow λ. (Lagrange/trust-region equivalence: λ is multiplier for ‖Δx‖²≤h².)
- Meiron refinement: per-variable damping (λ scaled per parameter by its sensitivity) — equivalent to using diag(JᵀJ) instead of I (Marquardt) or normalizing variables (optiland scaling). Curvatures and thicknesses have wildly different units/sensitivities → uniform λI penalizes them unequally → scale.
- land on code: residual_vector + scipy lm.

## Design-decision → why
- squared residuals: ± deviations equally harmful; gives smooth quadratic-near-min, normal equations solvable in one step if linear.
- operands not direct image metric: ray aberrations/Seidel give per-ray residuals = the f vector (LM needs vector not scalar); aberration coeffs avoid ray failure, fast early.
- finite-diff Jacobian: ray trace is a black-box composition of Snells; one extra trace per variable; cheap relative to m rays.
- Gauss–Newton not Newton: drop second-derivative term f·∂²f (small near solution, expensive); JᵀJ is free from J.
- damping λI vs trust region: equivalent (λ ↔ radius h); λI easiest to implement, always SPD so solvable even if rank-deficient.
- per-parameter / scaled damping (Meiron): variables differ in units/sensitivity; uniform I is wrong metric; diag(JᵀJ) or variable normalization fixes it.
- adapt λ: too big = crawl, too small = diverge; trust ratio adapts.
- m≥n for lm: needs overdetermined for JᵀJ full rank-ish; lens design naturally m≫n (many rays).
