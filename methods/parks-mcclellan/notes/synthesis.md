# Synthesis — Parks–McClellan / Remez-exchange equiripple FIR design

## Pain point (research question)
Design a length-N linear-phase FIR filter whose magnitude response best approximates a
desired piecewise-constant spec (passbands→1, stopbands→0) over prescribed bands, with
**direct control of band edges** and **minimum worst-case error**. Pre-1972 methods either
don't minimize the max error (window, frequency-sampling) or can't fix the band edges
(Herrmann/Hofstetter "maximal ripple").

## Three sources (all retrieved & read this run)
1. PRIMARY: McClellan, Parks, Rabiner 1973, "A Computer Program for Designing Optimum FIR
   Linear Phase Digital Filters," IEEE Trans. Audio Electroacoust. AU-21(6):506-526.
   → refs/MPR1973_computer_program.pdf (read pp.506-514 fully: 4 linear-phase cases,
   Q(f)P(f) unification, weighted-minimax (13)-(15), ALTERNATION THEOREM, Remez flowcharts
   Fig.5-6 = DEV via barycentric Lagrange, extremal search/exchange, endpoint search,
   inverse-DFT coefficient recovery). This is the FIRPM source. The companion theory paper is
   Parks & McClellan 1972 "Chebyshev Approximation for Nonrecursive Digital Filters with
   Linear Phase," IEEE Trans. Circuit Theory CT-19:189-194 (paywalled IEEE; its content —
   minimax formulation + alternation theorem + Remez — is reproduced in the 1973 paper §II
   which I read in full, and cross-checked vs Wikipedia explainer).
2. BACKGROUND: Rabiner 1971 survey "Techniques for Designing Finite-Duration Impulse-Response
   Digital Filters," IEEE Trans. Comm. Tech. COM-19(2):188-195 → refs/fir_design_trans_comm.pdf
   (window: Gibbs, fixed % overshoot, Hamming/Blackman/Kaiser sidelobe specs; frequency
   sampling: DFT, optimize transition samples by LP; equiripple Herrmann/Hofstetter: solve
   N-1 nonlinear eqns on extrema). Plus Rabiner 1972 "Linear Program Design of FIR Filters"
   COM-20 → refs/045 (LP can fix band edges but is slow, ~100 params). Plus Rabiner-Herrmann
   1973 even-N → refs/PM_optimal_design.pdf and Rabiner-Schafer freq-sampling → refs/034.
   Chebyshev/minimax theory + alternation (equioscillation) theorem; Remez 1934 exchange.
3. EXPLAINER: Wikipedia "Parks–McClellan filter design algorithm" (alternation count L+2,
   Remez iteration steps, barycentric Lagrange, Hofstetter maximal-ripple lineage, Remez 1934);
   Columbia Dan Ellis E4810 L09 → refs/columbia (rectangular window = best ISE/least-squares
   but ~9% Gibbs overshoot, "not optimal by minimax criterion").

## Canonical code (1.4)
- scipy `signal.remez` Python wrapper → code/scipy_fir_filter_design.py (lines 780-953).
- The engine: `_sigtoolsmodule.cc` `remez()` + `pre_remez()` → code/sigtoolsmodule.cc.
  Header credits "converted from an original FORTRAN by McClellan/Parks/Rabiner." This IS the
  1973 program. Map:
  * lagrange_interp (lines 67-81) = Lagrange weights b_k (the "AD" array), product form.
  * freq_eval (90-106) = barycentric Lagrange evaluation of P(f): P=Σ ad_j y_j/(xf-x_j) ÷ Σ ad_j/(xf-x_j), xf=cos(2πf).
  * remez (124-427): x[j]=cos(2π F_j); ad[j]=lagrange_interp; DEV = Σ ad_j D_j / Σ (-1)^{j-1} ad_j/W_j (lines 166-176); y[j]=D_j + (-1)^{j-1}DEV/W_j (184-188); convergence dev<=devl (189-193); extremal search L200-L370 (the exchange); coefficient recovery via inverse DFT (330-393); for non-[0,0.5] bands the change-of-variable aa,bb + Chebyshev recursion (342-421).
  * pre_remez (472-661): neg/nodd → which of 4 cases; nfcns = # cosines; dense grid spacing 0.5/(LGRID·nfcns) (530-531); D̂=D/Q, Ŵ=W·Q via sin/cos "change" factors (575-597); initial iext = equispaced (600-604); after remez, fold alpha → h (617-656) with symmetry h[k]=±h[nfilt+1-j].

## Derivation chain (for reasoning.md, discovery order)
1. Want min-max (Chebyshev) error, not least-squares. Window gives best ISE but ~9% Gibbs
   overshoot independent of N → the worst-case ripple never shrinks → wrong objective.
   Frequency-sampling: exact at samples, but ripples between; optimizing transition samples by
   LP doesn't directly minimize band ripple and band edges drift. So: minimize
   ‖E‖ = max_F W(f)|D(f)-G(f)| directly.
2. Linear phase ⇒ h symmetric/antisymmetric ⇒ H(f)=G(f)e^{j(...)} with G(f) REAL. Four cases
   (sym/antisym × odd/even length). Each G(f) = trig polynomial. KEY: all four → G(f)=Q(f)P(f),
   P(f)=Σ_{k=0}^{n-1} α_k cos(2πkf) a pure cosine (=Chebyshev) polynomial, Q∈{1, cos πf, sin πf,
   sin 2πf}. Absorb Q into D̂=D/Q, Ŵ=W·Q ⇒ ONE problem: best weighted-Chebyshev cosine approx of D̂.
3. cos(2πkf)=T_k(cos 2πf): substitute x=cos(2πf), P becomes an ordinary polynomial of degree n-1
   in x on [-1,1]. So it's polynomial Chebyshev approximation — the classical theory applies.
4. ALTERNATION THEOREM (Chebyshev): P (degree n-1, i.e. r=nfcns cosine terms) is THE unique best
   weighted approx of D̂ on F iff the weighted error E(f)=Ŵ(f)[D̂(f)-P(f)] attains its max magnitude
   ‖E‖ at ≥ r+1 points F_1<...<F_{r+1} with ALTERNATING signs: E(F_i)=-E(F_{i+1}), |E(F_i)|=‖E‖.
   (Why r+1: P-Q' for a competitor would have ≥ r+1 sign changes ⇒ ≥ r+1 roots, but degree ≤ r-1
   ⇒ identically zero. Equioscillation ⇒ optimal.)
5. So if I KNEW the r+1 extremal frequencies, optimality forces r+1 equations:
   Ŵ(F_i)[D̂(F_i)-P(F_i)] = (-1)^i δ for a common level δ. r+1 eqns, r+1 unknowns (the α's + δ).
   Solve linearly. The clean trick (Parks-McClellan): treat P as interpolating the r values
   y_i = D̂(F_i) - (-1)^i δ/Ŵ(F_i) through the r+1 nodes, and δ falls out in closed form:
   δ = [Σ b_k D̂(F_k)] / [Σ (-1)^k b_k / Ŵ(F_k)], b_k = Lagrange weights = 1/Π_{j≠k}(x_k-x_j).
   Evaluate P anywhere by barycentric Lagrange — never form the α's during iteration.
6. REMEZ EXCHANGE: guess r+1 extremals (equispaced on dense grid). Compute δ & P on dense grid.
   Find the actual local maxima of |E| on the grid; these become the new reference set (keep r+1
   of them, alternating sign, largest magnitude). δ grows monotonically toward ‖E‖_opt each pass
   (|δ| nondecreasing); stop when the grid max no longer exceeds δ. Converges in ~6-12 passes.
7. After convergence, recover the α's by inverse DFT of P sampled at r equispaced points, then
   fold α → impulse response h via the case's symmetry. Land on scipy.signal.remez.

## Design-choice → why
- Min-max not LSQ: window/LSQ overshoot fixed ~9% regardless of N; minimax equalizes ripple →
  smaller passband/stopband deviation for given N.
- Reduce 4 cases to one cosine problem (Q·P): a single Remez core handles LP/HP/BP/BS/diff/Hilbert.
- x=cos(2πf): turns trig approx into polynomial approx ⇒ alternation theorem applies verbatim.
- Dense grid (16·(N+1)/2 pts): the continuous max over a band ≈ max over a fine grid; grid density 16
  is enough that grid extrema ≈ true extrema.
- Barycentric Lagrange eval: O(r) per grid point, no linear solve for coefficients each iteration;
  numerically stable.
- Closed-form δ: the alternation equations are solvable in one shot because P interpolates and δ is
  the single free level.
- Exchange whole reference set (multiple exchange) not one point: far faster convergence than
  single-point Remez.
- Weight W(f): lets you trade passband vs stopband ripple ratio (δ_p/δ_s = W_s/W_p).
- Initial equispaced extremals: cheap, generically in the right basin.
