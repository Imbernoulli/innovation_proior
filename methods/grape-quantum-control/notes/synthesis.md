# GRAPE — synthesis notes

## Pain point / research question
Design control pulses u_k(t) (rf fields) that steer a quantum system to realize a target:
a target density operator C (coherence/polarization transfer) or a target propagator U_F.
Dynamics: time-dependent Schrödinger / Liouville-von Neumann.
  - State: ρ̇ = -i[H0 + Σ u_k(t) H_k, ρ]   (Liouville–von Neumann)
  - Propagator: U̇ = -i(H0 + Σ u_k(t) H_k) U,  U(0)=1
Want max overlap of final state with target, in fixed time T.

Before GRAPE (in coupled-spin NMR): gradient of the objective wrt many control amplitudes
was almost always computed by FINITE DIFFERENCE. To get ∂Φ/∂u_k(j) for all m·N controls you
re-run the whole forward evolution once per perturbed control → ~ (m·N + 1) full propagations.
Paper's concrete number: N=500, m=4 → 2001 full time evolutions per gradient. Too costly →
people restricted to small pulse families (composite pulses, Gaussian cascades, splines, Fourier
expansions) with a few dozen parameters. The one prior analytic-gradient exception cited:
Levante et al. — analytic derivatives via eigenvalues/eigenfunctions of the total propagator.

## Lineage / load-bearing ancestors (cite by author/year, elaborate)
- Optimal control theory (Pontryagin maximum principle; Bryson & Ho 1975 "Applied optimal
  control"; Krotov 1996). Core: maximize a terminal cost subject to ẋ=f(x,u). Introduce a
  co-state/adjoint λ that satisfies a BACKWARD equation with terminal condition from the cost;
  then ∂(cost)/∂u(t) = ⟨λ, ∂f/∂u⟩ — gradient at ALL times from ONE forward (state) + ONE
  backward (adjoint) sweep. This is the structural trick GRAPE imports into spin dynamics.
- Liouville–von Neumann / time-dependent Schrödinger equation (Ernst/Bodenhausen/Wokaun;
  Abragam) — the dynamics being controlled; density operator inner product ⟨C|ρ⟩=tr{C†ρ}.
- Matrix-exponential derivative identity: d/dx e^{A+xB}|_{x=0} = e^A ∫_0^1 e^{Aτ} B e^{-Aτ} dτ.
  For a single slice U_j = exp(-iΔt H^{(j)}), perturbing u_k(j) gives, to first order,
  δU_j = -i Δt δu_k(j) H̄_k U_j with H̄_k Δt = ∫_0^{Δt} U_j(τ) H_k U_j(-τ) dτ; for small Δt,
  H̄_k ≈ H_k. This is the per-slice gradient seed.
- Prior NMR optimal control limited to UNCOUPLED spins (Bloch eqs): band-selective pulses,
  robust broadband excitation/inversion. Coupled-spin case (the interesting one) was wide open
  for gradient methods because FD over many slices was prohibitive.
- Quantum control landscape (Rabitz 2004; "trap-free almost always"): for a controllable
  finite-level system with adequate/unconstrained controls, critical points of the fidelity
  landscape are global optima or saddles, not genuine local traps. This is WHY a humble local
  gradient ascent works so well in practice. (Use carefully in-frame: at 2005 derivation time
  this is contemporaneous insight — frame as "the landscape seems benign / extremal points tend
  to be global", grounded, not overclaimed.)

## The derivation (must re-derive in reasoning.md)
1. Discretize T into N slices of Δt=T/N; u_k constant on slice j = u_k(j).
   Slice propagator U_j = exp{-iΔt(H0 + Σ_k u_k(j) H_k)}.
2. ρ(T) = U_N…U_1 ρo U_1†…U_N†.  Φo = ⟨C|ρ(T)⟩.
3. Cyclic invariance of trace → split at slice j:
   Φo = ⟨ U_{j+1}†…U_N† C U_N…U_{j+1} | U_j…U_1 ρo U_1†…U_j† ⟩ = ⟨λ_j | ρ_j⟩
   where ρ_j = forward-propagated state to t=jΔt, λ_j = BACKWARD-propagated target to t=jΔt.
4. Perturb u_k(j): δU_j = -iΔt δu_k(j) H_k U_j (small Δt). Only slice j's factor in ρ_j moves.
   δρ_j = -iΔt δu_k(j)[H_k, ρ_j]. So
   ∂Φo/∂u_k(j) = -⟨λ_j | iΔt [H_k, ρ_j]⟩.        (Eq. 12)
   KEY: ρ_j for all j from ONE forward sweep; λ_j for all j from ONE backward sweep → 2 sweeps
   for the WHOLE m×N gradient, not (mN+1). Adjoint trick.
5. Update: u_k(j) ← u_k(j) + ε ∂Φo/∂u_k(j). Ascent. (conjugate-grad / L-BFGS later.)

Non-hermitian operators (Φo not real):
  - Φ1 = Re⟨C|ρ(T)⟩; gradient = -⟨λ^x_j|iΔt[H_k,ρ^x_j]⟩ - ⟨λ^y_j|iΔt[H_k,ρ^y_j]⟩  (Eq.17)
  - Φ2 = |⟨C|ρ(T)⟩|² ;  ∂Φ2/∂u_k(j) = -2Re{⟨λ_j|iΔt[H_k,ρ_j]⟩ ⟨ρ_N|C⟩}              (Eq.18)

Relaxation (Liouville space, ρ̇=L̂ρ, L̂=-iĤ+Γ̂): same structure, L̂_j=exp(L̂Δt),
  ρ(T)=L̂_N…L̂_1 ρo, Φo=⟨λ_j|ρ_j⟩, ∂Φo/∂u_k(j)=-⟨λ_j|iΔt[H_k,ρ_j]⟩.                  (Eqs.19-24)

UNITARY synthesis (the task's focus):
  U̇=-i(H0+Σu_k H_k)U, U(0)=1.  U(T)=U_N…U_1.
  Phase-sensitive: minimize ‖U_F - U(T)‖² ⇔ maximize Φ3 = Re⟨U_F|U(T)⟩
     = Re⟨ U_{j+1}†…U_N† U_F | U_j…U_1 ⟩ = Re⟨P_j|X_j⟩,
     X_j = U_j…U_1 (forward propagator to jΔt), P_j = U_{j+1}†…U_N† U_F (backward-prop target).
     ∂Φ3/∂u_k(j) = -Re⟨P_j | iΔt H_k X_j⟩.                                            (Eq.28)
  Phase-insensitive (target up to global phase e^{iϕ}): maximize Φ4 = |⟨U_F|U(T)⟩|²
     = ⟨P_j|X_j⟩⟨X_j|P_j⟩.
     ∂Φ4/∂u_k(j) = -2Re{⟨P_j|iΔt H_k X_j⟩ ⟨X_j|P_j⟩}.                                 (Eq.31)

rf-power penalty: Φrf = α Σ_{j,k} u_k(j)² Δt → ∂Φrf/∂u_k(j) = -2α u_k(j)Δt (sign in paper: the
  penalty is subtracted, gradient term -2α u Δt). Amplitude clamp: reset to max after update.
Robustness: sample parameters ω_p, Φtot=Σ_p Φ(ω_p), gradient = Σ_p per-system gradients (Eq.36).

## Why design choices
- Piecewise-constant controls: makes each slice a single matrix exponential; makes U a product
  of unitaries; makes the gradient a clean per-slice expression. Alternative (continuous
  parameterized families) is what limited prior work.
- Adjoint (forward+backward) vs finite difference: the whole point — O(1) sweeps not O(mN).
- |⟨·⟩|² (Φ4) vs Re⟨·⟩ (Φ3): physically the global phase of a propagator is unobservable, so
  Φ4 is the right gate fidelity; Φ3 is "of theoretical interest". Φ4 quotients out the phase.
- Normalized overlap 1/d in code (_overlap divides by dimension): makes fidelity ∈[0,1],
  f_PSU = (1/d)|tr{U_F† U}|.
- ε small fixed step → ascent guaranteed locally; conjugate gradient / L-BFGS to speed up;
  add noise to escape critical points. Trap-free landscape (adequate controls) ⇒ local ascent
  lands on global optimum in practice.

## Canonical code mapping (qutip_qtrl/grape.py grape_unitary)
- _overlap(A,B) = tr(A†B)/dim.
- U_list[idx] = (-1j * H_idx(idx) * dt).expm()   # slice propagators
- Forward sweep U_f (=X_m) appended; Backward sweep U_b accumulates U_list[..].dag() →
  U_b_list[m] (the P without the target U applied yet).
- P = U_b_list[m] @ U      (= P_j, target U_F backprop'd)
- Q = 1j*dt*H_ops[j] @ U_f_list[m]   (= iΔt H_k X_j)
- phase_sensitive: du = -_overlap(P,Q)              = -Re part used → Φ3 grad (Eq.28)
- not phase_sensitive: du = -2 _overlap(P,Q) _overlap(U_f_list[m],P)  → Φ4 grad (Eq.31)
- alpha penalty: du += -2*alpha*u*dt                (Eq.33)
- update: u[r+1] = u[r] + eps*du.real
QuTiP default optimizer (pulseoptim) uses L-BFGS-B (scipy) on these gradients; the grape.py
module above is the bare steepest-ascent version, closest to the original algorithm.

## Empirical facts (context.md only; recall in reasoning, never fabricate)
- FD cost: N=500,m=4 → 2001 full evolutions per gradient (paper's own arithmetic about the
  PRIOR FD method — this is a pre-method fact about existing practice).
- Prior NMR opt limited to uncoupled (Bloch) spins; coupled-spin gradient opt mostly FD.
- Levante et al.: analytic derivatives via eigen-decomposition of total propagator (prior art).
- Validity condition for the first-order/H̄_k≈H_k step: Δt ≪ ‖H0+Σu_k H_k‖^{-1}.
- Control-landscape (Rabitz 2004): controllable + adequate controls ⇒ critical points are
  global optima/saddles, "almost always trap-free". Use in-frame as a benign-landscape premise.
- Do NOT fabricate any achieved gate fidelity / transfer-efficiency numbers (those are the
  method's own results — excluded).

## Code scaffold (pre-method) ↔ final code correspondence
Scaffold: build_slice_propagators(stub), forward/backward sweeps (stub: # the cheap-gradient
trick goes here), objective(stub), gradient(# TODO), ascent loop. Final fills each: U_j=expm,
two sweeps, Φ4, Eq.31 gradient, u += eps*grad.
