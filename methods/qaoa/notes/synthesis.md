# QAOA synthesis notes

## Pain point / research question
Combinatorial optimization: objective C(z) = Σ_α C_α(z), z an n-bit string, C_α a local
clause (depends on few bits), C_α=1 if satisfied. MaxSat / approximate optimization. Want
z with C(z) near max. NP-hard in general; want a quantum algorithm with a *tunable* knob
trading circuit depth for solution quality, implementable with locality no worse than the
objective's locality.

## Ancestors (load-bearing)
1. **Adiabatic quantum computation / quantum annealing (Farhi, Goldstone, Gutmann, Sipser
   2000, quant-ph/0001106).** Encode the optimum as the ground state of a problem
   Hamiltonian H_P (diagonal in computational basis). Start in the easy-to-prepare ground
   state of a "beginning" Hamiltonian H_B = Σ_i ½(1−σˣ_i), whose ground state is the uniform
   superposition |+⟩^⊗n. Interpolate H(t) = (1−t/T)H_B + (t/T)H_P. Adiabatic theorem: if the
   spectral gap E_1(s)−E_0(s) stays positive, then lim_{T→∞}|⟨ground(s=1)|ψ(T)⟩|=1.
   Quantitative condition: T ≫ ℰ/g_min², where g_min = min_s gap and ℰ = max_s |⟨1|dH̃/ds|0⟩|.
   So **runtime scales like 1/g_min²**. PAIN: gap can be exponentially small; runtime
   exponential; and the analog continuous evolution requires a long coherent run — too deep
   for near-term hardware.
2. **Adiabatic theorem** — gives the 1/gap² scaling above.
3. **Trotterization (Lie–Trotter product formula).** e^{−i(A+B)t} ≈ (e^{−iA t/N} e^{−iB t/N})^N
   with error O(t²/N). Lets us approximate continuous evolution under H(t)=A+B by alternating
   short evolutions under A and B. Applied to the adiabatic path: alternate U(C,γ)=e^{−iγC}
   and U(B,β)=e^{−iβB}.
4. **Variational principle.** For any state |ψ⟩ and Hermitian C, ⟨ψ|C|ψ⟩ ≤ C_max (max
   eigenvalue). So maximizing ⟨ψ(θ)|C|ψ(θ)⟩ over a parameterized family is a sound bound:
   you can only do as well as the true max, and any improvement is real. Variational quantum
   eigensolver flavor: parameterize a circuit, optimize parameters classically against a
   quantum-measured objective.
5. **MaxCut, E3LIN2, classical approx baselines** — Goemans-Williamson 0.878 SDP; Halperin et al
   cubic-graph MaxCut; Håstad bounded-occurrence; Trevisan inapproximability.

## The method (re-derivation skeleton)
- Adiabatic path encodes optimum in ground state but runtime ∝ 1/gap² and depth too large.
- Trotterize the path: replace continuous H(t) by p alternating layers
  U(B,β_p)U(C,γ_p)···U(B,β_1)U(C,γ_1). With C the objective operator (here C, *maximize* — so
  we seek the *top* of the spectrum; Farhi uses −C / top-state language) and B = Σσˣ.
- Initial state |s⟩ = |+⟩^⊗n (ground state of H_B, the highest-energy state of C-direction
  start). Cost unitary U(C,γ)=e^{−iγC}=∏_α e^{−iγC_α} (commuting, locality = clause locality;
  γ∈[0,2π] since C integer-valued). Mixer U(B,β)=e^{−iβB}=∏_j e^{−iβσˣ_j}, β∈[0,π].
- The Trotter angles would have to be small and p large for a faithful adiabatic
  approximation. KEY MOVE: stop insisting the angles trace the adiabatic schedule. Make
  (γ,β) **free variational parameters**. Define F_p(γ,β)=⟨γ,β|C|γ,β⟩, M_p=max F_p. Then
  M_p ≥ M_{p−1} (the (p−1) optimum is a constrained p-instance with γ_p=β_p=0... actually
  with the last layer trivial), and lim_{p→∞} M_p = max_z C(z) (because the adiabatic path is
  recovered as a special case of the parameter family — Trotter + adiabatic theorem +
  Perron–Frobenius gives the limit).
- Algorithm: pick p; find good (γ,β) (classical preprocessing for fixed p + bounded degree,
  OR grid/optimization with quantum evaluation of F_p); prepare |γ,β⟩; measure in
  computational basis → string z; repeat. Mean of C(z) = F_p. Concentration: Var(C) ≤
  O(m) so std ~ √m, sample mean within 1 of F_p with O(m²) (or O(m log m)) reps.

## Locality / classical preprocessing (the fixed-p trick)
Each edge term U†...C_⟨jk⟩...U only involves qubits within graph-distance p of edge ⟨jk⟩
(operators outside commute through and cancel). So F_p = Σ_g w_g f_g(γ,β), sum over subgraph
*types* g, w_g = count of type g, f_g computed on a Hilbert space of size 2^{q_tree},
q_tree = 2[((v−1)^{p+1}−1)/((v−1)−1)] (or 2p+2 if v=2), independent of n. So for fixed p and
bounded degree the angles can be optimized by a classical computer whose resources don't grow
with n. Partial derivatives of F_p are O(m²+mn)-bounded, so a poly grid finds the max.

## MaxCut p=1 on 3-regular graphs (the provable ratio)
C = Σ_⟨jk⟩ ½(1 − σᶻ_j σᶻ_k). For p=1, possible subgraphs of an edge: g4 (crossed square,
4 vertices), g5 (5 vertices, in isolated triangles & crossed squares), g6 (tree, 6 vertices).
F_1 = S f_{g4} + (4S+3T) f_{g5} + (3n/2 − 5S − 3T) f_{g6}, S=#crossed squares, T=#isolated
triangles, 3T+4S ≤ n. Best cut ≤ (3n/2 − S − T). Ratio ≥ M_1(1,s,t)/(3/2 − s − t),
s=S/n, t=T/n, 4s+3t≤1. Minimized at s=t=0 → **0.6924**. (Not as good as classical
Halperin et al, but a genuine provable quantum result.)

DERIVED single-edge expectation (triangle-free, each endpoint degree 3, d=2 other neighbors):
⟨C_e⟩ = ½ + ¼ sin(4β) sin(γ)(cos^{d_u}γ + cos^{d_v}γ), d_u=d_v=2.
Maximize → 0.6924496... at γ≈0.616, β≈0.393. **I verified this numerically reproduces 0.6924.**

## Ring of disagrees (2-regular, all p)
2-regular connected = ring. C max = n (even) or n−1 (odd). Single subgraph type (segment of
2p+2 qubits). Numerics: M_p = n(2p+1)/(2p+2) for p=1..6 (3/4,5/6,7/8,9/10,11/12,13/14). So
ratio → 1 as p→∞, *independent of n*, depth 3p (independent of n). Note: p=1 state giving
3/4 has *exponentially small* overlap with optimal strings — QAOA is not the QAA in disguise.

## QAA vs QAOA contrast (Sec on relation to adiabatic)
- Trotterized adiabatic evolution = alternation of U(C,γ),U(B,β), sum of angles = total run
  time. Faithful approx wants small angles + long run time ⇒ large p. This proves
  lim M_p = max C.
- QAA success prob is NOT monotonic in T (Crosson et al 2014 example; rises then drops).
  QAOA's M_p is monotonic in p (M_p ≥ M_{p−1}) — a structural advantage.
- Example where QAA fails (trapped in false min at Hamming weight w=n for subexponential T,
  Farhi 2002 symmetric objective) but QAOA p=1 concentrates near true min w=0.
- Perron–Frobenius: B has non-negative off-diagonal elements, so top state is non-degenerate
  with a gap to the next, guaranteeing adiabatic success for the top-of-spectrum path.

## E3LIN2 application (1412.6062) — full p=1 derivation
Max E3LIN2: each clause x_a+x_b+x_c = 0 or 1 mod 2, exactly 3 vars, each bit in ≤ D+1 eqns.
Objective operator C = ½ Σ_{a<b<c} d_{abc} Z_a Z_b Z_c, d∈{0,±1}. p=1, β=π/4 chosen for
analysis. Evaluate ⟨−γ,β|C|−γ,β⟩ (odd in γ). β=π/4 turns e^{iβB}Z_1Z_2Z_3e^{−iβB} into
Y_1Y_2Y_3. Conjugate by clause-123 contribution: cos(γd)Y_1Y_2Y_3 + sin(γd)X_1X_2X_3.
Insert complete sets, X's are off-diagonal → flip bits → ⟨s̄|e^{−2iγ(z_1C_1+z_2C_2+z_3C_3)}|s̄⟩.
Sum over z gives 4 cosines; full term = ⅛ d_{123} E_z[Σ sin(γ(d_{123}±c_1±c_2±c_3))]. Taylor:
½ d²γ + P^k + R^k. Linear coeff = ½ d²=½ always. Remainder bounded via Dinur et al Theorem 5
(E[|c|^{k+2}] ≤ (k+1)^{k+2}(E[c²])^{(k+2)/2}) and E[c_i²] ≤ D. With |γ|≤1/(10D^{1/2}),
|R| ≤ (9D^{1/2}|γ|)^{k+2}. Lower-bound positive term via Dinur Cor 2.7: max over
γ_r=cos(πr/k)/(10D^{1/2}) of |m/2 γ + P^k| ≥ m/(20 D^{1/2} k). Take k=5 ln D →
result m/(101 D^{1/2} ln D), i.e. (½ + 1/(101 D^{1/2} ln D))m equations satisfiable, found
by searching only 5 ln D values of γ.
Typical case (random d signs): E_d[⟨...⟩] = (m/2) sinγ cos^{3D}γ lower bound; γ=g/D^{1/2},
maximize (m/2)(g/D^{1/2})exp(−3g²/2) at g=1/√3 → (½ + 1/(2√(3e) D^{1/2}))m. Variance O(m), so
typical. NOTE: these are the proposed method's *own* results (its provable approximation
guarantees) — per the empirical discipline these are forward-looking guarantees the method
*derives*, not benchmark wins; but to be safe I keep the headline ratios as the math the
derivation produces, and do NOT present any hardware/benchmark numbers. The 0.6924 and the
E3LIN2 bounds are analytic consequences of the construction, derived on the page.

## Design decisions → why
- **H_B = Σσˣ / mixer = e^{−iβΣσˣ}**: it's the standard transverse field; its ground state
  (uniform |+⟩) is trivially preparable, it has the required Perron–Frobenius non-negative
  off-diagonal structure (single-qubit X connects all basis states ⇒ irreducible ⇒
  non-degenerate extremal eigenstate with a gap), and e^{−iβσˣ} is a single-qubit gate (RX),
  so the mixer is depth-1 in single-qubit gates. Alternatives (e.g. nothing) don't move
  amplitude between basis states; the X mixer is the minimal driver that does.
- **Initial state |+⟩^⊗n**: ground state of H_B; in the adiabatic story you must start in the
  starting Hamiltonian's ground state, and this is the unique easy one. Also β=γ=0 gives a
  uniform random string = classical guessing baseline, so QAOA strictly contains "guess."
- **Cost unitary e^{−iγC}=∏ e^{−iγC_α}**: terms commute (diagonal), so it's exact, no Trotter
  error *within* the cost layer; locality of each gate = locality of the clause (e.g. ZZ for
  MaxCut edge, ZZZ for E3LIN2). γ∈[0,2π] suffices because C has integer spectrum.
- **Alternating C then B**: this is exactly the Trotter splitting of the adiabatic
  H(t)=A·B + B·C path; one γ + one β per layer is the minimal Trotter step.
- **2p free parameters (one γ, one β per layer)**: the adiabatic schedule would *fix* the
  angles (small, monotone). Freeing them can only help (the adiabatic schedule is one point
  in the search space) and lets a short circuit (small p) punch above the faithful-Trotter
  requirement. Monotonicity M_p ≥ M_{p−1} because layer p with γ_p=β_p=0 reduces to p−1.
- **β=π/4 in E3LIN2**: pure analytic convenience — it sends e^{iβB}Z e^{−iβB} to exactly Y
  (a π/4 rotation maps Z→Y under the X-generated rotation), collapsing the algebra. The
  conclusions only need *some* good (γ,β); β=π/4 is a provable choice, not the optimum.
- **Classical optimizer for angles**: F_p is a smooth function of 2p angles with bounded
  derivatives; for fixed p the landscape has no peaks too narrow for a poly grid, and for
  bounded-degree fixed-p the f_g's are n-independent so the whole F_p is classically
  computable → optimize offline, then run the quantum computer once at the best angles.

## Canonical code (PennyLane), grounded
- qaoa.x_mixer: H = Σ_w X(w). qaoa.maxcut: cost H = Σ_⟨ij⟩ ½(Z_i Z_j − I).
- cost_layer(γ,H) = ApproxTimeEvolution(H,γ,1) = e^{−iγH}; mixer_layer(β,H)=e^{−iβH}.
- MaxCut demo: U_C(γ): for edge CNOT;RZ(γ);CNOT (implements e^{−iγ/2 Z_iZ_j} up to const).
  U_B(β): RX(2β) on each wire. circuit: Hadamards, then for each layer U_C,U_B; objective
  −0.5(|E| − ⟨Σ Z_iZ_j⟩); GradientDescentOptimizer, params shape (2,p), ~30 steps; sample
  bitstrings at optimal params, take most frequent → cut.
