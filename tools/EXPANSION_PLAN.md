# Frontier-Engineering Expansion — Blueprint

Extending **Innovation Prior** beyond ML into the **seminal scientific methods behind the
[EinsiaLab/Frontier-Engineering](https://github.com/EinsiaLab/Frontier-Engineering) tasks**. A
Frontier task is an engineering-optimization problem with a *continuous verifier*; we do **not**
reconstruct the task's optimization loop — we reconstruct the **discovery reasoning of the canonical
method each task is built on**. The domain is the carrier; the reasoning trace is the product.

Granularity = **hybrid**: one trace per *distinct method* (tasks that share a method collapse to one),
plus a short per-instance-family historical note where instance sets differ. Every trace obeys the
**three-source bottom line** (primary full text + background ancestors + third-party explainer, even
for pre-arXiv classics) and the **empirical-content discipline** (observed phenomena → `context.md`;
the narrator recalls/derives, never fabricates an experiment). See the `paper-to-reasoning` skill.

## A. Done so far (23 anchors)

| Frontier domain | Method (slug) | Status |
|---|---|---|
| StructuralOptimization | SIMP (`simp`), MMA (`mma-svanberg`) | live |
| Optics | Gerchberg–Saxton (`gerchberg-saxton`), Dammann grating (`dammann-grating`) | live |
| InventoryOptimization | Guaranteed-Service Model (`guaranteed-service-model`), (s,S)/K-convexity (`scarf-ss-policy`) | live |
| PyPortfolioOpt | Markowitz MVO (`markowitz`), CVaR (`cvar`) | live |
| ReactionOptimisation | NSGA-II (`nsga-ii`), Bayesian opt / EGO (`bayesopt-ego`) | live |
| QuantumComputing | QAOA (`qaoa`), SABRE routing (`sabre-routing`), gridsynth (`gridsynth`) | live |
| JobShop | Shifting bottleneck (`shifting-bottleneck`) | live |
| Robotics | DWA (`dwa`), Vásárhelyi flocking (`vasarhelyi-flocking`), LQR (`lqr`) | live |
| Aerodynamics | Gappy-POD sensors (`gappy-pod-sensors`), Jameson adjoint (`jameson-adjoint`) | live |
| CommunicationEngineering | LDPC + BP (`ldpc`) | live |
| EnergyStorage | SPMe (`spme-battery`) | live |
| Cryptographic | Keccak / sponge (`keccak`) | live |
| KernelEngineering | TriMul / AlphaFold2 (`trimul`) | live |

## B. Methodology spectrum (the reusable "reasoning moves")

The real axis of value. Each is a domain-agnostic scientific-thinking primitive; `→` lists the anchors
that exemplify it (done unless marked ◻ = candidate).

1. **Discrete→continuous relaxation + penalty** → `simp`; ◻ discrete-MIP rebalancing
2. **Adjoint / reverse-mode sensitivity** → `jameson-adjoint`; ◻ differentiable-sim AM, ◻ FWI, ◻ photonic inverse design
3. **Importance sampling / rare-event** → `ldpc` (error floor); ◻ PMD, ◻ Rayleigh deep-fade, ◻ Hamming-IS
4. **Alternating projections (POCS) / fixed point** → `gerchberg-saxton`; ◻ Fienup HIO, ◻ Sinkhorn OT, ◻ ART/CT
5. **Lagrangian duality / KKT / optimality criteria** → `simp` (OC), `mma-svanberg` (dual subproblem)
6. **DP on special structure (tree/serial/lattice)** → `guaranteed-service-model`, `scarf-ss-policy`; ◻ Selinger join-order DP, ◻ Needleman–Wunsch
7. **Surrogate / Bayesian optimization** → `bayesopt-ego`; ◻ TPE, ◻ multi-fidelity BO
8. **Multi-objective / Pareto** → `nsga-ii`; ◻ SPEA2/MOEA-D, ◻ hypervolume/ε-constraint
9. **Continuation / homotopy / annealing** → `simp` (penalty continuation); ◻ simulated annealing, ◻ graduated non-convexity
10. **Convex relaxation / robust & stochastic programming** → `cvar`, `markowitz`; ◻ OPF SOCP/SDP (Lavaei–Low), ◻ chance constraints
11. **IO-aware / memory-hierarchy algorithm redesign** → `trimul` (the op); ◻ FlashAttention*, ◻ MallocLab allocator, ◻ blocked GEMM
12. **Spectral / modal decomposition + ROM** → `gappy-pod-sensors`, `spme-battery` (asymptotic ROM); ◻ Zernike AO, ◻ DEIM
13. **Optimal experimental design / sensor placement** → `gappy-pod-sensors`; ◻ submodular greedy, ◻ muon-detector placement
14. **Receding-horizon / optimal control** → `lqr`; ◻ MPC, ◻ iLQR/DDP, ◻ powered-descent lossless convexification
15. **Graph / routing / scheduling heuristics** → `shifting-bottleneck`, `sabre-routing`, `dwa`; ◻ VRP/Lin–Kernighan, ◻ weather-routing isochrones, ◻ spectrum bin-packing
16. **Variational / energy minimization** → ◻ conformer generation, ◻ force-field torsion fitting, ◻ protein/RNA design
17. **Minimax / exchange (Chebyshev approximation)** → ◻ Parks–McClellan FIR (currently the only clean carrier; an open gap)
18. **Column generation / decomposition (Benders, Dantzig–Wolfe)** → ◻ cutting-stock Gilmore–Gomory, ◻ unit-commitment MILP
19. **Number-theoretic / algebraic synthesis** → `gridsynth`; ◻ NTT, ◻ lattice reduction (LLL)
20. **Permutation/sponge & structural crypto design** → `keccak`; ◻ AES/Rijndael, ◻ Merkle trees
21. **Distance-geometry consistency / message passing** → `trimul`; ◻ MSA progressive alignment, ◻ belief propagation (general)
22. **Bio-inspired collective dynamics + meta-optimization** → `vasarhelyi-flocking`; ◻ Reynolds boids, ◻ ACO/PSO

**Thinnest coverage (highest marginal value to add next): #16, #17, #18** — variational energy
minimization, minimax/exchange, and decomposition/column-generation are each represented by ≤0 done
anchors. Prefer these over a 4th optics or quantum trace.

## C. Next candidate seeds (by carrier availability)

**Have a ready continuous verifier (from the Frontier source repos):**
- OPF convex relaxation (PyPSA) — #10
- Lens design / damped-least-squares (optiland) — #2/#9
- Multiple-sequence alignment / Needleman–Wunsch (Sequoya) — #6
- Reservoir SDP (calvin), weather routing (WeatherRoutingTool/halem) — #6/#15
- Multi-energy unit-commitment MILP (MESMO) — #18

**Clean classics regardless of repo:**
- Parks–McClellan FIR (#17), Gilmore–Gomory cutting stock (#18), Kalman filter (#12/#14),
  compressed sensing / ISTA-FISTA (#1/#10), Selinger join-order DP (#6), Sinkhorn OT (#4),
  lossless-convexification powered descent (#14, complements Astrodynamics),
  GRAPE quantum optimal control (#2), AES/Rijndael (#20), Lin–Kernighan TSP (#15).

## D. Process

- Generation: parallel `paper-to-reasoning` subagents (skill v2.1.0). Each runs its own §2.6 Codex
  review; if Codex is rate-limited it writes a `not_run` marker and falls back to manual re-derivation.
- Review status: durable per-method `methods/<slug>/results/.codex_review.json` (git-tracked) +
  rollup `tools/codex_review_status.json` + report `tools/CODEX_REVIEW_STATUS.md`. The legacy
  `.codex_done` sentinel is gitignored and unreliable.
- Re-gate the `not_run` set with the sequential gate: `bash tools/codex_gate_run.sh <slug...>`.
