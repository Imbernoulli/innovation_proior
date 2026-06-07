# Synthesis — SABRE (SWAP-based bidirectional heuristic routing)

## Pain point / problem (first principles)
- Quantum algorithms in the circuit model assume a 2-qubit gate (CNOT) can act on ANY pair of logical qubits.
- NISQ superconducting devices: physical qubits laid out on a planar chip; a 2-qubit gate is only physically possible between qubits joined by a coupler. Coupling graph G(V,E) is sparse (degree 4–6 on IBM Q20 Tokyo).
- So a logical circuit must be (a) given an INITIAL mapping π: logical→physical, and (b) have its mapping changed over time by inserting SWAP gates (each SWAP = 3 CNOTs) to bring interacting logical qubits onto adjacent physical qubits.
- Cost of inserted SWAPs: more gates → more error (CNOT error ~1e-2, SWAP=3 CNOT); more depth → more decoherence (coherence ~50–100 µs bounds sequential gate count). So minimize #added gates AND depth.
- Formal: given circuit + coupling graph, find initial mapping + mapping transition (SWAP insertions) satisfying all 2-qubit constraints, minimizing added gates and depth. Proven NP-Complete (Siraichi 2018).

## Established empirical/context facts (→ context.md, recalled in reasoning, never "measured")
- IBM Q20 Tokyo: ~20 qubits, planar coupling graph, symmetric CNOT (either direction) on connected pairs; single-qubit error ~4.43e-3, measurement ~8.47e-2, CNOT ~3.00e-2; lifetime ~50µs avg, coherence up to ~100µs.
- SWAP decomposes into 3 CNOTs. SWAP exchanges states of two qubits.
- Early symmetric assumption justified: chen2014 showed superconducting couplings can be symmetric; Q20 supports CNOT both directions, so only SWAP insertion needed (no Reverse/Bridge — those were for asymmetric QX2/QX3/QX5).
- Initial mapping proven to strongly affect final quality (Siraichi 2018, Zulehner 2018).
- NNC (nearest neighbor cost) heuristic widely used in prior routing work (Saeedi 2011, Wille 2016, Zulehner 2018).
- Token swapping: the abstract problem of permuting tokens on a graph by adjacent swaps; NP-hard; qubit routing is a time-extended variant. (context for why exact is hard, greedy is the practical answer.)

## Baselines / ancestors (load-bearing) — with their LIMITATION
1. **Exact / solver formulations** (Maslov 2008 graph-iso; Chakrabarti 2011 graph partition; Shafaei 2013/14 MinLA/MIP; Wille 2014 pseudo-boolean; Venturelli 2017/18 temporal planning). LIMIT: general solvers, can't exploit problem structure, very long runtime, tiny circuits only.
2. **Heuristic routers on ideal 1D/2D lattices** (Saeedi 2011, Lin 2015 PAQCS, Wille 2016 look-ahead, Kole 2016/18). LIMIT: assume regular lattice; not applicable to irregular/restricted NISQ coupling.
3. **Siraichi et al. 2018 (CGO) — "Qubit Allocation".** Proved NP-complete. Exact DP solution (exponential, ≤8 qubits). Heuristic: initial mapping by counting #2-qubit-gates between logical pairs, match to physical degree (NO temporal info). Movement: resolve one CNOT at a time greedily by #2-qubit gates, no global lookahead. Transforms: Reverse / Bridge / Swap (for asymmetric QX). LIMIT: oversimplified heuristic (worse than IBM's), exponential exact part, no global initial mapping.
4. **IBM QISKit mapper.** Partition circuit into layers of non-overlapping gates; random search (heuristic-guided) for satisfying mapping per layer; insert SWAPs. LIMIT: random search infeasible/slow for many circuits.
5. **Zulehner, Paler, Wille 2018 (DATE/TCAD) — BKA (Best Known Algorithm).** Partition circuit into depth layers l0,l1,...; for each transition determine compliant mapping σ̂^i from σ0^i via **A\*** search over combinations of concurrently-applicable SWAPs; admissible heuristic h(σ)=max over CNOTs in layer of (shortest-path-1)·7 (+4 for reversal); g(x)=cost so far. Look-ahead scheme. Initial mapping fixed by gates at the BEGINNING of circuit (layer 0). LIMIT: per-layer mapping search is m!/(m-n)! ⇒ O(exp(N)) time & space; out-of-memory on 16–20 qubit circuits; initial mapping local (only first gates), no global optimization.

## The SABRE answers (method, in discovery order)
### 1. SWAP-based search (vs mapping-based exhaustive)
- Observation: exhaustive search over combinations of concurrent SWAPs/mappings is mostly redundant — only SWAPs touching a qubit currently *blocking* an executable gate can help.
- Build circuit DAG of 2-qubit gates (single-qubit gates are free, no dependency). Front layer F = gates with 0 unexecuted predecessors (indegree 0).
- Main loop: while F nonempty: execute all gates in F whose two logical qubits map to adjacent physical qubits; remove them, add newly-resolved successors. If none executable, generate candidate SWAPs = those involving a physical neighbor of a qubit in F (edges adjacent to front-layer qubits). Score each, pick best, apply, update π. Repeat.
- Candidate count O(N) (worst case all qubits in F, each ≤ degree neighbors). Per 2-qubit gate need ≤ diameter ≈ O(√N) swaps; H cost is O(N). Total per gate O(N·N·√N)=O(N^2.5). Search space O(exp(N))→O(N): exponential speedup.

### 2. Heuristic cost function H (built up in 3 stages)
- H_basic = Σ_{gate∈F} D[π(q1)][π(q2)] — sum of coupling-graph distances of front-layer gates. D = all-pairs shortest path via Floyd–Warshall on G (edge weight 1 = 1 SWAP). Minimizing pulls front-layer qubits together. (D = NNC; the -1 offset that makes adjacent=0 is dropped, doesn't change argmin.)
- **Look-ahead / Extended set E**: H_basic is myopic — a SWAP good for F may push apart gates just behind it. Add E = next ~|E| successors of F-gates in DAG. Normalize each sum by |F|, |E| (different sizes). Weight W∈[0,1) on E term (E less urgent than F). H_LA = (1/|F|)Σ_F D + W·(1/|E|)Σ_E D. Don't look further: mapping drifts too much to estimate far-future cost.
- **Decay** (parallelism/depth control): piling swaps on the same qubit serializes them (depth↑). Maintain decay(q), bumped by δ each time q is swapped, reset every DECAY_RESET_INTERVAL=5 steps or when a gate executes. H = max(decay(q1),decay(q2)) · H_LA. Penalizes reusing recently-swapped qubits ⇒ tends to pick non-overlapping (parallel) SWAPs. Tuning δ trades #gates vs depth.
- Final H (Eq. eq:Hopt): H = max(decay(SWAP.q1),decay(SWAP.q2)) · { (1/|F|)Σ_F D + W·(1/|E|)Σ_E D }.

### 3. Reverse-traversal initial mapping (the key SABRE trick)
- Initial mapping matters a lot but no global method existed (Siraichi: gate counts; Zulehner: first gates only).
- Quantum circuits are reversible: reverse circuit has SAME 2-qubit gates, order reversed. Key symmetry: if you run routing forward from a random initial map you END at some final map π_f; that π_f is a GOOD map for the END of the circuit. Feed π_f as the INITIAL map to route the REVERSE circuit; you end at a map that is good for the END of the reverse circuit = the BEGINNING of the original. So the reverse traversal's final map = a refined INITIAL map for the original, informed by the whole circuit, with gates near the start weighted most (they're at the end of the reverse pass, closest to where the map settles) but the rest not ignored.
- Procedure: random π0 → forward traverse original → π_f → use as init for reverse circuit → forward-backward-forward (3 traversals); fake_run (don't keep the SWAPs, only the layout) during layout. Repeat for several random seeds, keep best.
- This is SabreLayout in qiskit (calls SabreSwap with fake_run=True, decay heuristic, forward/backward alternation).

## Design-decision → why table
- DAG over 2-qubit gates only / single-qubit free: single-qubit gates don't constrain mapping (act on one qubit). Front layer = ready-to-run set.
- Candidate SWAPs only adjacent to F qubits: SWAPs inside the idle set can't resolve any current blocker — pruning that gives the exponential→linear speedup. Rejected: search all SWAP combos (Zulehner) = exp.
- D from Floyd–Warshall (O(N^3) once): need all-pairs distance; N small (hundreds) so cubic fine. Edge=1 SWAP ⇒ D = min #SWAPs to co-locate.
- Sum (not max) over F: want to make progress on ALL front gates, not just the worst (Zulehner used max for admissibility in A*; here we're greedy, not A*, so sum better reflects total work).
- Normalize F and E sums by size: |F| and |E| differ; without normalization the larger set dominates arbitrarily.
- W<1 on E: F must execute first; E is a softer hint. W=0.5 in evaluation.
- |E| bounded (~20): larger E = more compute, and far-future estimate is unreliable as mapping drifts.
- decay = max(not sum/product) of the two swapped qubits' decay: a SWAP is "recent/overlapping" if EITHER endpoint was just moved.
- decay reset every 5 steps / on gate execution: keep penalty local in time; forget old penalties so search isn't permanently biased.
- Tie-break random among equal-min SWAPs (seeded): avoids systematic bias; enables multi-trial diversity.
- Multiple random initial seeds + keep best: greedy + random init ⇒ run several, the reverse-traversal polishing makes each good, take best.
- fake_run in layout: during initial-mapping search we only care about the resulting layout, not the routed circuit, so don't emit SWAPs.

## Canonical code
- qiskit-terra 0.18.3 `qiskit/transpiler/passes/routing/sabre_swap.py` — pure Python SabreSwap (faithful): EXTENDED_SET_SIZE=20, EXTENDED_SET_WEIGHT=0.5, DECAY_RATE=0.001, DECAY_RESET_INTERVAL=5; _obtain_swaps, _obtain_extended_set, _score_heuristic(basic/lookahead/decay), _compute_cost.
- qiskit-terra 0.18.3 `qiskit/transpiler/passes/layout/sabre_layout.py` — SabreLayout: random init, forward/backward iterations, SabreSwap(fake_run=True, "decay"), _compose_layouts.
```
```
