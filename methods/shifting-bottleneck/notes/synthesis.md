# Synthesis — Shifting Bottleneck Procedure (ABZ 1988)

## Primary source (read in full)
- Adams, Balas, Zawack 1988, "The Shifting Bottleneck Procedure for Job Shop Scheduling",
  Management Science 34(3):391–401 (JSTOR 2632051). PDF: refs/adams-balas-zawack-1988.pdf.
  Read all 7 sections + tables + references.

## The problem (Section 1)
- Jobs, machines; each job is a fixed sequence (route) of operations; op i has duration d_i,
  start time t_i. Each machine processes one op at a time. Minimize makespan = min t_n (n = dummy
  finish, 0 = dummy start).
- Formulation (P):
  min t_n
  s.t. t_j - t_i >= d_i  for (i,j) in A      [conjunctive: job precedence]
       t_i >= 0
       t_j - t_i >= d_i  OR  t_i - t_j >= d_j  for (i,j) in E_k, k in M  [disjunctive: machine]
- Disjunctive graph G=(N,A,E) [Roy & Sussman 1964]: nodes = operations, conjunctive arcs A =
  precedence, disjunctive edges E = pairs on same machine. Weight on arc out of i = d_i.
- E decomposes into cliques E_k, one per machine. A *selection* S_k orients each disjunctive edge
  in E_k; acyclic selection = a sequence on machine k. Complete selection S = union of S_k.
- D_S = (N, A ∪ S): orient all machines. **Makespan of best schedule for S = length of longest
  path in D_S.** Problem = find acyclic complete selection S minimizing longest path. (NP-hard;
  Garey-Johnson 1979.)

## The approach (Sections 2–3)
- NP-hard to orient all machines at once. Decompose: sequence machines ONE at a time.
- M_0 = machines already sequenced (start empty). For each unsequenced machine k in M\M_0:
  build single-machine subproblem by (i) replacing E_p, p in M_0, by their fixed selections S_p,
  and (ii) DELETING the disjunctive arcs E_p for p in M\M_0, p≠k (other unsequenced machines
  ignored). This yields a 1-machine max-lateness problem P(k,M_0).
- **Bottleneck = machine m with the LARGEST optimal subproblem value:**
  v(m,M_0) = max{ v(k,M_0) : k in M\M_0 }, where v = optimal Lmax of P(k,M_0).
  Rationale: criticality (Balas 1969) is a yes/no property; we need a *degree*. The marginal
  utility of sequencing a machine in reducing makespan is the ideal but impractical to assess;
  the optimal 1-machine Lmax is the operational proxy.
- Algorithm skeleton:
  Step 1: identify bottleneck m among M\M_0, sequence it optimally, M_0 ← M_0 ∪ {m}.
  Step 2: reoptimize each critical machine k in M_0 (solve P(k, M_0\{k})), keeping others fixed.
          If M_0 = M stop; else go to 1.

## Heads and tails (Section 3) — the load-bearing derivation
- P(k,M_0) = single machine, op i has besides d_i a **release time r_i** (head) and **due date
  f_i** (tail / delivery): r_i = L(0,i), f_i = L(0,n) - L(i,n) + d_i, where L(i,j) = length of
  longest path from i to j in D_T, T = union of S_p over p in M_0. L(0,n) = current makespan.
- Equivalently the 1|rj,qj|Cmax form (P*(k,M_0)):
  min t_n  s.t.  t_n - t_i >= d_i + q_i ;  t_i >= r_i ;  disjunction on E_k.
  with q_i = L(0,n) - f_i = tail (delivery time after op).
- Three-machine view: each job processed on three machines in order; first & third have infinite
  capacity (process head r_i and tail q_i respectively), middle = machine k (one at a time, p=d_i).
- Computing r_i, q_i = two longest-path passes in D_T. Naively O(α) with α=O(n^2) (D_T dense:
  complete subgraph per sequenced machine — its transitive closure). But an acyclic complete
  digraph IS the transitive closure of its unique Hamilton path, so only the n-1 Hamilton-path
  arcs matter; delete the rest → D_T* where every non-source/sink node has 1–2 predecessors →
  labeling algorithm runs in **O(n)**. In implementation: keep a "job list" (ops per job) and a
  "machine list" (ops per sequenced machine); each node has its 1–2 neighbors on these lists.

## Solving the one-machine problem (Section 4) — Carlier 1982
- P*(k,M_0) is the single-machine 1|rj,qj|Cmax = 1|rj|Lmax, NP-hard in strong sense but solvable
  fast by branch-and-bound (Carlier 1982; closely related to McMahon-Florian 1975).
- Carlier's iterative step (Schrage / MWKR-style heuristic for upper bound + critical set):
  - t := min{ r_j : j in N* }, Q := N*. Repeat: among ready jobs (r_j <= t) in Q, pick j with
    largest q_j (ties → largest d_j); schedule it: t_j := t, Q := Q\{j}; then
    t := max{ t_j + d_j , min{ r_i : i in Q } }. Stop when Q empty.
  - From the schedule's critical path get critical sequence j(1..p) (most-arc one if several).
    Let k = largest index in 1..p with d_{j(k)} < q_{j(p)}. The "critical job" is j(k).
    J := { j(k+1), ..., j(p) }.
- Lower bound: h(J) := min{ r_i : i in J } + sum{ d_i : i in J } + min{ q_i : i in J }.
  (A block J all-or-nothing: must wait for earliest release, run all of them, then earliest tail.)
- Branching on j(k): either j(k) goes BEFORE all of J (give it new tail q'_{j(k)} large enough to
  force it before J) or AFTER all of J (new head r'_{j(k)} large). Prune by h(J) >= incumbent.
  Trees are small, rarely exceed 2n nodes.
- The Columbia lecture notes (refs/columbia-bb-1rjLmax.pdf) give the simpler relaxation lower
  bound: 1|rj,pmtn|Lmax solved by **preemptive EDD** — preemption only helps, so this is a valid
  LB for the non-preemptive problem at each B&B node. (Worked example: 4 jobs, LB=5, optimum=7.)
- Cycle wrinkle: occasionally the optimal selection of P(k,M_0) creates a cycle when added to D_T.
  Handled by re-solving that subproblem subject to the offending precedence constraint (rare).

## Local reoptimization (Section 5)
- Given M_0, order k(1)..k(p). For i=1..p: solve P*(k(i), M_0\{k(i)}) and substitute the new
  selection S_{k(i)}. Up to ~3 full cycles per M_0; at the last step (|M_0|=|M|) continue until no
  improvement. After each full cycle, reorder M_0 by decreasing subproblem value.
- Cost = O(max{|M|^2, γ|M|}) one-machine problems, γ = gap between SB makespan and optimum.
- Extra refinement: temporarily remove the last α noncritical machines (S_k has no arc on a
  critical path), α = min(|M_0|^{1/2}, #noncritical), then reinsert one by one — finds extra gains.

## Selective enumeration (Section 6, SBII)
- Apply SB to nodes of a partial enumeration tree: node = a way of sequencing M_0; a node's
  successors = the f(l) highest-ranking (by v(k,M_0)) subproblems, l = |M_0| = level, f decreasing.
  Penalty function discards nodes whose choices deviate too far from the bottleneck. Search =
  breadth-first to level l* = ceil(|M|^{1/2}), then depth-first by best group. Best value kept as
  upper bound for pruning.

## Code grounding
- Disjunctive graph + O(n) longest path + makespan: code/sbp_corsini/graph.py (forward longest
  path = heads; reverse longest path = tails), code/sbp_rafaellucas/code/classes.py (_forward S,
  _backward Cp; computeLmax by permutation for tiny machines).
- Full SBP loop (bottleneck selection by max Lmax, reoptimize cycles, noncritical reopt, last
  reopt): code/sbp_corsini/ShiftingBottleneck.py. head = long_to[op] - p; tail = ms - long_from[op]
  + p. Bottleneck = argmax lmax. max_reopt=3 (as in Adams 1988).
- 1|rj|Lmax subproblem: Corsini solves via OR-Tools CP-SAT (solve_lmax: interval vars, NoOverlap,
  minimize max(end - tail)); the paper uses Carlier's B&B. Both valid; CP-SAT is the clean modern
  drop-in.
- Makespan EVALUATION of final full schedule: job_shop_lib ORToolsSolver
  (code/job_shop_lib/.../constraint_programming/_ortools_solver.py) and OR-Tools jobshop_sat.py —
  interval vars per op, end==start+dur, job precedence end<=next.start, AddNoOverlap per machine,
  AddMaxEquality makespan, minimize.

## Design decisions → why
- One machine at a time (not all): full orientation NP-hard; single-machine relaxation is the
  cheapest nontrivial relaxation that still couples ops through the graph (heads/tails carry the
  rest of the shop).
- Bottleneck = max Lmax (not criticality): need a *degree* of bottleneck, not yes/no. Subproblem
  optimum value is computable proxy for marginal makespan impact. Sequencing the worst machine
  first commits the hardest decision while the graph is least constrained.
- Heads/tails from longest paths: they summarize everything OUTSIDE machine k (jobs' earlier ops =
  head, later ops = tail) into a single 1-machine instance — exact given the current fixed
  selections.
- Ignore other unsequenced machines (delete their E_p): they aren't oriented yet, so they'd add no
  valid precedence; including unoriented disjunctions would make the subproblem the full NP-hard
  problem again. Their omission makes heads/tails *optimistic* (a relaxation) → v(k,M_0) lower-
  bounds true impact, consistent with using it as a bottleneck score and the first-level LB.
- Lmax minimization (vs Cmax on the subproblem): the subproblem's "makespan with tails" =
  Cmax_sub = current_ms + Lmax. Minimizing Lmax = minimizing the new graph longest path through k.
  Lmax >= 0 means sequencing k pushes makespan up by exactly Lmax.
- Carlier B&B for the subproblem: 1|rj|Lmax is NP-hard but tiny in practice; Schrage gives a near-
  optimal start, h(J) gives tight bounds, trees ~2n. Exact solution matters because the bottleneck
  RANKING depends on it.
- Reoptimize after each insertion: a newly sequenced machine changes every other machine's
  heads/tails, so earlier (greedy, myopic) sequences may now be improvable — local search on the
  decomposition.
- O(n) longest path: longest-path recompute is the inner loop, run for every candidate machine
  every iteration; D_T* (Hamilton-path-only) reduction makes it linear → overall efficiency.

## Empirical / hardness facts (context only — pre-method, sourced)
- Job shop is NP-hard / NP-complete (Garey-Johnson 1979); 10×10 hard while 300–400-city TSP and
  100k-var set cover solve exactly — i.e. JSP is among the worst in practice (stated in §2).
- ft10 = the Muth-Thompson 1963 10-job/10-machine instance, "notorious", defied solution >20 years
  (optimum 930, proved optimal by Carlier-Pinson 1986/Lenstra 1986). Lawrence 1984 = the "la"
  family. These are the yardstick instances (also ft06, abz, orb, swv, ta, yn families later).
- Disjunctive IP / its LP relaxation gives very poor lower bounds for JSP (quoted in King-
  Hildebrand 2024, refs/jss-ip-sbp-dd-2024.pdf) — motivates combinatorial decomposition.
- DO NOT put the proposed method's own benchmark wins (Tables 1–2 makespans) in reasoning.
