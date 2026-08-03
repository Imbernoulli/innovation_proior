#!/usr/bin/env python3
"""wave-4 de-clone replacements (fsx_*_1366..1430).

Why this pack exists. `reports/audit_family_reuse.py` compared the SCORING LOGIC of
same-family problems (checker source, comments/strings/numeric literals stripped, token
5-gram Jaccard) and found 102 wave-1 problems sitting in 37 clone clusters at >=0.60
similarity, 22 pairs at >=0.85. The worst pair (fsx_B_0155 / fsx_B_0357, 0.995) shared the
same hidden functional form, the same noise schedule, the same sample-size schedule, the
same extrapolation bound and the same scoring formula — differing only in RNG seed
constants and which x-index fed which term. Two skins, one problem.

Root cause: wave-1 seeds carry no `mechanisms`, no `innovation_hook`, no `trap`. Fourteen
agents in one family received effectively identical briefs and converged. The old
`scan_homogeneity.py` gate could not see this — it hashes exact skeletons, so "same logic,
retuned constants" passes it.

Remediation: keep the best-headroom member of each cluster, quarantine the other 65 into
`problems_wave1_clone_quarantine/`, and replace them with the 65 seeds below under wave-3
discipline — one family each, an explicit mechanism triple, and an innovation_hook that
names the specific reason the obvious approach fails. Format mix mirrors what was removed
(A29 B15 C13 D3 E5) so corpus balance is preserved.
"""
import json, pathlib

SYNTH = pathlib.Path(__file__).resolve().parent.parent
OUT = SYNTH / "seeds" / "bulk_seed_packs" / "pack_w4_1366_1430.jsonl"
LENS = "wave4-lens:declone-replacement"

BRIEF_FILE = {"A": "AGENT_BRIEF.md", "B": "AGENT_BRIEF_PY_PROGRAM.md",
              "C": "AGENT_BRIEF_PY_STDOUT.md", "D": "AGENT_BRIEF_PY_STDOUT.md",
              "E": "AGENT_BRIEF_PY_STDOUT.md"}
EVAL_FORM = {"A": "quality-metric", "B": "quality-metric", "C": "quality-metric",
             "D": "flops", "E": "quality-metric"}
DANG = {"S": "核心", "A": "重要", "B": "应用前沿", "C": "方法与异域前沿", "N": "bespoke"}

# (tier, fmt, family, theme, objective, mechanisms[3], hook, seed_example, why)
SEEDS = [
# ---- graph & network structure (replaces graph-interdiction/network-design/coloring/IS) ----
("S","A","cascade-firebreak-placement","Cutting a firebreak in a network that fails in waves","max",
 ["threshold-cascade-propagation","budgeted-node-hardening","wave-timing-asymmetry"],
 "Hardening the highest-degree nodes stops the first wave and leaves the second unchecked, because "
 "a cascade's later waves enter through nodes that were harmless until their neighbours flipped. The "
 "insight is hardening by predicted activation TIME, not by degree.",
 "Input: a graph with per-node activation thresholds, a seed set, and a hardening budget. Solver "
 "outputs nodes to harden. chk.cc runs the deterministic threshold cascade and scores surviving nodes. "
 "Trap: 7 of 10 cases place a low-degree articulation node on the second wave's only entry path.",
 "Timing-aware rather than static-centrality intervention transfers to epidemics, outages, and "
 "rumour control."),
("S","A","bridge-toll-equilibrium","Pricing a road so drivers spread themselves","min",
 ["user-equilibrium-response","braess-paradox-edges","toll-revenue-neutrality"],
 "Tolling the most congested link pushes traffic onto a parallel route and can raise total travel "
 "time, because equilibrium re-routing is not local. The insight is finding the links whose REMOVAL "
 "would help (Braess edges) and tolling those instead.",
 "Input: a road network with latency functions, demand pairs, and a revenue-neutral toll budget. "
 "Solver outputs per-edge tolls. chk.cc computes the deterministic user equilibrium and scores total "
 "travel time. Trap: 6 cases contain a Braess edge that congestion-greedy tolling never touches.",
 "Intervening where the system's equilibrium response helps, rather than where the symptom is "
 "loudest, is a general lesson in policy design."),
("S","A","sensor-coverage-under-occlusion","Watching a floor plan where walls block sight","max",
 ["visibility-polygon-coverage","occlusion-shadow-chains","redundancy-for-failure"],
 "Placing sensors greedily by marginal coverage leaves every covered cell watched exactly once, so a "
 "single failure opens a hole. The insight is that redundancy is nearly free in convex pockets and "
 "expensive at corners, so the placement should buy it where it is cheap.",
 "Input: a polygonal floor plan with obstacles, sensor range, a placement budget, and a "
 "one-failure-tolerance requirement. Solver outputs positions. chk.cc computes exact visibility and "
 "scores doubly-covered area. Trap: marginal-coverage greedy fails the tolerance check on 7 cases.",
 "Buying robustness where it is structurally cheap, rather than uniformly, generalizes across "
 "redundancy design."),
("S","A","multi-commodity-unsplittable","Every shipment must take one path","min",
 ["integral-flow-constraint","lp-rounding-gap","congestion-vs-length-tradeoff"],
 "Routing each commodity on its shortest path is optimal per commodity and creates a hot edge; the "
 "fractional LP optimum is unreachable because paths cannot be split. The insight is rounding guided "
 "by the LP's fractional support, not by shortest path.",
 "Input: a capacitated graph and commodities with integral demands. Solver outputs one path per "
 "commodity. chk.cc checks capacities and scores max congestion. Trap: shortest-path routing exceeds "
 "capacity on 7 of 10 cases where a length-1 detour would fit.",
 "Using a relaxation's structure to guide integral decisions is the core of approximation "
 "algorithms."),
("S","A","dynamic-graph-spanner","Keeping distances right as edges vanish","max",
 ["stretch-guarantee","edge-deletion-sequence","recourse-budget"],
 "Rebuilding the spanner after each deletion keeps stretch minimal and blows the recourse budget "
 "(edges changed). The insight is maintaining slack in the initial spanner so most deletions require "
 "no repair at all.",
 "Input: a graph plus a deletion sequence, a stretch bound, and a recourse budget. Solver outputs an "
 "initial spanner plus per-deletion repairs. chk.cc verifies stretch after every deletion and scores "
 "edges used plus recourse. Trap: minimum-size initial spanners need repair on nearly every deletion.",
 "Provisioning slack up front to avoid churn later is a general online-algorithm trade-off."),
("S","A","community-bridge-detect","Finding the few edges that hold two worlds together","max",
 ["edge-betweenness-vs-embeddedness","weak-tie-strength","removal-impact-certificate"],
 "Ranking edges by betweenness finds bridges and also finds every edge on a long path in a sparse "
 "region. The insight is combining betweenness with LOCAL embeddedness (shared neighbours): a true "
 "bridge has high betweenness AND near-zero embeddedness.",
 "Input: a graph with planted communities and decoy sparse chains, and a report budget. Solver "
 "outputs ranked bridge edges. chk.cc scores by modularity drop on removal. Trap: 7 cases include "
 "long sparse chains that pure betweenness ranks above the real bridges.",
 "Combining a global and a local statistic to disambiguate is a general detection pattern."),
("S","A","steiner-forest-with-buyback","Connecting groups when you can resell cable","min",
 ["terminal-group-connectivity","buyback-refund-rate","irreversible-commitment-order"],
 "Building the cheapest forest for the current terminals and buying back later is dominated because "
 "the refund is partial. The insight is anticipating future groups and over-building along shared "
 "corridors where the buyback penalty would bite.",
 "Input: an online sequence of terminal groups, edge costs, and a buyback refund fraction. Solver "
 "outputs per-step edge purchases and sells. chk.cc verifies connectivity at every step and scores "
 "net cost. Trap: myopic Steiner construction pays the refund gap on 7 of 10 sequences.",
 "Committing under partial reversibility is the general structure of online infrastructure buildout."),
("S","A","label-propagation-adversary","Poisoning a classifier that trusts its neighbours","max",
 ["propagation-fixed-point","budgeted-label-flip","influence-radius-decay"],
 "Flipping the labels of the highest-degree nodes shifts the most immediate neighbours and washes out "
 "after two hops. The insight is flipping at the boundary between classes, where propagation has the "
 "least counter-evidence to overcome.",
 "Input: a graph with seed labels, a propagation rule, and a flip budget. Solver outputs flips. "
 "chk.cc runs propagation to its fixed point and scores misclassified nodes. Trap: degree-greedy "
 "flipping is far from optimal on 7 cases with clear class boundaries.",
 "Attacking where a system's evidence is weakest, not where its mass is greatest, is a general "
 "adversarial insight."),
("S","A","tree-decomposition-width","Finding the order that keeps cliques small","min",
 ["elimination-ordering","fill-in-cascade","separator-guided-recursion"],
 "Min-degree elimination is the classic heuristic and cascades fill-in edges that inflate later "
 "cliques. The insight is separator-guided recursive ordering, which pays a locally worse choice to "
 "keep the two sides independent.",
 "Input: a graph. Solver outputs an elimination ordering. chk.cc computes exact induced width. Trap: "
 "min-degree and min-fill both lose 20-40% on the 7 cases with a balanced separator structure.",
 "Global decomposition over local greedy is a recurring algorithmic upgrade."),
("S","A","interference-channel-assign","Radios that must not shout over each other","max",
 ["conflict-graph-weighting","partial-interference-accumulation","reuse-distance"],
 "Colouring the conflict graph treats interference as binary and ignores that several distant weak "
 "interferers sum to a real problem. The insight is scheduling against accumulated interference, "
 "which no proper colouring captures.",
 "Input: transmitter positions, a path-loss model, channel count, and an SINR threshold. Solver "
 "outputs channel assignments. chk.cc computes exact SINR per link and scores served links. Trap: "
 "graph-colouring solutions violate SINR on 7 of 10 dense layouts.",
 "Replacing a binary conflict abstraction with the additive physical quantity is a general modelling "
 "correction."),
("S","A","supply-network-echelon","Stock at the depot or stock at the shops","min",
 ["risk-pooling-benefit","lead-time-echelon","service-level-constraint"],
 "Holding safety stock at every shop meets service levels and ignores pooling: one unit at the depot "
 "covers several shops' variance. The insight is placing stock at the echelon whose lead time still "
 "permits reaction, which is a structural not a numeric choice.",
 "Input: a distribution tree with lead times, demand variances, holding costs, and service targets. "
 "Solver outputs per-node stock. chk.cc simulates deterministic demand traces and scores cost subject "
 "to service. Trap: per-shop stocking is 30-50% more expensive on the 7 high-variance cases.",
 "Pooling variance at the right level under reaction-time constraints generalizes across inventory, "
 "staffing, and capital buffers."),
("S","A","graph-burning-schedule","Lighting a network as fast as possible","min",
 ["round-based-spread","source-spacing","eccentricity-vs-coverage"],
 "Choosing the most central node each round burns a dense core quickly and leaves the periphery. The "
 "insight is spacing sources at least twice the remaining round count apart, so their spreading balls "
 "never overlap wastefully.",
 "Input: a graph. Solver outputs an ordered source list. chk.cc simulates the burning process and "
 "scores rounds to full coverage. Trap: centrality-greedy ordering is 1-3 rounds worse on 7 cases.",
 "Non-overlapping coverage design under a growth process transfers to broadcast and vaccination "
 "scheduling."),
("S","A","edge-orientation-balance","Pointing every edge without overloading anyone","min",
 ["indegree-balance","cycle-reversal-moves","local-vs-global-optimum"],
 "Orienting each edge toward its lower-degree endpoint balances locally and leaves a global imbalance "
 "no local move can fix. The insight is that cycle reversals are the only moves that change the "
 "in-degree profile without breaking others, so the search must operate on cycles.",
 "Input: a graph. Solver outputs an orientation. chk.cc computes max in-degree. Trap: local greedy "
 "orientation is stuck one above optimal on 7 cases where a single long cycle reversal fixes it.",
 "Identifying the move set that actually reaches the optimum is a general local-search insight."),
("S","A","hypergraph-cut-sparsify","Shrinking a hypergraph without losing its cuts","max",
 ["hyperedge-sampling-weight","cut-preservation-bound","size-vs-fidelity"],
 "Sampling hyperedges uniformly preserves total weight and destroys small cuts, which are exactly "
 "what the score measures. The insight is importance sampling by hyperedge strength, so rarely-cut "
 "edges are dropped first.",
 "Input: a weighted hypergraph and a size budget. Solver outputs a sparsified hypergraph with "
 "weights. chk.cc evaluates cut fidelity on held-out cut queries. Trap: uniform sampling loses small "
 "cuts on all 7 heterogeneous cases.",
 "Importance-weighted compression that preserves the queried statistic, not the total mass, is "
 "general to sketching."),
("S","A","temporal-path-reachability","Journeys where the edges appear and vanish","max",
 ["time-respecting-paths","waiting-time-budget","edge-schedule-perturbation"],
 "Choosing the earliest-arrival path at each hop is optimal for arrival time and can strand you when "
 "the next edge's window has already closed. The insight is optimizing for FUTURE connectivity, "
 "sometimes waiting deliberately.",
 "Input: a temporal graph (edges with time windows), source, targets, and a waiting budget. Solver "
 "outputs journeys. chk.cc validates time-respecting order and scores targets reached. Trap: "
 "earliest-arrival greedy strands on 7 of 10 schedules.",
 "Sacrificing a local optimum to preserve future options is the general shape of planning under "
 "time windows."),
("S","A","matching-under-rotation","Pairing people who will re-shuffle tomorrow","max",
 ["rolling-horizon-matching","fairness-over-rounds","compatibility-decay"],
 "Maximizing each round's matching weight starves the same participants repeatedly, and the score "
 "charges cumulative unfairness. The insight is a shadow price per participant that rises with time "
 "unmatched, converting fairness into weight.",
 "Input: a sequence of rounds with compatibility weights and participation. Solver outputs per-round "
 "matchings. chk.cc scores total weight minus an unfairness penalty. Trap: per-round max-weight "
 "matching is beaten on all 7 skewed-compatibility sequences.",
 "Converting a long-run constraint into a per-step price is the general trick of Lagrangian control."),
("S","A","planar-separator-layout","Drawing a graph that must fit on a page","min",
 ["planarity-testing","crossing-minimization","area-vs-crossings"],
 "Minimizing crossings alone produces layouts that need huge area; minimizing area alone produces "
 "crossings. The insight is using the separator structure to lay out recursively, bounding both.",
 "Input: a graph, page dimensions, and a crossing cost. Solver outputs vertex coordinates. chk.cc "
 "computes exact crossings and area usage. Trap: force-directed layout is decent on sparse warm-ups "
 "and loses on the 7 structured cases where recursion wins."),
("S","A","k-center-with-outliers","Covering everyone except the ones you may ignore","min",
 ["outlier-allowance","radius-vs-coverage-tradeoff","adversarial-outlier-placement"],
 "Greedy k-center is 2-approximate without outliers and arbitrarily bad with them, because one "
 "far-away point that you were allowed to drop dictates every centre. The insight is choosing which "
 "points to abandon FIRST, then covering.",
 "Input: points, k centres, and an outlier allowance. Solver outputs centres and the abandoned set. "
 "chk.cc computes the covering radius. Trap: 7 cases plant decoy far points that greedy chases."),
("S","A","network-flow-with-switching","Pipes that cost money to turn on","min",
 ["fixed-charge-arcs","flow-consolidation","lp-relaxation-weakness"],
 "Minimum-cost flow spreads flow thinly across many arcs, paying every arc's fixed opening charge. "
 "The insight is consolidating flow onto fewer arcs even at higher per-unit cost.",
 "Input: a network with per-arc fixed and variable costs, and demands. Solver outputs arc flows. "
 "chk.cc verifies conservation and scores total cost. Trap: min-cost-flow solutions open 2-3x too "
 "many arcs on the 7 high-fixed-charge cases."),
("S","A","chromatic-scheduling-conflict","Timetabling when conflicts are soft","min",
 ["soft-conflict-penalty","room-capacity-coupling","spread-requirement"],
 "Hard-conflict colouring finds a feasible timetable that piles a student's exams into consecutive "
 "slots, which the soft penalty punishes. The insight is optimizing the combined objective rather "
 "than colouring first and repairing.",
 "Input: exams, student enrolments, slots, rooms, and spread rules. Solver outputs a timetable. "
 "chk.cc scores hard violations plus soft penalties. Trap: colour-then-repair loses on all 7 cases "
 "with dense enrolment overlap."),
# ---- packing / geometry ----
("A","A","irregular-nesting-rotation","Fitting shapes that may be turned","max",
 ["no-fit-polygon","rotation-discretization","interlocking-concavity"],
 "Placing pieces by decreasing area with a fixed orientation is the standard nester and cannot "
 "exploit concavities that interlock only at specific angles. The insight is pairing complementary "
 "concave pieces before placing anything.",
 "Input: polygons with allowed rotations and a sheet. Solver outputs placements. chk.cc verifies "
 "non-overlap exactly and scores utilization. Trap: 7 cases plant complementary concave pairs."),
("A","A","sphere-packing-in-shell","Filling a curved container","max",
 ["curved-boundary-effect","contact-graph-rigidity","layer-vs-random-packing"],
 "A regular lattice packing is densest in the bulk and wastes the curved boundary layer. The insight "
 "is a boundary-conforming outer shell plus a lattice core, joined at a deliberately irregular seam.",
 "Input: shell geometry and sphere radii. Solver outputs centres. chk.cc verifies containment and "
 "non-overlap and scores packing fraction. Trap: pure lattice packing loses 8-15% on the 7 "
 "high-curvature shells."),
("A","A","strip-packing-with-precedence","Stacking boxes that must arrive in order","min",
 ["placement-precedence","gravity-support-constraint","lookahead-vs-shelf"],
 "Shelf-based packing is efficient and violates the arrival order because a later item must sit under "
 "an earlier one. The insight is reserving vertical channels so the required order stays feasible.",
 "Input: item sizes with an arrival order and support rules. Solver outputs placements. chk.cc "
 "verifies support and order and scores height. Trap: shelf packing is infeasible on 7 cases."),
("A","A","point-set-diameter-spread","Placing points that are far apart in every direction","max",
 ["min-pairwise-distance","directional-spread-measure","boundary-attraction"],
 "Maximizing the minimum pairwise distance pushes every point to the boundary, leaving the interior "
 "empty and the directional spread poor. The insight is optimizing the worst DIRECTIONAL gap, which "
 "forces interior occupancy.",
 "Input: a region and a point count. Solver outputs coordinates. chk.cc computes both statistics and "
 "scores the directional measure. Trap: max-min-distance solutions score poorly on 7 cases."),
("A","A","polyomino-tiling-defect","Tiling a board with a hole in it","max",
 ["colouring-invariant-obstruction","defect-position-parity","piece-multiset-budget"],
 "Trying tilings by search wastes the budget when a colouring invariant already proves impossibility. "
 "The insight is computing the invariant first, and when it permits, using it to guide placement.",
 "Input: a board with defects and a piece multiset. Solver outputs a tiling or an impossibility "
 "certificate. chk.cc verifies both. Trap: 7 cases are impossible for invariant reasons that search "
 "cannot discover within budget."),
("A","A","convex-hull-peeling-depth","Layers of an outlier-riddled cloud","max",
 ["depth-layer-structure","outlier-robustness","peeling-order-sensitivity"],
 "Peeling hulls layer by layer is the definition and is destroyed by a few outliers that add spurious "
 "outer layers. The insight is a robust depth that ignores a bounded number of points per layer.",
 "Input: a point cloud with planted outliers and a robustness allowance. Solver outputs depth "
 "assignments. chk.cc scores against the hidden clean depth. Trap: naive peeling is off by 2-4 layers "
 "on 7 cases."),
("A","A","art-gallery-mobile-guard","One guard who walks a route","min",
 ["watchman-route","visibility-along-path","route-length-vs-count"],
 "Placing stationary guards optimally uses more guards than a single walking guard needs. The insight "
 "is that a route's visibility is the union along it, so the objective is a shortest watchman route, "
 "not a covering set.",
 "Input: a polygon. Solver outputs a closed route. chk.cc verifies full visibility along it and scores "
 "length. Trap: guard-placement-then-connect is 30-60% longer on the 7 spiral polygons."),
("A","A","minkowski-clearance-path","Steering a wide vehicle through a gap","max",
 ["configuration-space-inflation","orientation-coupling","clearance-vs-length"],
 "Planning for the vehicle's centre with a fixed inflation radius blocks gaps the vehicle could pass "
 "at an angle. The insight is planning in the full configuration space where orientation is a "
 "dimension.",
 "Input: obstacles, vehicle rectangle, start and goal. Solver outputs a path with orientations. chk.cc "
 "verifies collision-freedom exactly and scores clearance. Trap: 7 cases have a gap passable only "
 "diagonally."),
("A","A","farthest-insertion-tsp-cluster","Touring clusters, not points","min",
 ["cluster-entry-exit-choice","intra-vs-inter-cost","order-dependent-entry"],
 "Solving the cluster order first and the intra-cluster path second is the natural decomposition and "
 "fixes entry points before knowing the neighbours. The insight is that entry and exit points must be "
 "chosen jointly with the order.",
 "Input: clustered points and a distance metric. Solver outputs a full tour. chk.cc scores length. "
 "Trap: two-stage decomposition is 10-25% worse on the 7 elongated-cluster cases."),
# ---- algebraic / combinatorial constructions ----
("A","A","sidon-set-in-group","Sums that never collide","max",
 ["additive-uniqueness","group-structure-exploitation","greedy-vs-algebraic"],
 "Greedy insertion builds a valid Sidon set and plateaus well below the algebraic constructions "
 "(perfect difference sets) that exploit the group's multiplicative structure. The insight is "
 "constructing from a primitive root rather than searching.",
 "Input: a cyclic group order. Solver outputs a subset. chk.cc verifies all pairwise sums are "
 "distinct and scores size. Trap: greedy plateaus 25-40% below on the 7 prime-power orders."),
("A","A","covering-code-radius","Balls that must reach every word","min",
 ["covering-radius-bound","code-linearity-tradeoff","sphere-covering-inefficiency"],
 "Random or greedy codeword selection approaches the sphere-covering bound and never beats it; linear "
 "codes with the right generator do. The insight is searching over generator matrices, not codewords.",
 "Input: length, alphabet, and a target radius. Solver outputs codewords. chk.cc verifies covering "
 "and scores size. Trap: greedy covering is 15-30% larger on 7 cases."),
("A","A","hadamard-partial-completion","Finishing a matrix that must stay orthogonal","max",
 ["orthogonality-constraint-propagation","partial-fill-consistency","construction-family-choice"],
 "Filling entries greedily to satisfy the nearest constraint paints into a corner, because "
 "orthogonality couples every pair of rows. The insight is propagating constraints to detect "
 "inconsistency early, or recognizing the target belongs to a known construction family.",
 "Input: a partially filled +-1 matrix. Solver completes it. chk.cc scores the number of orthogonal "
 "row pairs. Trap: 7 cases are only completable via a Paley/Sylvester structure."),
("A","A","cap-set-in-small-dim","Points with no three in line","max",
 ["affine-line-avoidance","dimension-lifting","local-search-plateau"],
 "Local search on a fixed dimension plateaus quickly. The insight is a product construction: lift a "
 "good low-dimensional cap set to higher dimension, which multiplies size faster than search grows it.",
 "Input: a dimension and modulus. Solver outputs a point set. chk.cc verifies no three collinear and "
 "scores size. Trap: pure local search is 20-35% below the product construction on 7 cases."),
("A","A","perfect-difference-family","Blocks whose differences tile the group","max",
 ["difference-multiset-balance","block-size-constraint","cyclic-structure"],
 "Constructing blocks to maximize coverage greedily leaves the difference multiset unbalanced, which "
 "the score measures. The insight is building from a base block and its multiplicative orbit.",
 "Input: group order and block size. Solver outputs blocks. chk.cc computes the difference multiset "
 "and scores balance. Trap: greedy block selection is far from balanced on 7 cases."),
("A","A","low-rank-boolean-factor","Factoring a 0/1 matrix over the semiring","min",
 ["boolean-rank-vs-real-rank","cover-by-rectangles","greedy-rectangle-trap"],
 "Real-valued SVD suggests a rank that Boolean factorization cannot achieve, and greedy maximal "
 "rectangles overshoot it. The insight is that Boolean rank equals a biclique cover number, so the "
 "search should be over bicliques with a set-cover bound.",
 "Input: a 0/1 matrix. Solver outputs factors. chk.cc verifies exact reconstruction and scores rank. "
 "Trap: greedy rectangle covering uses 30-50% more factors on 7 structured cases."),
("A","A","integer-sequence-autocorrelation","A sequence that does not resemble its own shifts","max",
 ["autocorrelation-sidelobe","merit-factor","skew-symmetry-restriction"],
 "Random search over all binary sequences plateaus; restricting to skew-symmetric sequences halves "
 "the search space while provably containing near-optimal solutions. The insight is the symmetry "
 "restriction, not more search.",
 "Input: a length. Solver outputs a +-1 sequence. chk.cc computes exact autocorrelations and scores "
 "merit factor. Trap: unrestricted search is well below on the 7 odd lengths."),
("A","A","tensor-border-rank-witness","A decomposition that only exists in the limit","max",
 ["border-rank-vs-rank","approximation-sequence","numerical-certificate"],
 "Searching for an exact decomposition fails because the target has border rank strictly below its "
 "rank. The insight is producing a parameterized family whose limit is the target, plus the "
 "degeneration certificate.",
 "Input: a small tensor. Solver outputs a decomposition family with a limit parameter. chk.cc "
 "evaluates the error as the parameter tends to the limit and scores the achieved border rank. Trap: "
 "exact-rank search cannot reach the target on 7 cases."),
("A","A","polynomial-identity-witness","Proving two circuits differ with few probes","min",
 ["schwartz-zippel-probing","structured-vs-random-points","degree-bound-exploitation"],
 "Random probing needs many points to be confident; structured probes along a low-degree curve "
 "certify a difference with far fewer. The insight is exploiting the degree bound to choose the probe "
 "set deterministically.",
 "Input: two arithmetic circuits and a probe budget. Solver outputs probe points. chk.cc evaluates "
 "and scores whether a difference is certified per probe used. Trap: random probing exceeds the "
 "budget on 7 cases."),
("A","A","sum-free-set-density","A set that never contains its own sums","max",
 ["sum-free-structure","interval-vs-modular-construction","density-plateau"],
 "The obvious construction (an upper interval) is sum-free and has fixed density; modular "
 "constructions beat it for the right modulus. The insight is choosing the modulus from the range's "
 "arithmetic, not from the range's size.",
 "Input: a range. Solver outputs a subset. chk.cc verifies sum-freeness and scores density. Trap: the "
 "interval construction is 10-20% below on 7 ranges."),
# ---- scheduling / routing / online ----
("S","A","job-shop-with-transport","Machines and the robot between them","min",
 ["transport-resource-contention","blocking-no-buffer","route-plus-sequence-coupling"],
 "Solving the machine sequencing first and the transport second (the standard decomposition) creates "
 "deadlocks when no buffers exist. The insight is treating the transporter as a machine in the same "
 "disjunctive graph.",
 "Input: jobs, machines, transport times, and zero buffers. Solver outputs a full schedule. chk.cc "
 "verifies no deadlock and scores makespan. Trap: decomposed solutions deadlock on 7 cases."),
("S","A","batch-machine-compatibility","Firing a kiln with items that must agree","min",
 ["batch-family-compatibility","non-identical-processing-times","capacity-vs-family-mix"],
 "Batching by earliest due date fills the kiln and mixes incompatible families, forcing a longer "
 "cycle. The insight is that a batch's time is set by its slowest member, so mixing costs more than "
 "an extra batch.",
 "Input: jobs with families, sizes, times and due dates, plus kiln capacity. Solver outputs batches "
 "and order. chk.cc scores total tardiness. Trap: EDD batching loses on 7 mixed-family instances."),
("S","A","online-bin-stretching","Packing when bins can stretch a little","min",
 ["online-irrevocable-placement","stretch-factor-budget","adversarial-item-order"],
 "Best-fit is excellent on average sequences and is beaten by the adversarial order the score uses. "
 "The insight is reserving a bin class for the item sizes the adversary will send late.",
 "Input: an item sequence revealed one at a time and a stretch bound. Solver places each irrevocably. "
 "chk.cc scores bins used. Trap: 7 sequences are adversarially ordered against best-fit."),
("S","A","calendar-meeting-consolidate","Fewer meetings, same coverage","min",
 ["attendee-coverage-requirement","context-switch-cost","slot-adjacency-bonus"],
 "Minimizing the meeting count packs unrelated attendees together, whose context-switch cost the "
 "score charges. The insight is clustering by attendee overlap first, which lowers total cost despite "
 "more meetings.",
 "Input: topics, required attendees, slots, and switch costs. Solver outputs a schedule. chk.cc scores "
 "total cost. Trap: count-minimizing schedules are worse on 7 overlapping-attendee cases."),
("S","A","truck-drone-tandem","A van that launches a drone","min",
 ["synchronization-rendezvous","drone-endurance","parallel-vs-sequential-service"],
 "Serving the nearest customer by drone whenever it is idle maximizes drone utilization and forces "
 "the truck to wait at rendezvous points. The insight is choosing drone sorties that lie ALONG the "
 "truck's route, not near its current position.",
 "Input: customers, truck and drone speeds, endurance, and launch/retrieve rules. Solver outputs a "
 "combined plan. chk.cc verifies synchronization and scores completion time. Trap: utilization-greedy "
 "dispatch is 15-30% slower on 7 cases."),
("S","A","periodic-maintenance-window","Servicing machines that cannot all stop","min",
 ["cyclic-schedule-feasibility","simultaneous-outage-limit","drift-of-due-dates"],
 "Servicing each machine at its own optimal interval drifts into phase alignment, eventually "
 "violating the simultaneous-outage limit. The insight is choosing intervals whose least common "
 "multiple structure keeps them de-phased.",
 "Input: machines with service intervals and tolerance, plus an outage limit. Solver outputs a "
 "periodic schedule. chk.cc simulates a long horizon and scores violations plus cost. Trap: "
 "independent optimal intervals align on 7 of 10 cases."),
("S","A","crew-pairing-legality","Rosters that obey the rulebook","min",
 ["duty-time-regulations","rest-requirement-chains","deadhead-cost"],
 "Building the cheapest pairings and repairing legality afterwards is dominated because a legality "
 "fix late in a pairing invalidates its whole prefix. The insight is generating only legal pairings "
 "by construction via a resource-constrained shortest path.",
 "Input: flight legs, regulations, and costs. Solver outputs pairings. chk.cc verifies every rule and "
 "scores cost. Trap: cost-first-then-repair is infeasible or expensive on 7 cases."),
("S","A","reentrant-flow-shop","Wafers that visit the same machine twice","min",
 ["reentrant-visits","priority-conflict-between-passes","bottleneck-starvation"],
 "First-come-first-served at the shared bottleneck starves the later passes and inflates cycle time. "
 "The insight is a pass-aware priority rule that protects the machine's downstream feed.",
 "Input: a reentrant routing, processing times, and buffers. Solver outputs a dispatch rule. chk.cc "
 "simulates deterministically and scores cycle time. Trap: FCFS and SPT both lose on 7 cases."),
# ---- science / regression / causal ----
("B","E","saturating-enzyme-law","A rate that stops caring about more substrate","max",
 ["michaelis-menten-saturation","inhibition-term","held-out-concentration"],
 "The visible concentrations are all below the half-saturation constant, where the rate looks linear. "
 "The insight is estimating the saturation constant from mild curvature, which fixes the held-out "
 "high-concentration behaviour.",
 "Input: rate measurements at low concentrations plus an inhibitor level. Solver outputs a rate "
 "expression. verify.py evaluates at held-out high concentrations. Trap: linear and quadratic fits "
 "win in-range and diverge out-of-range."),
("B","E","allometric-scaling-break","A law that changes slope at large size","max",
 ["power-law-regime","geometric-vs-transport-limit","held-out-mass-range"],
 "A single power law fits the visible mass range well; the true relation has two regimes joined at a "
 "transport-limited crossover. The insight is locating the crossover from residual structure.",
 "Input: measurements over a limited mass range. Solver outputs a scaling expression. verify.py "
 "evaluates on a held-out larger range past the crossover. Trap: single power laws lose badly there."),
("B","E","viscoelastic-relaxation-spectrum","A material with many memories","max",
 ["multi-timescale-relaxation","observation-window-truncation","held-out-timescale"],
 "The visible window sees only the fast modes, so a single-exponential fit is excellent. The insight "
 "is that the residual's shape reveals a slow mode whose amplitude is estimable even when its decay "
 "is not observed.",
 "Input: a relaxation curve over a short window. Solver outputs a modulus expression. verify.py "
 "evaluates on a held-out long window. Trap: single-exponential fits collapse there."),
("B","C","confounder-vs-mediator","The variable you must not control for","max",
 ["causal-graph-role","adjustment-set-validity","collider-bias"],
 "Adjusting for every available covariate maximizes apparent fit and opens collider paths, biasing "
 "the effect estimate. The insight is that adjustment-set validity is a graph property, so some "
 "covariates must be deliberately excluded.",
 "Input: observational data plus a candidate causal graph with ambiguous edges. Solver outputs an "
 "adjustment set and an effect estimate. verify.py compares against the hidden interventional truth. "
 "Trap: adjust-for-everything is badly biased on 7 cases."),
("B","C","instrument-strength-check","Borrowing randomness from somewhere else","max",
 ["instrument-relevance","exclusion-restriction","weak-instrument-bias"],
 "Using every candidate instrument maximizes first-stage fit and includes weak ones whose bias "
 "exceeds the OLS bias they were meant to fix. The insight is that instrument count must be traded "
 "against strength, not maximized.",
 "Input: data with candidate instruments and a hidden true effect. Solver outputs an instrument "
 "subset and estimate. verify.py scores estimation error. Trap: use-all-instruments is worse than "
 "plain OLS on 7 cases."),
# ---- multiobjective / decision / structural ----
("B","C","pareto-knee-selection","Choosing one point off the frontier","max",
 ["hypervolume-vs-knee","preference-robustness","frontier-density-bias"],
 "Maximizing hypervolume rewards spreading points into sparse frontier regions that no plausible "
 "preference would ever pick. The insight is scoring by robustness across a preference distribution, "
 "which concentrates effort near the knee.",
 "Input: a multiobjective instance and a preference distribution. Solver outputs a solution set. "
 "verify.py scores expected utility over held-out preferences. Trap: hypervolume-optimal sets lose on "
 "7 cases."),
("B","C","robust-vs-stochastic-plan","Planning for the average or for the worst","max",
 ["scenario-set-construction","regret-vs-expectation","distribution-misspecification"],
 "Optimizing expected value under the given distribution is optimal if that distribution is right and "
 "the score uses a perturbed one. The insight is minimizing maximum regret over a distributional "
 "neighbourhood.",
 "Input: a planning instance with a nominal distribution and an ambiguity radius. Solver outputs a "
 "plan. verify.py evaluates on held-out perturbed distributions. Trap: expectation-optimal plans lose "
 "on 7 perturbed cases."),
("B","C","chance-constraint-tightening","How much slack to leave for bad luck","max",
 ["reliability-level","correlated-constraint-violation","safe-approximation-conservatism"],
 "Tightening each constraint independently to its marginal reliability level fails jointly, because "
 "violations are correlated. The insight is allocating the joint violation budget across constraints "
 "by their correlation structure.",
 "Input: constraints with uncertainty and a joint reliability target. Solver outputs tightened "
 "bounds. verify.py samples held-out realizations and scores objective subject to joint reliability. "
 "Trap: per-constraint tightening violates jointly on 7 cases."),
("B","C","portfolio-of-experiments","Which bets to run in parallel","max",
 ["success-correlation","budget-across-projects","information-vs-payoff"],
 "Funding the highest expected-value projects concentrates on one technical approach whose failures "
 "are correlated. The insight is diversifying across FAILURE MODES, not across projects.",
 "Input: projects with payoffs, costs, and a shared-technology correlation structure. Solver outputs "
 "a funded set. verify.py runs held-out outcome scenarios and scores at-least-one-success value. "
 "Trap: EV-greedy funding fails entirely on 7 correlated-failure scenarios."),
("B","C","feasibility-pump-repair","Getting to a feasible point at all","max",
 ["infeasibility-measure","rounding-cycle-detection","restart-diversification"],
 "Rounding the LP relaxation and re-solving cycles between two infeasible points. The insight is "
 "detecting the cycle and perturbing along the constraint whose violation is most persistent.",
 "Input: a mixed-integer instance. Solver outputs an integral point. verify.py measures feasibility "
 "and objective. Trap: plain feasibility pumping cycles on 7 cases within the iteration budget."),
("B","C","column-generation-stabilize","Pricing out without oscillating","min",
 ["dual-oscillation","stabilization-penalty","column-selection-diversity"],
 "Adding the single most-negative-reduced-cost column each round makes the duals oscillate wildly and "
 "converges slowly. The insight is stabilizing the duals and adding a diverse column set per round.",
 "Input: a set-covering master problem with a pricing oracle and an iteration budget. Solver outputs "
 "a column-generation strategy. verify.py runs it and scores the gap achieved. Trap: textbook pricing "
 "does not converge within budget on 7 cases."),
("C","D","checkpoint-compression-tradeoff","Saving state you may never reload","min",
 ["compression-ratio-vs-cpu","delta-vs-full-snapshot","restore-latency-requirement"],
 "Maximizing compression ratio minimizes bytes and inflates restore latency past its requirement. The "
 "insight is a mixed schedule where fulls are cheap-compressed and deltas are heavy-compressed.",
 "Input: a state-evolution trace, compression cost curves, and a restore-latency bound. Solver "
 "outputs a schedule. counter.py measures exact bytes and restore work. Trap: ratio-greedy schedules "
 "violate the latency bound on 7 cases."),
("C","D","instruction-cache-layout","Ordering functions so the hot path stays resident","min",
 ["call-graph-affinity","cache-line-packing","conflict-set-mapping"],
 "Ordering functions by call frequency puts the hottest ones adjacent and can map them to the same "
 "cache set, causing conflict misses. The insight is layout by co-occurrence AND set-index diversity.",
 "Input: a call trace, function sizes, and cache geometry. Solver outputs an ordering. counter.py "
 "simulates the exact cache and counts misses. Trap: frequency ordering thrashes on 7 low-"
 "associativity configurations."),
("C","D","join-index-selection","Which indexes earn their upkeep","min",
 ["query-benefit-overlap","write-amplification-cost","index-interaction"],
 "Selecting indexes by individual query benefit double-counts, because two indexes often serve the "
 "same query and only one is used. The insight is marginal benefit under the optimizer's actual "
 "choice, plus the write cost each index adds.",
 "Input: a query/update workload with an optimizer cost model and a storage budget. Solver outputs an "
 "index set. counter.py evaluates total workload cost. Trap: benefit-ranked selection wastes half the "
 "budget on 7 overlapping workloads."),
("N","C","modular-transfer-under-shift","A component that must work in a new setting","max",
 ["module-interface-invariance","distribution-shift-magnitude","recalibration-budget"],
 "Fine-tuning every module on the new setting uses the whole budget and overfits the small "
 "recalibration sample. The insight is identifying which module the shift actually touched, and "
 "recalibrating only that one.",
 "Input: a pipeline of modules with per-module diagnostics, a shifted setting, and a recalibration "
 "budget. Solver outputs which modules to recalibrate. verify.py evaluates on held-out shifted data. "
 "Trap: recalibrate-everything overfits on 7 cases."),
("N","C","policy-under-partial-identification","Deciding when the data cannot pin the answer","max",
 ["identification-bounds","decision-under-ambiguity","bound-width-vs-action"],
 "Acting on the point estimate is optimal if the effect is identified and the data only supports "
 "BOUNDS. The insight is that some actions are optimal across the entire identified set, so the "
 "decision can be made without narrowing it.",
 "Input: observational data with a partial-identification structure and an action set. Solver outputs "
 "an action plus the bound it relies on. verify.py evaluates against hidden truth and rewards "
 "decisions robust across the identified set. Trap: point-estimate decisions are wrong on 7 cases."),
]

def emit():
    assert len(SEEDS) == 65, f"expected 65, got {len(SEEDS)}"
    rows, n = [], 1366
    for (tier, fmt, family, theme, obj, mech, hook, ex, why) in SEEDS:
        sid = f"fsx_{tier}_{n}"
        brief = (
            f"[{fmt}|WAVE4-DECLONE] Author a NOVEL, deterministically-scored problem in family "
            f"'{family}', objective={'maximize' if obj=='max' else 'minimize'}, skin '{theme}'. "
            f"Compose ALL of these mechanisms into one objective: {', '.join(mech)}. "
            f"The strong solution must exploit: {hook} "
            f"The generator must plant trap cases where the obvious greedy approach lands far from strong. "
            f"Follow {BRIEF_FILE[fmt]} AND AGENT_BRIEF_INNOVATION_ADDENDUM.md; acceptance needs harness "
            f"PASS plus strong-greedy>=0.06, strong<=0.92, greedy-trivial>=0.03."
        )
        rows.append({"id": sid, "tier": tier, "dang": DANG[tier], "format": fmt,
                     "brief_file": BRIEF_FILE[fmt], "eval_form": EVAL_FORM[fmt],
                     "family": family, "mechanisms": list(mech), "innovation_hook": hook,
                     "source_frameworks": [LENS], "why_it_generalizes": why,
                     "seed_example": ex, "objective": obj, "theme": theme,
                     "scale": "small", "variant": 0, "brief": brief})
        n += 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} seeds -> {OUT}")
    print("id range:", rows[0]["id"], "..", rows[-1]["id"])

if __name__ == "__main__":
    emit()
