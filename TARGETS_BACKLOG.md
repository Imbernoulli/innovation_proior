# Innovation-Prior Targets — Combinatorial Optimization & Classic CS

Wave-based backlog for the "innovation prior" expansion: each target is a **TASK** (combinatorial
optimization / mathematical construction / classic-CS algorithm design), and each subagent
reconstructs the **principled, insight-driven method** for it (NOT a brute-force / evolutionary
search output — that anti-pattern is what we teach the model to transcend). Everything web-grounded
per the `paper-to-reasoning` skill.

Subagents write `methods/<slug>/results/{context,reasoning,answer}.md`; raw materials under
`methods/<slug>/{src,refs,code,notes}/` (gitignored). `methods.json` is updated centrally (not by
subagents) to avoid concurrent-write conflicts. Status: ☐ todo · ◐ launched · ✓ done · ⚠ needs check.

## Wave 1 — launched (AlphaEvolve-core + anchors)

| status | slug | task |
| --- | --- | --- |
| ◐ | circle-packing-in-square | n equal circles in unit square, maximize radius |
| ◐ | kissing-number | max unit spheres touching a central one (LP bound, E8/Leech) |
| ◐ | cap-set | no-3-AP sets in F_3^n (polynomial method / slice rank) |
| ◐ | erdos-minimum-overlap | minimum overlap constant via variational/LP method |
| ◐ | fast-matrix-multiplication | subcubic matmul (Strassen + tensor rank) |
| ◐ | maxcut-sdp | Goemans–Williamson SDP + hyperplane rounding |
| ◐ | heilbronn-triangle | maximize smallest triangle area of n points |
| ◐ | low-autocorrelation-sequences | merit factor, Legendre construction + character sums |
| ◐ | christofides-tsp | metric TSP 3/2-approximation |
| ◐ | hyperloglog | distinct counting in sublinear space |

## Wave 2 — AlphaEvolve math / geometry / packing

| status | slug | task |
| --- | --- | --- |
| ☐ | sphere-packing-lattices | densest lattice/sphere packings; LP bounds (Cohn–Elkies) |
| ☐ | tammes-problem | spread n points on a sphere maximizing min distance |
| ☐ | thomson-problem | min-energy point configurations on a sphere |
| ☐ | sums-and-differences-sets | MSTD sets / Erdős sum-difference constructions |
| ☐ | sum-free-sets | largest sum-free subset; density bounds |
| ☐ | constructive-ramsey | explicit Ramsey-graph lower-bound constructions |
| ☐ | finite-field-kakeya | Kakeya sets over finite fields (polynomial method) |
| ☐ | autocorrelation-inequalities | Erdős autocorrelation / nonnegative-cosine extremal constants |
| ☐ | moser-spindle-chromatic | Hadwiger–Nelson / chromatic number of the plane constructions |
| ☐ | minimum-energy-riesz | Riesz s-energy minimization configurations |

## Wave 3 — fast algebra & numerical methods

| status | slug | task |
| --- | --- | --- |
| ☐ | karatsuba-multiplication | subquadratic integer multiplication |
| ☐ | toom-cook-multiplication | evaluate/interpolate generalization of Karatsuba |
| ☐ | schonhage-strassen | FFT-based near-linear integer multiplication |
| ☐ | harvey-hoeven-multiplication | O(n log n) integer multiplication |
| ☐ | cooley-tukey-fft | O(n log n) DFT |
| ☐ | winograd-convolution | minimal-filtering convolution (used in CNN kernels) |
| ☐ | strassen-tensor-decomposition | matmul tensor rank / border rank framing |
| ☐ | toeplitz-superfast-solver | superfast Toeplitz/structured-matrix solvers |
| ☐ | montgomery-multiplication | division-free modular multiplication |
| ☐ | barrett-reduction | modular reduction via precomputed reciprocal |

## Wave 4 — combinatorial optimization & approximation

| status | slug | task |
| --- | --- | --- |
| ☐ | karlin-klein-tsp | (3/2−ε) metric TSP via randomized rounding |
| ☐ | submodular-greedy | (1−1/e) greedy for monotone submodular maximization |
| ☐ | algorithmic-lll | Moser–Tardos algorithmic Lovász Local Lemma |
| ☐ | jain-iterative-rounding | iterative LP rounding for survivable network design |
| ☐ | karmarkar-karp-binpacking | asymptotic FPTAS for bin packing |
| ☐ | primal-dual-steiner-forest | primal-dual 2-approximation for Steiner forest |
| ☐ | ellipsoid-method | polynomial-time LP via shrinking ellipsoids |
| ☐ | interior-point-lp | Karmarkar / central-path interior-point LP |
| ☐ | lin-kernighan-tsp | variable-depth local search for TSP |
| ☐ | held-karp-lower-bound | Lagrangian 1-tree bound for TSP |
| ☐ | metric-embedding-bourgain | Bourgain embedding into ℓ_p / metric methods |
| ☐ | multiplicative-weights | multiplicative-weights / online learning meta-method |

## Wave 5 — online, streaming & randomized

| status | slug | task |
| --- | --- | --- |
| ☐ | k-server-problem | competitive k-server / work-function algorithm |
| ☐ | online-bipartite-ranking | Karp–Vazirani–Vazirani RANKING (1−1/e) |
| ☐ | adwords-online-matching | online budgeted matching / AdWords |
| ☐ | ski-rental | rent-or-buy competitive analysis |
| ☐ | paging-marking | competitive paging / marking algorithm |
| ☐ | count-min-sketch | frequency estimation in sublinear space |
| ☐ | ams-frequency-moments | AMS sketch for F_2 / frequency moments |
| ☐ | minhash-lsh | Jaccard similarity / locality-sensitive hashing |
| ☐ | karger-min-cut | randomized contraction global min cut |
| ☐ | reservoir-sampling | uniform sampling from a stream |
| ☐ | johnson-lindenstrauss | distance-preserving random projection |
| ☐ | miller-rabin-primality | randomized primality testing |

## Wave 6 — codes, designs, spectral & expanders

| status | slug | task |
| --- | --- | --- |
| ☐ | reed-solomon-codes | algebraic error-correction via polynomial evaluation |
| ☐ | ldpc-belief-propagation | sparse-graph codes + message-passing decoding |
| ☐ | polar-codes | channel polarization (Arıkan) |
| ☐ | expander-ramanujan | explicit expander / Ramanujan graph constructions |
| ☐ | zig-zag-product | combinatorial expander construction |
| ☐ | spectral-sparsification | Spielman–Srivastava graph sparsifiers |
| ☐ | laplacian-solver | near-linear-time SDD/Laplacian solvers |
| ☐ | steiner-systems-construction | combinatorial design / Steiner-system constructions |
| ☐ | fks-perfect-hashing | worst-case O(1) static dictionary |
| ☐ | cuckoo-hashing | worst-case O(1) lookup hashing |

## Wave 7 — sorting networks, search, SAT, scheduling, systems kernels

| status | slug | task |
| --- | --- | --- |
| ☐ | aks-sorting-network | O(log n)-depth sorting network |
| ☐ | batcher-bitonic-sort | bitonic / odd-even merge networks |
| ☐ | cdcl-sat | conflict-driven clause learning |
| ☐ | dpll-backtracking | DPLL with unit propagation |
| ☐ | astar-admissible-heuristics | A* / heuristic-guided search |
| ☐ | pattern-database-heuristics | additive pattern databases (Korf) |
| ☐ | alpha-beta-pruning | minimax with pruning |
| ☐ | monte-carlo-tree-search | UCT / MCTS |
| ☐ | simplex-method | vertex-walking LP |
| ☐ | branch-and-bound | exact discrete optimization via bounding |
| ☐ | gomory-cutting-planes | integer programming via valid inequalities |
| ☐ | dancing-links-exact-cover | Knuth's Algorithm X |
| ☐ | gemm-tiling-autotuning | cache/register tiling + autotuning for GEMM |
| ☐ | register-allocation-coloring | Chaitin graph-coloring register allocation |
| ☐ | list-scheduling | list scheduling for instruction/task scheduling |
| ☐ | polyhedral-loop-optimization | polyhedral model loop transforms |
| ☐ | consistent-hashing | minimal-disruption distributed hashing |
| ☐ | work-stealing-scheduler | provably efficient parallel scheduling |

## Wave 8+ — overflow pool (continue past 80 while tokens remain)

Continuation policy: **after the first 80 are done, keep generating from this pool (and extend it)
as long as token budget remains.** Refill in bounded waves on completion notifications.

| status | slug | task |
| --- | --- | --- |
| ☐ | suffix-automaton | smallest automaton recognizing all substrings |
| ☐ | suffix-array-dc3 | linear-time suffix-array construction (skew/DC3) |
| ☐ | fm-index | search compressed text via BWT + backward search |
| ☐ | aho-corasick | multi-pattern matching automaton |
| ☐ | fibonacci-heap | O(1) amortized decrease-key for Dijkstra/Prim |
| ☐ | link-cut-trees | dynamic-tree path queries |
| ☐ | van-emde-boas | O(log log u) integer priority queue |
| ☐ | fusion-tree | beat the comparison bound for integer keys |
| ☐ | persistent-data-structures | fully/partially persistent structures (fat node / path copying) |
| ☐ | union-find-analysis | inverse-Ackermann amortized analysis |
| ☐ | splay-tree | self-adjusting amortized-optimal BST |
| ☐ | treap-randomized-bst | balance via random priorities |
| ☐ | smawk-algorithm | row minima of totally monotone matrices |
| ☐ | knuth-yao-dp-speedup | quadrangle-inequality DP speedup |
| ☐ | convex-hull-trick | DP optimization via lower envelope of lines |
| ☐ | li-chao-tree | online lower-envelope queries |
| ☐ | mo-algorithm | offline query reordering on sqrt blocks |
| ☐ | heavy-path-fft-string | bitset / FFT string-matching speedups |
| ☐ | edmonds-arborescence | min-cost directed spanning tree |
| ☐ | stoer-wagner-min-cut | global min cut without flows |
| ☐ | gale-shapley-stable-matching | deferred-acceptance stable matching |
| ☐ | hungarian-assignment | min-cost bipartite assignment via dual potentials |
| ☐ | hopcroft-dfa-minimization | partition-refinement DFA minimization |
| ☐ | thompson-nfa | regex → NFA simulation |
| ☐ | earley-parser | parse any context-free grammar |
| ☐ | pratt-parsing | precedence parsing via binding powers |

Extend this pool further (more classic algorithms / open combinatorial problems) if it empties
before the budget does.

## ⚠ QUALITY GATE (added after audit) — grounding is mandatory

The project's point is to reconstruct the **inventor's actual path to the method** (pain → tools →
wall → self-correction → the idea), and that path MUST be grounded in retrieved sources, not memory.
Audit of the first 30 found 9 with an EMPTY `refs/` → almost certainly written from memory → must be
redone. New hard rule for every (re)generation prompt:

1. **Author's primary source, full text, downloaded to `refs/`** — non-negotiable. Old papers without
   arXiv are fine; get the PDF/scan/book chapter and READ IT IN FULL. The trace must follow the
   author's own development/motivation, not a textbook restatement.
2. **Antecedent method(s) it builds on / reacts against** — download + read; reconstruct what they did
   and exactly where they fell short (best-effort; if unfindable, say so explicitly in `notes/`).
3. **At least one analysis/explainer dissecting the idea's origin** — best-effort, same caveat.
4. The agent's final report MUST list the actual files in `refs/`. **No primary source in `refs/` →
   do NOT write `reasoning.md`** (report the blocker instead).

### REDO list (refs/ empty in first batch — regenerate with the gate above)
`maxcut-sdp` `thomson-problem` `autocorrelation-inequalities` `chromatic-number-plane`
`karatsuba-multiplication` `winograd-convolution` `barrett-reduction` `held-karp-bound` `ski-rental`

### Watch (weak grounding — recheck, maybe redo)
`montgomery-multiplication` (1 ref only) · `erdos-minimum-overlap` (2 refs, thin 1871-word reasoning)

### Still PARTIAL / MISSING (regenerate with the gate)
PARTIAL: `sphere-packing-lattices` `sidon-sets` `schonhage-strassen` `karlin-klein-tsp`
`multiplicative-weights` `minhash-lsh` `karger-min-cut` `reservoir-sampling`
MISSING: `toom-cook-multiplication` `harvey-hoeven-multiplication` `toeplitz-solvers` `algorithmic-lll`
`karmarkar-karp-binpacking` `primal-dual-steiner-forest` `ellipsoid-method` `interior-point-lp`
`lin-kernighan-tsp` `bourgain-embedding` `k-server-problem` `online-bipartite-ranking`
`adwords-online` `competitive-paging` `ams-frequency-moments`

### Operational lessons
- Concurrency ~10–12 is the safe ceiling; 42-at-once tripped server-side rate limiting and killed
  many agents before they wrote anything. Refill in waves of ~8 on completion, keep ≤~14 in flight.
- Session usage limit hit twice; on reset, resume from the lists above.

## ❓ OPEN: 60 orphan methods (have reasoning.md, NOT in methods.json, refs empty locally)

60 method dirs have results/reasoning.md but are absent from methods.json and have empty refs/
(adapter, awq, blip2, bpe, chain-of-thought, deepseek-r1, rag, react, whisper, vllm, … and
aho-corasick, ldpc-belief-propagation). These are NOT from this session's batch. Likely cause:
generated on another machine by a colleague/Kimi and the gitignored refs/ were never pushed — so
"refs empty locally" does NOT by itself prove they were written from memory. **User is checking with
the colleague before we touch them.** DO NOT modify/redo/ingest these pending that confirmation.

EXCEPTION already actioned: aho-corasick and ldpc-belief-propagation are being REDONE now regardless,
because the user found a hard in-frame violation in them (reasoning.md titled "Reconstruction of the
Reasoning Process" / "Reasoning Trace" with ## Step/## Part section structure — exposing the
meta-reasoning of constructing the trace instead of being the inventor's first-person discovery).
Full-scan result: those two are the ONLY reasoning.md files in the whole repo with markdown-header
(C_rsn_header) violations.

RULE reaffirmed by user: reasoning/answer must NOT expose "the reasoning for constructing this
reasoning" — no "Reconstruction of…/Reasoning Trace" titles, no Step/Part scaffolding, no markdown
headers in reasoning.md (only the five allowed section headers in context.md). reasoning.md IS the
inventor's first-person discovery monologue, not an essay about reconstructing one.

## ⛔ STOP-ADVANCING (user directive) — no new generation after the current wave

User said to stop pushing forward after the wave launched alongside this note (the 6 partial-restarts
+ 6 missing: karger-min-cut, karlin-klein-tsp, minhash-lsh, multiplicative-weights, reservoir-sampling,
schonhage-strassen, toom-cook, algorithmic-lll, karmarkar-karp-binpacking, primal-dual-steiner-forest,
interior-point-lp, lin-kernighan-tsp). Let those + in-flight Codex sweeps finish. Then: NO new topics
(don't start the remaining missing: harvey-hoeven, toeplitz, bourgain, k-server, adwords,
competitive-paging, sphere-packing, sidon, or the SELF_ACCOUNT_CANDIDATES backlog). Remaining work is
finishing/verification only: drain in-flight, run Codex sweep on done-but-no-sentinel, then hand off.

## ▶ HANDOFF SNAPSHOT (newest) — 62 complete, 37 Codex-confirmed, methods.json DONE (290)

- methods.json: all 62 completed batch methods are now registered (254→290; added 36 that were missing,
  26 were already in). Validated: JSON parses, 0 dangling entries. Only lin-kernighan-tsp not yet in
  (still generating) — add it when it lands.
- Still NEED CODEX (~25): re-derive the no-sentinel list from disk (command in this file) and run §2.6
  in small waves once the Codex usage limit clears (it returned "try again Jun 7" mid-session).
- lin-kernighan-tsp: finish generation, then add its methods.json row.
- Everything else below is the prior, still-valid detail.

## ▶ HANDOFF SNAPSHOT (prior) — batch 63 slugs: 62 complete (3/3), 32 Codex-confirmed

Generation is essentially DONE for the batch. Remaining work is ONLY: (a) finish lin-kernighan-tsp
(still 0/3, in flight), (b) run the Codex gate on the 30 complete-but-unconfirmed methods below when
the Codex/usage limit clears (Codex started returning "usage limit" mid-session — interior-point-lp
fell back to a verified MANUAL re-derivation and deliberately did NOT fake a sentinel). Do NOT start
any new topics (STOP-ADVANCING still in force).

**NEED CODEX (30)** — complete 3/3, no `.codex_done`; run §2.6 each in small waves (~6), verify, sentinel:
algorithmic-lll, christofides-tsp, chromatic-number-plane, circle-packing-in-square, constructive-ramsey,
cooley-tukey-fft, count-min-sketch, ellipsoid-method, erdos-minimum-overlap, fast-matrix-multiplication,
finite-field-kakeya, heilbronn-triangle, hyperloglog, iterative-rounding-sndp, johnson-lindenstrauss,
karger-min-cut, karlin-klein-tsp, karmarkar-karp-binpacking, kissing-number, low-autocorrelation-sequences,
miller-rabin-primality, minhash-lsh, montgomery-multiplication, online-bipartite-ranking,
primal-dual-steiner-forest, reservoir-sampling, schonhage-strassen, submodular-greedy, sum-free-sets,
sums-and-differences-sets

**STILL GENERATING:** lin-kernighan-tsp (let it finish; it self-runs Codex).
**Note:** 6 sweep agents (fast-matmul, kissing, finite-field-kakeya, heilbronn, cooley-tukey-fft,
miller-rabin) were launched but likely hit the Codex usage limit; re-check their sentinels — any that
didn't land are already in the NEED-CODEX list above.
**Also pending earlier:** add each completed method's row to `methods.json` (`{slug,title,domain,arxiv|doi|url}`)
and commit in batches — NOT yet done for this batch (subagents were told never to touch methods.json).

## ⧗ PENDING CODEX SWEEP — verifiable review for the unconfirmed (run when generation wave drains)

`.codex_done` sentinel is the ONLY trusted proof of a real Codex pass (agents' self-reports are not
trusted — the shared serial Codex queue + rate limits caused several to claim a pass without one, the
same failure SESSION_REVIEW_NOTES.md warned about). Codex-confirmed in the new batch: 11
(ams, autocorrelation, barrett, cap-set, held-karp, karatsuba, maxcut, ski-rental, tammes, thomson,
winograd). The following 34 have reasoning.md but NO confirmed sentinel — run a dedicated, write-enabled
Codex review-and-fix over each, in SMALL waves (~6) to not clog the serial queue, then verify the
sentinel landed:

christofides-tsp, chromatic-number-plane, circle-packing-in-square, constructive-ramsey,
cooley-tukey-fft, count-min-sketch, dijkstra-shortest-path, ellipsoid-method, erdos-minimum-overlap,
fast-matrix-multiplication, finite-field-kakeya, hamming-codes, heilbronn-triangle, hyperloglog,
iterative-rounding-sndp, johnson-lindenstrauss, karger-min-cut, karlin-klein-tsp, kissing-number,
knuth-morris-pratt, low-autocorrelation-sequences, miller-rabin-primality, minhash-lsh,
montgomery-multiplication, multiplicative-weights, online-bipartite-ranking, quicksort,
renormalization-group, reservoir-sampling, schonhage-strassen, simplex-method, submodular-greedy,
sum-free-sets, sums-and-differences-sets

SWEEP-DISPATCH RULE (learned the hard way): only launch a Codex-sweep agent for a method that has
ALREADY RETURNED a completion notification AND still lacks `.codex_done`. Never sweep an in-flight
method — its generation agent may be running its own Codex pass on the same 3 files (observed with
dna/pcr/black-scholes: sweep collided with the still-finishing generation agent). Re-derive the
no-sentinel list from disk immediately before each sweep wave, minus anything still in flight.

Concurrency-write hazard noted: multiple agents appending to SELF_ACCOUNT_SOURCES.md concurrently
race on that file (cap-set agent observed it being "repeatedly rewritten"); no loss yet, but prefer
serializing self-account appends or have agents write to per-method notes and merge centrally.

## ✓ QUALITY BENCHMARK — karatsuba-multiplication (the bar to match)

The karatsuba REDO is the reference standard for what "grounded in the author's self-account" should
produce. Its reasoning.md OPENS from Karatsuba's real motive (per his 1995 retrospective): he set out
to PROVE Kolmogorov's n² lower bound, built the tightest 4-sub-multiplication recursive scheme, got
n² and thought it confirmed the conjecture — then "stare at where the n² came from… it came entirely
from the 4," realized the construction built to confirm the wall instead demolished it, attacked
"is four forced?", saw the middle coefficient only needs the SUM of cross terms, and recovered it
from (a₁+a₂)(b₁+b₂) minus the corners. It also uses the primary paper's actual SQUARING formulation
(not the textbook two-product one). refs/: 1962 Doklady Russian ORIGINAL + 1995 retrospective (OCR) +
Ofman companion + 3 analyses; honest gap note on the unobtainable 1963 English translation.
Match this: real motive → genuine wall → self-correction → the idea falls out, all in the author's path.

## ★ Priority add-ons — methods with strong AUTHOR SELF-ACCOUNTS (do these early)

Added per the "author records own thinking" insight. These have a known first-person account /
manuscript / award lecture, so their `reasoning.md` can be backed by the inventor's real narrated
path (see `SELF_ACCOUNT_SOURCES.md` and skill §1.2b). High value, do ahead of generic backlog.

| status | slug | self-account hook |
| --- | --- | --- |
| ☐ | quicksort | Hoare's own account of inventing it (1959–60, Moscow, machine-translation project) |
| ☐ | dijkstra-shortest-path | Dijkstra's retrospective ("designed it in ~20 min at a café"); EWD archive |
| ☐ | dijkstra-self-stabilization | EWD391a and related EWDs; his own framing |
| ☐ | hamming-codes | Hamming's *You and Your Research* + his book recount the Friday-night frustration |
| ☐ | knuth-morris-pratt | Knuth's account of deriving it from Cook's theorem; his writings |
| ☐ | dancing-links | Knuth's own paper is itself a first-person exposition |
| ☐ | fft-cooley-tukey-history | Cooley's "how the FFT came to be" historical note (distinct from the algorithm slug) |
| ☐ | tex-line-breaking | Knuth–Plass; Knuth's *Digital Typography* recounts the design path |
| ☐ | simplex-method | Dantzig's own retrospective on inventing the simplex method |
| ☐ | renormalization-group | Wilson's Nobel lecture narrates the path |
| ☐ | nash-embedding | Nash's account / Villani & others on the genesis |

(When doing any EXISTING backlog method, still check `SELF_ACCOUNT_SOURCES.md` first — many already
have a usable self-account even if not listed here.)

## Notes

- Wave 1 launched as background subagents; verify the first completed result end-to-end (output dir,
  in-frame voice, web-grounding, math) BEFORE scaling — a systematic mistake should not be amplified 80×.
- After each method's three files land and pass review, add its `methods.json` entry centrally:
  `{ "slug": ..., "title": ..., "domain": ..., "arxiv": <id|DOI|url> }` and commit in batches.
- Suggested domains: `Combinatorial optimization`, `Computational geometry`, `Extremal combinatorics`,
  `Fast algorithms`, `Approximation algorithms`, `Streaming & sketching`, `Coding & expanders`,
  `Search & SAT`, `Systems & compilers`.
- The earlier `ALGORITHM_CANDIDATES.md` was memory-drafted and pedagogical; this file supersedes it
  for the task-oriented, web-grounded effort.
