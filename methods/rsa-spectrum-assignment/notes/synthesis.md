# Synthesis — RSA (Routing and Spectrum Assignment) in elastic optical networks

## Source status (three-source discipline)
- PRIMARY content (Christodoulopoulos/Tomkos/Varvarigos 2011 EBA; Wang/Cao/Pan 2011 RSA): the original JLT/INFOCOM PDFs are paywalled (IEEE Xplore / Optica) and could not be retrieved as full text after several queries (ResearchGate/SemanticScholar = abstract walls). Their *content* — the path-based ILP with k-shortest-paths, the RMLSA distance-adaptive variant, the Most-Used heuristic, the "minimize max subcarrier index" objective, the SPSR (shortest-path + first-fit, demands in decreasing size) heuristic — is faithfully reproduced and cited as [17],[1],[19] in the two secondaries below. **Gap flagged.**
- BACKGROUND + faithful equations: Talebi/Rouskas et al. "Spectrum management techniques for EONs: A survey" (Optical Switching and Networking 13, 2014), retrieved full text from rouskas.wordpress.ncsu.edu. Gives Definition 1 (RSA), the three constraints, guard carriers, NP-hardness via interval-graph coloring [21] and multiprocessor scheduling P3|fix_j|Cmax [22,24], the "channel" concept [31], First-Fit/Random/Best-Fit, SPSR/BLSA heuristics. This is the load-bearing equations source.
- THIRD-PARTY explainer: Chatterjee/Sarma/Oki "RSA in EONs: A Tutorial" (IEEE Comm. Surveys & Tutorials 17(3), 2015), full text retrieved. Gives RWA→RSA reduction & NP-hardness, continuity/contiguity worked example (Fig 14), routing decomposition (FR/FAR/LCR/AR), spectrum policies (First Fit, Random, Last, First-Last, Least Used, Most Used, Exact Fit), guard band = 2 slots.
- Exact-RMSA ILP arXiv 2309.07621 (David et al.): faithful modern CC + non-overlap + guard-band ILP constraints (continuity zde=zdp, contiguity via no two demands overlapping interval [zde, zde+Fde-1]). Cross-check for the constraint logic.
- Optical-networks overview arXiv 2408.07478: 12.5 GHz slot, slot width = 12.5*m, modulation/reach tradeoff (BPSK/QPSK long reach, 16-QAM short reach, fewer slots), the three constraints restated.
- Canonical code structure: Optical-Networks-Group/rsa-rl (github) is a First-Fit RSA simulator; standard structure = networkx graph, per-link slot occupancy boolean array, k-shortest-paths, first-fit contiguous scan. I will ground the heuristic code in this structure and the ILP in a PuLP/standard channel-based formulation.

## The problem (precise)
Fixed grid (WDM): each link's spectrum is a comb of fixed-width wavelengths; a lightpath needs ONE wavelength, same on every link of its route (wavelength continuity) → Routing-and-Wavelength-Assignment (RWA). WA on a path = interval/graph coloring; NP-hard in general.

Elastic / flex grid (EON): the comb is sliced into fine frequency slots (FS), 12.5 GHz each. A demand of bit-rate r, using modulation m with spectral efficiency b(m) bits/s/Hz, needs n = ceil(r / (b(m)*FS_width)) contiguous slots (plus modulation table: BPSK/QPSK/8QAM/16QAM trade slots for reach). RSA = assign each demand a route + a block of slots subject to:
1. **Contiguity**: the n slots are consecutive (a single interval [s, s+n-1]).
2. **Continuity**: the SAME interval on every link of the demand's route (no spectrum converters).
3. **Non-overlap**: two demands sharing any link get disjoint intervals.
4. **Guard band**: between two demands sharing a link, G empty slots separate them (filter roll-off / crosstalk). Equivalently each demand reserves n+G slots, or non-overlap is enforced with a G-slot gap.
Objective (offline): minimize max occupied slot index over all links (≡ minimize total used spectrum width). 

## Why it's hard (in-frame, derivable)
- Drop continuity (single link): SA on one link with widths n_d and guard = interval scheduling / interval graph coloring where each "color span" is n_d wide. This is the contiguous-block / strip-packing flavor; the interval-chromatic-number problem. On a path of ≥4 links it is NP-hard (Talebi [21],[22]); on ≤3 links poly. Reduction: SA ≡ multiprocessor scheduling P|fix_j|Cmax — task j needs a contiguous time block of length p_j=n_j run *simultaneously* on a fixed set fix_j of machines (= links of its route), minimize makespan Cmax (= max slot index). P3|fix_j|Cmax strongly NP-hard.
- RWA reduces to RSA (set all widths = 1 slot, FS count = wavelength count), and RWA is NP-hard → RSA NP-hard.
- The contiguity constraint is what makes it harder than RWA/coloring: in coloring any free color works; here the n colors must be *adjacent and identical across links*. Fragmentation: free slots exist but not as a contiguous aligned block → blocking.

## The intellectual moves
1. **Decompose R | SA.** Routing is hard to couple, so fix k candidate shortest paths per demand (offline) or route first then assign (online). Then SA alone.
2. **Channel trick** (eliminates explicit contiguity constraints): pre-enumerate, for each width t, all contiguous channels c = [start, start+t-1]. A demand picks ONE channel; contiguity is automatic (a channel is contiguous by construction), continuity is automatic if the channel index is shared along the whole path. Non-overlap becomes: two demands sharing a link must pick channels whose slot-sets are disjoint (including guard). Reduces the ILP from per-slot to per-channel binary variables.
3. **Greedy heuristics** when ILP too big: order demands (decreasing size or decreasing path length — biggest/hardest first, the classic "place the hard ones while room exists" first-fit-decreasing intuition from bin packing), then First-Fit slot scan (lowest-index contiguous block free on every route link, with guard). Variants: Most-Used (pick the slot index most used elsewhere in the network → pack reuse, leave clean blocks), Random-Fit, Last-Fit, Exact-Fit, First-Last-Fit.

## Design decisions → why
- **Why decompose R then SA?** Joint RSA ILP intractable beyond ~6 nodes; decomposition is compact/scalable but loses optimality. k-shortest-paths bounds routing choices.
- **Why minimize MAX index (not total)?** Max index on a link = the spectrum width that link must physically support = the cost driver; equals makespan Cmax in the scheduling analogy. (Total-slots is an alternative objective.)
- **Why FFD order (decreasing size)?** Same reason as bin-packing first-fit-decreasing: large demands need a big contiguous gap; placing them last, after the spectrum is fragmented, blocks them. Place big/hard first while contiguous room remains.
- **Why First-Fit (lowest index)?** Packs all demands toward index 0, leaving a long contiguous free tail at the high indices for future/large demands; no global info needed; lowest blocking + lowest complexity among simple policies.
- **Why Most-Used?** Allocates the slot index already used on the most fiber links network-wide → maximizes spatial reuse of that index, concentrates usage, keeps other indices fully clean → fewer fragmented links. Needs global info, higher complexity. (Christodoulopoulos [1] heuristic.)
- **Why guard band G slots?** BV-WSS/filter roll-off is not a brick wall; adjacent channels need a guard (typ. 1–2 slots, tutorial says 2) to avoid inter-channel crosstalk and to let the WSS filter separate them.
- **Why the channel formulation?** Direct contiguity constraints blow up the ILP; pre-enumerating contiguous channels folds contiguity+continuity into the variable definition, giving a compact set-packing-like ILP (David/Velasco style) → big speedup.
- **Modulation table (RMLSA):** higher-order QAM = more bits/symbol = fewer slots, but lower OSNR margin = shorter reach. Choose the most-efficient modulation whose reach ≥ path length → fewest slots. Demand-feasibility / reach (the OSNR/NLI that decides whether a modulation survives a route) is supplied by the physical-layer GN model — cited as the input that sets reach, NOT re-derived here.

## Code plan
- `modulation_slots(rate, path_len, table, slot_width, guard)` → n slots from the most efficient feasible modulation.
- Heuristic: build per-link occupancy; sort demands FFD; for each, over its k paths, First-Fit scan lowest start s such that [s, s+n+G-1] free on every link; commit; report max occupied index.
- ILP (channel-based, PuLP): variables x[d,p,c] = demand d on path p using channel c (channel = contiguous slot interval of width n_d+G). Constraints: each demand exactly one (d,p,c); for every link e and slot index k, sum over demands whose chosen channel covers k ≤ 1 (non-overlap incl. guard built into channel width). Objective: minimize Z with Z ≥ (top slot of chosen channel) for all demands → min max index. Demonstrate on a small 4-node network.
