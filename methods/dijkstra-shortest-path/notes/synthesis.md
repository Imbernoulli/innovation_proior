# Synthesis — Dijkstra's Shortest Path

## Self-account backbone (highest priority)

Three first-person sources, all consistent:

- **OH 330** (Frana oral history, Aug 2, 2001) — fullest version. Key facts:
  - 1956: festive opening / official inauguration of ARMAC (Automatische Rekenmachine
    MAthematische Centrum). Needed a *demonstration*.
  - Predecessor ARRA was so unreliable the only safe demo was random-number generation.
    ARMAC more reliable → "could try something more ambitious."
  - **Demo constraint (the design driver):** "for a demonstration for non-computing people
    you have to have a problem statement that non-mathematicians can understand, even they
    have to understand the answer." → shortest route between two Dutch cities.
  - **Reduced roadmap: 64 cities**, "so that in the coding, 6 bits would suffice to identify
    a city." (2^6 = 64.) This is a hard, quoted machine constraint.
  - "What's the shortest way to travel from Rotterdam to Groningen, in general: from given
    city to given city."
  - **"designed in about twenty minutes"** at a **café terrace in Amsterdam**, shopping with
    his young fiancée (Ria Debets, herself a programmer), tired, over a cup of coffee, "just
    thinking about whether I could do this."
  - **"I designed it without pencil and paper… one of the advantages of designing without
    pencil and paper is that you are almost forced to avoid all avoidable complexities."**
  - Published 1959 (three years late) in Numerische Mathematik (Bauer's new journal),
    bundled with the shortest sub-spanning tree (designed for hardware friends to minimize
    backpanel copper wiring on the X1).
  - At the time shortest-path "was hardly considered mathematics: there was a finite number
    of ways of going from A to B and obviously there is a shortest one, so what's all the
    fuss about?"

- **CACM 2010** (Misa & Frana, published, edited) — same story, "20 minutes," same quotes
  ("without pencil and paper you are almost forced to avoid all avoidable complexities").

- **EWD 1166 "From my Life"** — "The algorithm for The Shortest Path was designed for the
  purpose of demonstrating the power of the ARMAC at its official inauguration in 1956…
  designed without pencil and paper, while I had a cup of coffee with my wife on a sunny
  cafe terrace in Amsterdam, only designed for a demo." Also: "discrete algorithms had not
  yet acquired mathematical respectability, and there were no suitable journals."

**How the account shapes the trace:** the *demo + ARMAC + no-pencil* constraints are the
engine of the derivation. The problem must be lay-understandable (shortest road route) and
small enough to run on ARMAC (64 cities, 6-bit codes, store data for at most ~n branches).
Designing in the head with no paper forces the simplest possible bookkeeping — no sorting
all the edges first, no storing the whole adjacency matrix — which is *exactly* the
efficiency advantage the 1959 paper claims over Ford/Berge and Kruskal/Loberman-Weinberger.
So "avoid all avoidable complexities" is not a flourish; it is the selection pressure that
picks the grow-the-known-region-by-the-nearest-frontier-node formulation.

## Primary source (the 1959 paper, Problem 2)

Sets, in Dijkstra's own words:
- **A** — nodes whose minimum-length path from P is *known*; added in order of increasing
  minimum path length from P.
- **B** — nodes connected to at least one node of A but not yet in A (the frontier).
- **C** — the rest.
- Branch sets I (in known minimal paths), II (one per B-node: the candidate branch from
  which the next I-branch is chosen), III (rest).

Loop: transfer P to A. Repeat:
- **Step 1 (relaxation):** for each branch r from the just-settled node to nodes R in B or
  C: if R∈B, check whether r gives a shorter P→R path than the current candidate in II; if
  so replace, else reject. If R∈C, move R to B and add r to II.
- **Step 2 (settle nearest):** each B-node has a distance from P via I-branches + one
  II-branch; the **minimum-distance B-node** is moved to A and its branch from II to I.
  Stop when Q enters A.

Remark 1: works for directed branches (length depends on direction).
Remark 2: for each I/II branch, record its two nodes (in order of increasing distance from
P) and the distance from P to the farther node — actual for I, best-so-far for II.

Claim of superiority over Ford [3] (Rand P-923, 1956) as described by Berge [4]: "irrespective
of the number of branches, we need not store the data for all branches simultaneously but
only those for the branches in sets I and II, and this number is always less than n.
Furthermore, the amount of work to be done seems to be considerably less."

## The correctness insight (why settle-the-nearest is safe; nonnegative lengths)

Dijkstra's own justification: "if R is a node on the minimal path from P to Q, knowledge of
the latter implies the knowledge of the minimal path from P to R" — optimal substructure.
Settling the nearest unfinished frontier node is safe **because branch lengths are
nonnegative**: the smallest tentative distance among frontier nodes cannot later be beaten,
since any other route to it must first leave the settled region through some frontier node
whose tentative distance is already ≥ this one, and the remaining branches only add
nonnegative length. (Road distances are nonnegative — this is automatic in the demo, never
flagged as an assumption, which is itself telling.)

## Antecedents (best-effort)

- **Ford, L.R., "Network flow theory," Rand P-923 (1956)** — the label-correcting method:
  keep a tentative distance for *every* node, repeatedly relax *any* edge that violates
  d[v] ≤ d[u] + w(u,v) until no edge does. Correct for nonnegative (and detects/handles via
  iteration) but stores/scans all edges and may relax a node many times; no fixed settle
  order. Described in **Berge, C., "Théorie des graphes…" (1958), pp. 68–69**, which is
  Dijkstra's actual point of contact (he cites Berge's description, not Ford directly).
- **Bellman (1958), Shimbel (1955), Moore (1957)** — same label-correcting family
  (Bellman–Ford), parallel to Dijkstra; he does not cite these.
- **MST companion (Problem 1):** Kruskal (1956) and Loberman–Weinberger (1957) sort all
  ½n(n−1) edges first → store all edges. Dijkstra's Prim-like growth stores ≤ n branches.
  (Prim 1957 / Jarník 1930 parallel; he cites neither.) Relevant only as the sibling
  problem he bundled into the same paper; the trace centers on Problem 2.

## Design decisions → why

| decision | why (in-frame) | rejected alternative + failure |
|---|---|---|
| grow ONE known region outward from P | optimal substructure: a shortest P→Q path's prefixes are shortest P→R paths, so build them in increasing-length order | recompute each path independently → redundant |
| settle the *nearest* frontier node, mark it final | nonneg lengths ⇒ nothing can later beat the current smallest tentative distance | settle arbitrary frontier node → can be wrong (a cheaper detour may exist) |
| keep only ONE candidate branch per frontier node (set II) | only the best-so-far distance to a frontier node matters; the route into it is determined by the predecessor | keep all incoming branches (Ford/Berge) → store all edges, more memory + work |
| relax only from the just-settled node | its distance is now final, so it's the only node that can improve its neighbours' best-so-far | re-relax everything every pass (Ford) → O(VE) work, many revisits |
| nonnegative branch lengths | road distances; makes the greedy settle provably safe | negative lengths → greedy settle invalid (out of scope; never arises for roads) |
| store ≤ n branches (I ∪ II), not the full graph | ARMAC memory; "avoid all avoidable complexities"; no pencil & paper | store/sort all edges first (Kruskal, Loberman–Weinberger) → blows the memory budget |
| (code) min-heap keyed by tentative distance; lazy deletion | Step 2's "pick the minimum-distance B-node" IS extract-min; heap gives O((V+E)log V) | linear scan of B each step → O(V²) (fine for 64 cities, but the heap is the clean general form) |

## Canonical code (grounded)

Lazy-deletion heapq Dijkstra (TheAlgorithms/Python `graphs/dijkstra.py` structure):
push (0, source); pop min; skip if already visited (lazy delete of stale entries); on first
pop a node is settled with final distance (== Step 2); relax neighbours by pushing
(dist+w, v) (== Step 1). This maps piece-for-piece onto A/B/II.
