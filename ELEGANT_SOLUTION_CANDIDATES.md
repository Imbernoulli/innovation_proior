# Elegant-Solution Reasoning Traces — Candidate Backlog

Goal: extend Innovation Prior with **famously elegant competition / expert solutions**, reconstructed as
first-person discovery traces.

**Hard gate (non-negotiable — same rule as the paper-to-reasoning grounding discipline):** a candidate is
buildable only if BOTH are publicly retrievable today — (1) the **original solution/proof itself**, and
(2) a **documented analysis of the idea behind it** (an editorial, lecture notes, retrospective, "why it
works / how you'd find it" writeup). Never reconstruct a solution from memory. A well-sourced "not
recoverable" is a valid result and is recorded as such below.

Proposed new domains: `Math olympiad`, `Competition math` (Putnam), `Informatics olympiad`,
`Physics olympiad`, `Linguistics olympiad`. Entries use `"arxiv": "N/A"` like the existing pure-math traces.

---

## A. IMO Special Prize — complete enumeration + buildability

Authoritative source: official IMO Hall of Fame data (`imo-official.org/hall-of-fame/overall.json`) plus a
per-year sweep of `imo-official.org/results/individual/year/<YYYY>/`. **50 awards, 44 unique recipients,
across 18 years (1965–2005)** (1980 = none; the IMO was cancelled). The award became rare after the early
1980s: only **1988, 1995, 2005** in the modern era. (Per-person totals imply 53; the year-sweep caught 50 —
Rickard's 3rd, Ruzsa's 2nd, van Leeuwen's 2nd land in years where null-score records slipped the parser.)

**The core obstacle:** the official record stores year + contestant + per-problem scores, but **not which
problem the prize was for, nor the solution text.** So most awards are NOT faithfully reconstructable.

### Tier 1 — the actual prize-winning elegant solution is documented  *(build these)*
| Year | Contestant | Problem | Idea | Status | Sources |
|---|---|---|---|---|---|
| 1988 | Emanouil Atanassov (BGR) | P6 | **Vieta jumping** — `ab+1 ∣ a²+b² ⇒` quotient is a perfect square, by root-flip infinite descent | **generating** → `methods/imo-1988-vieta-jumping/` | Wikipedia "Vieta jumping"; WFNMC writeup; AoPS |
| 2005 | Iurie Boreico (MDA) | P3 | **sign-isolating / SOS** — `Σ (x⁵−x²)/(x⁵+y²+z²) ≥ 0` under `xyz≥1` | **generating** → `methods/imo-2005-boreico-sos/` | Evan Chen IMO-2005 notes (attributes to Boreico); AoPS |
| 1986 | Joseph Keane (USA) | P3 | **monovariant / semi-invariant** — pentagon sign-flip process terminates, via a strictly-decreasing nonnegative integer quantity | buildable | MAA FOCUS 6(4) (Maurer — names Keane + P3, **primary**); AoPS "Solution 2 (Semi-Invariants)"; Matt Baker blog "The Pentagon Problem" |

### Tier 2 — problem firmly identified + a documented elegant solution to that exact problem exists  *(buildable; the trace reconstructs the idea's discovery, not the contestant's exact script)*
| Year | Contestant | Problem | Idea | Strength | Sources |
|---|---|---|---|---|---|
| 1995 | Nikolay Nikolov (BGR) | P6 | **roots-of-unity filter** — # of `p`-subsets of `{1..2p}` with sum `≡0 (mod p)`; credited (folklore) as ROUF's origin | strong technique arc | Yu & Feng "Roots of Unity Filter" handout; Ray Li lecture; AoPS; Kalva `isoln956` |
| 1978 | Richard Borcherds (UNK) | P6 | **Schur-type iterated-pigeonhole** — 1978 names in 6 countries ⇒ some number = sum/double within its country | problem ID from a **primary** source; famous mathematician | UK 1978 team report (imo-register.org.uk/1978-report-ms.html); Kalva `isoln786`; AoPS |
| 1969 | Simon Norton (UNK) | P2 | `f(x)=Σ 2^{1−k}cos(a_k+x)`; two zeros ⇒ `x₁−x₂ ∈ πℤ` | problem ID from UK report | imo-register.org.uk/1969-report-st.html; Kalva |
| 1969 | David Aldous (UNK) | P5 | `n>4` points, no 3 collinear ⇒ `≥ (n−3)(n−4)/2` convex quadrilaterals | problem ID from UK report; famous probabilist | imo-register.org.uk/1969-report-st.html; Kalva |

### Tier 3 — record lost, NOT faithfully reconstructable
1965–68 batch (Lovász, Pelikán, Babai, Norton'67, Misiurewicz, Figiel, Csirmaz, Livshic, Georgiev…) — the
prize is confirmed for several (Pelikán via his own Rényi obituary; Norton'67 "certificate of special
elegance" via the UK report) **but no source names the problem.** Also Nazarov 1984, van Leeuwen 1978 (×2),
Rickard (×3), Burmeister (×2), Ruzsa (×2), Steiner, Kröger, Leeb, Keane, etc. — problem and/or solution
undocumented. The 1965–66 years have no detailed UK team report, which was the decisive source type for 1969.

---

## B. The ONLY other genuine "special prize for elegance" outside the IMO: IOL (Linguistics)

The **International Linguistics Olympiad has a formal "Best Solution" award** (and a Solvers' Choice Award),
and it publishes BOTH the winning contestant solution AND official solution booklets whose "Commentary"
sections are pure idea-analysis ("you need an anchor; once you have it, the rest follows by elimination…").

- **IOL 2023 Individual P5 "Supyire"** — decipher a mixed base-5/10/20/80/400 numeral grammar from a few
  examples. Best solution: `ioling.org/best_solutions/iol-2023-i-5-best-Tam.en.pdf`; commentary booklet:
  `ioling.org/booklets/iol-2023-indiv-sol.us.pdf`.
- **IOL 2024 P3 "Komnzo"** — same structure (best-solution PDF + commentary booklet).
- Reservoir: 1–2 published best solutions per problem per year (2019–2025) + a commentary booklet each.

**Checked and FAILED the gate:** IPhO *does* award a "Most Original Solution" prize (real — confirmed in
official board minutes, e.g. 2014), **but** the minutes never name the problem, give no solution text, and
publish no idea-analysis → not recoverable. IChO / IOAA / IBO special prizes reward highest **score**, not
elegance. IOI has **no** elegance award (only a Distinguished Service Award) — though its official solution
PDFs and editorials often embed the idea-analysis (see §C). So "special prize *for elegance*" with a
recoverable solution is essentially **IMO + IOL only.**

---

## C. Famously-elegant + documented-idea solutions (no award, but pass the two-source gate)

### Competition math — Putnam
- **Putnam 1992 A6** — 4 random points on a sphere; `P(center ∈ tetrahedron) = 1/8` via the antipodal
  coin-flip / pair-of-lines reframe. Idea-analysis: **3Blue1Brown "The hardest problem on the hardest test"**
  (an entire video on the *discovery process*). Solutions + commentary: Kedlaya–Poonen–Vakil book; Kedlaya
  Putnam archive (`kskedlaya.org/putnam-archive`); a second analysis at laurentlessard.com.
- **Putnam 2016 A4** — tromino/tetromino tiling bound by coloring the `mn` odd-odd squares (each tile covers
  ≤1) ⇒ lower bound `mn`. Official solution (Kedlaya archive) + Beni Bogoșel's insight blog.

### Informatics — IOI (no elegance award, but elegant + heavily editorialized)
- **IOI 2016 "Aliens"** — the **"Aliens trick"** (Lagrangian relaxation: penalty `λ` per piece + binary
  search; convex cost). The penalty/convexity insight is *in the official PDF*; re-analyzed by USACO-Guide /
  SOI / Codeforces. *[strongest CP pick]*
- **IOI 2014 "Game"** — the one-liner `++c[max(u,v)] == max(u,v)`: commit an edge only when forced. Official
  solution PDF (with the no-cycle proof) + Bruce Merry's adversary-strategy analysis.
- **IOI 2011 "Race"** — **centroid decomposition**. Solution (robert1003 blog) + participant retrospective.
- **IOI 2006 "Joining Points"** — median-balanced divide & conquer → `O(n log n)`. Official solution PDF +
  Codeforces "most beautiful solutions" framing.

### Informatics — 中国国家集训队论文 (the project owner's prototype)
> Correction: the "one paper per technique, written by team members" tradition is the **informatics (NOI)**
> tradition, not math. China's *math* olympiad has no per-author technique-paper tradition (it uses TST
> problem sets and book series). The math analogue is self-authored handouts (see below).

- **Suffix Arrays — 罗穗骞 (2009)** — `O(n log n)` SA + height/LCP. *[flagship; closest literal match]*
- **Size Balanced Tree — 陈启峰 (2007)** — balance a BST by subtree size (author later co-created PixelCNN).
- **Suffix Automaton — 张天扬 (2015)**; **Link-Cut Tree — 陈首元 (2006)**; **Matrix exponentiation — 俞华程 (2008)**.
- Corpus: `github.com/OI-wiki/libs` (集训队历年论文), `github.com/Fesdrer/NOIpaper`. Idea-analysis: `oi-wiki.org`.

### Physics — Physics Cup / Kalda
- **Physics Cup 2022 P3 "Photon rocket"** — solve via **relativistic 4-momentum invariants** instead of
  Lorentz transforms ("geometry, not coordinate rotation"). Winner's PDF + editorial contrasting it with
  brute force. *[cleanest physics Tier-1]*
- **Physics Cup 2021 P2** — minimal-eccentricity orbit via the **Laplace–Runge–Lenz (eccentricity) vector**,
  collapsing the geometry to one line. Best-solution PDFs + academic-head solution + a "best solutions"
  commentary page (`physicscup.ee`). *[strong; genuine elegant-solution recognition]*
- **Physics Cup 2022 P4 "Image of a point"** — recognize the optics is **projective**; locate the image by
  **cross-ratio / harmonic conjugates** (ruler only). Winner's PDF + editorial.
- **Kalda booklets** (Mechanics / Kinematics) — enumerated "ideas" each with a derivation; pick one specific
  idea to reconstruct. Idea-analysis is the booklet's explicit design (`ioc.ee/~kalda/ipho`).

### Math — self-authored olympiad handouts (global analogue of 集训队论文)
- **Evan Chen** — Barycentric Coordinates; SOS ("A Dumbass's Perspective"); Monsters (functional-equation
  pathologies). **Yufei Zhao** — Algebraic Techniques in Combinatorics (the dimension method); Lifting the
  Exponent. All on `web.evanchen.cc` / `yufeizhao.com`.

### Information theory (weaker — "really a paper / textbook", low special-prize character)
- Entropy proof of Bregman's theorem (Radhakrishnan; Galvin tutorial `arxiv.org/abs/1406.7872`); the
  100-prisoners cycle-following strategy; the Kraft-inequality "gold-dust" proof.

---

## D. Existing collections / analyses  *(answer to "has anyone collected/analyzed these?")*

No single compilation of **IMO special-prize solutions** exists. Adjacent collections that DO pair elegant
solutions with idea-analysis:
- **3Blue1Brown** discovery videos (Putnam sphere; IMO 2011 windmill).
- **Tim Gowers** "how to discover a proof" essays + weblog.
- **Kedlaya–Poonen–Vakil**, *The Putnam Competition 1985–2000: Problems, Solutions, and Commentary* (+ Kedlaya's free archive).
- **Aigner–Ziegler**, *Proofs from THE BOOK*; **Cut-the-Knot** (catalogs of celebrated proofs, lighter on discovery narrative).
- **OI-wiki / NOIpaper** GitHub repos (集训队论文 corpus); **Kalda** booklet index; **Evan Chen / Yufei Zhao** handout hubs.

---

## Recommended wave-2 build order

1. **IMO 1986 — Keane — P3 monovariant** (Tier-1: his actual idea is documented; completes the IMO Tier-1 set).
2. **IMO 1995 — Nikolov — roots-of-unity filter** (strong technique-discovery arc).
3. **IMO 1978 — Borcherds — P6 pigeonhole** (primary-source problem ID; famous mathematician).
4. **Putnam 1992 A6 — sphere/tetrahedron** (3B1B = ready-made discovery analysis).
5. **IOL 2023 P5 "Supyire"** (the only other real elegance award; new domain).
6. **IOI 2016 "Aliens"** (named trick + rich analysis).
7. **集训队论文: Suffix Arrays — 罗穗骞 2009** (the prototype, literal match).
8. **Physics Cup 2022 P3 — photon rocket / relativistic invariants** (cleanest physics Tier-1).

Optional follow-ons: IMO 1969 Norton P2 / Aldous P5; Putnam 2016 A4; IOI 2014 Game / 2011 Race; Evan Chen
barycentric; Yufei Zhao dimension method; a Kalda mechanics idea; SBT (陈启峰 2007); Physics Cup 2021 P2.

---

## Status (built so far)

**Wave 1 (live):** `imo-1988-vieta-jumping`, `imo-2005-boreico-sos` — fully Codex-reviewed.
**Wave 2 (live, 10):** `imo-1986-keane-monovariant`, `imo-1995-nikolov-rouf`, `imo-1978-borcherds-pigeonhole`,
`putnam-1992-a6-sphere`, `ioi-2016-aliens-trick`, `iol-2023-supyire`, `noi-suffix-array-luo`,
`physicscup-2022-photon-rocket`, `evanchen-barycentric`, `yufeizhao-dimension-method`. All use minimal
problem-only `context.md`. (Supyire + barycentric Codex-reviewed inline; the other 8 in a batch Codex pass.)

New domains in use: `Math olympiad`, `Competition math`, `Informatics olympiad`, `Physics olympiad`,
`Linguistics olympiad`.

---

## Wave-3+ backlog (broad-net research — every row passed the two-source gate, URLs verified)

### Competitions
| Field | Item | Elegant idea | Solution | Idea-analysis |
|---|---|---|---|---|
| Math | **IMO 2011 P2 Windmill** | side-count past the oriented line is a rotation invariant | imo-official SL2011 | 3B1B windmills lesson/video *(top pick)* |
| Physics | **Physics Cup 2023 P4 (Spaceship)** | chained-4-velocity Minkowski invariant collapses four boosts | physicscup.ee PC2023 P4 Oros | Kalda intended-solution PDF |
| Physics | **Physics Cup 2022 P4 (Image of a point)** | a ray is its own image ⇒ projective cross-ratio (ruler only) | physicscup.ee 2022 P4 | editorial + Kalda GeoGebra |
| Physics | **Physics Cup 2024 P4 (Satellites)** | velocity hodograph is a circle (Runge–Lenz) | physicscup.ee PC2024 P4 | organizer editorial |
| Physics | **Physics Cup 2023 P3 (MHD)** | Alfvén flux-freezing into a co-moving tube | physicscup.ee PC2023 P3 | Foster expert essay |
| CS | **IOI 2014 Game** | adversary invariant; one-liner `++c[max(u,v)]==max(u,v)` | ioi.te.lv game-solution | gagguy/DMOJ editorial |
| CS | **IOI 2011 Race** | centroid decomposition → O(n log n) paths | ioinformatics race.pdf | gangsterveggies + USACO Guide |
| CS | **BalkanOI 2011 Time is Money** | min of Σcost·Σtime at a hull vertex; MST probes | koosaga repo | koosaga.com/82 + CF 62896 |
| CS | **IOI 2006 Joining Points** | median-balanced bichromatic D&C | ioi.te.lv points_sol | official PDF (single-source) |
| Math | **Putnam 2018 B4** | x_n=cos(F_n·b); Pisano-period periodicity | Kedlaya 2018s | Bogoșel discovery post |
| Math | **Putnam 2016 A4** | color odd-odd squares ⇒ mn tiling bound | Kedlaya 2016s | Bogoșel post |
| Math | **Putnam 2020 B6** | Beatty partition bounds sign-run length | Kedlaya 2020s | Evan Chen twitch ep058 |
| Math | **IMO 2017 P3 (hunter/rabbit)** | rabbit = expanding cloud; symmetric points force growth | imo-official SL2017 (C5) | Grozev "a motivation" |
| Math | **IMO 2009 P6 (grasshopper)** | induction with x=s−aₙ | imo-official SL2009 | Tao mini-polymath |
| Ling | **IOL 2025 P1 Dzongkha** | parallel base-20 numeral systems w/ overcounting | ioling 2025 indiv-sol | author slides |
| Ling | **IOL 2022 P5 Proto-Chamic** | reverse reconstruction; tonogenesis | ioling 2022 indiv-sol | author slides |

### Expert writeups & famously elegant proofs
| Field | Item | Idea | Original | Idea-analysis |
|---|---|---|---|---|
| CS (集训队论文) | **陈启峰 SBT (2007)** | balance a BST by subtree size | Fesdrer/NOIpaper PDF | oi-wiki/ds/sbt |
| CS (集训队论文) | **漆子超 centroid decomp (2009)** | recurse on the tree centroid | Fesdrer/NOIpaper PDF | oi-wiki/graph/tree-divide |
| CS (集训队论文) | **胡伯涛 min-cut models (2007)** | project-selection ↔ min-cut | Fesdrer/NOIpaper PDF | oi-wiki/graph/flow/min-cut |
| CS (集训队论文) | **许智磊 suffix array (2004)** | original O(n log n) SA + height | OI-Public-Library PDF | oi-wiki/string/sa |
| CS (集训队论文) | **俞华程 matrix exponentiation (2008)** | accelerate linear recurrences | Fesdrer/NOIpaper PDF | oi-wiki matrix |
| CS (集训队论文) | **翁文涛 palindromic tree / eertree (2017)** | automaton of all palindromic substrings | OI-Public-Library 2017 PDF | oi-wiki/string/pam |
| Math (proof) | **Zagier one-sentence two-squares** | involution with one fixed point | Zagier 1990 (MPIM) | Mathologer windmill video |
| Math (handout) | **Yufei Zhao — Lifting the Exponent** | vₚ(aⁿ−bⁿ) via one lemma | yufeizhao.com/olympiad | self-contained |
| Math (handout) | **Mildorf — Olympiad Inequalities (SOS)** | SOS + smoothing toolkit | AoPS Mildorf PDF | Evan Chen Ineq |
| Math (proof) | **Bregman permanent — entropy proof** | entropy of random matching + chain rule | Galvin tutorial (arXiv 1406.7872) | same tutorial |
| Math (proof) | **Crossing-number inequality** | random vertex-subsampling amplification | Wikipedia | Tao blog |
| Math (proof) | **Furstenberg topological primes** | clopen APs ⇒ contradiction | Wikipedia | Richeson "Division by Zero" |
| Math (proof) | **Monsky's theorem** | 2-adic 3-coloring + Sperner | Wikipedia | UChicago REU exposition |
| Math (proof) | **Sylvester–Gallai (Kelly)** | minimal point-line distance | Wikipedia | Tao blog (Melchior sibling) |
| Math (handout) | **Evan Chen — Monsters** | functional-equation pathologies | web.evanchen.cc/handouts | self-contained |
| Physics (Kalda) | **infinite-ladder self-similarity** | R = f(R) | ioc.ee circuits booklet | physoly solutions |
| Physics (Kalda) | **fastest-path via Huygens** | min-time path = wavefront | ioc.ee kinematics booklet | physoly "Solution 2" |
| Prob (folklore) | **100 prisoners (cycle-following)** | follow your permutation cycle | Wikipedia | Veritasium video |

**Fetch caveats (for builders):** 集训队论文 via the jsdelivr CDN path (`cdn.jsdelivr.net/gh/Fesdrer/NOIpaper@main/...`)
— not bot-blocked, unlike github.com UI. `ioc.ee`/`physoly.tech` have expired TLS → `curl -k` or McGill mirror.
`web.evanchen.cc` and AoPS 403 bots → browser User-Agent. Codeforces/DMOJ 403 → `r.jina.ai` proxy. IMO shortlist
PDFs 403 to curl but load via WebFetch. IOL idea-analysis lives in the author **slide decks**, not the terse answer keys.
Not recoverable: 陈丹琦 plug-DP & 陈首元 LCT (doc/ppt only, no PDF); Mo's algorithm (never a formal 论文).
