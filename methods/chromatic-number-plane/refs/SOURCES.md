# refs/ — retrieved sources for chromatic-number-plane (this run, web-grounded)

This file lists every artifact retrieved THIS run into refs/, with provenance and what
each contributes. The earlier attempt left refs/ empty (memory-based); this redo grounds
every load-bearing claim in retrieved full text.

## PRIMARY SOURCE (full text, read in full)
- `degrey-2018-arxiv-1804.02385.pdf` — Aubrey D.N.J. de Grey, "The chromatic number of
  the plane is at least 5", arXiv:1804.02385. Downloaded from https://arxiv.org/pdf/1804.02385 .
  The full LaTeX source is also present in `../src/CNP5final.tex` (with figure PDFs
  `UD5p1.pdf`..`UD5p9.pdf`, `UD5p7.png`). Read end to end: Introduction; §2 the four
  4-colourings of H; §3 J(31 vtx)/K(61)/L(121) construction forcing a monochromatic
  triple; §4 M via spindle-density (T,U,V,W → 1345-vtx M), N (20425 vtx), the custom
  DFS 4-colourability tester; §5 shrinking to G (1581 vtx) and SAT verification.

## ANTECEDENTS (best-effort)
- `antecedent-hadwiger-nelson-history.md` — Hadwiger–Nelson problem history, retrieved
  from https://en.wikipedia.org/wiki/Hadwiger%E2%80%93Nelson_problem . Nelson 1950 posed
  CNP; Isbell 1950 7-colour upper bound via hexagonal tiling; de Bruijn–Erdős 1951
  reduction (under AC) to finite unit-distance graphs; long-standing 4 ≤ χ ≤ 7; de Grey
  2018 raised the lower bound to 5.
- `antecedent-moser-spindle.md` — Moser spindle, retrieved from
  https://en.wikipedia.org/wiki/Moser_spindle . 7 vertices, 11 edges, two 60/120 rhombi
  sharing an acute vertex with the far acute vertices 1 apart; forcing argument for χ=4;
  the smallest 4-chromatic unit-distance graph; Moser & Moser 1961.

## ANALYSIS (best-effort: how the construction was found / verified)
- `analysis-quanta-degrey.md` — Quanta "Decades-Old Graph Problem Yields to Amateur
  Mathematician", https://www.quantamagazine.org/decades-old-graph-problem-yields-to-amateur-mathematician-20180417/ .
  De Grey's motivation (Othello/competitive players, "rest from my real job", Christmas),
  built on the Moser spindle, combined many copies "delicate process, minimal computer
  assistance", first a 20425-vtx non-4-colourable graph, shrank to 1581, computer-verified.
- `analysis-polymath16-first-thread.md` — Dustin Mixon, "Polymath16, first thread:
  Simplifying de Grey's graph",
  https://dustingmixon.wordpress.com/2018/04/14/polymath16-first-thread-simplifying-de-greys-graph/ .
  H/L/M/N modular construction; M prevents a √3-triangle from being monochromatic; SAT
  verification of the 1581 graph; vertex-removal shrinking; Heule SAT/UNSAT proofs to
  874 then 826 vertices.
- `analysis-gil-kalai-announcement.md` — Gil Kalai, "Aubrey de Grey: The chromatic number
  of the plane is at least 5",
  https://gilkalai.wordpress.com/2018/04/10/aubrey-de-grey-the-chromatic-number-of-the-plane-is-at-least-5/ .
  Announcement; initial 1567 → 1581 after a code-bug fix; G = two copies of S_a; built on
  Moser spindle; independent SAT-solver confirmations; Polymath16 launched.

## GAPS / NOTES
- The Quanta explainer "Four Is Not Enough"
  (https://www.quantamagazine.org/the-numbers-and-geometry-behind-a-math-coloring-puzzle-20180618/)
  was checked but is introductory only — it does NOT give the geometric forcing detail
  (H 4-colourings, arcsin rotations, spindle density). It is therefore NOT relied on for
  any technical claim; all technical detail is taken from the primary source itself
  (src/CNP5final.tex) and the Polymath16 thread. Recorded here as a checked-but-thin source.
