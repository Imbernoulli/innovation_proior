# Innovation Addendum (wave-2b) — read AFTER your format brief

This corpus trains RL to AMPLIFY a model's disposition for scientist-style innovation.
The reward signal must therefore make *insight* pay and make *recipe-recombination* plateau.
On top of your format brief's contract, your problem MUST satisfy:

## 1. The obvious approach must be a trap, not the answer
- Identify the ONE approach an average strong coder writes first (single greedy pass /
  textbook algorithm / uniform random + repair). That is your `greedy` tier.
- Your generator MUST include cases (≥3 of the 10) engineered so this obvious approach
  lands FAR from what an insightful strategy achieves: planted structure it cannot see,
  interaction terms it ignores, a regime change it does not model.
- Your `strong` tier embodies one genuine insight (not just "greedy + more iterations"):
  a reformulation, an invariant, a decomposition, an exchange argument, a dual/relaxation
  view, exploiting the planted structure.

## 2. Innovation headroom is measured — hit these numbers
After the harness PASSes, check validation.json metrics and ALSO enforce:
- `strong - greedy >= 0.06` (mean ratio) — the insight must visibly beat the recipe.
- `strong <= 0.92` — reference solutions must NOT saturate the score; the RL policy
  needs room ABOVE your strong solution. If strong hits 1.0, rescale the checker's
  baseline/cap so the ceiling stays open.
- `greedy - trivial >= 0.03` — the recipe still beats do-nothing (sanity of the ladder).
These are acceptance criteria; the batch auditor rejects problems that miss them.

## 3. No benchmark cloning, no wave-1 duplication
- FrontierCS / ALE-Bench / MLS-Bench define the FORM (write one program; deterministic
  graded score; no known optimum) — NEVER copy their specific tasks or skins.
- Do not re-create a family that already exists in seeds/seed_list.jsonl (your spec's
  family was pre-checked to be new — do not drift back to a classic during authoring).
- One textbook mechanism in a costume is a REJECT. Compose ≥2 mechanisms so no single
  named algorithm is "the intended solution".

## 4. Statement discipline (RL prompt hygiene)
- Statement ≤ ~700 words. It fully specifies feasibility + the SHAPE of the objective,
  but the exact bonus coefficients/interaction tables may live in the input (so the
  solver must read and exploit them, not pattern-match the statement).
- State the scoring formula's structure honestly (solvers must be able to reason about
  what improves the score); hide only tunable constants, never the mechanism.
- Time limit 2–5s, memory ≤512m, each .in ≤ 5 MB, checker O(input) — reward latency
  matters for RL (10 cases must score in well under a minute).

## 5. Determinism (unchanged, absolute)
Seed all randomness from testId; no wall-time, no GPU, no network, no dict-order
dependence. Same submission ⇒ same score, forever, on any machine.
