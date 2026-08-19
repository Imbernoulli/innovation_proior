# Changelog

## 2026-08-18 — svfix(W3_primary_plus_ancestors)

Fixed a factual/attribution error at the decisive step (blossom shrink + lift) in
`results/reasoning.md`. The worked 8-vertex example's lift paragraph said the
alternating route from the entry vertex (`3`) to the base (`1`) was "consistent
with" the internal matching `{1-2, 4-5}` — but under `{1-2, 4-5}`, edge `3-2` is
*unmatched* and `2-1` is *matched*, the reverse of what the passage then asserted
(`3-2` matched, `2-1` unmatched). The route must be read off the matching that is
still in force at the moment of the flip — the one the blossom was actually shrunk
with, `{2-3, 4-5}` (base `1` exposed) — under which `3-2` is matched and `2-1` is
not, exactly as the symmetric-difference computation two sentences later already
(correctly) used. `{1-2, 4-5}` is real, but it is the *result* of flipping along
that route, not what determines it. The generalizing paragraph right after ("the
stored circuit supplies the unique internal matching leaving that vertex exposed,
so the lift is forced") repeated the same inversion and is corrected likewise.

Grounded the correction against `refs/explainers/abdulaziz-2024-formal-correctness-blossom.txt`
(Abdulaziz & Mehlhorn 2024, "A Formal Correctness Proof of Edmonds' Blossom
Shrinking Algorithm"), whose `replace_cycle`/`stem2vert_path` description states
the route-finding step "returns the path leading to the base of the blossom
starting with a matching edge" — i.e. from the *stored* (pre-flip) matching, not
the post-flip one — and against Edmonds 1965 Theorem 4.14 (`refs/primary/edmonds-1965-paths-trees-flowers.txt`),
which independently confirms that the matching left after the lift is "the maximum
matching of B which leaves b1 exposed" (the boundary vertex), the *outcome* of the
flip rather than its cause. This ref was previously read (see `notes/source_matrix.md`)
but not load-bearing at this specific step; it now is.

The lifted path (`8-3-2-1`), the resulting matching (`{4-5, 6-7, 8-3, 1-2}`), the
5-cycle enumeration table, and the landing (method + code) were already correct
and are unchanged.
