# Rooftop Garden Trellis: Inducing a Nesting Rule that Generalizes to Taller Trellises

A vertical-farming co-op grows plants on rooftop **trellises**. A trellis is a stack of
planter **modules** read bottom-to-top; each module is one of `n_types` kinds (encoded as
integers `0..n_types-1`). Every plant **variety** thrives only on trellises whose module
arrangement obeys a hidden *nesting rule* — a **regular (finite-state) predicate** over the
module string.

The co-op has measured, for a batch of **short** trellises, whether the variety **thrived**
(label `1`) or **wilted** (label `0`). They want to predict thriving on **taller** trellises
they have not built yet.

Your job: from the labeled short trellises, **induce a rule** — expressed as a deterministic
finite automaton (DFA) — that the evaluator runs on **hidden** test trellises. Half of the
hidden tests are fresh **short** trellises (in-distribution generalization); half are much
**taller** trellises (**length-OOD** generalization). A rule that merely *memorizes* the
training trellises scores near the majority baseline; a rule that recovers the underlying
finite-state **structure** generalizes to the never-seen taller lengths. The hidden rule is
randomized per instance, so it cannot be hand-coded — it must be *learned* from the examples.

This is the classic synthetic-algorithmic-generalization split (in-distribution vs
length-OOD) that rewards generalization over memorization.

## Candidate program contract

Your solution is a **standalone program**: read ONE JSON object (the public instance) from
**stdin**, write ONE JSON object (your induced DFA) to **stdout**. It runs in an isolated
subprocess and sees only the public instance (the training trellises).

```python
import sys, json
inst = json.load(sys.stdin)
# ... induce a DFA from inst["train"] ...
print(json.dumps({"start": s0, "accept": accept, "trans": trans}))
```

### Public instance (stdin)

```json
{
  "name": "trellis101",
  "n_types": 2,                 // module alphabet size D; symbols are ints 0..D-1
  "max_states": 64,            // your DFA may use at most this many states
  "train": [                    // labeled SHORT trellises
    [[0,1,1,0], 1],
    [[1,0,0], 0],
    ...
  ]
}
```

Each training entry is `[seq, label]`, where `seq` is a list of ints in `[0, n_types)` and
`label` is `0` or `1`.

### Answer (stdout) — a complete deterministic finite automaton

```json
{
  "start": 0,                             // int in [0, K)
  "accept": [1, 0, 1],                    // length K; accept[i]=1 => state i is accepting
  "trans": [[1,2],[0,2],[2,2]]            // K rows, each of length n_types; entries in [0,K)
}
```

Let `K = len(trans) = len(accept)` be the number of states. The DFA reads a trellis
left-to-right starting from `start`, moving `state -> trans[state][symbol]`; the trellis is
predicted **thrives** (`1`) iff the final state is accepting.

A submission is **valid** iff:

- `1 <= K <= max_states`;
- `accept` is a length-`K` list of `0`/`1`;
- `trans` is a `K`-row list, every row of length `n_types`, every entry an int in `[0, K)`;
- `start` is an int in `[0, K)`.

Any malformed submission, a crash, a timeout, or non-JSON output makes that instance score
`0.0`.

## Objective

**Maximize** exact-match accuracy of your DFA on the hidden test trellises, in **both**
regimes at once, across a fixed, seeded family of **12 instances** (varying alphabet size,
hidden-rule size, amount of training data, and trellis lengths). For each instance the
evaluator computes:

- `id_acc`  = accuracy on fresh **short** trellises (disjoint from training),
- `ood_acc` = accuracy on much **taller** trellises,
- `obj = gmean(id_acc, ood_acc) = sqrt(id_acc * ood_acc)`.

The geometric mean forces **both** regimes to be good: a memorizer with high `id_acc` but
chance `ood_acc` earns a low `obj`.

## Scoring (deterministic)

For each instance the evaluator computes, itself:

- `base`   = `gmean` of the **majority-label** classifier (predict the train-majority label
  on every test trellis),
- `oracle` = `1.0` (the true hidden DFA labels every test trellis correctly),

and normalizes with an affine anchor:

```
r = clamp( 0.1 + 0.9 * (obj - base) / max(1e-9, oracle - base), 0, 1 )
```

- Matching the majority baseline scores ≈ `0.1`; perfectly recovering the hidden rule on all
  lengths scores `1.0`; doing worse than the baseline scores below `0.1`.
- Because exact DFA identification from finite data is **under-determined** (and hard on the
  sparser / larger instances), even good grammar-induction learners stay strictly below `1.0`
  on much of the family — there is real headroom.

The reported **Ratio** is the mean of `r` over all 12 instances; the **Vector** lists the
per-instance scores.

## Suggested strategies

1. **Majority label** (baseline): a one-state DFA that always predicts the train majority —
   no structure, no generalization (≈ `0.1`).
2. **Suffix-window classifier**: a DFA whose state remembers only the last `k` modules,
   labeled by local majority. Captures short-range patterns and beats the baseline, but
   cannot represent the global finite-state rule.
3. **RPNI-style state merging**: build the prefix-tree acceptor, then merge states in
   shortlex order (folding for determinism, rejecting label-conflicting merges) to recover
   the underlying automaton — which generalizes to the much taller OOD trellises.
4. **Evidence-driven / Occam-biased variants**: EDSM-style scored merges, blue-fringe
   heuristics, or picking the smallest hypothesis consistent with the data to squeeze more
   generalization out of the under-determined, sparse instances.
