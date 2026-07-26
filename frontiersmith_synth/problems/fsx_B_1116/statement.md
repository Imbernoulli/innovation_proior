# Rung-Ladder Counterpoint: One Policy, Ten Cantus Firmi, One Shared Style Bonus

You are writing first-species counterpoint against a fixed **cantus firmus**
(cf) melody, one note of counterpoint (cp) per cf note. Pitches are integers
on a diatonic **rung ladder**: rung `k+7` is the same scale-degree class as
rung `k`. There are **10 fixed, seeded cantus firmi** in this corpus. Your
submission is a single **standalone program**, invoked once per cantus firmus
(a fresh, isolated subprocess call each time — no memory between calls).

```python
import sys, json
inst = json.load(sys.stdin)
# ... decide a full counterpoint line ...
print(json.dumps({"cp": [ ... ]}))
```

### Public input (stdin)

```json
{
  "instance_id": 3, "corpus_size": 10, "name": "cf3",
  "cantus": [5, 6, 5, 3, 4, 6, 7, 9, 8, 9],
  "cp_range": [-3, 22],
  "rules": {
    "consonant_classes": [0, 2, 4, 5],
    "perfect_classes": [0, 4],
    "boundary_classes": [0, 4],
    "max_leap": 6,
    "min_contrary_frac": 0.22,
    "climax_buckets": 5
  }
}
```
`cantus` is the fixed melody (length `L`). `cp_range` bounds every cp pitch.
`rules` is the **species rule table** — read it, don't assume it: it may
differ per instance.

### Answer (stdout): `{"cp": [int, ...]}`, length `L`, each within `cp_range`.

### Hard rules (any violation zeroes this instance)
- Every vertical distance `(cp[i]-cantus[i]) mod 7` must be in
  `consonant_classes`.
- The first and last vertical distance must be in `boundary_classes`.
- No two **consecutive** verticals may both be in `perfect_classes` with the
  *same* class value (no repeated perfect unisons/octaves/fifths in a row).
- `|cp[i+1]-cp[i]| <= max_leap` for every step.
- **Contrary-motion quota**: over all `L-1` melodic moves, count a move as
  contrary when cp and cantus move in strictly opposite nonzero directions.
  The fraction of contrary moves must be `>= min_contrary_frac`. This is a
  *whole-piece* requirement, not a per-note one.
- The maximum pitch in your `cp` line must occur **exactly once** (a single,
  unambiguous climax).

### Scoring (deterministic, higher is better)
An instance that violates a hard rule scores 0. A valid instance scores a
base amount plus three bonuses, each worth roughly a comparable share of the
remaining headroom:
1. **Contrary excess** — how far your contrary-motion fraction sits above
   the required `min_contrary_frac` (0 at the quota, maxed out near 100%
   contrary).
2. **Interval diversity** — the Shannon entropy (normalized) of your line's
   melodic step sizes, bucketed into step / skip / leap. A line that only
   ever moves by one size earns nothing here.
3. **Corpus-level style bonus (shared)** — after all 10 instances have been
   answered, the grader locates each valid instance's unique climax as a
   relative position in the piece, buckets it into `climax_buckets` bins
   (evenly spanning start-to-end), and computes the normalized entropy of
   *where climaxes land across the whole 10-piece corpus*. This single
   number is added to **every** valid instance's score. All 10 cantus firmi
   in this corpus have their natural high point in a similar relative
   region — a policy that resolves every piece "the same way" will cluster
   every climax there and collapse this shared bonus for the whole corpus,
   even though no single subprocess call ever sees another instance's
   answer or output.

The reported `Ratio` is the mean per-instance score; `Vector` lists the 10
per-instance scores.

### What makes this hard
Reaching the contrary-motion quota requires real planning across the whole
line, not a one-note-at-a-time reflex — burning your register room on
locally-tempting contrary moves early can make the quota unreachable later.
Guaranteeing a *unique* climax means deciding, in some sense, ahead of time
which note is allowed to be the highest. And the corpus bonus can only be
earned by treating the **whole 10-piece corpus** as the object you are
optimizing, using each instance's own `instance_id`/`corpus_size` to
deliberately vary your structural choices instance to instance — not by
optimizing any single cantus firmus in isolation.
