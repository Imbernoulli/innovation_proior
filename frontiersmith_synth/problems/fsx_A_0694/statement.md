# Apprentice's Montage: Interference-Aware Skill Curriculum

## Story

An apprentice must master `K` skills before the final trial. You are the mentor:
you plan a training montage of `T` drills, one skill per time step. Progress is
governed by two coupled effects.

**Diminishing returns.** Drilling skill `j` closes part of the remaining gap to
mastery: `p_j <- p_j + gain_j * (1 - p_j)`, where `p_j` is skill `j`'s
proficiency (starts in `[0,1)`, mastery is `1`). The first few drills on a raw
skill help enormously; drilling an already-strong skill barely moves it -- past a
point, more reps on a saturated skill are wasted montage time.

**Interference.** Every drill also touches every OTHER skill: drilling `j` sets
`p_i <- p_i * interfere[j][i]` for every `i != j`, where `interfere[j][i] <= 1`.
Skills in the same "reinforcement clique" barely erode each other
(`interfere` close to 1). Antagonist skills erode each other hard, and this
erosion happens on EVERY drill of either one -- a skill keeps losing ground even
while it sits untouched, as long as its antagonists are being trained.

You only pass the apprentice if their WEAKEST skill is good, so your score is the
**maximin final proficiency**: `min_i p_i` after all `T` drills. A skill's fate
hinges on two things: how many times it was drilled before its LAST drill (its
post-drill peak), and how much interference battered it AFTER that last drill --
so the very last time you touch a skill determines how well it survives to the
final trial. Planning the montage is really a graph-partitioning problem in
time: which skills to block together, which to keep apart, and whose final
refresher to save for last.

## Input (public instance, one JSON object on stdin)

```json
{
  "name": "apprentice101",
  "K": 5,
  "T": 22,
  "p0": [0.11, 0.27, 0.09, 0.31, 0.14],
  "gain": [0.35, 0.22, 0.48, 0.19, 0.40],
  "interfere": [[1.0, 0.99, 0.87, 0.9, 0.98], ...]
}
```

- `K` (int): number of skills.
- `T` (int): montage length, `T > K`.
- `p0` (list of `K` floats in `[0,1)`): starting proficiency per skill.
- `gain` (list of `K` floats in `(0,1)`): diminishing-returns rate per skill.
- `interfere` (list of `K` lists of `K` floats): `interfere[j][i] <= 1` is the
  multiplier applied to skill `i`'s proficiency whenever skill `j` is drilled
  (for `i != j`). Diagonal entries are `1.0` and unused.

## Output (one JSON object on stdout)

```json
{"sequence": [3, 3, 1, 4, 0, 2, 3, ...]}
```

- Exactly `T` integers, each in `[0, K)`: the skill drilled at every time step,
  in order. Repeats are required (`T > K`).

Any of the following makes the instance score `0.0`: wrong length, an
out-of-range or non-integer entry, a crash, a timeout, or output that is not the
JSON object above.

## Objective and scoring (deterministic)

For each instance the evaluator simulates your sequence with the exact rule
above (diminishing returns on the drilled skill, multiplicative interference on
every other skill, applied at every step) and computes:

- `y_cand` = the maximin final proficiency, `min_i p_i`, reached by your sequence.
- `y_base` = the maximin proficiency reached by the weak baseline "drill only the
  initially-weakest skill for the entire montage" (repeat one index `T` times).
- `y_ideal` = a loose upper reference: split the `T` drills as evenly as
  possible across the `K` skills with NO interference at all (pretend it away),
  and take the maximin of that. Real interference only hurts, so this stays a
  comfortable, generally-unreachable ceiling.

```
r = clamp( 0.1 + 0.9 * (y_cand - y_base) / max(1e-9, y_ideal - y_base), 0, 1 )
```

Reproducing the weak baseline scores about `0.1`; doing worse scores `0`;
outplanning it scores higher, capped at `1.0` (never actually reached, since
`y_ideal` ignores interference entirely). Your final score is the mean of `r`
over 10 instances with varied `K`, `T`, clique shapes (some contiguous by skill
index, some interleaved so a naive fixed-order pass keeps colliding with
antagonists), and some tight-budget / held-out cases.

## Notes

- Every number you need -- `p0`, `gain`, `interfere` -- is in the public
  instance; nothing about scoring is hidden except which construction the
  evaluator itself used for `y_base`/`y_ideal`.
- Scoring never measures wall-clock time; treat the per-instance limit as a
  compute budget for search/refinement.
- Your program runs in an isolated subprocess and sees only the public instance
  above.
