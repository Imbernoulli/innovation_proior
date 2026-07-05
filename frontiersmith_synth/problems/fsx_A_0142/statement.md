# Tell Qadesh Dig: Crating the Catalogued Finds

A season at the **Tell Qadesh** excavation has finished. Every cell of the dig grid
has been swept and each recovered artifact catalogued with an integer **mass**. The
finds must now be sealed into identical archival crates for the museum convoy.

Each crate has **two** hard limits:

- a **mass capacity `C`** — the total mass of the artifacts inside may not exceed `C`
  (any heavier and the crate buckles in transit), and
- a **slot limit `K`** — at most `K` artifacts fit, because every find gets its own
  padded, humidity-sealed slot regardless of how small it is.

Sealing and hauling one crate costs one crate. You are the registrar: pack the whole
catalogue into **as few crates as possible** while respecting both limits.

This is one-dimensional bin packing with an extra **cardinality (k-item)** constraint.
Because a crate is bounded on *both* mass and slot count, a rule that watches only
mass wastes crates whenever slots bind, and a rule that watches only counts wastes
crates whenever mass binds — there is no single greedy that wins on every instance.

## Program contract

Write a **standalone program**: read ONE JSON object (the public instance) from
`stdin`, write ONE JSON object (your answer) to `stdout`.

```python
import sys, json
inst = json.load(sys.stdin)   # public instance ONLY
# ...compute a crating...
print(json.dumps({"assign": assign}))
```

Your program is run in an isolated subprocess, once per instance.

### Public instance (stdin)

```json
{
  "name": "qadesh101",
  "capacity": 24,          // C: max total mass per crate
  "slots": 6,              // K: max artifacts per crate
  "n": 30,                 // number of artifacts
  "masses": [7, 3, 19, ...] // n integer masses, each 1 <= m_i <= C
}
```

### Answer (stdout)

```json
{ "assign": [c_0, c_1, ..., c_{n-1}] }
```

`c_i >= 0` is the crate index artifact `i` is sealed into. Crate indices need not be
contiguous; a crate "exists" iff at least one artifact is assigned to it, and the
number of **distinct non-empty crates** is your cost.

## Validity

An answer is valid iff `assign` is a list of exactly `n` non-negative integers and,
for **every** crate, the summed mass does not exceed `C` **and** the artifact count
does not exceed `K`. Invalid output, wrong length, an overfilled or over-slotted
crate, a crash, a timeout, or non-JSON makes that instance score `0.0`.

## Objective and scoring (deterministic)

You **minimize** the number of crates. For each instance the evaluator computes:

- `q_lb  = max(ceil(sum(masses)/C), ceil(n/K))` — a lower bound (usually unreachable),
- `q_base` — crates used by a **first-fit** registrar in catalogue order (mass- and
  slot-aware): the weak reference,
- `q_cand` — crates used by your crating.

Your per-instance score is

```
r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_lb), 0, 1 )
```

So reproducing first-fit scores about `0.1`, reaching the lower bound scores `1.0`,
and doing worse than first-fit scores below `0.1`. The final score is the mean of `r`
over a fixed family of 12 instances (uniform / light-sherd / near-half-mass / bimodal
mass distributions, with slot limits that make either the mass or the slot constraint
bind, plus larger held-out digs). The lower bound is loose, so even strong packers
stay below `1.0` — there is real headroom and no easy optimum.
