# Drone Delivery Swarm: Mission-Validity Rule Induction

## Setting

A drone-delivery swarm executes **missions**. A mission is a sequence of atomic
commands drawn from an alphabet of `2*K` tokens, encoding `K` depots:

| token       | meaning                                            |
|-------------|----------------------------------------------------|
| `2*i`       | **LAUNCH** a drone toward depot `i` (an *open* of type `i`) |
| `2*i + 1`   | **RETURN** a drone from depot `i` (a *close* of type `i`)   |

Some missions are **valid** (label `1`) and some are **invalid** (label `0`)
according to a fixed but **hidden** rule. You are given a set of **labelled**
example missions (the *train* split) and must predict the labels of a set of
**unlabelled** missions (the *queries* split). You must **induce the rule** from
the examples.

The hidden rule (which you must recover, not read) is:

> A mission is valid **iff** (1) it is *well-nested* — every RETURN matches the
> most recently LAUNCHed, still-airborne drone of the **same depot**, and the
> swarm is empty when the mission ends (a Dyck-`K` word) — **and** (2) the
> maximum number of drones airborne **at any one instant** (the maximum nesting
> depth) never exceeds a per-swarm **airspace capacity `D`**. `D` is not given
> and differs per instance.

### Generalization (the point)

Some instances are **length-OOD**: the query missions are far **longer** than any
training mission. A classifier that memorises surface patterns of training
missions collapses on these; one that recovers the abstract, length-independent
rule generalizes. The final score is a **geometric mean** across instances, so
you must do well on *both* the in-distribution and the longer out-of-distribution
instances.

## Candidate contract (isolated stdin -> stdout program)

Your program reads ONE JSON object (the **public** instance) from stdin and
writes ONE JSON array (your predicted labels) to stdout. It runs in an isolated
subprocess and only ever sees the public view; the query labels and the hidden
capacity `D` stay in the evaluator.

### Public instance (stdin)

```json
{
  "num_types": 3,
  "train":   [{"seq": [0, 1, 2, 3], "label": 1}, {"seq": [1, 0], "label": 0}, ...],
  "queries": [[0, 0, 1, 1], [2, 3, 2], ...],
  "regime":  "ID",
  "seed":    1000101
}
```

- `num_types` (int `K`): number of depots; tokens range over `0 .. 2*K-1`.
- `train`: labelled missions. `seq` is the token list, `label` is `0` or `1`.
- `queries`: the missions you must classify (no labels given).
- `regime`: `"ID"` or `"OOD_LENGTH"` (informational; OOD queries are longer).
- `seed`: an integer you MAY use to seed your own RNG (kept deterministic).

### Answer (stdout)

```json
[1, 0, 0, 1, ...]
```

A JSON array of length `len(queries)`; entry `i` is your predicted label
(`0` or `1`) for `queries[i]`. Wrong length, or any entry outside `{0, 1}`,
makes that instance score **0**.

## Objective

**Maximize** the geometric mean, over all instances, of a per-instance
normalized accuracy.

For each instance let `acc_cand` be your classification accuracy against the
hidden query labels, and let `acc_base` be the accuracy of the evaluator's
baseline (predict the **majority train label** for every query). Then

```
r = clamp( 0.1 + 0.9 * (acc_cand - acc_base) / max(1 - acc_base, 0.15), 0, 1 )
```

so reproducing the baseline maps to `~0.1` and a perfect classifier maps to
`~1.0`. Valid instances are floored to a small positive value so the geometric
mean stays defined; an instance where your program crashes, times out, returns
the wrong shape, or emits a non-`{0,1}` label scores exactly `0.0`, which drives
the whole geometric mean to `0`.

The evaluator prints:

```
Ratio: <geometric mean of per-instance r>
Vector: [r_1, r_2, ...]
```

## Notes

- **Deterministic**: the instance family is fixed and seeded; there is no
  wall-time or hardware in the score.
- Multiple strategies are viable: majority guessing (weak), well-nesting only
  (ignores capacity), a capacity-aware inducer (strong), or richer learners that
  extrapolate the threshold / model the grammar. There is no free perfect
  optimum — the true capacity boundary is never exhibited by the training valids,
  so it must be extrapolated, and the longer OOD queries punish memorization.
