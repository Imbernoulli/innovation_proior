# Harbor Container Port -- Berth-Code Prediction (length-generalization)

A single quay crane at a harbor works one **LIFO stack** of shipping containers.
A *manifest* is an ordered list of moves; each move is either

- **LOAD(t)** -- place a container of bay-type `t` (`0..K-1`) on top of the stack, or
- **UNLOAD** -- remove the container currently on top.

Every manifest is a well-nested (balanced) **Dyck-K** word: no UNLOAD is ever
issued on an empty stack, and the stack is empty when the manifest ends.

Each time a container is **UNLOADED** the port stamps it with a hidden **berth
code** in `0..C-1`. The code is a fixed but hidden **local law** of two things:
the bay-type of the removed container (`top`) and the bay-type of the container
it was resting on (`under`), where `under` is the newly exposed top of the stack
-- or the special **quay** value `K` when the stack becomes empty:

```
code = LAW( top, under )        # deterministic, independent of manifest length
```

The same law governs manifests of every length, so a law inferred from short
manifests should transfer to long ones. You are given labelled **training
manifests** at one length and must predict the berth codes for **query
manifests** at that same in-distribution length (`id`) *and* at a longer
out-of-distribution length (`ood`).

## Program contract (isolated)
Your program reads ONE JSON public instance from **stdin** and writes ONE JSON
answer to **stdout**. It runs in an isolated sandbox and only ever sees the
public instance below; the hidden law and the query berth codes stay in the
grader.

### Public instance JSON
```json
{
  "K": 5,                    // number of bay-types (0..K-1)
  "C": 5,                    // number of berth codes (0..C-1)
  "quay": 5,                 // the `under` value when the stack becomes empty (== K)
  "train": [                 // labelled manifests, all the ID length
    {"moves": [2,0,-1,-1, ..], "codes": [c0, c1, ..]},
    ..
  ],
  "queries": {
    "id":  [ [moves..], .. ],   // query manifests, ID length
    "ood": [ [moves..], .. ]    // query manifests, longer OOD length
  }
}
```
In `moves`, an integer `>= 0` is a **LOAD** of that bay-type and `-1` is an
**UNLOAD**. In each training entry, `codes` lists the berth codes in the order
the UNLOADs occur (so `len(codes)` equals the number of `-1`s in `moves`).

### Answer JSON
```json
{
  "predictions": {
    "id":  [ [codes..], .. ],   // one list per queries.id  manifest
    "ood": [ [codes..], .. ]    // one list per queries.ood manifest
  }
}
```
For each query manifest, your predicted list must contain exactly one berth code
(an integer in `0..C-1`) per UNLOAD move, in order. The number of predicted
lists must match the number of query manifests, and each list length must match
that manifest's UNLOAD count. Any shape / range / type violation scores 0 for
that instance.

## Objective
For each instance let `acc_id` and `acc_ood` be the fraction of correctly
predicted berth codes over all UNLOAD positions of the `id` / `ood` query
manifests respectively. The instance objective is the geometric mean

```
obj = sqrt(acc_id * acc_ood)
```

which rewards a stack-based rule that generalizes to the longer, deeper OOD
manifests rather than memorizing absolute positions. The per-instance normalized
score is `min(1, 0.1 * obj / baseline)`, where `baseline` is the accuracy of
stamping the single most-frequent training berth code everywhere. The reported
`Ratio` is the mean over 8 instances (larger `K` and longer OOD manifests are
the harder cases).

## Strategy hints
- A constant / majority-code prediction scores about `0.1`.
- To know `top` and `under` at each UNLOAD you must correctly **replay the LIFO
  stack**; a positional or purely local heuristic breaks on the longer OOD
  manifests.
- Learning the law at increasing resolution -- global mode, then keyed on `top`,
  then on the full `(top, under)` pair -- monotonically improves accuracy.
- The full `(top, under)` law is not fully observable from the limited, skewed
  training manifests: deeper OOD stacks surface rare/unseen pairs, so unseen
  contexts must be handled by **backoff / smoothing**. Better handling of unseen
  pairs keeps improving the score, and no strategy reaches a perfect stamping.
