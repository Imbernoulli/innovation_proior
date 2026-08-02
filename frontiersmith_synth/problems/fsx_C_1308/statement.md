# Reviewing What Matters With Too Few Reviewers

A moderation queue holds far more items than the review team can look at in a
shift. Every item already has a moderation MODEL's violation-probability score.
Anything you don't send to a human is **auto-actioned**: removed iff
`model_score >= auto_threshold`, kept up otherwise. That rule is right most of
the time, but not always — and getting it wrong costs something: missing a real
violation costs that item's category **severity**; wrongly removing a fine item
costs that category's **appeal cost**.

Sending an item to a human reviewer instead buys a probabilistic re-decision.
Each reviewer has a **base accuracy**, but it decays with every consecutive item
you place on their queue that shift (**fatigue** — no rotation, no recovery: it
resets only if you simply give them fewer items in a row), and every item also
has an **ambiguity** value that erodes accuracy further regardless of who
reviews it (some content is intrinsically hard to adjudicate). A reviewer placed
at queue position `k` on item `i` decides correctly with probability

```
eff_acc(reviewer, k, i) = max(floor_acc,
                               base_accuracy - fatigue_rate * k - difficulty_penalty * ambiguity[i])
```

If they get it wrong, the same severity/appeal cost applies as above. You have
far fewer total reviewer-slots than items, so you must choose which items are
worth spending a slot on, and in what order to hand them to which reviewer.

## Candidate program contract

Standalone program: read ONE JSON object (the public instance) from **stdin**,
write ONE JSON object (your answer) to **stdout**. Runs in an isolated
subprocess; sees only the public instance.

```python
import sys, json
inst = json.load(sys.stdin)
# ... decide a schedule ...
print(json.dumps({"schedule": schedule}))
```

### Public instance (stdin)

```json
{
  "name": "modq_101",
  "n": 50,
  "auto_threshold": 0.5,
  "floor_acc": 0.74,
  "difficulty_penalty": 0.12,
  "categories": {
    "spam": {"severity": 0.10, "appeal_cost": 0.06},
    "misinfo": {"severity": 0.28, "appeal_cost": 0.17},
    "harassment": {"severity": 0.40, "appeal_cost": 0.24},
    "hate_speech": {"severity": 0.55, "appeal_cost": 0.30},
    "graphic_violence": {"severity": 0.60, "appeal_cost": 0.32}
  },
  "items": [
    {"id": 0, "category": "harassment", "model_score": 0.53, "ambiguity": 0.61},
    ...
  ],
  "reviewers": [
    {"id": 0, "base_accuracy": 0.95, "fatigue_rate": 0.02, "capacity": 6},
    ...
  ]
}
```

### Answer (stdout)

```json
{ "schedule": { "0": [7, 2, 41], "1": [13], "2": [] } }
```

- Keys are reviewer ids as strings; the list for a reviewer is the **order**
  they work those items this shift (index 0 = position `k=0`, freshest).
- Every item id appears **at most once** across the whole schedule.
- A reviewer's list length must not exceed that reviewer's `capacity`.
- Any item id not present in any list is auto-actioned.
- Wrong length/type, an out-of-range or repeated item id, a reviewer list over
  capacity, an unknown reviewer key, a crash, timeout, or non-JSON output makes
  that instance score `0.0`.

## Scoring (deterministic)

For every item, the evaluator computes a **realized expected cost** — every
probability above is used as an expected-value weight, not a random draw, so
scoring is fully deterministic:

- unreviewed: `0` if the auto-rule's decision matches the (hidden) true label,
  else the category's severity (missed violation) or appeal cost (wrongful
  removal);
- reviewed at `(reviewer, k)`: `(1 - eff_acc) * loss`, where `loss` is the
  category's severity if the item truly is a violation, else its appeal cost.

Summing over all items gives `cost_cand`. The evaluator also computes, itself,
`cost_auto` (cost of the empty schedule) and `cost_oracle` (cost of an internal
reference policy that — unlike you — is allowed to see the hidden true labels
to greedily pick a strong subset + assignment; a stretch target, not always
reachable). Your score for the instance is

```
r = clamp( 0.1 + 0.9 * (cost_auto - cost_cand) / max(1e-6, cost_auto - cost_oracle), 0, 1 )
```

Doing nothing scores `~0.1`. Reviewing badly — e.g. spending reviewer slots on
items the auto-rule was already very likely to get right — can score **below**
`0.1`. The reported **Ratio** is the mean `r` over 10 fixed, seeded instances
(some larger / scarcer, held out for generalization); **Vector** lists the
per-instance scores.

## Suggested strategies

1. **Do nothing**: everything auto-actioned.
2. **Confidence-ranked routing**: review the highest `model_score` items first,
   filling reviewers to capacity in order.
3. **Value-ranked routing with rotation**: estimate each item's expected cost
   reduction from review (blend `model_score`, category severity/appeal cost,
   and ambiguity), skip items where review would likely hurt, and spread
   chosen items round-robin across reviewers so no one absorbs a long fatiguing
   run.
4. **Local refinement**: perturb a value-ranked schedule (seeded, deterministic)
   to squeeze out further reductions.
