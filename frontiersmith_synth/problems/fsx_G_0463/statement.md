# Credit-Scoring Decision Tree: Design the Split

## Setting

A consumer-credit lender approves or denies loans by predicting, from an
applicant's financial profile, whether the applicant will **DEFAULT** (label `1`)
or **REPAY** (label `0`). The lender's model is a **binary decision tree**: each
internal node tests one numeric feature against a threshold, and each leaf predicts
a class.

The entire modelling decision is the **split criterion** and the **stopping rule**
that a greedy, top-down tree builder uses. That is what you design. You are given a
labelled **training** sample of applicants; you must return a fully-built decision
tree (the tree your greedy criterion produces). Your tree is then scored by its
**accuracy on a held-out test set** of applicants drawn from the same lending
population, which you never see.

The population's default risk is concentrated in several **disjoint risk regions**
(e.g. high debt-to-income combined with high utilization; a history of prior
defaults; young thin-file applicants with many recent inquiries), and labels carry
irreducible noise. So a tree that is too shallow underfits the separate regions,
while a tree grown until pure memorises the training noise and generalises worse —
neither "always predict the common class" nor "grow until pure" is optimal.

## Candidate program (isolated stdin -> stdout)

Read ONE JSON object from stdin (the PUBLIC instance) and write ONE JSON object to
stdout. Your program runs in a sandboxed subprocess and sees only the public
instance below — never the held-out test set.

### Input (stdin)
```json
{
  "name": "portfolio101",
  "n_features": 8,
  "feature_names": ["age", "annual_income", "debt_to_income",
                    "credit_utilization", "num_prior_defaults",
                    "employment_years", "num_open_accounts",
                    "recent_inquiries"],
  "X_train": [[34.2, 51000.0, 0.31, 0.44, 0.0, 6.1, 4.0, 2.0], ...],
  "y_train": [0, 1, 0, ...]
}
```
`X_train` has `M` rows, each a list of `n_features` numbers; `y_train[i]` is the
0/1 label of row `i`.

### Output (stdout)
```json
{"nodes": [node0, node1, ...]}
```
`node0` is the ROOT. Each node is either:
- a **LEAF**: `{"leaf": 0}` or `{"leaf": 1}`
- an **INTERNAL** node: `{"feature": j, "threshold": t, "left": a, "right": b}`

To classify a row `x`: start at node `0`; at an internal node, go to child `left`
if `x[feature] <= threshold`, else child `right`; stop at a leaf and predict its
class.

### Validity
A tree is valid iff `nodes` is a non-empty list of at most **4096** node objects;
every leaf class is `0` or `1`; every internal node has an integer `feature` in
`[0, n_features)`, a **finite** numeric `threshold`, and integer child indices
`left`, `right` in `[0, len(nodes))`; and the graph reachable from the root is
**acyclic** (so classification always ends at a leaf). Any violation — wrong shape,
out-of-range index, non-finite threshold, a cycle, a crash, a timeout, or non-JSON
output — scores `0.0` on that instance.

## Objective (MAXIMIZE)

For each instance let `acc_cand` be your tree's accuracy on the held-out test set,
and `acc_base` the accuracy of the majority-class rule on that test set. The
per-instance score is
```
r = clamp( 0.1 + 0.9 * (acc_cand - acc_base) / (1.0 - acc_base), 0, 1 )
```
The majority-class rule scores ~0.1; carving the population's risk regions well
scores higher; scoring worse than the majority rule scores below 0.1. Because
labels are noisy, perfect accuracy is unreachable, so strong trees stay below 1.0.
The reported **Ratio** is the mean of `r` over all instances (a mix of easier and
harder / larger, noisier held-out portfolios).

## Scoring is deterministic

All applicant data is generated from fixed seeds; the evaluator re-runs to identical
results. No wall-clock or resource timing enters the score (timeouts are a safety
bound only).
