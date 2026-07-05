# Fleet-Telemetry Unsupervised Anomaly Scorer

## Background
A fleet of machines (long-haul trucks, generators, turbines) each streams a vector of
sensor readings per operating snapshot: exhaust-gas temperature, vibration RMS, oil
pressure, coolant flow, rpm, voltage, fuel flow, boost, torque, ambient. Under normal
operation these sensors are **correlated** (rpm drives fuel flow, temperature tracks
load, ...). A small fraction of snapshots are **faults**, and you do not know which.

Your job: build an **unsupervised anomaly scorer**. Given the unlabeled telemetry
matrix `X` for one fleet snapshot log, output a real-valued **anomaly score per row**
(row = one snapshot). **Higher score = more anomalous.** Your scores are graded by
ROC-AUC against the hidden fault labels; you never see the labels.

## Why it is open-ended
The faults are drawn from **diverse regimes**, and no single simple detector wins them
all:
- **global / point** — a sensor pegged far into its tail (a plain univariate z-score
  catches these);
- **correlation-break** — every individual sensor still looks normal, but the joint
  structure is destroyed (sensors that should move together do not) → needs a
  **multivariate** view;
- **contextual** — one sensor is reflected against its mean so it contradicts its
  correlated partners; marginals still look normal — a thin off-manifold shift;
- **density** — normal operation is **bimodal** (two operating modes); the fault is a
  scattered, isolated low-density outlier offset from a mode in a *random full-dimensional
  direction*, so no single sensor is extreme → a univariate view is weak; it only stands
  out in the **joint / local-density** view (Mahalanobis / kNN / LOF);
- **mixtures** of the above.

A strong entry blends a global multivariate model with a local-density model. There is
no closed-form optimum; many strategies are viable (Mahalanobis / whitening, PCA
reconstruction error, kNN / LOF, isolation-forest-style random splits, one-class
boundaries, and rank ensembles of these).

## Candidate program contract
Read ONE JSON **public instance** from stdin; write ONE JSON **answer** to stdout.

### Public instance (stdin)
```json
{
  "instance_id": 1101,
  "n": 420,                      // number of snapshots (rows)
  "d": 6,                        // number of sensors (columns)
  "X": [[...], ...],             // n x d matrix of floats (unlabeled telemetry)
  "feature_names": ["egt", "vib_rms", "oil_prs", "coolant", "rpm", "volt"],
  "contamination": 0.1,          // approximate fault fraction (a domain prior; use or ignore)
  "note": "unsupervised: labels are hidden; higher score = more anomalous"
}
```

### Answer (stdout)
```json
{ "scores": [s_0, s_1, ..., s_{n-1}] }
```
A list of exactly `n` finite floats, one per row, higher = more anomalous. Any
malformed / non-finite / wrong-length answer scores **0** on that instance.

Example skeleton:
```python
import sys, json
inst = json.load(sys.stdin)
X, n, d = inst["X"], inst["n"], inst["d"]
scores = [0.0] * n            # ... compute anomaly scores ...
print(json.dumps({"scores": scores}))
```

## Objective — MAXIMIZE
For each instance the evaluator computes ROC-AUC of your scores vs the hidden labels
(exact Mann-Whitney form with tie handling; a constant score gives AUC exactly 0.5).
Per-instance reward is an affine lift over chance:

```
reward = clip( (AUC - 0.45) / 0.5 , 0, 1 )
```

so AUC 0.50 → 0.10, AUC 0.70 → 0.50, AUC 0.90 → 0.90, AUC ≥ 0.95 → 1.00. The reported
**Ratio** is the mean reward across all instances (a diverse public set plus harder,
held-out instances — higher dimension, subtler shifts, lower contamination), so a
scorer that overfits one regime cannot reach the top.

## Rules
- Deterministic scoring; the same submission always gets the same Ratio.
- Your program runs **OS-sandboxed** in its own namespaces (no network, no filesystem
  access to the judge). It sees only the public instance and returns only its scores.
- Reject-on-garbage: non-finite or wrong-length outputs score 0.
