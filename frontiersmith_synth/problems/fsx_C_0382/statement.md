# Telescope Array Focus Calibration (length-generalization)

A segmented telescope is an **array** of `L` adaptive mirrors read out in order.
Each mirror reports a raw focus state `x[i]` (an integer in `0..K-1`). The array
has a fixed but **hidden local calibration law** that maps raw states to the
*corrected* focus state `y[i]`. The law is **order-3 and length-invariant**: the
corrected state of mirror `i` depends only on the raw states of mirror `i` and
its two upstream neighbours,

```
y[i] = LAW( x[i-2], x[i-1], x[i] )        with x[<0] = boundary
```

The same law governs arrays of every length, so a calibration inferred from
short arrays should transfer to long ones. You are given labelled **training
arrays** at one length and must predict the corrected states for **query
arrays** at that same in-distribution length (`id`) *and* at a longer
out-of-distribution length (`ood`).

## Program contract (isolated)
Your program reads ONE JSON public instance from **stdin** and writes ONE JSON
answer to **stdout**. It runs in an isolated sandbox and only ever sees the
public instance below; the calibration law and the query targets stay in the
grader.

### Public instance JSON
```json
{
  "K": 6,                                  // alphabet size (states 0..K-1)
  "boundary": 0,                           // value of x[i] for i < 0
  "train": [ {"x": [..], "y": [..]}, .. ], // labelled arrays, all length L_id
  "queries": {
    "id":  [ [x..], .. ],                  // query inputs, length L_id
    "ood": [ [x..], .. ]                   // query inputs, longer length L_ood
  }
}
```

### Answer JSON
```json
{
  "predictions": {
    "id":  [ [y..], .. ],   // one predicted array per queries.id  (same lengths)
    "ood": [ [y..], .. ]    // one predicted array per queries.ood (same lengths)
  }
}
```
Every predicted value must be an integer in `0..K-1`, each predicted array must
match the length of its query input, and the counts must match. Any shape /
range / type violation scores 0 for that instance.

## Objective
For each instance let `acc_id` and `acc_ood` be the fraction of correctly
predicted mirror states over all `id` / `ood` query positions respectively. The
instance objective is the geometric mean

```
obj = sqrt(acc_id * acc_ood)
```

which rewards models that generalize to the longer OOD arrays rather than
memorizing absolute positions. The per-instance normalized score is
`min(1, 0.1 * obj / baseline)`, where `baseline` is the accuracy of predicting
the single most-frequent training state everywhere. The reported `Ratio` is the
mean over 8 instances (increasing `K` and OOD length are the harder cases).

## Strategy hints
- A constant / majority prediction scores about `0.1`.
- Inferring a length-invariant **local** rule of increasing order (unigram →
  bigram → trigram) monotonically improves both ID and OOD accuracy.
- The full order-3 law is not fully observable from the limited, skewed
  training data, so contexts unseen in training must be handled by
  backoff / smoothing — better handling of unseen contexts keeps improving the
  score, and no strategy reaches a perfect calibration.
