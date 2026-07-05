# Geothermal Casing-Log Validity — In-Distribution vs Length-OOD Generalization

## Story
A geothermal drilling rig records a **completion log**: a stream of casing operations. Each operation
either **sets** (opens) a casing of one of `K` sizes, or **seals** (closes) a casing of a given size.
A log is **structurally valid** iff

1. it is a properly nested, properly matched sequence — every *seal* closes the most recently
   *set*, still-open casing **of the same size** (a typed-Dyck word), **and**
2. the nesting depth never exceeds the rig limit `Dmax`.

You are building a **tiny fixed classifier** to flag invalid logs. To keep it deployable on the rig
controller, the model class is frozen to a **single linear unit** over a fixed menu of
order-invariant summary features. Your job is to choose the classifier's weights so it
**generalizes** — it is trained/tuned on **short** logs (in-distribution) but must also flag much
**longer** logs (length-OOD), where cues that merely correlate with validity on short logs break down.

## You are a candidate program
Read ONE public instance (JSON) from **stdin**, write ONE answer (JSON) to **stdout**.

### Public instance schema
```json
{
  "m": 6,
  "feature_names": ["abs_balance","max_depth","mean_depth","length","n_open","open_frac"],
  "Dmax": 5,
  "K": 3,
  "train": [[f0,f1,f2,f3,f4,f5, label], ...]
}
```
- `train` is a labelled sample of **short** (in-distribution) logs. Each row is the 6 features
  followed by `label` ∈ {0 (invalid), 1 (valid)}.
- The 6 features are computed by the evaluator and are all **order-invariant / bag-style**:
  `abs_balance` = |#set − #seal|; `max_depth`, `mean_depth` = max/mean of the running set−seal
  balance; `length` = #operations; `n_open` = #set; `open_frac` = n_open/length.
- `Dmax`, `K` are the rig depth limit and number of casing sizes for this instance.

### Answer schema
```json
{"w": [w0,w1,w2,w3,w4,w5], "b": <float>}
```
The evaluator predicts each hidden log as **valid** iff `w·f + b > 0`, else invalid.

## Scoring (deterministic, isolated)
Your `{w,b}` is applied to **hidden** test logs in three buckets: **id** (short, held out),
**ood_med** (~2–3× longer), **ood_long** (~4–5× longer), each 50% valid. Per bucket we measure
exact-match accuracy; the instance objective is the **geometric mean** of the three accuracies
(so ignoring the OOD buckets is heavily penalized). This is converted to a normalized score in
`[0,1]` against the trivial always-valid baseline, `min(1, 0.1·baseline_err / your_err)`, and
averaged over 10 seeded instances (varying `K`, `Dmax`, and the log distribution).

The candidate runs in an isolated sandbox and sees **only** the public instance — the hidden test
logs and their labels never leave the evaluator.

## Why it's open-ended
- **No easy optimum.** The features are order-invariant, so the linear class *cannot* detect a
  wrong-size seal that preserves the operation counts and depth profile — an irreducible error
  floor keeps the best achievable score well below 1.
- **Multiple viable strategies.** Threshold the spurious `length` cue (great ID, chance on OOD);
  fit logistic regression on all features (risks over-weighting length); or hand-pick the
  length/order-invariant structural cues (`abs_balance`, `max_depth`) that transfer to OOD. The
  ID-vs-OOD gap directly rewards generalization over memorization.
