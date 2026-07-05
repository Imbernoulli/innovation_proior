# Masked-Entry Imputation of Clinical Cohort Records

## Background
You are given a de-identified electronic-health-record cohort: a matrix `X` of `N`
patients (rows) by `D` clinical variables (columns) such as age, BMI, systolic /
diastolic blood pressure, resting heart rate, fasting glucose, HbA1c, cholesterol
panels, creatinine, eGFR, and so on. These variables are **not independent**: a small
number of latent physiological factors (a metabolic axis, a cardiovascular axis, a
renal axis, ...) drive many columns at once, so the true matrix is approximately
**low rank** plus per-measurement noise.

A seeded subset of the cells has been **held out (masked)**. You see the matrix with
those cells blanked and must reconstruct them as accurately as possible. This is the
classic missing-value imputation task: score = reconstruction error on the masked
held-out entries.

## Program contract
Your program is a **standalone process**: read ONE JSON object (the public instance)
from stdin, write ONE JSON object (your answer) to stdout. You never see the true
values of the masked cells.

### Public instance (stdin)
```json
{
  "N": 45,                       // number of patients (rows)
  "D": 12,                       // number of clinical variables (columns)
  "names": ["age", "bmi", ...],  // length-D column names
  "matrix": [[..], [..], ...],   // N x D; a cell is a number if observed, null if masked
  "masked": [[i, j], ...]        // coordinates of the held-out cells, row-major sorted
}
```
A cell `matrix[i][j] == null` exactly for the coordinates listed in `masked`. Every
column keeps at least one observed entry.

### Answer (stdout)
```json
{ "preds": [v0, v1, ...] }
```
`preds` must be a list of finite real numbers, one per entry of `masked` and in the
**same order** as `masked`: `preds[k]` is your imputed value for cell
`masked[k] = [i, j]`.

## Objective (MINIMIZE)
Let `sd_j` be the standard deviation of the observed entries in column `j`. The score
of your answer on an instance is the **normalized RMSE** over the masked cells:
```
obj = sqrt( mean over masked (i,j) of  ((pred_ij - true_ij) / sd_j)^2 )
```
Normalizing by `sd_j` makes columns on different clinical scales (e.g. potassium ~4
vs. cholesterol ~190) contribute comparably.

## Scoring
The evaluator computes a baseline `b` = `obj` of the **column-mean** imputation
(predict every masked cell as its column's observed mean). Your normalized ratio on an
instance is
```
r = min(1, 0.1 * b / obj)
```
so the column-mean rule scores ~0.1 and an imputation `k` times more accurate scores
`min(1, 0.1*k)`. Any malformed answer -- wrong length, wrong type, or a non-finite
value (`nan`/`inf`) -- scores 0 on that instance. The reported `Ratio` is the mean of
`r` over all (public and harder held-out) instances.

## Notes / strategy hints
- Predicting the column mean ignores all correlation between variables; you can do
  strictly better by exploiting it.
- Viable approaches include nearest-neighbour patients, per-column regression against
  the other columns, and iterative low-rank matrix completion (SoftImpute / EM-PCA).
- The larger, noisier held-out instances are less recoverable, so no method reaches a
  perfect score -- there is genuine headroom and no single dominant optimum.
- Scoring is deterministic; there is no time/GPU component.
