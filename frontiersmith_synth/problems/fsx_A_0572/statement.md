# The Aliased Assay: Which Sample to Test

A materials lab must identify an unknown sample and report one scalar property of
it (say a yield strength). Prior work has narrowed the sample to **K** candidate
identities. Identity `h` has a prior belief `prior[h]` (weights sum to 1) and, if
true, implies the property value `theta[h]`.

The lab owns a catalogue of **M** assays. Running assay `j` costs `cost[j]`
lab-hours. Because every instrument has finite resolution (its noise level), an
assay returns only a **coarse reading**: assay `j` reports `read[j][h]` for
identity `h`, and two identities are told apart by assay `j` **iff their readings
differ** (`read[j][h] != read[j][h']`). Cheap coarse assays lump many
near-identities into one reading; a few precise assays cost more but split a
subtle alias.

## What you choose

Pick a **set `S` of assays to run** (a design). It must satisfy the hard budget

    sum_{j in S} cost[j] <= budget.

After running `S`, the true identity produces a reading vector
`(read[j][*])_{j in S}`. This narrows the candidates to the **confusion class** —
every identity that produces that *same* reading vector. Your best estimate of
the property is then the prior-weighted mean of `theta` over that class.

## Objective (minimise)

Averaged over the prior on the true identity, the posterior estimation error is
the **within-class prior-weighted variance of theta**:

    residual(S) = sum over confusion classes C of
                    sum_{h in C} prior[h] * (theta[h] - mean_C)^2,
    mean_C = ( sum_{h in C} prior[h]*theta[h] ) / ( sum_{h in C} prior[h] ).

You minimise

    J(S) = residual(S) + gamma * ( sum_{j in S} cost[j] ).

Running nothing (`S` empty) is feasible: every identity shares the empty reading
vector, so `residual = ` the full prior variance of `theta` and `J = ` that
variance. Splitting a confusion class (some assay in `S` separates two of its
members) can only lower `residual`.

The exact readings, costs, prior, `theta`, `budget` and `gamma` are all in the
input — read them and exploit their structure.

## Input (one JSON object on stdin)

    {
      "name":  str,
      "K":     int,                  # number of candidate identities
      "prior": [float, ... K],       # prior weights, sum to 1
      "theta": [float, ... K],       # property value implied by each identity
      "M":     int,                  # number of assays
      "cost":  [int,  ... M],        # lab-hours per assay (>0)
      "read":  [[int, ... K], ... M],# read[j][h] = reading of identity h under assay j
      "budget": int,
      "gamma":  float                # cost weight in the objective
    }

## Output (one JSON object on stdout)

    {"probes": [j0, j1, ...]}        # the SET of assay indices you run

Valid iff `probes` is a list of **distinct** integers in `[0, M)` whose total
cost is `<= budget`. The empty list is valid. Any violation (bad type,
out-of-range index, duplicate, over budget, crash, timeout, non-JSON) scores
`0.0` on that instance.

## Scoring

Deterministic; no wall-clock. Let `b = J(empty design)` (the baseline). For your
design `S` the instance score is

    r = min( 1.0, 0.1 * b / max(J(S), 1e-12) ).

Running nothing reproduces the baseline and scores `0.1`. Lower `J` scores
higher; because eliminating every alias requires buying enough costly assays that
`gamma * cost` keeps `J` well above `0.1*b`, even excellent designs stay below
`1.0`. Your final score is the mean of `r` over all instances.

## What makes this hard

The obvious recipe — buy the assay with the largest information gain (posterior
entropy reduction) per lab-hour — is a **trap**. Information gain depends only on
the prior weights of what an assay separates, never on how far apart the `theta`
values are. Cheap assays that split high-prior but `theta`-clustered identities
look excellent yet barely reduce `residual`; the alias that dominates the error
is a `theta`-extreme pair that only a costly assay can break. Under the budget,
information-per-cost spends everything in the wrong place. The winning move is to
value each assay by the `theta`-variance it removes from the surviving confusion
classes.
