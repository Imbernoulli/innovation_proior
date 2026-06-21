# Difference-in-Differences

Difference-in-differences estimates a treatment effect by comparing how the treated group changes over time with how a comparison group changes over the same time interval.

For treated group `G = 1`, comparison group `G = 0`, pre-period `Post = 0`, and post-period `Post = 1`:

```text
tau =
  [E(Y | G = 1, Post = 1) - E(Y | G = 1, Post = 0)]
  -
  [E(Y | G = 0, Post = 1) - E(Y | G = 0, Post = 0)].
```

The first within-group change removes stable group level. The second change estimates common time movement. Subtracting the second from the first isolates the treated group's post-treatment departure from its projected untreated path.

In potential-outcomes form, this sample contrast equals the treated post-period effect plus any untreated-trend mismatch:

```text
tau_DID =
  E[Y^1(1) - Y^0(1) | G = 1]
  +
  {E[Y^0(1) - Y^0(0) | G = 1]
   - E[Y^0(1) - Y^0(0) | G = 0]}.
```

The second bracket is zero only under parallel untreated trends. If it is positive, negative, or large enough to cross zero, the estimate is biased in that direction and can even flip sign.

The saturated two-period regression is:

```text
Y_it = alpha + gamma G_i + lambda Post_t + tau (G_i * Post_t) + epsilon_it.
```

Here `gamma` absorbs fixed group differences, `lambda` absorbs common time shocks, and `tau` is the treatment-timing contrast.

The four regression cells are:

```text
G = 0, Post = 0: alpha
G = 0, Post = 1: alpha + lambda
G = 1, Post = 0: alpha + gamma
G = 1, Post = 1: alpha + gamma + lambda + tau
```

Taking the treated change minus the comparison change leaves exactly `tau`.

In a two-wave store-level implementation, the same estimand can be run as a first-difference regression:

```text
Delta Y_i = a + b'X_i + c Treated_i + e_i,
```

where `Delta Y_i = Y_i,post - Y_i,pre`. Card and Krueger's primary application uses this form for change in full-time-equivalent employment, with a New Jersey dummy as `Treated_i`; their exposure version replaces that dummy by `GAP_i`, the proportional starting-wage increase needed to reach `$5.05`, with `GAP_i = 0` for Pennsylvania stores and for New Jersey stores already at or above the new minimum.

Identification requires the treated group and comparison group to have the same untreated trend over the estimation window:

```text
E[Y^0(1) - Y^0(0) | G = 1] = E[Y^0(1) - Y^0(0) | G = 0].
```

The groups can differ in levels. The design fails when treatment timing coincides with a treated-group-specific shock, anticipation, spillovers, composition changes, or a counterfactual trend that the comparison group does not approximate.
