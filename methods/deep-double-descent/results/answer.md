# Deep Double Descent

## Core Definition

Treat a training procedure `T` as the whole map from a labeled training set to a predictor: architecture, optimizer, number of steps, augmentation, regularization, and any other training choices.

For distribution `D` and small `epsilon`, define Effective Model Complexity:

```text
EMC_{D,epsilon}(T)
  = max { n : E_{S ~ D^n}[ Error_S(T(S)) ] <= epsilon }.
```

I use `epsilon = 0.1` heuristically as "approximately zero" training error. EMC rises with width, training time, weaker regularization, and easier fitting conditions. It also depends on the actual labels and the training procedure, which is why it can move with label noise, augmentation, and epochs in ways that VC dimension or Rademacher complexity cannot.

## Generalized Hypothesis

For a natural data distribution `D`, neural-network training procedure `T`, and training sample size `n`:

- If `EMC_{D,epsilon}(T) << n`, increasing effective complexity decreases test error.
- If `EMC_{D,epsilon}(T) >> n`, increasing effective complexity also decreases test error.
- If `EMC_{D,epsilon}(T) approx n`, increasing effective complexity can either increase or decrease test error.

The interpolation threshold is therefore not the end of generalization. It is the critical boundary where the procedure is just able to fit the training set. Test error peaks or plateaus near that boundary, then can descend again in the over-parameterized regime.

## Three Crossings

Model-wise: fix a long training budget and vary model width. Width changes EMC. Test error follows the classical descent/ascent up to the interpolation threshold, then descends again. This creates real regimes where bigger models are worse, but only near the critical region.

Epoch-wise: fix a sufficiently large model and vary training time. Training longer changes EMC even though parameter count is fixed. Test error can fall, rise near the epoch where train error becomes approximately zero, then fall again. Thus continued training can reverse apparent overfitting.

Sample-wise: fix model and procedure, then vary `n`. More data usually lowers the curve, but it also shifts the interpolation threshold because more samples require more effective complexity to fit. Near the critical region these effects can cancel, and in some settings more data hurts.

## Mechanism

At `EMC approx n`, the learner has little slack. In linear and random-feature settings, this corresponds to a poorly conditioned or nearly unique interpolating solution, so fitting noise or misspecification can destroy global structure. In the over-parameterized regime, many interpolants exist, and the training rule can select a better one, such as the minimum-norm solution in least squares or random features.

The deep-network mechanism is not fully proved. I use the linear/random-feature theory as an anchor and treat label noise as an amplifier for the underlying sensitivity to noise or model misspecification.

The contribution is this coordinate change: organize generalization by the distance between sample count and the procedure's effective ability to fit, not by parameter count alone.
