# Grokking: delayed generalization after memorization

Build a finite symbolic rule-learning testbed from complete binary operation tables. For each equation `a op b = c`, encode every element and operator as an opaque token, reveal a random fraction of table cells for training, and hold out the remaining cells from the same table. Train a small causal decoder and measure whether held-out cells are filled by the learned rule after the train cells are already memorized.

## Exact Operation Family

Use prime `p = 97` for modular tasks:

- `x + y mod p`
- `x - y mod p`
- `x / y mod p`, with `0 <= x < p` and `0 < y < p`
- `x / y mod p` if `y` is odd, otherwise `x - y mod p`
- `x^2 + y^2 mod p`
- `x^2 + xy + y^2 mod p`
- `x^2 + xy + y^2 + x mod p`
- `x^3 + xy mod p`
- `x^3 + xy^2 + y mod p`

Use the symmetric group tasks:

- `x * y` for `x, y in S_5`
- `x * y * x^{-1}` for `x, y in S_5`
- `x * y * x` for `x, y in S_5`

The primitive-root check is that nonzero multiplication mod `p` is addition mod `p - 1` under relabeling, so abstract-symbol learners should treat `x + y mod (p - 1)` like `x * y mod p`, and `x - y mod (p - 1)` like `x / y mod p`.

## Training Recipe

For the main experiments, use a 2-layer decoder-only transformer with width `128`, `4` attention heads, causal masking, and about `4e5` non-embedding parameters. Score only the right-hand side after the equals sign, via a shifted language-model objective over the answer symbol and trailing EOS token.

The common configuration is AdamW with learning rate `1e-3`, betas `(0.9, 0.98)`, epsilon `1e-8`, linear warmup over `10` updates, minibatch size `min(512, ceil(train_size / 2))`, weight decay `1`, and `1e5` gradient updates. The learning-time curves use `5e5` updates. The dramatic modular-division curve uses `50%` training data, Adam with no weight decay, and `1e6` updates to make the late transition visible.

A bare-bones training script's own defaults are not all the configuration defaults above: it may default to `train_data_pct=5` and `weight_decay=0`, so the runs above must pass the desired operator, train percentage, max steps, and weight decay explicitly.

## Signature the Setup Is Built to Detect

In the modular-division run with half the table revealed, the design predicts that train accuracy reaches near-perfect well before validation accuracy moves off chance, so a snapshot taken early would look like ordinary overfitting. The test of the core claim is what continued optimization, run for several further orders of magnitude of steps, does to the validation curve: if delayed generalization is real, validation accuracy should eventually rise from near-chance and reach the same near-perfect level training reached much earlier, with validation loss first climbing while the model is confidently wrong on held-out cells and then undergoing a second descent long after training loss has already gone flat.

A complementary sweep of training data fraction, on a task such as `S_5` product, is meant to discriminate "less data generalizes worse" from "less data mainly costs optimization time." The test is a matched-budget comparison: track the number of steps needed to reach a high validation-accuracy threshold and the number of steps needed to reach a high train-accuracy threshold separately as the fraction shrinks. If final validation accuracy stays close to its ceiling across a range of fractions while the time to reach that ceiling grows sharply as the fraction shrinks, and the time to fit the training set does not grow the same way, that is the signature that the fraction is charging optimization time for generalization rather than blocking it.

## Mechanism Probes

Four probes are designed to test, not assume, what pushes the optimizer from the memorizing solution toward the generalizing one:

- **Intervention sweep on `S_5`.** Matched-budget comparisons with weight decay on and off, decay-to-zero versus decay-to-initialization, minibatch or injected weight/update noise on and off, and learning rate varied within roughly one order of magnitude. The hypothesis is that weight decay is the single strongest lever, and that decay-to-origin beating decay-to-initialization, if it does, would rule out "stay near where you started" as the whole explanation and point instead to a low-norm prior plus trajectory geometry.
- **Capacity control.** Replace a controlled range of training answers (`k` from `0` up to several thousand) with answers sampled from other training equations, and check whether the network still reaches full training accuracy on the corrupted set. Reaching it, and at a similar rate regardless of `k`, would show the model has spare capacity to memorize rule-consistent examples and explicit lies alike, ruling out "not enough capacity to memorize exceptions" as the reason generalization ever happens; the companion prediction is that small `k` should barely move whether a run generalizes, while large `k` should shrink the range of training fractions that still do.
- **Representation check.** Inspect the output-layer embeddings for structure that would only appear if the model had recovered the algebra rather than interpolating it — a roughly circular arrangement for modular addition, with adding a fixed residue tracing a consistent walk around it, and clustering along cosets of a subgroup or its conjugates for `S_5`. Cleaner structure in weight-decayed runs than in undecayed ones would be a second, independent place for the weight-decay hypothesis to hold up or fail.
- **Sharpness correlation.** Compute a sharpness proxy (loss sensitivity to small parameter perturbations) alongside validation accuracy across `S_5` runs stopped at a fixed time, and take the Spearman correlation between them. A negative, statistically significant correlation is the predicted signature if flatter minima are where generalization lives; a null or positive correlation would be a real problem for that story rather than something to explain away.

## Canonical Implementation Contract

The reference implementation uses `ArithmeticDataset.make_data` to wrap each equation as:

```text
<|eos|> a op b = c <|eos|>
```

Training feeds all tokens except the final EOS as input and uses the one-token-shifted sequence as target. Loss and accuracy are computed only on target positions after `=`, so a binary operation row must predict both `c` and the trailing EOS. The division generator is equivalent to `c = a * y^{-1} mod p`; in code it enumerates quotient `c` and renders `(y * c mod p) / y = c`.

The reference transformer uses sinusoidal position encodings, a lower-triangular causal mask, two decoder blocks for the main setting, ReLU feed-forward layers, no dropout by default, and no bias in the attention/feed-forward/output linear maps.

The custom AdamW update applies decoupled decay before the Adam step:

```text
if weight_decay_form == "to_zero":
    p <- (1 - lr_t * weight_decay) p
elif weight_decay_form == "to_init":
    p <- p + lr_t * weight_decay * (p_init - p)

m_t <- beta1 m_{t-1} + (1 - beta1) g_t
v_t <- beta2 v_{t-1} + (1 - beta2) g_t^2
direction <- m_t / (sqrt(v_t / (1 - beta2^t)) + eps)
p <- p - (lr_t / (1 - beta1^t)) direction
```

If `noise_factor > 0`, Gaussian noise is added to `direction` before the final subtraction. The signs are therefore: weight decay to zero shrinks parameters, decay to initialization moves toward saved initial weights, and the Adam direction is subtracted.
