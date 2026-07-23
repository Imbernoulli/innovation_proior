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

## Empirical Signature

In the modular-division run with half the table revealed, train accuracy becomes close to perfect before `10^3` steps while validation accuracy remains near chance. With continued optimization, validation accuracy shows little evidence of generalization until about `10^5` steps and reaches the train level close to `10^6` steps. Validation loss rises during the overfitting phase and then undergoes a second descent.

On `S_5` product near `25-30%` training data, reducing the training fraction by `1%` raises the median time to `99%` validation accuracy by about `40-50%`, while time to `99%` train accuracy stays around `10^3-10^4` updates. Less data can therefore keep final generalization intact while greatly increasing optimization time.

## Mechanism Probes

Weight decay is the strongest intervention in the `S_5` ablation, more than halving the data needed to generalize compared with most alternatives. Decay to initialization helps, but decay to the origin helps more. Minibatch noise and injected Gaussian update or weight noise also help. Learning rate must stay within roughly one order of magnitude.

Capacity is not the bottleneck: after replacing `k in [0, 10, 100, 1000, 2000, 3000]` training answers with answers sampled from other training equations, the network still reaches `100%` train accuracy. Small numbers of outliers, up to about `1000`, do not strongly affect generalization; larger numbers shrink the range of train fractions that generalize.

Representation and sharpness checks support the same interpretation. Modular-addition output weights form a circular structure, adding `8` walks around that circle, and `S_5` output weights cluster into cosets of a subgroup or its conjugates. On fixed-time `S_5` runs, validation accuracy and the sharpness proxy have Spearman correlation `-0.79548` with `p < 0.000014`.

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
