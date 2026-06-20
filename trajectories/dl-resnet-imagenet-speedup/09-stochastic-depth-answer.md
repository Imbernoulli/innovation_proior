**Problem (from step 8).** The spatial-shrink levers (resizing, ColOut) are tapped; the depth axis is
untouched. ResNet-50 runs all its residual bottleneck blocks on every example every step. Can the network be
made effectively shallower *during training* without removing capacity from the final model?

**Key idea — Stochastic Depth (block-wise).** A residual block computes `out = x + F(x)`; the skip path alone
carries a valid signal, so dropping `F(x)` doesn't break the network. Give each block an independent
probability `p` of dropping its transformation each forward pass — with probability `p`, `out = x` and the
block's three convs (and their backward pass) are skipped. A random subset of blocks is active per step, so
effective depth and per-step compute drop. Use a **linear** drop distribution (≈0 for early, load-bearing
blocks; largest for the redundant deep blocks) with peak `drop_rate=0.2` for ResNet-50.

**Why it works.** Skipped blocks mean fewer convolutions run in the forward and backward passes, cutting
wall-clock per step roughly in proportion to the drop fraction. It also regularizes — training an ensemble of
shallower sub-networks, discouraging block co-adaptation (dropout at block granularity) — though that's a
bonus here given the regularizers already stacked. Two correctness points: the linear distribution protects
the foundational early features; and at **inference** every block runs (deterministic) but its transform is
scaled by `(1 − drop_rate)` to match the expected contribution it had during training, fixing the
train/inference activation-scale mismatch. Requires skip connections (so ResNet-50 qualifies); orthogonal to
the spatial-shrink methods. Config: `target_layer_name='ResNetBottleneck'`, `drop_rate=0.2`,
`drop_distribution='linear'`, no drop warmup.

**Change / code.** Model surgery swaps bottleneck blocks for stochastic ones; the core is the stochastic
bottleneck forward — bernoulli-drop the transform in training, scale by `(1 − drop_rate)` at inference.

```python
def block_stochastic_forward(self, x):
    """ResNet Bottleneck forward where the layers are randomly skipped with
       probability ``drop_rate`` during training."""
    identity = x
    sample = (not self.training) or bool(torch.bernoulli(1 - self.drop_rate))

    if sample:
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)

        if not self.training:
            out = out * (1 - self.drop_rate)        # inference: scale to match train expectation

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
    else:                                            # transform dropped: identity only
        if self.downsample is not None:
            out = self.relu(self.downsample(identity))
        else:
            out = identity
    return out
```
