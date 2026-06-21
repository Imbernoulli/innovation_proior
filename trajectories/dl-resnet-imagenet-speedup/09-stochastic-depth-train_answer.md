ColOut bought another $\sim 11\%$ throughput by shaving rows and columns off every image — but it, like progressive resizing before it, attacks the *spatial* dimension, because conv cost scales with area. There is a whole other dimension of the compute I have not touched: *depth*. ResNet-50 is a stack of residual bottleneck blocks, and every block runs its three convolutions on every example on every step, forward and backward. Can I make the network effectively *shallower* during training without permanently removing capacity from the final model? The structure that makes this possible is the residual form itself. A residual block computes $\text{out} = x + F(x)$: the input $x$ flows along the identity skip connection and $F(x)$ — the block's three convs — is added as a learned correction on top. The defining property is that the skip path *alone* already carries a valid signal; $F(x)$ is a refinement. So if, for a given step, I simply skip $F(x)$ for some block and pass $x$ straight through the identity, the network still produces a sensible output — it is the same network minus that block's refinement for that step. The residual structure means I can drop a block's transformation without breaking anything, because the skip connection keeps the information flowing.

The method I propose is **block-wise Stochastic Depth**: during training, give each residual block an independent probability $p$ of *dropping its transformation*. On each forward pass, with probability $p$, replace $\text{out} = x + F(x)$ by $\text{out} = x$, skipping the block's convs entirely and therefore their backward pass too. The block is randomly present or absent each step, so across the network and across steps a random subset of blocks is active at any moment, and the *effective depth* the gradient flows through is reduced — fewer convolutions actually run per step, on average by the drop fraction. That is the speedup: less forward and backward compute, cutting wall-clock per step roughly in proportion to how many blocks are dropped. It is also a regularizer from the same mechanism: randomly dropping blocks trains the network as an ensemble of many shallower sub-networks (each step uses a different random subset), and no single block can rely on any particular other always being present, which discourages co-adaptation — the same logic as dropout, applied at the granularity of whole residual blocks. By now I have several regularizers stacked (decoupled weight decay, label smoothing, the augmentation in ColOut), and composing regularizers gives diminishing returns, so the regularization here is a bonus; the reason I am reaching for it is the compute reduction.

Two things have to be right. The first is *how the drop probability is distributed across blocks*. A constant rate for every block is not ideal: the early blocks do foundational feature extraction that everything downstream depends on, so dropping them is more damaging, while the later blocks are more redundant and safer to drop. So I use a *linear* drop distribution — drop probability increasing with depth, near zero for the first blocks and largest for the last — which protects the load-bearing early features while harvesting most of the speedup from the deeper, more droppable blocks. For ResNet-50 a peak `drop_rate=0.2` with the linear distribution gives the best wall-clock reduction with minimal accuracy loss; the tolerable rate scales with depth (a ResNet-101 can take nearly double, having more redundant blocks). The second is the *train/inference mismatch*. At inference I want the full deterministic network, every block present — but if I just turn all blocks on at full strength, there is a scale problem: during training each block's transformation was present only a fraction $(1-p)$ of the time, so the downstream layers learned to expect $F(x)$ contributing on average only that fraction of the time. Including $F(x)$ every time at full strength makes activations systematically larger than the network was trained on. The fix is the standard dropout-style rescaling: at inference always run the block, but scale its transformation output by $(1-p)$ to match the expected contribution it had during training. Train-time, Bernoulli-drop the whole transform; inference-time, keep it but multiply by $(1-p)$ — keeping the statistics consistent across the train/test boundary.

I implement this by model surgery, replacing ResNet-50's bottleneck blocks with stochastic versions. When training, the forward samples a Bernoulli$(1-p)$ and either runs the full $\text{conv1}\to\text{bn1}\to\text{relu}\to\text{conv2}\to\text{bn2}\to\text{relu}\to\text{conv3}\to\text{bn3}$ transform and adds it to the (possibly downsampled) identity, or skips straight to the identity; when not training, it always runs the transform but scales it by $(1-p)$ before adding. Stochastic depth needs skip connections to work — you cannot drop a block in a plain feedforward net without disconnecting it — so it applies only to architectures with residual connections, which ResNet-50 is, and it is orthogonal to all the spatial-shrinking methods, so it composes with them. The config targets the bottleneck blocks by name with block-wise dropping and the linear distribution, no warmup: `target_layer_name='ResNetBottleneck'`, `drop_rate=0.2`, `drop_distribution='linear'`. What I expect is a modest reduction in wall-clock per step — about $5\%$ at `drop_rate=0.2` on ResNet-50, since I am only skipping a fraction of the deeper blocks on average — at accuracy held close to baseline. The core is the stochastic bottleneck forward.

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
