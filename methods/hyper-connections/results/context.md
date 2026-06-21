## Research Question

Large decoder-only Transformers are built by repeating the same pattern: a residual stream carries the
model state through attention and feed-forward sublayers, and normalization is placed either before or
after each residual addition. The residual path lets very deep stacks train, and it fixes how each layer's
input and output are connected. The question is whether that connection rule can be made learnable and
input-aware while leaving attention, the feed-forward network, embeddings, and the output head unchanged.

## The Residual Highway

The ordinary Pre-Norm block can be written as `h_k = h_{k-1} + T_k(Norm(h_{k-1}))`. It carries an
identity route: a signal can move forward and a gradient can move backward through many layers without
being repeatedly transformed. Every block preserves the previous state with coefficient one and adds the
new branch output with coefficient one, the same for every layer, token, and current representation.

Post-Norm changes the placement: `h_k = Norm(h_{k-1} + T_k(h_{k-1}))`. The combined state is
renormalized after each addition, so a new layer output does not become small merely because the
unnormalized residual stream has grown. Here normalization sits on the path that gradients cross at each
block.

## The Normalization Seesaw

Pre-Norm and Post-Norm sit at opposite points of the same trade-off. Under Pre-Norm the normalized
branch contribution is added into an accumulated unnormalized stream, and deeper-layer features tend to
become similar. Under Post-Norm the layer outputs stay more balanced in the forward representation, with
repeated normalization on the residual path.

A single residual stream gives one connection pattern per block. Changing a scalar branch weight moves the
model along this trade-off; it keeps the long path and the recent branch output on the same coefficient.

## Existing Pressure Points

Several residual and normalization variants frame the connection rule. Residual networks use an identity
path for deep optimization. Layer normalization and RMSNorm provide scale control, and normalization
placement sets the gradient route. Transformer training analyses characterize the Pre-Norm/Post-Norm
trade-off as a stability-versus-collapse question. Two-stream or widened-state variants add extra state
alongside the residual stream, with attention to the cost of that state at billion-parameter pretraining
and to keeping the sublayer definitions fixed.

Parallel transformer blocks and two-hop residual variants show that useful depth structure is not always
purely sequential: attention and feed-forward outputs sometimes behave as if they bypass or run beside
neighboring computations.

## Setting for a Replacement Rule

A connection rule at each residual site accepts the same sublayer `T`, provides a single input vector to
that sublayer, and returns a hidden state of the same model dimension to the surrounding Transformer.
Reducing to ordinary Pre-Norm behavior at initialization keeps training starting from a familiar
architecture.

Overhead is a consideration: connection parameters small compared with attention and feed-forward
weights, additional computation as a small mixing operation rather than another large projection, and
extra activations handled by recomputation. The setting also allows connection strengths to vary across
depth, and across tokens, so that the routing through a deep stack can differ by layer or input.
