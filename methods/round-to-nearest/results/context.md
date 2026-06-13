# Context: compressing trained network weights to low-bit integers

## Research question

A neural network has already been trained; its weights live in fp16 or fp32. The
problem is to shrink and speed up inference by representing the large linear-layer
weight matrices with very small signed integers, such as 4 bits per weight or even 3,
while changing the layer outputs as little as possible.

Two constraints make the problem narrow. First, no retraining or fine-tuning is
available: the weights are fixed, and the conversion must be a post-training
operation. Second, the representation must be a genuine fixed-precision integer
format, not an arbitrary lookup table or a learned codebook that is awkward for
matmul kernels. Each stored code has to decode to an approximate real value by a
simple arithmetic rule. The open design question is how to choose the integer codes
and any shared parameters for each weight tensor so the model keeps most of its
behavior under an extremely coarse grid.

## Background

Low-bit integer inference is attractive because model serving is often limited by
weight storage, memory bandwidth, and multiply-accumulate throughput. Replacing
fp16 or fp32 weights with 4-bit codes can cut weight traffic by several times, and
integer kernels are a natural match for SIMD and accelerator hardware. The price is
quantization error: with only 16 representable values at 4 bits and 8 at 3 bits, the
distance between neighboring levels can be large relative to many trained weights.

The standard hardware-friendly family is uniform affine quantization. A real value
is represented by an integer code through
```
r ~= S (q - Z),
```
where `S > 0` is a shared scale and `Z` is an integer zero-point. Uniform spacing is
the important property: the decoded real levels are evenly separated, so converting
back to real values is cheap, and integer matrix multiplication can be expressed as
integer accumulation plus scale factors. The zero-point exists so that real zero can
be represented exactly, which is important for padded activations and sparse-looking
values.

The clipping range can be placed in different ways. Asymmetric ranges use the data
minimum and maximum, which can be a good fit for skewed quantities such as
post-ReLU activations. Symmetric ranges center the grid around zero and simplify the
integer arithmetic because the zero-point vanishes, but they may spend levels on
both signs even when a tensor is one-sided. Signed integer formats also have one
extra negative code. A common restricted-range convention keeps the full
container range available for clamping while choosing the scale so that the
effective grid avoids that most-negative endpoint; the decoded levels then remain
symmetric around zero, and the integer kernel avoids a problematic extreme
product.

Granularity is another independent choice. One set of quantization parameters can
serve a whole tensor, or the tensor can be split so smaller regions get their own
parameters. Finer granularity spends more metadata, because more scales or offsets
must be stored, but it can prevent a small part of a tensor with a large range from
dictating the resolution everywhere else.

Weights and activations differ sharply. Weights are fixed after training, so their
ranges are available directly from the tensors. Activations depend on input data, so
activation quantization usually needs calibration batches or runtime statistics. A
weight-only compression rule can choose whether to ignore calibration entirely or use
it only to estimate which weight errors are most damaging to layer outputs.

A known diagnostic failure for simple post-training quantization is range imbalance.
Large, over-parameterized networks may tolerate a crude conversion, but smaller or
efficiency-tuned models can degrade sharply. Two causes are especially important:
output channels in the same layer can have weight ranges differing by more than
100x, and individual outlier weights can stretch a shared range so most other
weights get a coarse representation. Any low-overhead method has to confront those
range effects without turning the conversion into another training run.

## Baselines

**Floating-point deployment.** Keep every trained weight in fp16 or fp32 and run the
ordinary linear layers. This preserves the trained model exactly up to normal
inference precision, but it leaves the memory footprint and bandwidth cost
unchanged.

**Per-tensor affine post-training quantization.** Pick one affine grid for an entire
weight tensor from its global min/max range, map every weight to an integer code on
that grid, and dequantize with the same shared parameters. It is simple and cheap,
but a single wide range can waste nearly all resolution on channels whose weights
occupy a much smaller interval. Outliers have the same effect: they enlarge the
range while most weights are forced onto fewer useful levels.

**Non-uniform or codebook compression.** Cluster weights, use logarithmic/power-of-two
levels, or otherwise place representable values unevenly. These schemes can match a
weight distribution more closely at a fixed bit budget, but the decoded value is no
longer just a shared multiply and offset. That makes fast dense matmul kernels and
portable deployment harder.

**Binary, ternary, and shift-style networks.** Restrict weights to one bit, three
levels, or powers of two so multiplications become signs, masks, or shifts. These
formats can be efficient in specialized kernels, but they are a much stronger
constraint on the trained model and usually require training or fine-tuning around
the constraint.

**Quantization-aware training or post-quantization fine-tuning.** Insert simulated
quantization into training, or run additional optimization after converting the
weights, so the floating-point parameters adapt to the discrete grid. This can
recover accuracy, especially at low precision, but it violates the zero-retraining
constraint and can be too expensive for very large models.

## Evaluation settings

- **Model**: a pretrained decoder-only transformer language model, Mistral-7B-v0.1
  (32 layers, 4096 hidden size, 14336 FFN hidden size, grouped-query attention;
  about 7.24B parameters), loaded from a fixed HuggingFace snapshot in fp16.
- **What is quantized**: linear weight matrices inside each transformer block,
  including attention projections `q_proj`, `k_proj`, `v_proj`, `o_proj` and MLP
  matrices `gate_proj`, `up_proj`, `down_proj`. Embeddings, normalization layers,
  and the LM head remain in fp16.
- **Bit-widths / grouping knobs**: 4-bit and 3-bit signed integer weights, with the
  conversion allowed to use whole rows or contiguous blocks of input columns as its
  parameter-sharing units.
- **Calibration stream**: 128 sequences of 2048 tokens from the WikiText-2 training
  split are available to methods that want activation statistics or layer-output
  reconstruction data.
- **Metric**: WikiText-2 perplexity of the converted model and its increase over the
  fp16 baseline. Secondary metric: wall-clock time for the conversion.
- **Protocol**: load the fp16 model, optionally stream calibration data through each
  layer to gather statistics, convert each selected linear layer in place, then
  evaluate perplexity. No gradient updates to the weights are allowed.

## Code framework

The harness already exists: it loads the model, can stream calibration batches
through a layer, visits each linear layer, asks a per-layer object for a
quantized-then-dequantized weight matrix of the same shape and dtype, and writes that
matrix back. The unsettled part is the body of the weight conversion rule: how the
real-valued weight matrix becomes integer codes and then a reconstructed float matrix.

```python
import torch


def quantize_tensor(x, scale, zero_point, qmin, qmax):
    """Map a float tensor to integer codes given scale, zero-point, and range."""
    # TODO: the real-to-integer code rule we will design
    pass


def dequantize_tensor(x_int, scale, zero_point):
    """Map integer codes back to approximate float values."""
    # TODO: the integer-to-real reconstruction rule paired with quantize_tensor
    pass


def find_scale_zero(weight, num_bits=4, group_size=-1, symmetric=True):
    """Compute the conversion parameters for a weight matrix.

    The signed B-bit code range is fixed by the bit-width; per-group reshaping is
    bookkeeping. The open question is how to set (scale, zero_point) from the
    weights so that mapping onto this grid does the least harm."""
    qmin = -(1 << (num_bits - 1))
    qmax = (1 << (num_bits - 1)) - 1
    # TODO: choose (scale, zero_point) per channel or per group of columns
    pass


class LayerQuantizer:
    """Owns one linear layer; may consume calibration inputs; returns the
    reconstructed weight matrix. The strategy is what we are designing."""

    def __init__(self, layer, num_bits=4, group_size=-1):
        self.layer = layer
        self.num_bits = num_bits
        self.group_size = group_size
        self.out_features, self.in_features = layer.weight.shape
        self.dev = layer.weight.device
        self.nsamples = 0
        # buffers for any calibration statistics a strategy might accumulate
        # TODO: whatever state the chosen strategy needs (or none)

    def add_batch(self, inp):
        """Receive one batch of calibration inputs for this layer.

        A strategy may accumulate statistics here, or may ignore the data."""
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        self.nsamples += inp.shape[0]
        # TODO: gather whatever the conversion rule needs from inputs, if anything

    def quantize(self):
        """Produce the quantized-then-dequantized weight (same shape & dtype)."""
        W = self.layer.weight.data.clone().float()
        # TODO: turn W into its low-bit reconstruction using the rules above
        pass

    def free(self):
        """Release any calibration buffers."""
        pass
```
