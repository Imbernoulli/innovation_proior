Channels Last — systems-level throughput method, tagged `Computer Vision`, `Math Equivalent` in the method
card. It changes the memory format of activation and weight tensors to NHWC (channels last) rather than the
default NCHW.

The card states the effect qualitatively rather than with a single ResNet-50 number: "NVIDIA GPUs natively
perform convolution operations in NHWC format, so storing the tensors this way eliminates transpositions
that would otherwise need to take place, increasing throughput." Critically: "This is a systems-level method
that does not change the math or outcome of training in any way." And: "In our experiments, Channels Last
improves the attainable tradeoffs between training speed and the final quality of the trained model. We
recommend Channels Last for training convolutional networks."

| metric | effect (ResNet-50 / ImageNet) | direction |
|---|---|---|
| top-1 accuracy | unchanged (math-equivalent) | n/a |
| GPU throughput | increased (eliminates NCHW↔NHWC transposes) | higher is better |
| hyperparameters | none | — |

No separately logged percentage throughput number is given in the card for ResNet-50; the documented effect
is "increased throughput" at zero accuracy cost. Caveat from the card: a model with layers that cannot
support channels-last incurs overhead from PyTorch switching formats back and forth (noted for UNet);
ResNet-50 is not affected.

(Provenance: `docs/source/method_cards/channels_last.md`.)
