# Cutout Synthesis

Primary source: DeVries and Taylor, "Improved Regularization of Convolutional Neural Networks with Cutout," arXiv:1708.04552. Canonical code: `uoguelph-mlrg/Cutout`, especially `util/cutout.py` and `train.py`.

## Core diagnosis

Dropout's dense-layer story is model averaging and co-adaptation prevention, but in convolutional layers pointwise removal is weakened by spatially correlated neighbors. Dropping one activation often leaves the same evidence available nearby, so convolutional dropout behaves more like noise robustness. Feature-map dropout variants respond to spatial correlation, but they operate after feature maps exist and can leave the visual cue present in other maps.

## Method reconstruction

The method removes visual evidence at the input, before any feature map is computed. It chooses a center uniformly from valid image coordinates, builds a clipped square mask around that center, multiplies the normalized tensor by the mask, and leaves the label unchanged. The zero fill is meaningful only because the transform is placed after per-channel normalization, so zero is approximately the dataset mean.

The paper stresses the boundary case: the square is allowed to lie partly outside the image. This does not create truly clean images, but it does create lightly occluded examples. A comparable alternative is to force the patch inside the image and apply it only with 50% probability, which supplies genuinely unmodified images.

## Exact code facts

Reference code:

- `y = np.random.randint(h)` and `x = np.random.randint(w)`.
- `y1 = np.clip(y - length // 2, 0, h)` and `y2 = np.clip(y + length // 2, 0, h)`, with analogous x bounds.
- `mask[y1:y2, x1:x2] = 0.`
- `mask = torch.from_numpy(mask)`; `mask = mask.expand_as(img)`; `img = img * mask`.
- The transform is appended after `transforms.ToTensor()` and `Normalize(...)`.

For even lengths, the unclipped patch side is exactly `length`. For odd lengths, the implementation masks `2 * (length // 2)` pixels per side, because Python slicing excludes `y2` and `x2`. The paper reports even tuned values, so this off-by-one-for-odd-inputs is a code-fidelity note rather than an experimental inconsistency.

## Hyperparameters and settings

- CIFAR-10: `length=16`.
- CIFAR-100: `length=8`.
- SVHN: `length=20`.
- STL-10: paper reports `24` without augmentation and `32` with augmentation.
- CIFAR schedule: 200 epochs, batch 128, SGD with Nesterov momentum 0.9, weight decay 5e-4, initial learning rate 0.1, multiplied by 0.2 after epochs 60, 120, and 160.
- SVHN schedule: 160 epochs, initial learning rate 0.01, multiplied by 0.1 after epochs 80 and 120.

## Audit notes

The repaired deliverables remove the previous "clean images by edge clipping" phrasing, which was too strong: edge clipping gives lightly occluded images, while the 50% constrained alternative gives unmodified images. The answer now states the odd-length case exactly as implemented. The context prompt avoids naming the method and avoids telling the reader to transplant context-encoder erasing directly into supervised classification.
