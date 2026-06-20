**Problem (from step 7).** SE + EMA refilled accuracy headroom; time to convert it to speed again.
Progressive resizing only shrinks images early — the expensive late phase is back at full 224×224. Want a
cheap spatial shrink that applies *throughout* training, including the full-resolution phase.

**Key idea — ColOut.** Instead of resampling (interpolation), just *delete* a fraction of the rows and
columns of each image and stitch what's left together. Dropping ~15% of rows and ~15% of columns keeps ~0.85
× 0.85 ≈ 0.72 of the pixels (a ~28% spatial-area cut, hence a similar conv-cost cut), and because natural
images are spatially redundant the content survives — a cat minus 15% of its rows/cols is still obviously a
cat. The dropped rows/cols are random each time the image is seen, so it doubles as a mild augmentation.

**Why it works.** Conv cost scales with spatial area, so removing rows/cols cheapens every step (late phase
included), while the high correlation between adjacent rows/cols means a modest drop barely changes the
semantics. The augmentation variability costs a small accuracy decrease — a genuine quality-for-speed trade,
which is why it goes here, on top of banked headroom. Run it **batch-wise on the GPU** (same rows/cols
dropped across the minibatch → uniform sizes → one GPU slice, called once per batch): this offloads work onto
the GPU that progressive resizing left underutilized and avoids piling onto the CPU/data-loader bottleneck,
giving ~11% throughput in CPU-bound contexts. (Per-sample on CPU is more diverse but aggravates exactly the
bottleneck progressive resizing created.) Diminishing returns when composed with progressive resizing (both
shrink size) and with the other regularizers. Suggested config: `p_row=0.15`, `p_col=0.15`, `batch=True`.

**Change / code.** The functional core `colout_batch` — pick which rows/cols to keep, then slice.

```python
def colout_batch(sample, p_row=0.15, p_col=0.15, resize_target='auto'):
    sample = ensure_tuple(sample)
    input = sample[0]
    X_tensor = image_as_type(input, torch.Tensor)

    # Get the dimensions of the image
    row_size = X_tensor.shape[-2]
    col_size = X_tensor.shape[-1]

    # Determine how many rows and columns to keep
    kept_row_size = int((1 - p_row) * row_size)
    kept_col_size = int((1 - p_col) * col_size)

    # Randomly choose indices to keep. Must be sorted for slicing
    kept_row_idx = sorted(torch.randperm(row_size)[:kept_row_size].numpy())
    kept_col_idx = sorted(torch.randperm(col_size)[:kept_col_size].numpy())

    # Keep only the selected rows and columns
    X_colout = X_tensor[..., kept_row_idx, :]
    X_colout = X_colout[..., :, kept_col_idx]
    X_colout = image_as_type(X_colout, type(input))
    return X_colout
```
