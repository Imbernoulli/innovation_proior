Squeeze-and-Excitation refilled the accuracy tank and EMA tightened the variance, so I have quality headroom again — which means it is time to spend it on speed again. Progressive resizing already showed me the lever: smaller images make the convolutions cheaper, because conv cost scales with spatial area. But progressive resizing only shrinks images in the *early* phase; by the late phase it is back to full $224\times224$ and full cost. Is there a way to shave the spatial size a little *throughout* training, including the expensive late phase, without the heavy-handedness of resizing everything down? Resizing downsamples by interpolation, blending neighboring pixels to change the resolution. There is a cruder, cheaper way to make an image tensor smaller: just *delete* some of its rows and columns and stitch what is left back together. The image is then physically smaller — fewer pixels along each axis — so every convolution over it is cheaper, by roughly the product of the kept fractions, and unlike interpolation I am not resampling anything, just slicing out whole rows and columns.

The method I propose is **ColOut**: drop a random fraction of the rows and a random fraction of the columns of each image. Does deleting rows and columns destroy the image? If I drop a *small* fraction — say $15\%$ of rows and $15\%$ of columns — mostly no. Natural images are spatially redundant: adjacent rows and columns are highly correlated, so removing a scattered $15\%$ of them barely changes the content. A cat with $15\%$ of its rows and columns snipped out is still obviously a cat, just slightly compressed and missing a few stripes of pixels. Meanwhile the size drops meaningfully — keeping $85\%$ of rows and $85\%$ of columns means roughly $0.85 \times 0.85 \approx 0.72$ of the pixels, a $\sim 28\%$ reduction in spatial area and hence a similar cut to conv cost — while the semantic content survives. That is the trade: a modest, near-content-preserving shrink that is cheap to compute and applies on every step, full-resolution phase included. And there is a second benefit for free: which rows and columns I drop is *random* each time the image is seen, so the same training image appears slightly differently across epochs. That is a data augmentation — a mild regularizer, like a random crop or erasing but operating by deleting scattered lines rather than a contiguous block. ColOut does double duty: shrinking the image for speed and augmenting it for a touch of regularization. The augmentation cuts the other way too, though — the variability from randomly dropping lines is a distortion the model must be robust to, and on already-well-learned content that distortion is a small accuracy tax. So this is firmly a quality-for-speed trade, which is exactly why it goes here, on top of the headroom SE and EMA banked: I have accuracy to spend, and ColOut is one of the cheapest ways to convert it into throughput.

The design choice that determines how much speed I get is whether I drop the *same* rows and columns for every image in a minibatch or different ones per image. The per-example version drops independent rows/columns for each image — more diverse augmentation, but it produces images of *different* sizes within a batch (each loses different lines), so it cannot be a single batched tensor op and must run on the CPU inside the dataloader, per-sample. The batch version drops the *same* rows and columns for the whole minibatch — slightly less diverse, so slightly lower accuracy, but because every image in the batch ends up the same size I can do it as one GPU tensor slice on the whole batch at once, immediately before the model, called once per batch. Which one I want depends on where my bottleneck is, and I know where it is, because progressive resizing put it there: the early small-image phase is *CPU/data-loader bound*, the GPU starved waiting for images. Run ColOut per-sample on the CPU and I pile more work onto the exact resource that is already the bottleneck — terrible. Run it batch-wise on the GPU and I offload the work onto the underutilized GPU and barely touch the CPU, which is why the batch-wise GPU version yields roughly an $11\%$ throughput increase for ResNet-50 on ImageNet in CPU-bottlenecked contexts: it is called once per batch and its operations are on the GPU. So for this stacked recipe, where the data pipeline is the constraint, the batch-wise GPU version is the right call — accept the slightly lower accuracy from dropping the same lines across the batch in exchange for the large throughput win and for not aggravating the CPU bottleneck. The settings are `p_row=0.15`, `p_col=0.15`, `batch=True`.

One interaction to note: ColOut and progressive resizing both shrink the spatial dimensions, so they show *diminishing returns* when composed — the second size-reducer cannot cut as much as it would alone, because the first already shrank things — and both are mild regularizers, so combining them gives sublinear accuracy effects. I should not expect ColOut's standalone numbers to stack linearly on top of progressive resizing, but a meaningful chunk of its throughput gain survives, and that is worth the small accuracy cost given the headroom I am holding. The core is the row/column-dropping operation: pick which rows and columns to keep (sorted, for slicing), then slice.

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
