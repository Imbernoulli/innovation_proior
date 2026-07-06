```python
import torch
import triton
import triton.language as tl


@triton.jit
def _row_softmax_kernel(x_ptr, y_ptr, n_cols: tl.constexpr, x_stride_m: tl.constexpr, x_stride_n: tl.constexpr, block_size: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, block_size)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * x_stride_m + cols * x_stride_n, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    y = num / den

    tl.store(y_ptr + row * n_cols + cols, y, mask=mask)


def softmax(x):
    if x.dim() != 2:
        raise ValueError("softmax expects a 2D tensor")
    if not x.is_cuda:
        raise ValueError("softmax expects a CUDA tensor")
    if x.dtype is not torch.float32:
        raise ValueError("softmax expects torch.float32 input")

    m, n = x.shape
    y = torch.empty((m, n), device=x.device, dtype=x.dtype)

    if m == 0 or n == 0:
        return y
    if n > 8192:
        raise ValueError("softmax supports N <= 8192")

    block_size = triton.next_power_of_2(n)
    num_warps = 4
    if block_size >= 2048:
        num_warps = 8
    elif block_size <= 64:
        num_warps = 1

    _row_softmax_kernel[(m,)](
        x,
        y,
        n,
        x.stride(0),
        x.stride(1),
        block_size,
        num_warps=num_warps,
    )
    return y
```