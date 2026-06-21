I propose the canonical method name "RetNet," short for Retentive Network, and identify its central operator as multi-scale retention. The problem it addresses is the familiar impossible triangle of autoregressive language modeling: we want training-time parallelism like Transformers, constant-time-and-memory inference like RNNs, and quality that remains competitive with standard attention. Standard softmax attention achieves the first and third of these properties but fails the second, because every decode step must compare the current query with every previous key, producing a key-value cache that grows linearly with sequence length and latency that grows in the same way. Recurrent networks achieve constant inference but forfeit parallel training. Efficient attention variants and state-space models recover one missing corner while giving up another. RetNet is designed to sit in all three corners at once by replacing attention with a single operator that has three mathematically equivalent evaluation modes.

The retention operator is derived from a linear recurrence with a state matrix. Start with S_n = A S_{n-1} + k_n^T v_n and read out o_n = q_n S_n, where k_n and q_n are content projections of the input. Unrolling the recurrence gives o_n = sum_{m<=n} q_n A^{n-m} k_m^T v_m. This expression already looks like causal attention, except the relative-distance weight is carried by a matrix power rather than a softmax score. Diagonalizing A as a rotation-and-decay per dimension and absorbing the change-of-basis matrices into the query and key projections reduces A^{n-m} to a scalar decay gamma^{n-m} multiplied by a rotary relative phase e^{i(n-m)theta}. The decay enforces causality and provides a finite memory horizon, while the rotation supplies the relative position encoding without requiring an additive positional embedding.

For training, the operator is evaluated in a fully parallel form. Let Q, K, V be the usual projections after applying the rotary encoding, and let D be the lower-triangular matrix whose entry (n,m) equals gamma^{n-m} when n is at least m and zero otherwise. Then the retention output is (QK^T circle D) V. This is the same shape as a standard attention forward, two matrix multiplications and an elementwise mask, but the softmax is gone and is replaced by explicit distance-based decay. Because the entire expression is a product of per-position terms, it can be computed over the whole sequence in one GPU-friendly call.

For inference, the same operator is evaluated as a fixed-size recurrence. The state S is updated by S_n = gamma S_{n-1} + K_n^T V_n, and the output at step n is Q_n S_n. The state has constant size d_k by d_v, independent of the generated length, so each decode step performs constant work and stores constant memory. The equivalence with the parallel form follows by unrolling: S_n equals sum_{m<=n} gamma^{n-m} K_m^T V_m, and multiplying by Q_n gives exactly row n of the parallel expression because the causal mask D is the same statement as only past positions being included in the state.

For long-sequence training, a chunkwise form splits the difference. The sequence is divided into fixed-size blocks. Within each block the parallel form is used, preserving matrix-multiplication parallelism; across blocks the recurrence is used, keeping memory linear in sequence length rather than quadratic. The bookkeeping is subtle but exact: when folding a block into the recurrent state, each key is pre-weighted by the decay from its local position to the block boundary, and when a later block reads that state it scales by the decay from the boundary to the query position. The two exponents add up to the true relative distance, so the chunkwise form computes exactly the same retention map as the parallel form.

A single decay rate would restrict the model to one memory horizon, so RetNet uses multiple heads with different gamma values, ranging from quickly forgetting to almost constant. This multi-scale retention lets some heads track local syntactic dependencies while others maintain long-range discourse. Different gamma values produce outputs with different scales, so per-head normalization, implemented as RMSNorm on each head separately, balances their variances. Deleting softmax also removes its nonlinearity, so a content-dependent swish output gate is applied after the head mixing to restore nonlinear, data-dependent gating.

In practice the retention layer is inserted into a Transformer-style pre-norm residual block: layer normalization, multi-scale retention, residual connection, layer normalization, feed-forward network, residual connection. Position embeddings can be dropped because the rotary phase already encodes relative position. The value dimension is often widened to give the recurrent state more capacity, and the feed-forward intermediate dimension is adjusted to keep parameter counts comparable to a Transformer of the same width and depth.

The code below implements a minimal, self-contained version of the core idea. It provides both the parallel and recurrent evaluation modes for a single-head-equivalent retention layer and verifies that they produce the same output up to numerical tolerance.

```python
import torch
import torch.nn as nn


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin


class MinimalRetention(nn.Module):
    def __init__(self, d_model=64, n_heads=4, gamma=0.96):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dk = d_model // n_heads
        self.gamma = gamma
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def _project(self, x):
        b, n, _ = x.shape
        q = self.Wq(x).view(b, n, self.n_heads, self.dk).transpose(1, 2)
        k = self.Wk(x).view(b, n, self.n_heads, self.dk).transpose(1, 2)
        v = self.Wv(x).view(b, n, self.n_heads, self.dk).transpose(1, 2)
        return q, k, v

    def _rope(self, q, k, n):
        pos = torch.arange(n, dtype=torch.float32, device=q.device)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dk, 2, device=q.device) / self.dk))
        angles = pos[:, None] * inv_freq[None, :]
        cos = torch.cos(angles).repeat_interleave(2, dim=-1)
        sin = torch.sin(angles).repeat_interleave(2, dim=-1)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin)

    def parallel(self, x):
        b, n, _ = x.shape
        q, k, v = self._project(x)
        q, k = self._rope(q, k, n)
        distances = torch.arange(n, device=x.device).view(n, 1) - torch.arange(n, device=x.device).view(1, n)
        D = torch.tril(self.gamma ** distances)
        scores = (q @ k.transpose(-2, -1)) * D.unsqueeze(0)
        out = scores @ v
        return out.transpose(1, 2).reshape(b, n, self.d_model)

    def recurrent(self, x):
        b, n, _ = x.shape
        q, k, v = self._project(x)
        q, k = self._rope(q, k, n)
        S = torch.zeros(b, self.n_heads, self.dk, self.dk, device=x.device)
        outputs = []
        for t in range(n):
            S = self.gamma * S + k[:, :, t:t+1].transpose(-2, -1) @ v[:, :, t:t+1]
            o = q[:, :, t:t+1] @ S
            outputs.append(o.squeeze(2))
        return torch.stack(outputs, dim=1).reshape(b, n, self.d_model)

    def forward(self, x, mode="parallel"):
        out = self.parallel(x) if mode == "parallel" else self.recurrent(x)
        return self.Wo(out)


if __name__ == "__main__":
    torch.manual_seed(0)
    layer = MinimalRetention(d_model=64, n_heads=4, gamma=0.96)
    x = torch.randn(2, 32, 64)
    y_par = layer(x, mode="parallel")
    y_rec = layer(x, mode="recurrent")
    diff = (y_par - y_rec).abs().max().item()
    print("max diff parallel vs recurrent:", diff)
    assert torch.allclose(y_par, y_rec, atol=1e-5)
```
