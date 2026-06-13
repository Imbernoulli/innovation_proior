# Context: Efficient sequence modeling for long-range dependencies

## Research question

How can a single, general-purpose sequence layer model dependencies that span tens of thousands of time steps, while remaining cheap enough to train and run at scale?

Real-world signals — raw audio at 16 kHz, long documents, pixel-by-pixel images, biomedical time series — routinely contain dependencies that stretch over 10,000 or more steps. The dominant sequence architectures each break down on this regime in a different way, and there is no single layer that simultaneously (i) has a principled mechanism for unbounded memory, (ii) trains in parallel over the sequence, (iii) runs as a constant-time-per-step recurrence at inference, and (iv) scales roughly linearly in both the sequence length L and any internal state size N. A solution would have to deliver all four at once: the memory of an RNN without its vanishing gradients, the parallel training of a convolution or attention without their cost or finite context, and the fast autoregressive stepping of a recurrence.

The benchmark that crystallizes the difficulty: on a suite of synthetic and real long-range tasks, no existing model does meaningfully better than chance on the hardest one (a length-16,384 image-path classification task), and the whole field clusters far below what the tasks should allow.

## Background

**Why long-range dependencies are hard.** A recurrent model with state update x_k = f(x_{k-1}, u_k) propagates information across time by repeated application of the same map. Backpropagating through L steps multiplies L Jacobians; if their product shrinks or grows geometrically, gradients vanish or explode (Pascanu, Mikolov, Bengio 2013), so signal from step 0 cannot reliably influence the loss at step L when L is large. Gating (LSTM, GRU) and orthogonal/unitary or Lipschitz-constrained recurrences (Arjovsky et al. 2016; Erichson et al. 2021) mitigate but do not eliminate this; effective memory remains bounded and training stays sequential.

**Continuous-time linear state space models.** A foundational model from control theory and signal processing maps a 1-D input signal u(t) to a 1-D output y(t) through an N-dimensional latent state x(t):

    x'(t) = A x(t) + B u(t)
    y(t)  = C x(t) + D u(t).

A is the state (transition) matrix, B the input map, C the output map, D a feedthrough term. Because the system is linear and time-invariant, it is completely characterized by its impulse response, and convolving the input with that response gives the output — connecting the state space view to a convolution. The feedthrough D u acts as a simple skip connection and is usually set aside.

**HiPPO: principled memory through orthogonal polynomials.** A line of work on continuous-time memorization derived specific A matrices that make the latent state optimally compress the history of the input. The Legendre Memory Unit (Voelker, Kajić, Eliasmith 2019), motivated by approximating a spiking neural model, fixed such an A and built a recurrent cell around it. HiPPO (Gu, Dao, Ermon, Rudra, Ré 2020) generalized this into a framework: for a chosen weighting measure over the past, there is an A (and B) such that the state x(t) holds the coefficients of the best polynomial approximation of the input history, and is updated online by an ODE. The most important member — built on scaled Legendre polynomials (HiPPO-LegS) — is

    A_nk = -[ (2n+1)^{1/2}(2k+1)^{1/2}   if n > k
              n+1                         if n = k
              0                           if n < k ].

Empirically this is decisive: swapping a random A for this matrix in a state space layer lifted sequential-MNIST accuracy from about 60% to 98%. The intuition for *why* a plain SSM fails without it: a linear first-order ODE solves to matrix exponentials, so a generic A gives state trajectories that decay or blow up exponentially — the continuous analogue of vanishing/exploding gradients — whereas the HiPPO A is exactly tuned so the state retains a faithful, bounded summary of the whole past.

**Discretization.** To run on a sampled sequence (u_0, u_1, …) with u_k = u(kΔ), the continuous SSM is discretized with a step size Δ. The bilinear (trapezoidal / Tustin 1947) rule yields a linear recurrence

    x_k = Ā x_{k-1} + B̄ u_k,   y_k = C̄ x_k,
    Ā = (I − Δ/2·A)^{-1}(I + Δ/2·A),   B̄ = (I − Δ/2·A)^{-1} Δ B,   C̄ = C.

This is the discrete analogue of the continuous map and can be stepped like an RNN.

**Two views, two costs.** The same discrete LTI system has a convolutional form: unrolling the recurrence from x_{-1}=0 gives y_k = Σ_j C̄ Ā^{k−j} B̄ u_j, i.e. y = K̄ ∗ u with kernel

    K̄ = (C̄B̄, C̄ĀB̄, C̄Ā²B̄, …, C̄Ā^{L−1}B̄) ∈ R^L.

If K̄ is known, the convolution is one FFT pair, O(L log L), fully parallel — ideal for training. The recurrence is ideal for inference (O(1) state update per step). The obstacle is *materializing* K̄: built naively it requires L successive multiplications by Ā, costing O(N²L) time and O(NL) memory, far above the Ω(L+N) one should hope for.

**The immediate predecessor and its wall.** A deep Linear State Space Layer (Gu, Johnson, Goel, Saab, Dao, Rudra, Ré 2021) used the full SSM as a trainable layer, learning A, B, C by gradient descent and unifying the continuous, recurrent, and convolutional views. Initialized with HiPPO, it proved that deep SSMs can in principle handle long-range dependencies. But it computed the kernel essentially naively, inheriting the O(N²L) time and O(NL) memory blow-up — at N = 256 it used orders of magnitude more memory than a comparable RNN or CNN. A faster algorithm was proposed (based on the characteristic polynomial of A and its inverse modulo x^L) but is numerically unstable: for A near the identity, the characteristic polynomial (1−x)^N has coefficients up to about 2^N, and its inverse modulo x^L is larger still, so the intermediate quantities overflow. Likewise, directly diagonalizing the HiPPO matrix is hopeless: its eigenvector matrix has entries of magnitude up to 2^{4N/3} (e.g. an entry C(4i,2i) ≈ 2^{4i}), so the change of basis is catastrophically ill-conditioned.

## Baselines

**Gated RNNs (LSTM/GRU).** Maintain a hidden state with multiplicative gates regulating information flow. Core gap: still sequential at train time, and despite gating, gradients over very long horizons remain unreliable; effective context is limited well below 10k steps.

**Orthogonal / unitary / Lipschitz RNNs** (Arjovsky et al. 2016; Erichson et al. 2021). Constrain the recurrent map to have eigenvalues on or near the unit circle so the Jacobian product neither shrinks nor grows. Gap: the constraint restricts expressivity, training remains sequential, and benchmark performance on the hardest long-range tasks stays poor.

**Dilated / temporal convolutions** (WaveNet, van den Oord et al. 2016; TCN, Bai et al. 2018). Stack convolutions with exponentially growing dilation to enlarge the receptive field; fully parallel and stable. Gap: the receptive field is finite and grows only with depth/dilation, so genuinely global dependencies require many layers, and there is no single global filter.

**Transformers and efficient attention.** Self-attention relates all pairs of positions, giving global context, but costs O(L²) time and memory. The efficient-attention family — linear attention via kernel feature maps (Katharopoulos et al. 2020), random-feature approximations (Performer, Choromanski et al. 2020), locality-sensitive-hashing sparsity (Reformer) — reduces this toward O(L) or O(L log L). Gap: the approximations trade quality for speed and, on dedicated long-range benchmarks, still perform poorly; attention also has no constant-time recurrent inference mode.

**Deep Linear State Space Layer (LSSL)** (Gu et al. 2021). The closest prior method: a trainable deep SSM with HiPPO initialization, exposing recurrent/convolutional/continuous views. Core idea and math as in Background. Gap: O(N²L) time, O(NL) memory from naive kernel computation; the proposed fast algorithm and naive diagonalization are both numerically unstable. This is the wall the next step must break.

## Evaluation settings

The natural yardsticks for a long-range sequence layer, all predating any new method:

- **Long Range Arena (LRA)** (Tay et al. 2021): a benchmark suite for efficient sequence models — ListOps, byte-level text classification, byte-level document retrieval, sequential CIFAR-10 image classification, the Pathfinder spatial-reasoning task, and its length-16,384 extension Path-X — with sequence lengths from ~1k to 16k. Metric: classification accuracy; also wall-clock/throughput and memory for the "efficiency" axis.
- **Pixel-by-pixel image classification**: sequential MNIST, permuted MNIST, and sequential CIFAR-10 (flatten the image into a 1-D sequence of pixels). Metric: test accuracy.
- **Raw speech / audio classification** at the waveform level (e.g. length-16,000 spoken-command classification). Metric: test error.
- **Autoregressive density estimation / language modeling**: CIFAR-10 in bits-per-dimension; word-level perplexity on a long-document corpus (WikiText-103). Metric: bits/dim and perplexity, plus generation throughput.
- **Time-series forecasting** with long horizons. Metric: forecasting error.

Protocol axes that matter: parameter count, training compute and memory, training parallelizability, and per-step inference compute; ability to handle a change of input sampling rate without retraining.

## Code framework

The pre-existing primitives: PyTorch with autograd, an AdamW optimizer with cosine learning-rate schedule, FFT routines (`torch.fft`), and standard data pipelines for images/audio/text. The harness below stacks a sequence layer into a residual backbone — the sequence layer itself is the empty slot the method will fill.

```python
import torch
import torch.nn as nn

class SequenceLayer(nn.Module):
    """A drop-in 1-D sequence-to-sequence map: (B, H, L) -> (B, H, L).

    This is the slot the contribution occupies. What goes here — how the layer
    represents long-range structure, how it is trained (parallel) vs. run
    (stepwise), and how it is made computationally feasible — is exactly what
    has to be designed.
    """
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.h = d_model
        # TODO: parameters of the layer

    def forward(self, u):
        # TODO: map input sequence u -> output sequence y of the same shape
        pass

    def step(self, u_k, state):
        # TODO: constant-time-per-step recurrent form for autoregressive inference
        pass


class SequenceModel(nn.Module):
    """Encoder -> stack of (SequenceLayer + norm + residual) -> pool -> decoder."""
    def __init__(self, d_input, d_output, d_model=256, n_layers=4, dropout=0.2):
        super().__init__()
        self.encoder = nn.Linear(d_input, d_model)
        self.layers = nn.ModuleList(SequenceLayer(d_model) for _ in range(n_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layers))
        self.drops = nn.ModuleList(nn.Dropout(dropout) for _ in range(n_layers))
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        x = self.encoder(x).transpose(-1, -2)        # (B, H, L)
        for layer, norm, drop in zip(self.layers, self.norms, self.drops):
            z = layer(x)
            z = drop(z)
            x = z + x                                  # residual
            x = norm(x.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(-1, -2).mean(dim=1)           # mean pool over L
        return self.decoder(x)
```
