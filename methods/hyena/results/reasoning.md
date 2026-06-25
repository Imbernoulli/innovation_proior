OK, let me think this through from scratch. The thing that bothers me is the cost of attention. I write down one head of self-attention on a length-`L` sequence `u ∈ R^{L×D}`:

```
A(u) = softmax( (1/√D) · u Mq Mkᵀ uᵀ ),   y = A(u) u Mv.
```

The middle object `u Mq Mkᵀ uᵀ` is `L × L`. I have to form it, softmax it row-wise, and multiply it by `v = u Mv`. Every one of those steps is `O(L²)` in time, and just *holding* `A(u)` is `O(L²)` in memory. So the moment `L` gets large — a whole book, an hour of audio, a megapixel image flattened into a sequence — I'm dead. The quadratic term is the wall.

The lazy move is to approximate `A(u)`: make it low-rank, or sparse, or replace the softmax with a kernel feature map so the product associates the cheap way. People have done all of these. And they're faster. But the recurring observation is that they lose something — you have to bolt real attention layers back on to recover quality. So approximating the matrix is apparently throwing away the very thing that made attention good. That tells me I shouldn't be approximating `A(u)` at all. I should figure out *what property* of `A(u)` is load-bearing, and then build something cheap that has that property natively, instead of something cheap that pretends to be `A(u)`.

So what is actually doing the work in `y = A(u) v`? Let me stare at it. For a *fixed* input, `A(u)` is just a matrix and `y = A(u)v` is a linear map. But `A(u)` is not fixed — its entries are a function of `u` itself. So a single attention layer is not one linear operator, it's a whole *family* of linear operators, and the input chooses which member of the family acts on it. The map `v ↦ y` is linear, but the *choice* of map is nonlinear in `u`. The operator is conditioned on the data — let me call that data control. I'd bet this is what lets attention adapt to context it has never seen: it can reconfigure its own mixing matrix on the fly. So whatever I build, I want to keep *this*, not the softmax.

Two more things I don't want to lose. The parameters of attention are `Mq, Mk, Mv`, each `D × D` — there is no `L` in there. The operator looks at the entire sequence but its parameter count doesn't grow with sequence length. That decoupling is precious: it means I can pour parameters into the rest of the network instead of into the mixer. And: any position can talk to any other position. No locality cutoff is baked in (masking aside). Unrestricted context.

So the target is three things at once: an operator that is (a) data-controlled — its action is a nonlinear function of the input; (b) parameter-decoupled from `L`; (c) globally connected, no locality restriction — but costs less than `O(L²)`. I'm not going to approximate the matrix; I'm going to try to *construct* one with these properties out of cheap parts. Whether cheap parts can actually deliver all three at once is the open question — I'll have to check it, not assume it.

What cheap parts do I have? Two come to mind immediately. Element-wise multiplication — gating — is `O(L)` and trivially data-dependent: if I multiply a signal pointwise by a projection of the input, the result depends nonlinearly on the input. And convolution. A convolution `y_t = (h * u)_t = Σ_n h_{t-n} u_n` is a linear map. If I write it as a matrix-vector product, the matrix is the Toeplitz matrix `S_h` built from the filter — constant along diagonals. Convolution is the natural "mixing across positions" primitive.

Now, can a convolution give me unrestricted context? Only if the filter is long. A normal CNN filter has size `M ≪ L` — it's local; the output at `t` can only see inputs within `M` steps. In fact the memory is exactly `M`: `∂y_t/∂u_{t-n} = h_n`, which is zero for `n ≥ M`. So to get any-to-any reach I need a filter as long as the sequence itself — a *long* convolution, `M = L`.

But here's the immediate problem, and it's the parameter-decoupling property biting me. If I store the filter as `L` explicit tap values per channel, then my parameter count scales with `L`. That's exactly what attention avoided. A length-`L` explicit (FIR) filter costs `L` parameters; I've reintroduced the `L`-dependence I was trying to kill. So I can't store the filter as a list of numbers.

How do I make a filter as long as the sequence without as many parameters as the sequence? I need to *generate* the filter, not store it. Represent the filter value at position `t` as the output of a small parametric function of `t`:

```
h_t = γ_θ(t).
```

Now the filter has length `L` — I can evaluate `γ_θ` at `t = 0, 1, …, L-1` — but `θ` is fixed-size, independent of `L`. This is the implicit parametrization, and it's the move that recovers parameter decoupling. The filter's *length* is now a property of where I choose to evaluate it, not of how many parameters I have. Exactly the disentanglement I wanted: memory length decoupled from parameter count.

What should `γ_θ` be? One option I know works is to make `h` the impulse response of a linear state-space model: `x_{t+1} = A x_t + B u_t`, `y_t = C x_t + D u_t`. With `x_0 = 0`, the response is `h_t = 0` for `t < 0` and `h_t = C Aᵗ B + D δ_t` for `t ≥ 0`, with memory controlled by the spectral radius of `A` and a sublinear parameter count. People have gotten this to work well for long sequences. But two things nag me. First, it's a *fixed* linear time-invariant filter — the same `h` convolves every input. That's not data control; some other factor has to make the operator depend on `u`. Second, to actually materialize `h_t = C Aᵗ B` for long `L` you need `A` in a special structured form and careful numerics — it's an iterative construction that doesn't keep a GPU busy. I'd like something freer and more hardware-friendly.

So instead: let `γ_θ` just be a small feed-forward network. Feed it (an encoding of) the position `t`, and let it output the filter value `h_t`. A shallow MLP. Then `h` can be any smooth function of position — it can imitate an SSM kernel, a decaying exponential, an oscillation, whatever — and evaluating the whole length-`L` filter is a single batched forward pass of the MLP across all positions, which is exactly the kind of dense op hardware loves. Parameters: the MLP weights, fixed-size, no `L`. So:

```
h_t = (FFN ∘ PositionalEncoding)(t).
```

There's a subtlety I should not gloss over. If I feed the MLP the raw scalar `t`, it'll struggle, because MLPs are spectrally biased — they fit low-frequency functions easily and high-frequency ones very poorly. But a useful filter often has sharp, oscillatory structure — think of a filter that picks out a specific offset, which is basically a spike, i.e. lots of high frequencies. So a plain MLP-of-`t` will give me mushy, low-pass filters and nothing sharp. Two fixes, and I want both. First, encode `t` in a Fourier-feature basis before the MLP: feed in `[t, cos(2πf_jt/L), sin(2πf_jt/L)]` for frequencies `f_j` spread from almost zero up to the chosen band limit — a truncated complex-exponential encoding. That hands the MLP oscillatory features to combine, and the number of bands `K` sets roughly how high a frequency the filter can represent at initialization. Second, use a *periodic* activation inside the MLP — `sin(ω·)` rather than ReLU. Sine activations directly inject high-frequency content into what the network can represent, and the frequency `ω` of the sine gives me a cheap knob to widen the spectrum without paying for more Fourier bands (which would widen the MLP's input and cost parameters). I should not start with filters that are too smooth: a recall-like offset is a spike, and if those directions are absent at initialization then optimization has to create them from low-frequency pieces before it can use them.

Now the long-convolution cost. Direct evaluation of `y_t = Σ_n h_{t-n} u_n` for all `t` is a sum over `L` terms for each of `L` outputs — `O(L²)`. I'm back at the wall. But this is a convolution, and convolutions are exactly what the FFT was invented to accelerate. The convolution theorem: convolution in time is pointwise multiplication in frequency. Concretely, an aperiodic convolution can be turned into a *circular* one by zero-padding both signals (to length `≥ 2L-1`, so the wrap-around of the circular convolution lands in the zero region and doesn't corrupt the linear result). A circular convolution's matrix is circulant, and every circulant matrix is diagonalized by the DFT: `Ŝ_h = W⁻¹ D_H W`, where `W` is the DFT matrix and `D_H` is diagonal with the DFT of the filter on it. So

```
y = iFFT( FFT(h) ⊙ FFT(u) ),
```

and via Cooley–Tukey each FFT is `O(L log L)`. The whole long convolution is `O(L log L)` — and crucially I never form the `L × L` Toeplitz/circulant matrix; I only ever touch length-`L` vectors. Except I should be careful: the zero-padding trick is the load-bearing step and I've talked myself into it before being burned by an off-by-something. Let me actually check that padding to `2L` and keeping only the first `L` outputs reproduces the *linear* causal convolution, not a wrapped circular one. With `L=8`, a random length-8 filter `k` and signal `u`, I run the FFT route (`rfft` to size `2L`, pointwise multiply, `irfft`, slice `[:L]`) and compare against the brute-force causal sum `y_t = Σ_{n≤t} k_{t-n} u_n`. Max absolute difference: `2.4e-7`. That's float32 round-off, so the padded FFT convolution *is* the causal linear convolution. Good — the `O(L log L)` evaluation is real, and the long implicit FFT convolution buys me unrestricted context and sublinear parameters at subquadratic cost. That's two of the three properties — but only two; I haven't touched data control yet.

But not the third. A convolution — implicit or not — is linear time-invariant. The *same* filter acts on every input. `y = S_h u` is one fixed linear operator, period. There's no data control at all. If I stack convolutions, `y = S_{h_2} S_{h_1} u`, that's still just one fixed linear operator (the product of two Toeplitz matrices). I can compose as many convolutions as I like and I'll never escape "one fixed linear map." That's a dead end for matching attention — attention's whole point was that the operator *reconfigures itself per input*.

So I need to inject data dependence between the convolutions, and the cheap primitive I set aside earlier is exactly the tool: element-wise gating. Take a projection `x = u Mx` of the input and multiply pointwise. In matrix terms, pointwise multiplication by `x` is multiplication by the diagonal matrix `D_x = diag(x)`. And `D_x` depends on the input. Now interleave: gate, convolve, gate, convolve. The operator becomes a product

```
H(u) = D_x^N S_h^N ··· D_x^2 S_h^2 D_x^1 S_h^1,
```

where the `D_x^n = diag(x^n)` are data-controlled (one diagonal per projection of `u`) and the `S_h^n` are the fixed long-convolution Toeplitz matrices. `y = H(u) v` is linear in `v`, but `H(u)` itself is a nonlinear function of `u` through all those diagonals. That's data control — built, not approximated.

Before I get excited, I have to rule out a way this could be hollow. Here's the worry: is `D_x S_h` secretly just another convolution in disguise, so the whole chain folds back into one LTI map and the "data control" is an illusion? Look at it in the padded circular representation used by the FFT. A convolution factor has the form `S_h = W* D_Φ W`, diagonal in frequency. A gate `D_x` is diagonal in *time*. Write a piece of the chain: `D_q W* D_Ψ W D_k W* D_Φ W`. If the time-diagonal `D_q` and the frequency-basis change `W*` commuted, I could slide all the gates through and pull all the `W`'s together: I'd get `W* (D_q D_Ψ D_k D_Φ) W` — a single diagonal in frequency, i.e. *one* convolution, and all the data control would have evaporated.

So the whole thing hinges on whether `D_x` (diagonal in time) and the DFT fail to commute hard enough to keep the product from being a convolution. That's concrete enough to just compute. A circular convolution matrix is exactly a circulant matrix — constant along its (mod-`L`) diagonals. So the test is: take `L=4`, two filters `h^1, h^2`, build their circulant matrices `S_{h1}, S_{h2}`, pick a gate vector `x` and form `S_{h2} D_x S_{h1}`; is it circulant? I check `S_{h2} S_{h1}` first (no gate) — circulant, as it must be, since a product of convolutions is a convolution. Then with the gate inserted, `S_{h2} D_x S_{h1}`:

```
S_{h2} @ S_{h1}        circulant? True      (no gate -> still one convolution)
S_{h2} @ D_x @ S_{h1}  circulant? False     (gate inserted -> not a convolution)
```

The gated product is genuinely *not* circulant — its diagonals are not constant — so it is not any single convolution, of any filter. That's the thing I needed: the gate sandwiched between two convolutions cannot be absorbed into them, and the chain does not collapse to an LTI map. The interleaving is what makes it work — gating *between* convolutions, not gating that could be commuted away. So `H(u)` is a real input-conditioned operator, not a convolution wearing a costume.

And it stays cheap. I never materialize `H(u)`. I evaluate it by applying the factors right-to-left to `v`: convolve (one FFT conv, `O(L log L)`), gate (pointwise, `O(L)`), convolve, gate, … For `N` convolutions that's `O(N L log L)`. So I keep the subquadratic cost while getting the data control.

Let me write the recurrence cleanly. Take `N+1` linear projections of the input: `(v, x^1, …, x^N)`. One of them, `v`, plays the role of "value" — the thing the operator acts on. The rest are the gates. Define

```
z^1 = v
z^{n+1}_t = x^n_t · (h^n * z^n)_t,     n = 1, …, N
y = z^{N+1}.
```

Unrolled:

```
y = x^N ⊙ (h^N * (x^{N-1} ⊙ (h^{N-1} * ( … (x^1 ⊙ (h^1 * v)))))).
```

Each step: convolve the running signal with a long implicit filter, then gate it by the next projection. `N` convolutions, `N` gates, `N+1` projections. The depth `N` — call it the order — controls how long the product `H(u) = D_x^N S_h^N ··· D_x^1 S_h^1` is, and a longer product means a richer class of data-controlled matrices. This is the same intuition as fast structured-matrix factorizations (butterfly/Monarch decompositions), where you write a dense matrix as a product of cheap factors and the number of factors trades directly against the expressivity of the class you can represent — except here the factors are alternately data-controlled diagonals and Toeplitz convolutions. So order is an expressivity dial.

There's a frequency-domain reading of why alternating these two operations might be powerful, though I want to make sure I have the dual identity right before I lean on it. Convolution in time is pointwise multiply in frequency — that's the convolution theorem I already used. The claim I'm reaching for is the dual: pointwise multiply in time is convolution in frequency, `widehat{x ⊙ u}_ℓ = (1/L) Σ_r x̂_r û_{ℓ-r}` (circular convolution of the spectra) under the unnormalized DFT. I half-remember the `1/L`, so I check it: `L=6`, random `x, u`, compare `fft(x ⊙ u)` against `(1/L)` times the circular convolution of `fft(x)` and `fft(u)`. Max difference `3.3e-16` — the identity holds, `1/L` and all. So the recurrence really does alternate between convolving in time and convolving in frequency, up to that normalization. The time-domain convolution (multiply in frequency) stretches the memory, letting a broad context in; the time-domain gating (convolve in frequency) sharpens the selection of which frequency components survive. Spread the context, then select — repeatedly. That's a plausible reason it captures the recall-type behavior attention is good at.

Let me sanity-check the order against things I already know. Order `N=1`: `y = x^1 ⊙ (h^1 * v)` — a single gate and a single long convolution. That's exactly a gated-state-space layer (gating composed with one long convolution). Order `N=2`: `y = x^2 ⊙ (h^2 * (x^1 ⊙ (h^1 * v)))`. Rename `x^1 = k`, `x^2 = q`, `h^1 = φ`, `h^2 = ψ`:

```
z_t = k_t (φ * v)_t,
y_t = q_t (ψ * z)_t.
```

This is the H3 mechanism — two gates, two convolutions, `z_t = k_t (φ * v)_t` then `y_t = q_t (ψ * z)_t`, which is exactly the form I know H3 to have (with H3's surrogate matrix `A(q,k) = D_q S_ψ D_k S_φ`). So at least the low orders aren't exotic: order 1 lands on a gated single long convolution, order 2 lands on H3, and what I have is the general order-`N` recurrence with free-form implicit filters in place of H3's SSM filters. The known operators look like members of this family.

Let me actually derive the surrogate "attention" matrix for order 2, to see the data-controlled matrix explicitly and convince myself it really is attention-like. Expand the convolutions as sums (causal filters, so the sums run over the past):

```
z_t = k_t Σ_{m} φ_{t-m} v_m
y_t = q_t Σ_{m} ψ_{t-m} z_m
    = q_t Σ_{m} ψ_{t-m} k_m Σ_{n} φ_{m-n} v_n.
```

Now pull the sums apart. Move `ψ_{t-m} k_m` inside the inner sum and swap the order of summation so `v_n` is the outermost variable:

```
y_t = Σ_{n} ( q_t Σ_{m} ψ_{t-m} k_m φ_{m-n} ) v_n.
```

So `y = A(q,k) v` with

```
[A(q,k)]_{t,n} = q_t Σ_{m} ψ_{t-m} k_m φ_{m-n}.
```

That's a dense `L × L` matrix, and every entry is a function of the input through `q_t` and `k_m` — an attention-like matrix that I never have to form, since I evaluate it as gate–conv–gate–conv on `v`. In matrix factors I read off

```
A(q,k) = D_q S_ψ D_k S_φ,
```

`D_q = diag(q)`, `D_k = diag(k)` the data-controlled diagonals, `S_ψ`, `S_φ` the Toeplitz convolution matrices. That much is a clean factoring of the triple sum on paper, but the index gymnastics — swapping the order of summation, lining up `ψ_{t-m} k_m φ_{m-n}` — is exactly where I make sign or off-by-one mistakes, so I won't trust it until I've run it. Take `L=5`, random causal `q, k, v, φ, ψ`. Compute `y` two ways: (a) the recurrence, `z = k ⊙ (φ * v)` then `y = q ⊙ (ψ * z)` with explicit causal sums; (b) the matrix, build `A` from the triple sum `[A]_{t,n} = q_t Σ_m ψ_{t-m} k_m φ_{m-n}` and form `A v`.

```
recurrence : [-0.0027 -0.023   0.1359  0.2438  0.9744]
A(q,k) v   : [-0.0027 -0.023   0.1359  0.2438  0.9744]
max |diff| : 8.3e-17
```

They agree to machine precision, so the swap-and-collect was correct and `A(q,k)` really is the matrix the recurrence implements. I also build the factored form `D_q S_ψ D_k S_φ` directly and subtract it from `A`: max difference `1.1e-16`, so the factorization is right too. And one property I'll need later: is `A` lower-triangular? Checking `triu(A, 1)` — all zeros. So this order-2 operator is causal by construction, which I'll want for autoregressive use; I'll come back to why in general.

The same isolation should work if I stop pretending the sequence is only a finite vector and think of signals on a group `G`. Then

```
(φ * v)(t) = ∫_G φ(t - g) v(g) dg,
(ψ * z)(t) = ∫_G ψ(t - g) z(g) dg.
```

With `z(g) = k(g) ∫_G φ(g - τ) v(τ) dτ`, the output becomes

```
y(t) = q(t) ∫_G ψ(t - g) k(g) [∫_G φ(g - τ) v(τ) dτ] dg.
```

Swap the `g` and `τ` integrations and pull out the part that depends on `v(τ)`:

```
y(t) = ∫_G [q(t) ∫_G ψ(t - g) k(g) φ(g - τ) dg] v(τ) dτ.
```

So even in the continuous case the mechanism is a linear operator on `v`, with data-controlled kernel

```
K(t, τ) = q(t) ∫_G ψ(t - g) k(g) φ(g - τ) dg.
```

That is the same object as the discrete matrix entry, just with sums replaced by integrals. The input-dependent diagonals have become multiplication operators by `q` and `k`, and the Toeplitz matrices have become convolution operators; the data-controlled structure survives the change of domain.

Now causality, because I want to train autoregressive language models and the output at position `t` must not peek at the future. A discrete operator is causal exactly when its matrix is lower-triangular. `H(u)` is a product of diagonal matrices `D_x^n` (trivially lower-triangular) and Toeplitz matrices `S_h^n`. A diagonal times a lower-triangular is lower-triangular, and a product of lower-triangular matrices is lower-triangular. So `H(u)` is lower-triangular *iff* every `S_h^n` is lower-triangular, which happens iff every filter `h^n` is causal — `h^n_t = 0` for `t < 0`. So: make each implicit filter causal and the whole operator is causal. And I don't even need to constrain the MLP to output zeros for negative `t`. With the FFT route I just evaluate the filter at `t = 0, …, L-1` (never at negative `t`) and zero-pad input and filter to at least `2L-1` before the FFT; the implementation can use `2L`. The zero-padding turns the circular convolution back into the linear aperiodic one, and since I only ever use filter taps at non-negative offsets, no future information should leak.

That argument is clean for the abstract recurrence, but the implementation has a piece the argument doesn't cover: the short depthwise `Conv1d` with `padding = k-1` then a slice to `[:L]`, plus the interplay of the gates and the `2L` FFT padding all stacked together. Padding-then-truncate convolutions are a classic place to leak one step of future, so I don't want to certify causality from the matrix argument alone — I want to perturb the assembled module and watch. I build the real `SequenceMixer` (`D=4`, `L=16`, `order=3`, so `order-1 = 2` long filters), run it on a random input `u` to get `y0`, then add `5` to a single position `p=10` of the input and run again to get `y1`. If the operator is causal, only outputs at positions `≥ 10` may change:

```
output shape: (1, 16, 4)
positions whose output changed after perturbing pos 10: [10, 11, 12, 13, 14, 15]
all changed positions >= p? True
```

Nothing before position 10 moved. So the whole assembled mixer — short conv, gates, FFT long convs, all of it — is causal, not just the idealized factorization. The depthwise conv's left-padding-and-truncate happens to be causal too, which I'd half-worried about. Causality holds end to end.

Now the filter's shape, which I've been treating as fully free-form. Letting the MLP output anything is fine in principle, but I can help it by biasing the filters toward useful shapes, and there's one bias I'm fairly sure I want: decay. A filter that decays with distance — pays attention to recent context more than the distant past, smoothly — is both a good inductive bias and a way to give different channels different effective memory lengths. So multiply the MLP output by a window:

```
h_t = Window(t) · (FFN ∘ PositionalEncoding)(t),     Window(t) = exp(-α t) (+ bias).
```

I vary the decay rate `α` across channels, so at initialization some channels carry short filters and some carry long ones — a multi-scale set of memories, rather than committing every channel to the same length. The additive bias on the window matters: a pure `exp(-α t)` forces the filter to be essentially zero past a length set by `α`, hard-capping the memory; adding a small constant lets the filter retain a floor of long-range response if training wants it. There's a nice synergy I notice between the decay and the high-frequency content: a long, slowly-decaying envelope multiplied by a high-frequency oscillation is exactly a filter that can reach far back and *select a specific past position* — pick out the input at some offset — which is what a recall task needs. (It mirrors how the order-2 special case used a short shift-type filter and a longer diagonal-type filter together.) So the windowed, high-frequency implicit filter isn't just expressive, it's expressive in the directions the tasks reward.

One more practical piece. Before the long convolutions, I run the projections through a *short* explicit depthwise convolution (size 3). It's cheap and it gives the operator immediate local mixing / a learnable short shift on `q,k,v`-like projections, which the long filters don't need to spend their capacity on — it plays the role the short shift-SSM played in the order-2 case.

Let me total the cost for an input `u ∈ R^{L×D}` at order `N`. The projections are `N+1` linear maps `R^D → R^D` applied at every position: `O(N L D²)`. In the abstract recurrence, the convolutional factors cost `O(N D L log L)` if all of them are long FFT convolutions. In the implementation I want, the first factor is the cheap short depthwise convolution on the projected streams, and the remaining `N-1` factors are long FFT convolutions, so the same asymptotic bound in `L` applies. The gates and the final output projection are lower order. Together:

```
O(N D L (log L + D)).
```

Subquadratic in `L` — the wall is gone — while the operator is data-controlled, parameter-decoupled from `L` (the filters are implicit, the projections are `D × D`), and globally connected. All three attention properties, none of the `O(L²)`.

Now let me write it as code. The filter is a module that maps positions to a length-`L` kernel: a complex-exponential positional embedding, an MLP with sine activations, and an exponential-decay window with per-channel rates. The operator projects `u` into `order + 1` parts, runs a short depthwise conv, splits into gates and value, and then applies the implementation convention I want: the cheap short convolution supplies the first local convolutional factor, while the learned implicit filters supply the remaining `order - 1` long FFT convolutions. That is why the code generates `order - 1` long filters, gates by the later projections before those long convolutions, and uses the first projection as the final gate.

```python
import math
import torch
import torch.nn as nn
from einops import rearrange


def fftconv(u, k, D):
    # Long aperiodic convolution via the convolution theorem, O(L log L).
    # Pad to 2L so the circular FFT-convolution equals the linear (causal) one;
    # we only ever use filter taps at t = 0..L-1, so no future leaks back.
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]
    return (y + u * D.unsqueeze(-1)).to(dtype=u.dtype)   # learned residual term


class Sin(nn.Module):
    # Periodic activation: lets the filter MLP carry high-frequency content,
    # against the low-frequency bias of plain MLPs. w widens the spectrum cheaply.
    def __init__(self, dim, w=10):
        super().__init__()
        self.freq = nn.Parameter(w * torch.ones(1, dim))

    def forward(self, x):
        return torch.sin(self.freq * x)


class PositionalEmbedding(nn.Module):
    # Truncated complex-exponential (Fourier-feature) encoding of position t;
    # the number of bands sets the filter's spectral cut-off at init.
    def __init__(self, emb_dim, seq_len):
        super().__init__()
        t = torch.linspace(0, 1, seq_len)[None, :, None]
        bands = (emb_dim - 1) // 2
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len
        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.register_buffer("z", z)
        self.register_buffer("t", t)

    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]


class ExponentialModulation(nn.Module):
    # Window(t) = exp(-t * delta) + shift; delta spread across channels so
    # filters start with a spread of effective lengths (multi-scale memory).
    def __init__(self, d_model, fast_decay_pct=0.3, slow_decay_pct=1.5,
                 target=1e-2, shift=0.0):
        super().__init__()
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        self.register_buffer("deltas", deltas)

    def forward(self, t, h):
        return h * (torch.exp(-t * self.deltas.abs()) + self.shift)


class LongFilter(nn.Module):
    # Implicit long filter h_t = Window(t) * FFN(PositionalEncoding(t)):
    # length L, parameter count independent of L.
    def __init__(self, d_model, emb_dim=3, order=64, seq_len=1024,
                 w=10, num_inner_mlps=2):
        super().__init__()
        assert emb_dim % 2 != 0 and emb_dim >= 3
        self.d_model = d_model
        self.bias = nn.Parameter(torch.randn(d_model))         # learned residual term
        self.pos_emb = PositionalEmbedding(emb_dim, seq_len)
        act = Sin(dim=order, w=w)
        self.implicit_filter = nn.Sequential(nn.Linear(emb_dim, order), act)
        for _ in range(num_inner_mlps):
            self.implicit_filter.append(nn.Linear(order, order))
            self.implicit_filter.append(act)
        self.implicit_filter.append(nn.Linear(order, d_model, bias=False))
        self.modulation = ExponentialModulation(d_model)

    def filter(self, L):
        z, t = self.pos_emb(L)
        return self.modulation(t, self.implicit_filter(z))     # FFN then window

    def forward(self, x, L, k=None, bias=None):
        if k is None:
            k = self.filter(L)
        if k.dim() == 3:
            k = rearrange(k, "1 l d -> d l")
        bias = self.bias if bias is None else bias
        return fftconv(x, k, bias)


class SequenceMixer(nn.Module):
    def __init__(self, d_model, l_max, order=2, filter_order=64,
                 short_filter_order=3, dropout=0.0, **filter_args):
        super().__init__()
        assert order >= 2
        self.d_model, self.l_max, self.order = d_model, l_max, order
        self.in_proj = nn.Linear(d_model, (order + 1) * d_model)   # gates plus value
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.short_filter = nn.Conv1d(                              # cheap local mixing
            (order + 1) * d_model, (order + 1) * d_model,
            kernel_size=short_filter_order, groups=(order + 1) * d_model,
            padding=short_filter_order - 1,
        )
        self.filter_fn = LongFilter(                               # order - 1 long filters
            d_model * (order - 1), order=filter_order, seq_len=l_max, **filter_args
        )

    def forward(self, u):
        L = u.size(-2)
        u = self.in_proj(u)
        u = rearrange(u, "b l d -> b d l")
        uc = self.short_filter(u)[..., :L]
        *x, v = uc.split(self.d_model, dim=1)                      # x^1..x^N, then v

        k = self.filter_fn.filter(L)
        k = rearrange(k, "1 l (o d) -> o d l", o=self.order - 1)
        bias = rearrange(self.filter_fn.bias, "(o d) -> o d", o=self.order - 1)

        for o, x_i in enumerate(reversed(x[1:])):
            v = self.dropout(v * x_i)                              # data-controlled gate
            v = self.filter_fn(v, L, k=k[o], bias=bias[o])         # implicit long conv via FFT

        y = rearrange(v * x[0], "b d l -> b l d")                  # remaining gate
        return self.out_proj(y)


class Block(nn.Module):
    def __init__(self, d_model, l_max, ffn_mult=4, **kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mixer = SequenceMixer(d_model, l_max, **kwargs)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model), nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
```

So the whole thing chains back to one decision. Attention's power is data-controlled, parameter-light, global mixing, and its only sin is the `O(L²)` cost of building the mixing matrix. A long convolution gives global mixing cheaply — `O(L log L)` by FFT — and parameter-light if I generate the filter from positions with a small sine-activated MLP and a decay window instead of storing it; but a convolution alone is one fixed linear map, no data control. Element-wise gating is `O(L)` and input-dependent, and because a time-domain diagonal does not commute with the DFT, gating sandwiched *between* convolutions cannot be absorbed into them — it makes the product `D_x^N S_h^N ··· D_x^1 S_h^1` a genuine input-conditioned operator. The clean recurrence interleaves gates with convolutional factors over the projected input; the code realizes the first factor as a short depthwise projection convolution and the remaining factors as implicit long FFT convolutions. Evaluated this way, the mixer is data-controlled, globally connected, and parameter-decoupled at `O(N L log L)` scale, with no materialized attention matrix.
