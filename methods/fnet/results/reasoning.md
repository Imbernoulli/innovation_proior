I start by separating the cost of attention from the role it plays. In a Transformer encoder, the feed-forward sublayer acts independently at each token and mixes only hidden features. The self-attention sublayer is the part that moves information across token positions. Its dot products, softmax, heads, and value projections make a powerful content-dependent mixer, but the structural job I need from the sublayer is narrower: after it runs, each output position should be some function of the whole sequence, so the next per-token transformation can use information that has already crossed positions.

If the job is only "let every position see every other position," then approximating the softmax all-pairs computation is not the only route. I could replace the attention sublayer with a different operation that mixes positions and let the same residual, layer norm, and feed-forward machinery do the nonlinear modeling. The most direct thing to try is a learned dense matrix along the sequence axis, possibly paired with a learned dense matrix along the hidden axis. This is worth trying first not because I expect it to be the answer but because it isolates one variable: it removes query-key attention entirely while still giving every output position a learned linear combination of every input position. If an encoder built this way trains at all, then content-dependent dot products are not load-bearing for the cross-position mixing, and I can keep looking for something cheaper to put in the slot.

But the learned sequence matrix runs into a wall I can see without training anything. It is `N x N`, so multiplying by it is still quadratic in sequence length, exactly the cost I was trying to escape. Worse, its entries are tied to one chosen maximum length: a model trained at length 512 has a 512x512 mixing matrix and cannot be applied to a longer input without re-learning. So a learned dense mixer answers the first question — attention's specific form is not required — but it does not give me a length-friendly replacement. What I actually want is a *fixed* linear map that (i) sums over all input positions, so it is globally mixing, and (ii) has a fast multiplication algorithm so I am not paying the dense `N x N` cost. Those two requirements together point away from arbitrary learned matrices and toward structured transforms.

The discrete Fourier transform fits both requirements. With the forward convention used by NumPy and JAX, a length-`N` vector is mapped by

`X_k = sum_{n=0}^{N-1} x_n exp(-2*pi*i*n*k/N)`.

Each output coefficient is a sum over every input position, so the operation is dense and global, and it carries no learned parameters at all. The sign here is the negative-exponent forward transform, the one `jnp.fft.fftn` and `np.fft.fftn` compute. Direct multiplication by the DFT matrix is quadratic like any dense matrix, but the FFT computes the same transform in `O(N log N)` for the usual composite lengths — so requirement (ii) is met by the algorithm, not by sparsifying the matrix.

Before going further I want to check the "global" claim concretely rather than trust the formula, and I want to know how the transform should be applied to a two-axis array. The encoder input is not a single vector; it has a sequence axis and a hidden axis. The sequence axis is the one that must replace attention's position mixing. I could also transform the hidden axis — the feed-forward sublayer already mixes hidden features, so a hidden transform is not the essential move, but applying both one-dimensional transforms gives a separable two-dimensional transform. I want to confirm three things on a small array: that a 2D DFT really equals doing the two 1D axis transforms in succession, that the order does not matter, and that the resulting real operator is dense. Taking a 6x4 array `x`:

```
Xhidden = fft(x, axis=1);  Xboth = fft(Xhidden, axis=0)
Xfftn   = fftn(x)
allclose(Xboth, Xfftn)            -> True,  max|.| = 0.0
allclose(Xboth, fft(fft(x,0),1))  -> True,  max|.| ~ 1.8e-15
```

So `fftn` is exactly the composition of the per-axis transforms, and swapping the axis order changes nothing beyond floating-point noise. That is what I would expect from separability — the two 1D transforms act on different axes, so their matrices act on opposite sides of `x` and commute — but it is reassuring to see it land at machine precision rather than to assert it. The candidate sublayer is therefore a two-dimensional forward DFT over the hidden and sequence axes.

That transform is complex-valued, which is the next practical wall. I do not want to carry complex numbers into the feed-forward layers, the output heads, and the loss; that would touch the whole rest of the encoder. The cheap fix is to take the real part. The question is *where*. If I take the real part right after the first (hidden) transform and then run the second (sequence) transform, do I lose anything compared to taking the real part once at the very end? On a 5x3 array:

```
y_end = fft(fft(x, axis=1),        axis=0).real
y_mid = fft(fft(x, axis=1).real,   axis=0).real
allclose(y_end, y_mid) -> False,   max|y_end - y_mid| = 1.37
```

So the two are genuinely different — discarding the imaginary part after the first transform throws away information that the second transform would otherwise have used. Taking an absolute value would deform the operation even more. The clean choice is to keep the complex intermediate through both transforms and take the real part exactly once at the end:

`y = Re(F_seq(F_hidden(x)))`.

I also wanted to confirm that this real-at-the-end operator is still a dense global mixer and not accidentally something sparse. I build its implied real linear operator by pushing each standard basis array through `Re(fftn(.))` and reading off the columns: every one of the resulting entries is nonzero (fraction of nonzeros = 1.0). So after the real part is taken, each output coordinate still depends on every input coordinate. Requirement (i) survives the projection to the reals.

Next I have to be precise about normalization, because the model is supposed to admit two interchangeable implementations: an FFT path for speed and a cached DFT-matrix path for short sequences or for hardware without a good FFT kernel. A Vandermonde DFT matrix is sometimes written with a `1/sqrt(N)` factor for a unitary convention, so I should check which convention makes the matrix path agree with `fftn`. Building `Fseq = scipy.linalg.dft(N)` and `Fhid = scipy.linalg.dft(d)` and forming `Fseq @ x @ Fhid.T`:

```
allclose(Fseq @ x @ Fhid.T, fftn(x)) -> True,  max|.| ~ 1.9e-14
```

`scipy.linalg.dft` defaults to the *unnormalized* forward DFT, and with that convention the matrix path reproduces `fftn` to floating-point precision. To see the size of the trap, I insert the unitary factors and compare magnitudes:

```
ratio |fftn(x)| / |(Fseq/sqrt(N)) @ x @ (Fhid/sqrt(d)).T|  ->  4.899...
sqrt(N*d) with N=6, d=4                                    ->  4.899...
```
