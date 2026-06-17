Let me start from what actually hurts. I have a `d`-parameter network, `d` is north of a hundred million, and I want to train it across `M` workers. Each worker computes a stochastic gradient on its slice of data, and before anyone can take a step those gradients have to meet — go up to a parameter server, get aggregated, come back down. The usual accounting is a 32-bit float per coordinate on upload and another 32-bit float on the return path. With `d > 10^8` the arithmetic is cheap and the *communication* is the wall: `64 M d` bits an iteration, and it gets worse the more workers I add, which is exactly backwards from what I want parallelism to do. So the real problem isn't "optimize faster," it's "say less per coordinate." If I could get the message down to one bit per coordinate — and ideally one bit on the way *back* too — the communication wall basically disappears.

The crudest possible compression of a number is its sign: throw away the exponent and the mantissa entirely, keep one bit. So the temptation is just to send `sign` of each gradient coordinate and step `x_{k+1} = x_k - delta * sign(g~_k)`. That's as compressed as it gets. The question is whether it can possibly converge, and whether I can *prove* it converges, because the field is littered with compression schemes that work on a benchmark and have no theory, and with schemes that have theory that turns out to be vacuous.

Before I commit, let me look hard at why the existing approaches don't already solve this, because the gaps will tell me what a real solution has to do. Sign-based updates aren't new — Rprop, from Riedmiller and Braun, has been around for decades. Its move is to ignore the gradient magnitude and act only on its sign, adapting a per-weight step size: bump the step up by ~1.2 if a weight's last two gradient signs agree, cut it by ~0.5 if they flip. That's robust and fast in full batch, and it's the ancestor of RMSprop and Adam. But there's a famous reason it dies on minibatches, and I should internalize it because it's the trap I'm walking toward. SGD's whole logic is that, with a small enough learning rate, consecutive steps *average* the gradient over successive minibatches. A pure sign rule destroys that average. Picture a weight that gets gradient `+0.1` on nine minibatches and `-0.9` on the tenth. Its true average gradient is zero — it should sit still. But a sign rule sees nine `+` votes and one `-` vote and increments nine times, decrements once by about the same amount, so the weight marches a long way in the wrong direction. Magnitude-blindness and minibatch noise are a bad combination.

RMSprop's fix — which is what made the whole Adam family work — was to keep a moving average of the squared gradient and divide by its root, `g / sqrt(<g^2>)`, so adjacent minibatches divide by *similar* numbers and the average is restored. Notice what that division is, though: it's a soft, smoothed version of dividing by `|g|`, which is a soft version of taking the sign. Adam's step is `<g>_{beta1} / sqrt(<g^2>_{beta2})` — a mean over a root-mean-square. If I let both exponential-average timescales shrink to zero, `beta1, beta2 -> 0`, every average collapses to the current single sample and the step becomes `g~ / sqrt(g~^2) = g~/|g~| = sign(g~)`. So `sign(g~)` is literally the zero-memory limit of Adam. That's a comforting thought and a worrying one at once: comforting because the most successful optimizers in deep learning are *already* secretly sign-flavored, so the sign can't be crazy; worrying because the very averaging I'd be throwing away is what Hinton's example says I need. Whatever I prove will have to confront the biased, magnitude-blind sign head-on rather than smoothing it away.

Why not just keep the magnitude and compress it some other way? That's what the principled compression schemes do, and their gap is instructive. QSGD and TernGrad stochastically round each coordinate to a few discrete levels in such a way that the compressed gradient is an *unbiased* estimate of the true gradient. Unbiasedness is seductive because it lets you bootstrap standard SGD theory verbatim — the compressed gradient is still a valid stochastic gradient, just noisier. But "just noisier" is the whole problem: to make a one-bit message unbiased you have to randomize it hard, and that inflates the variance. For one-bit QSGD the variance blows up by a factor of order `sqrt(d)`, and with `d > 10^8` the SGD-style bound it inherits has a `sqrt(d)`-sized constant — it's technically a theorem and it says nothing. And the return trip is worse: when the server sums up a bunch of quantized updates the result isn't one-bit anymore, so the broadcast back to the workers picks up log factors, `(2 + log(2M+1)) M d` bits. Seide's 1-bit SGD takes the opposite tack — quantize to a thresholded sign and carry the discarded magnitude forward as an error-feedback residual added to the next minibatch. That's near-lossless in practice on speech nets, but it's a heuristic with no convergence guarantee; the error accumulator is a corrective bolt-on nobody had characterized.

So here's the fork. The unbiased route gives me theory but the theory is empty at scale, because unbiasing a one-bit message demands a variance explosion. The natural conclusion is exactly the opposite of what everyone reached for: don't unbias. Use the *biased* `sign(g~)` directly and confront the bias in the analysis. Bias is supposed to be the scary word, but stare at what the bias actually is. The sign of a coordinate is only *wrong* when the noise on that coordinate is large enough to flip it past zero — when `g~_{k,i}` lands on the opposite side of zero from the true `g_{k,i}`. That requires the noise to exceed the signal in magnitude. So the bias isn't some uncontrolled monster; it's governed by a per-coordinate signal-to-noise ratio. When a coordinate's true gradient is large relative to its noise, the sign is almost always right and I make good progress; when the gradient is small relative to noise, the sign can be a coin flip, but then I'm near a stationary point and going slightly the wrong way is cheap. That asymmetry is the thing to build the proof on, and it's a thing unbiased schemes literally cannot use, because they spend all their structure forcing the expectation to be exact instead of letting it be approximately-right-where-it-matters.

Now I need the right notion of smoothness, and the standard scalar one is going to be wrong for this. Usually one writes `|f(y) - [f(x) + g(x)^T(y-x)]| <= (L/2)||y-x||_2^2` with a single Lipschitz constant `L`. But a sign update steps by exactly `delta` in *every* coordinate at once — the move is `-delta * sign(g~)`, a vector of `±delta`, so `||x_{k+1}-x_k||_inf = delta` and the step lives on a box, not a ball. The natural majorization for a box-shaped step is coordinate-wise. So let me assume a per-coordinate smoothness: there's a vector `L = [L_1,...,L_d]` with `|f(y) - [f(x) + g(x)^T(y-x)]| <= (1/2) sum_i L_i (y_i - x_i)^2`. This is strictly finer than the scalar version — set `L := ||L||_inf = max_i L_i` and it collapses back to the standard `(L/2)||y-x||_2^2`, so I lose nothing. And I'll assume a per-coordinate variance bound on the oracle: unbiased, `E[g~(x)] = g(x)`, with `E[(g~_i - g_i)^2] <= sigma_i^2` for a vector `sigma`. Again the standard total-variance bound `sigma^2 := ||sigma||_2^2` falls out by summing. The reason I want these *fine-grained* is that the sign update is going to treat every coordinate identically, so whether it wins or loses will depend on how the gradient, the noise, and the curvature are *distributed across* coordinates — and a scalar `L` or scalar `sigma^2` has already thrown that distribution away.

Let me try to bound the one-step improvement and see what falls out. Plug the sign step into the coordinate smoothness inequality:

  f_{k+1} - f_k <= g_k^T (x_{k+1}-x_k) + sum_i (L_i/2)(x_{k+1}-x_k)_i^2.

The step is `x_{k+1}-x_k = -delta_k sign(g~_k)`, and each `(sign)_i^2 = 1`, so the second sum is just `delta_k^2 sum_i L_i / 2 = (delta_k^2/2)||L||_1`. The first term is `-delta_k g_k^T sign(g~_k)`. So

  f_{k+1} - f_k <= -delta_k g_k^T sign(g~_k) + (delta_k^2/2)||L||_1.

Now I have to deal with `g_k^T sign(g~_k)`, where the gradient `g_k` is true but the sign is taken on the *noisy* `g~_k`. Coordinate by coordinate, `g_{k,i} sign(g~_{k,i})` equals `+|g_{k,i}|` when the signs agree and `-|g_{k,i}|` when they disagree. Write that as one expression: `g_{k,i} sign(g~_{k,i}) = |g_{k,i}| - 2|g_{k,i}| I[sign(g~_{k,i}) != sign(g_{k,i})]`, where `I[.]` is the indicator. Summing, `g_k^T sign(g~_k) = ||g_k||_1 - 2 sum_i |g_{k,i}| I[disagree]`. Substitute:

  f_{k+1} - f_k <= -delta_k ||g_k||_1 + 2 delta_k sum_i |g_{k,i}| I[disagree] + (delta_k^2/2)||L||_1.

This is already telling me a lot. The good term is `-delta_k ||g_k||_1` — progress proportional to the `l_1` norm of the gradient. The cost is the curvature term `(delta_k^2/2)||L||_1` and the *error* term, the one carrying every coordinate where the noise flipped the sign. Take the expectation over the noise, conditioned on `x_k` (which fixes `g_k`), and the indicator becomes a probability:

  E[f_{k+1}-f_k | x_k] <= -delta_k ||g_k||_1 + 2 delta_k sum_i |g_{k,i}| P[sign(g~_{k,i}) != sign(g_{k,i})] + (delta_k^2/2)||L||_1.

Everything now rides on `|g_{k,i}| P[wrong sign]`. If `g_{k,i}=0`, that whole product is already zero, so there is nothing to divide by. Otherwise the signal-to-noise intuition can be cashed out cleanly. A wrong sign needs the noise to overpower the signal, so `P[sign(g~_{k,i}) != sign(g_{k,i})] <= P[|g~_{k,i} - g_{k,i}| >= |g_{k,i}|]`. Markov on the absolute deviation gives `<= E|g~_{k,i}-g_{k,i}| / |g_{k,i}|`. And I only control the *second* moment, not the first absolute moment, so Jensen pulls the expectation inside the square root the right way, `E|X| <= sqrt(E X^2)`: `<= sqrt(E[(g~_{k,i}-g_{k,i})^2]) / |g_{k,i}| = sigma_{k,i}/|g_{k,i}|`. Multiply through by `|g_{k,i}|` and the magnitude *cancels*:

  |g_{k,i}| P[sign(g~_{k,i}) != sign(g_{k,i})] <= sigma_{k,i}.

That cancellation is the whole trick. The error a flipped sign costs me is proportional to the gradient magnitude there, but the *probability* of flipping is inversely proportional to it, so the product is bounded purely by the noise scale `sigma_{k,i}` regardless of how big or small the gradient is. The bias problem just got tamed — not by removing the bias, but by showing its damage is capped by the noise. And `sigma_{k,i}` is the noise after averaging a minibatch of size `n_k`, so `sigma_{k,i} <= sigma_i / sqrt(n_k)`. Substituting:

  E[f_{k+1}-f_k | x_k] <= -delta_k ||g_k||_1 + 2 (delta_k/sqrt(n_k)) ||sigma||_1 + (delta_k^2/2)||L||_1.

Now I get to choose `delta_k` and `n_k`. The curvature cost is `(delta_k^2/2)||L||_1` and the progress is `delta_k ||g_k||_1`; balancing a quadratic-in-`delta` cost against a linear-in-`delta` gain over `K` steps wants `delta_k ~ 1/sqrt(||L||_1 K)`. And the noise term `2(delta_k/sqrt(n_k))||sigma||_1` is the one that won't vanish on its own — `sign` is biased, so unlike SGD I can't lean on a learning-rate decay alone to kill the noise; I have to actually *reduce* the noise, which means growing the batch. Set `n_k = K`. Then `delta_k = 1/sqrt(||L||_1 K)` and `n_k = K` give

  E[f_{k+1}-f_k | x_k] <= -(1/sqrt(||L||_1 K)) ||g_k||_1 + (2/(sqrt(||L||_1) K)) ||sigma||_1 + 1/(2K).

Take the full expectation over the trajectory and telescope `k = 0..K-1`. Since `f_0 - f* >= f_0 - E[f_K] = E sum_k (f_k - f_{k+1})`,

  f_0 - f* >= E sum_{k=0}^{K-1} [ (1/sqrt(||L||_1 K)) ||g_k||_1 - (1/(2 sqrt(||L||_1) K))(4||sigma||_1 + sqrt(||L||_1)) ]
           = sqrt(K/||L||_1) E[(1/K) sum_k ||g_k||_1] - (1/(2 sqrt(||L||_1)))(4||sigma||_1 + sqrt(||L||_1)).

Rearrange for the average gradient norm:

  E[(1/K) sum_k ||g_k||_1] <= (1/sqrt(K)) [ sqrt(||L||_1)(f_0 - f* + 1/2) + 2||sigma||_1 ].

Here `n_k = K` means it takes `N = K * K = O(K^2)` gradient calls to reach step `K`, so after squaring the `1/sqrt(K)` bound becomes `1/K = O(1/sqrt(N))`:

  E[(1/K) sum_k ||g_k||_1]^2 <= (1/sqrt(N)) [ sqrt(||L||_1)(f_0 - f* + 1/2) + 2||sigma||_1 ]^2.

There it is: the `1/sqrt(N)` rate, the SGD-class rate, for a one-bit-per-coordinate method on a non-convex objective. No `d` factor multiplying the noise, no variance explosion. The growing batch should bother me on the surface — it looks wasteful — but it's a systems *advantage*, not a cost: a large batch parallelizes, so `N` gradient calls happen in only `O(sqrt N)` iterations, which means `O(sqrt N)` rounds of the expensive communication rather than `O(N)`. Fewer rounds of the thing I'm trying to compress.

Now I should be honest and compare to SGD on the same footing, because "looks like the SGD rate" and "is competitive with SGD" are different claims. The catch is the bound I just proved is in `l_1`, while the standard SGD statement controls the average squared `l_2` norm: with `delta = 1/L`, `L = ||L||_inf`, `sigma^2 = ||sigma||_2^2`, plain SGD gives `E[(1/K) sum ||g_k||_2^2] <= (1/sqrt N)[2L(f_0-f*) + sigma^2]`. My translated sign bound will control the square of the expected average `l_2` norm, so the comparison is an upper-bound comparison in the same units, not literally the same stationarity measure. The density `phi(v) = ||v||_1^2/(d||v||_2^2)` is the bridge. It gives `||g||_1^2 = phi(g) d ||g||_2^2`, `||L||_1^2 <= phi(L) d^2 ||L||_inf^2 = phi(L) d^2 L^2`, and `||sigma||_1^2 = phi(sigma) d ||sigma||_2^2 = phi(sigma) d sigma^2`. Push these through (and `(a+b)^2 <= 2(a^2+b^2)` to split the square), and my signSGD bound becomes

  E[(1/K) sum ||g_k||_2]^2 <= (2/sqrt N)[ (sqrt(phi(L))/phi(g)) L (f_0-f*+1/2)^2 + 4 (phi(sigma)/phi(g)) sigma^2 ],

with `phi(g)` a lower bound on the gradient density along the trajectory. Lined up against the SGD bound, the shapes are similar except for two ratios of densities: `R_1 := sqrt(phi(L))/phi(g)` multiplying the curvature term and `R_2 := phi(sigma)/phi(g)` multiplying the noise term. So whether the magnitude-blind sign helps or hurts is entirely a question of *relative density*. If `R_2 >> 1` — noise much denser than the gradient — SGD's term is smaller and SGD wins on noise; if `R_2` is order one or less, signSGD's noise handling is as good or better, and it gets compression on top. There's a clean picture behind that. Imagine the gradient is dense but the noise is concentrated on a sparse set of extremely loud coordinates. Plain SGD will let those few screaming coordinates dominate the step — it'll random-walk along them and barely attend to the real signal unless you crush the learning rate. The sign update can't be screamed at: every coordinate moves by the same `delta`, so the sparse loud noise gets scaled *down* relative to the dense signal. In the toy geometry `f(x) = (1/2)||x||^2` on `R^100`, with `g(x) = x` dense and Gaussian noise only on the first coordinate, the density ratios predict exactly that regime.

The curvature ratio `R_1` is subtler and I shouldn't oversell it. The trouble is the SGD bound itself is *slack* under sparse curvature — when I bounded SGD I had to replace each `L_i` by `||L||_inf`, which throws away SGD's own benefit from sparse curvature. So `R_1` doesn't cleanly say "signSGD wins on sparse curvature"; what it reliably says is the *converse*: when `R_1 >> 1`, i.e. curvature much denser than the gradient, signSGD is in trouble, because it shoves every coordinate by `delta` including high-curvature coordinates that had a tiny gradient, and that overshoot is expensive. So read `R_1 >> 1` as the danger zone for the sign method. Putting it together: when gradients are dense, signSGD shrugs off sparse loud noise; when gradients are sparse, plain SGD is safer against dense curvature and noise.

Which regime are real networks in? This is the question I can actually answer before committing, because nothing about it requires the method to exist — I just need to measure the densities of `g` and `sigma` in a real net. Welford's algorithm gives me the exact gradient and per-coordinate variance in a single stable pass, so at every epoch of training a Resnet-20 on CIFAR-10 I do one extra pass and compute `phi(g)` and `phi(sigma)`. The finding is decisive for me: throughout training the gradient density and the noise density stay of the *same order*, and both are dense — `phi` nowhere near `1/d`. Same-order densities means `R_2` is order one, which is precisely the regime where signSGD should converge about as fast as SGD and pocket the compression. And it's not a CIFAR fluke — gradients come out dense across a spread of datasets and architectures. So the geometry of deep learning happens to be friendly to the magnitude-blind sign. (Curvature density I can't measure cheaply, so I leave `R_1` as a theoretical caveat rather than a measured one.)

So far I've used a batch as large as `K`, while practical training often stays around hundreds of examples per minibatch. I'd like a small-batch story, `n_k = 1`, but if I keep only the crude `P[wrong] <= sigma_i/|g_i|` bound, the noise term won't decay and the thing won't converge to a stationary point. To do better I need a tighter handle on the sign-flip probability, and for that I have to assume something about the noise *shape*. The reasonable assumption, and one I can check, is that the per-coordinate noise is unimodal and symmetric — which is exactly what the central limit theorem buys me as soon as the batch is more than a handful, and histograms of real per-coordinate gradient noise give me the empirical premise I need by looking close to Gaussian at batch 256. Under unimodality there's an old, sharp tail bound — Gauss's inequality: for a unimodal `X` with mode `nu` and `tau^2 = E[(X-nu)^2]`, `P[|X-nu| > k] <= (4/9)(tau^2/k^2)` when `k/tau > 2/sqrt3`, and `<= 1 - k/(sqrt3 tau)` otherwise. Symmetry makes the mode the mean and `tau` the standard deviation `sigma`. A wrong sign (say `g_i < 0`, without loss of generality) is `g~_i - g_i >= |g_i|`, and by symmetry that one-sided event is exactly half the two-sided `|g~_i - g_i| >= |g_i|`. So the flip probability is

  P[wrong] <= { (2/9)(1/S_i^2)   if S_i > 2/sqrt3 ;   1/2 - S_i/(2 sqrt3)   otherwise },

with `S_i := |g_i|/sigma_i` the per-coordinate signal-to-noise ratio. For every nonzero signal it is strictly below `1/2`, and at zero signal the descent cost is zero anyway. Now redo the one-step bound at `n_k = 1` with this case split. For high-SNR coordinates (`S_i > 2/sqrt3`, call that set `B_k`), `(2/9)(1/S_i^2) <= 1/6`, so their error contributes at most `2 delta_k sum_{B_k} |g_{k,i}|/6 = (delta_k/3) sum_{B_k}|g_{k,i}|`, which only nicks the `-delta_k|g_{k,i}|` progress, leaving `-(2 delta_k/3) sum_{B_k}|g_{k,i}|`. For low-SNR coordinates, plug `1/2 - S_i/(2 sqrt3)`: the `1/2` exactly cancels half the `2 delta_k|g_{k,i}|` against the progress, and the `-S_i/(2 sqrt3)` term leaves `-delta_k sum_{not B_k} g_{k,i}^2/(sqrt3 sigma_i)`. So the improvement is

  E[f_{k+1}-f_k|x_k] <= -(2 delta_k/3) sum_{B_k}|g_{k,i}| - delta_k sum_{not B_k} g_{k,i}^2/(sqrt3 sigma_i) + (delta_k^2/2)||L||_1.

A mixed norm appeared on its own: high-SNR coordinates converge in `l_1` (`|g|`), low-SNR ones in a variance-weighted `l_2` (`g^2/sigma`). That's structurally sensible — when the sign estimate is reliable you ride the `l_1` geometry; as the SNR shrinks and the sign degrades, you fall back to a gentler quadratic. With `delta_k = 1/sqrt(||L||_1 K)`, the low-SNR coefficient is exactly `1/sqrt(3||L||_1 K)`, and the high-SNR coefficient `2/(3 sqrt(||L||_1 K))` is larger because `2/3 > 1/sqrt3`, so I can weaken both terms to the same positive coefficient before telescoping. Using `N = K` for small batch, I get

  E[ min_k ( sum_{B_k}|g_{k,i}| + sum_{not B_k} g_{k,i}^2/sigma_i ) ] <= sqrt(3||L||_1/N) (f_0 - f* + 1/2).

So small-batch signSGD also hits the `1/sqrt N` rate, and again with no explicit dimension dependence; early in training (high SNR everywhere) it's `l_1` convergence, late (low SNR) it's `l_2`. The noise enters *linearly* here rather than quadratically as in SGD, which is a real structural difference in what the bound is measuring.

Now the part that pays for the whole exercise: making the *return* trip one-bit too. The single-machine update is `x_{k+1} = x_k - delta sign(g~)`. The obvious distributed version has each worker `m` send `sign(g~_m)` — one bit up — and the server step on `x_{k+1} = x_k - delta sum_m sign(g~_m)`. That compresses the *upload*, but `sum_m sign(g~_m)` is an integer in `[-M, M]`, so what the server broadcasts back is *not* one bit. Can I get one bit both ways? What if the server takes the sign of the vote count and broadcasts *that*:

  x_{k+1} = x_k - delta sign[ sum_{m=1}^M sign(g~_m) ].

Each worker votes its believed sign, the server tallies, takes the majority, and sends a single bit back. Majority vote — one bit up, one bit down. The elegance is suspicious, so I have to prove it doesn't cost convergence. Look back at where my whole signSGD argument hinged: the one inequality `|g_i| P[sign(g~_i) != sign(g_i)] <= sigma_i`. If I can show the *same* bound with `sign(g~_i)` replaced by the majority decision `sign[sum_m sign(g~_{m,i})]`, then the same descent argument goes through unchanged. For the no-loss coding-theory statement I should avoid exact ties — use an odd worker count or a neutral tie rule — because a deterministic `+1` tie rule is asymmetric and, for an even two-worker tie with a negative true sign, can count a one-one split as wrong. For SNR `S = |g_i|/sigma_i <= 1` the bound is trivial (the `sigma_i` on the right already dominates), so I only need `S > 1`. For one worker, the one-sided Cantelli inequality (one-sided Chebyshev, no shape assumption needed) gives failure probability `q = P[g~_i - g_i >= |g_i|] <= 1/(1 + g_i^2/sigma_i^2) = 1/(1+S^2)`, and for `S > 1` that's below `1/2`. Below one-half is the magic threshold: if each worker is independently right more often than wrong, the server is receiving a *repetition code* — the same true sign bit transmitted `M` times through independent noisy channels — and strict majority is the maximum-likelihood decoder of a repetition code, which only decreases the error probability. So the strict-majority decision is at least as reliable as one worker's in the no-tie decoder, the key inequality holds with the same `sigma_i`, and the convergence machinery is unchanged. When I mirror the implementation later, I still map a zero vote sum to `+1` with `>= 0`; that is the binary wire convention, not the symmetric tie break used by this no-loss argument.

That no-loss statement is conservative, though — it says "no worse," which makes the extra `M-1` workers look pointless, and that can't be the real story. Intuitively, when SNR is comfortably above one, each worker is right *much* more than half the time, and a repetition code drives the error down *exponentially* in the number of repeats; I should be able to turn that into an actual variance reduction. The hope is `|g_i| P[majority wrong] <= sigma_i/sqrt M`, i.e. `M` workers behave like one worker with `sqrt M` times less noise — exactly the `sqrt M` speedup that full-precision distributed SGD enjoys from averaging. But I have to be careful, because for a *skewed or bimodal* noise distribution majority vote can backfire: a variable with `P[X=50]=0.1, P[X=-1]=0.9` has positive mean `+0.1` yet `P[sign(X)=sign(mean)] = 0.1`, far below half, so each worker is usually *wrong* about the sign and adding workers would drive the error *up*, not down. That's precisely why the shape assumption matters, and it's why I lean on unimodal symmetry — under which Gauss gives me `q <= q~(S) < 1/2` strictly. The central limit theorem makes that assumption mild: even that nasty bimodal variable, averaged over a few tens of samples, looks Gaussian.

Let me actually push the `sqrt M` through. Let `Z` count the workers with the correct sign bit; `Z ~ Binomial(M, p)` with `p = 1 - q`. Majority is correct when `Z > M/2`, so I need `P[Z <= M/2] <= 1/(sqrt M S)`. Define the margin `eps := 1/2 - q = p - 1/2 >= eps~(S) := 1/2 - q~(S) > 0`. Work with the failures `Z_bar = M - Z`, mean `Mq`, variance `Mpq`. Then `P[Z <= M/2] = P[Z_bar >= M/2] = P[Z_bar - Mq >= M eps]`, and Cantelli gives `<= 1/(1 + (M eps)^2/(M p q)) = 1/(1 + M eps^2/(pq))`. Since `pq = (1/2 - eps)(1/2 + eps) = 1/4 - eps^2`, this is `1/(1 + M/((1/(4eps^2)) - 1))`. Now `1/(1 + x^2) <= 1/(2x)` with `x^2 = M/((1/(4eps^2))-1)` turns this into `P[Z<=M/2] <= sqrt((1/(4eps^2)) - 1)/(2 sqrt M)`. So I'm done if `sqrt((1/(4eps^2)) - 1) <= 2/S`, equivalently `(1/(4eps^2)) - 1 <= 4/S^2`, and since `eps >= eps~(S)` it suffices to check this with `eps~(S)`. Two cases. If `S <= 2/sqrt3`, then `eps~ = S/(2 sqrt3)`, so `1/(4 eps~^2) - 1 = 3/S^2 - 1 < 4/S^2`. If `S > 2/sqrt3`, then `eps~ = 1/2 - (2/9)/S^2`, and grinding the algebra, `1/(4 eps~^2) - 1 = (1/S^2)(8/9 - (16/81)/S^2)/(1 - (8/9)/S^2 + (16/81)/S^4) < (1/S^2)(8/9)/(1 - (8/9)/S^2) < 4/S^2`, the last step using `S > 2/sqrt3`. Both cases clear, so `P[Z <= M/2] <= 1/(sqrt M S)`, which is exactly `|g_i| P[majority wrong] <= sigma_i/sqrt M`. Feed that into the same descent machinery with `sigma` replaced everywhere by `sigma/sqrt M`:

  E[(1/K) sum ||g_k||_1]^2 <= (1/sqrt N)[ sqrt(||L||_1)(f_0-f*+1/2) + (2/sqrt M)||sigma||_1 ]^2,

with `N` counted per worker. The noise term shrank by `sqrt M`. So majority vote — one bit each way, `2 M d` bits an iteration against full-precision SGD's `64 M d` — gets the same variance-reduction speedup from `M` workers as full-precision distributed SGD under the symmetry assumption.

I haven't used momentum yet, and momentum is the thing practitioners reach for, so let me fold it in — and the cleanest place to put it is *inside* the sign. Instead of signing the raw stochastic gradient, accumulate a momentum `m_{k+1} = beta m_k + (1-beta) g~_k` and sign *that*: `x_{k+1} = x_k - delta sign(m_{k+1})`. Sign of the momentum — call it Signum. Why might this help, and what does it cost? Momentum is an average of recent gradients, so it has lower variance than a single sample — averaging beats down the noise term. But it averages in *stale* gradients evaluated at earlier, different points, and because the function is curved those stale gradients are biased estimates of the current gradient. So momentum buys variance reduction at the price of curvature-induced bias: a knob, `beta`, trading the two. Pushing `beta -> 1` averages over a longer horizon and kills the noise term, but inflates the bias; small `beta` is the reverse. I want to *prove* this, which is the hardest analysis in the lot, so let me first abstract what I actually need from a sign-based method.

All of these signed-update bounds have the same skeleton: I take the sign of *something*, `v_k`, and the only thing that matters is how often `sign(v_k)` disagrees with `sign(g_k)`. So let me state a master lemma once. For any update `x_{k+1} = x_k - delta_k sign(v_k)` with `v_k` any measurable function of the history, if for `k >= C` the expected sign-disagreement cost is controlled, `E[ sum_i |g_{k,i}| P[sign(v_{k,i}) != sign(g_{k,i}) | x_k] ] <= xi(k)` with `xi(k) -> 0`, then the same single-step-plus-telescope argument gives

  (1/(K-C)) sum_{k=C}^{K-1} E||g_k||_1 <= [ f_C - f* + 2 sum_k delta_k xi(k) + sum_k delta_k^2 ||L||_1/2 ] / [ (K-C) min_k delta_k ],

and with `delta_k = delta/sqrt k` and `xi(k) = kappa/sqrt k` this is `[ (f_C - f*)/delta + (2 kappa + ||L||_1 delta/2)(log K + 1) ] / (sqrt K - C/sqrt K)`. I get it by repeating the signSGD descent calculation with `v_k` in place of `g~_k`: coordinate smoothness, the same `g_k^T sign(v_k) = ||g_k||_1 - 2 sum |g_{k,i}| I[disagree]` identity, the `xi(k)` condition, telescoping, and division by `(K-C) min delta_k`. There's an even handier sufficient condition: `xi(k)` holds if `v_k` is a good *approximation* of `g_k` in expected absolute value, `sum_i E|v_k[i] - g_k[i]| <= xi(k)`. That's because `P[sign(a) != sign(b)] <= P[|a-b| > |b|]`, and `|b| P[|a-b| > |b|] <= E|a-b|` by Markov — the gradient magnitude cancels just like before. So now I only have to show the normalized stochastic momentum `m~_k`, whose sign matches the recursive state, tracks the true gradient `g_k` in `l_1` expected absolute deviation, with a rate `xi(k) = O(1/sqrt k)`. And I get a free pass that ordinary approximation arguments don't: the bound need not hold for every `x_k`, only on average, so a few bad iterates with small probability don't sink it — which is exactly what lets a momentum-induced bias be tolerable.

To estimate `m~_k - g_k`, split it into a bias and a variance piece against the *deterministic-gradient* momentum `m_k := ((1-beta)/(1-beta^{k+1})) sum_{t=0}^k beta^t g_{k-t}` (normalized so its weights sum to one). The actual recursive state differs from this normalized version by a positive scalar, so the sign is the same, and coordinatewise

  E|m~_k[i] - g_k[i]| <= E|m_k[i] - g_k[i]|  (bias)  +  E|m~_k[i] - m_k[i]|  (variance).

The variance piece first. `m~_k - m_k = ((1-beta)/(1-beta^{k+1})) sum_t beta^t Z_{k-t}` where `Z_l := g~_l - g_l` is the noise. Naively the `Z`'s are *dependent* — `Z_l` depends on `x_l`, which depends on earlier `Z`'s — so I can't just add variances. But `E[Z_{l} | history] = 0` (unbiased oracle), which makes `sum_l alpha_l Z_l` a martingale, and for a martingale the cross terms `E[Z_l Z_j]` vanish: conditioning the later one on the past kills it. The second moment is bounded coordinate-wise by `E[(sum_l alpha_l Z_l)^2] <= sum_l alpha_l^2 sigma_l^2` — I get to treat the `Z`'s *as if* independent for the purpose of this upper bound, even though they aren't. Apply Jensen and that martingale variance bound, accounting for the growing batch (the `t`-step-ago gradient was averaged over `n_{k-t} = k-t+1` samples):

  E|m~_k[i] - m_k[i]| <= ((1-beta)/(1-beta^{k+1})) sqrt( sum_{t=0}^k beta^{2t} sigma_i^2/(k-t+1) ).

Bound that inner sum by splitting at `t = k/2`: for the recent half (`t <= k/2`) the denominator `k-t+1 >= k/2+1`, and `sum beta^{2t} <= 1/(1-beta^2)`, giving `<= sigma_i^2/((k/2+1)(1-beta^2))`; for the stale half (`t > k/2`) the weight `beta^{2t} <= beta^k` is tiny and there are at most `k/2` terms, giving `<= (k/2) beta^k sigma_i^2`, which for `k >= C` is negligible (this is what the warmup `C` is *for*). Together `<= 3 sigma_i^2/((k+1)(1-beta^2))`, so `E|m~_k[i] - m_k[i]| <= 2 sqrt3 sqrt(1-beta) sigma_i/sqrt(k+1)`, and summing over `i` gives the same bound with `||sigma||_1`. The variance term carries a `sqrt(1-beta)` — sending `beta -> 1` shrinks it, confirming momentum averages noise away.

The bias piece. Since the weights sum to one, `m_k - g_k = ((1-beta)/(1-beta^{k+1})) sum_t beta^t (g_{k-t} - g_k)`, so `E|m_k - g_k| <= 2(1-beta) sum_{t=1}^k beta^t E||g_{k-t} - g_k||_1` (for `k >= C`). I need how far apart two gradients along the trajectory are, and here's where curvature enters. Sub-lemma: under coordinate smoothness, for any sign vector `s` and any `eps <= delta`, `||g(x + eps s) - g(x)||_1 <= 2 eps ||L||_1`. Proof: Taylor gives `g(x + eps s) - g(x) = [int_0^1 H(x + t eps s) dt] eps s = H eps s` for the averaged Hessian `H`. Let `v = sign(g(x+eps s)-g(x))` and split `H = H_+ - H_-` into its psd and nsd parts. Then `||g(x+eps s)-g(x)||_1 = v^T H eps s = eps<H_+^{1/2}v, H_+^{1/2}s> - eps<H_-^{1/2}v, H_-^{1/2}s> <= eps||H_+^{1/2}v|| ||H_+^{1/2}s|| + eps||H_-^{1/2}v|| ||H_-^{1/2}s||` by Cauchy-Schwarz. Coordinate smoothness means `-diag(L) ≺ H ≺ diag(L)`, so both `H_+ ≺ diag(L)` and `H_- ≺ diag(L)`, hence `s^T H_± s <= sum_i L_i = ||L||_1` for any sign vector — and `v`, `s` are both sign vectors, so each of the four `||H_±^{1/2}(.)||^2 <= ||L||_1` and each product `<= ||L||_1`, total `<= 2 eps ||L||_1`. Now every algorithm step is a sign vector scaled by `delta_{k-l-1} <= delta/sqrt(k-l)`, so chaining the sub-lemma across the `t` steps between `k-t` and `k`:

  ||g_{k-t} - g_k||_1 <= sum_{l=0}^{t-1} ||g_{k-l} - g_{k-l-1}||_1 <= 2||L||_1 sum_{l=0}^{t-1} delta_{k-l-1} <= 2||L||_1 delta int_{k-t}^k dx/sqrt x = 4||L||_1 delta(sqrt k - sqrt(k-t)) <= 4||L||_1 delta t/sqrt k,

the last step from `1 - sqrt(1 - t/k) <= t/k`. Substitute into the bias sum and use the geometric-series derivative `sum_t t beta^t = beta/(1-beta)^2`:

  sum_i E|m_k[i] - g_k[i]| = E||m_k - g_k||_1 <= 2(1-beta) sum_t beta^t (4||L||_1 delta t/sqrt k) <= (8(1-beta)||L||_1 delta/sqrt k)(beta/(1-beta)^2) <= 16 ||L||_1 delta/sqrt(k+1) * beta/(1-beta).

The bias term carries `beta/(1-beta)` — sending `beta -> 1` *blows it up*. So the two pieces pull opposite ways exactly as the intuition promised. Combine bias and variance:

  sum_i E|m~_k[i] - g_k[i]| <= (2/sqrt(k+1))( 8 ||L||_1 delta * beta/(1-beta) + sqrt3 ||sigma||_1 sqrt(1-beta) ).

This is `xi(k) = O(1/sqrt k)` — feed it into the master lemma with `delta_k = delta/sqrt(k+1)`, `n_k = k+1` (so `N = O(K^2)`), and `K =O(sqrt N)`, square both sides, and Signum converges at

  E[(1/(K-C)) sum_{k=C}^{K-1} ||g_k||_1]^2 = O( (1/sqrt N)[ (f_C-f*)/delta + (1+log N)( delta ||L||_1/(1-beta) + ||sigma||_1 sqrt(1-beta) ) ]^2 ).

The momentum reads cleanly as a bias-variance knob: the `||sigma||_1 sqrt(1-beta)` (variance) term and the `delta||L||_1/(1-beta)` (bias) term are the two halves, and `beta` slides between them. The `C` is a warmup — the variance bound needed `k >= C` so the stale-momentum tail is negligible — and the precise condition is the smallest `C` with `(C/2)beta^C <= (1/(1-beta^2))(1/(C+1))` and `beta^{C+1} <= 1/2`, which for `beta = 0.9` is `C = 54`, trivially short. During warmup I keep accumulating momentum but just step with `sign(g~)` (plain signSGD), so no iterations are wasted; switching optimizers after a short warmup is something practitioners already do anyway.

One discretization detail to pin down before code: `sign(0)`. The original ICML reproduction code used a sign function that maps zero to `0`; the later majority-vote/compressor path I am matching for the wire format uses `tensor >= 0`, so zero maps to `+1`. The `>= 0` convention keeps every communicated coordinate binary; in majority vote it also means an exact tie in the vote sum is broadcast as `+1`.

Now let me land this in the codec slot. The compression map is the sign of the gradient, stored as `uint8` bits in this simple implementation; the side information for reconstruction is just the shape; decode maps the bits back to `±1`; and the server-side aggregate is the majority vote, `sign(sum of the per-worker sign vectors)`, with ties following the same `>= 0` convention. Signum is the same encoder with a momentum buffer interposed before the sign. The update direction the codec returns is `±1` per coordinate, and the training loop scales it by the learning rate — so `delta` is the entire step size, exactly the trust-region radius of the `l_inf` box that the non-stochastic version is doing steepest descent on.

```python
import torch


class SignSGDCodec:
    """signSGD / majority-vote codec. Each coordinate is transmitted as a single
    sign bit; the server combines worker bits by majority vote (1-bit each way)."""

    def __init__(self):
        self.state = {}

    def encode(self, grad, name):
        # sign compression: the wire message is one bit per coordinate.
        # sign(0) -> +1 via (grad >= 0), so every coordinate emits a binary code.
        shape = grad.shape
        bits = (grad.flatten() >= 0).to(torch.uint8)        # sign(g~) stored as uint8
        return [bits], shape                                 # message, ctx(=shape)

    def aggregate(self, messages):
        # majority vote: sign(sum_m sign(g~_m)). Decode each worker's bits to +-1,
        # sum the votes, re-sign -> a single bit broadcast back to every worker.
        votes = sum(b.to(torch.float32) * 2 - 1 for [b] in messages)
        return [(votes >= 0).to(torch.uint8)]               # 1-bit return message

    def decode(self, received, ctx):
        # bits -> +-1 update direction, reshaped to the parameter's shape.
        shape = ctx
        [bits] = received
        return (bits.to(torch.float32) * 2 - 1).view(shape)  # -delta * this = the step


class SignumCodec(SignSGDCodec):
    """Signum: sign of the momentum. Accumulate m = beta*m + (1-beta)*g~ per
    parameter, then take its sign. Momentum lowers the noise (sigma sqrt(1-beta))
    at the cost of curvature bias (delta||L||_1/(1-beta)); beta tunes the trade."""

    def __init__(self, momentum=0.9):
        super().__init__()
        self.momentum = momentum
        self.buf = {}

    def encode(self, grad, name):
        shape = grad.shape
        g = grad.flatten()
        beta = self.momentum
        m = self.buf.get(name)
        if m is None:
            m = torch.zeros_like(g)
        m = beta * m + (1 - beta) * g                        # m_{k+1}=beta*m_k+(1-beta)*g~_k
        self.buf[name] = m
        bits = (m >= 0).to(torch.uint8)                      # sign(m_{k+1})
        return [bits], shape
```

Equivalently, when there's no parameter server and Signum is just a local optimizer, it's the same rule written as an update: form the rescaled gradient, keep a momentum state, take its sign, optionally apply the decoupled MXNet-style `wd_lh` shrink, then step by `delta`. Here I mirror the original MXNet/local-optimizer convention for exact zeros, so `sign(0)` produces a zero step.

```python
import torch


class Signum:
    """Local Signum optimizer: state = beta*state + (1-beta)*rescaled_grad.
    beta = 0 recovers signSGD. torch.sign mirrors the original ICML code:
    sign(0) produces a zero step."""

    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0,
                 decoupled_weight_decay=0.0):
        self.params = list(params)
        self.lr, self.momentum, self.wd = lr, momentum, weight_decay
        self.wd_lh = decoupled_weight_decay
        self.state = {id(p): None for p in self.params}

    @torch.no_grad()
    def step(self):
        beta = self.momentum
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            if self.wd != 0:
                g = g.add(p, alpha=self.wd)                  # rescaled_grad includes wd*weight
            if beta != 0:
                m = self.state[id(p)]
                if m is None:
                    m = torch.zeros_like(g)
                m.mul_(beta).add_(g, alpha=1 - beta)         # state=beta*state+(1-beta)*rescaled_grad
                self.state[id(p)] = m
                direction = torch.sign(m)
            else:
                direction = torch.sign(g)
            if self.wd_lh != 0:
                p.mul_(1 - self.lr * self.wd_lh)             # decoupled MXNet Signum shrink
            p.add_(direction, alpha=-self.lr)                # x <- x - delta*sign(...)
```

Let me trace the causal chain back. Communication, not computation, is the bottleneck in distributed training, so the goal was to say one bit per coordinate in both directions while keeping an SGD-class non-convex rate. The crudest one-bit message is the sign of the gradient, and the field's principled compressors avoided the *biased* sign by randomizing to unbias — which inflates variance by `sqrt(d)` and makes their guarantees vacuous at deep-learning scale. So I went the other way: keep the biased sign, and confront the bias. A flipped sign with nonzero true gradient needs the noise to beat the signal, so `P[wrong] <= sigma_i/|g_i|`, and the magnitude cancels — the damage of bias is capped purely by the noise scale `sigma_i`, independent of the gradient size. With coordinate-wise smoothness (the natural majorization for a box-shaped sign step), a one-step descent bound plus a growing batch and `delta ~ 1/sqrt(||L||_1 K)` telescoped to the `1/sqrt N` rate in `l_1` — and translating through the density `phi(v) = ||v||_1^2/(d||v||_2^2)` gave the ratios `R_1 = sqrt(phi(L))/phi(g)` and `R_2 = phi(sigma)/phi(g)`, which a Welford measurement puts in the favorable dense-gradient, dense-noise regime for real networks. Under unimodal-symmetric noise, Gauss's inequality sharpened the flip probability and gave a small-batch rate with a mixed `l_1`/variance-weighted-`l_2` norm. The return trip was compressed by majority vote, `sign[sum_m sign(g~_m)]`: a sign that's right more than half the time turns the workers into a repetition code, and under symmetry the binomial tail gives the full `sqrt M` variance reduction of distributed SGD with one bit each way. Folding momentum *inside* the sign gave Signum, where a martingale variance bound and a Hessian-splitting bias bound made `beta` an explicit bias-variance knob, all of it riding a single master lemma that only asks how often the signed quantity agrees with the true gradient. And the whole thing drops into the parameter-server codec as: encode = sign bit, aggregate = majority vote with the implementation's `>= 0` tie convention, decode = `±1` — one bit, both directions.
