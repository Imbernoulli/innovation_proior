OK. The style-based generator works embarrassingly well — faces at a megapixel that I can't tell from real — and yet there's this thing. Almost every image it makes has a blob in it. A little water-droplet smear. Sometimes I can't see it in the final RGB, but the moment I dump the intermediate feature maps it's right there, in *all* of them, like a watermark. Let me just stare at it for a while before I touch anything, because the existence of this artifact is the most interesting fact I have.

What do I actually know about it? It starts showing up around 64×64 resolution and it gets stronger the higher I go. It's in essentially every feature map at every layer past that point. And here's the part that bothers me most: it's *consistent*. The same stereotyped blob, image after image. The discriminator is a convolutional classifier whose entire job is to find anything that distinguishes fake from real. A consistent, repeated, easily-detectable blob is exactly the kind of thing a discriminator should crush. It doesn't. So either the discriminator can't see it — unlikely, it's a glaring feature — or the generator is producing it *on purpose* and the discriminator has decided the images are better with it than without it. The generator wants the blob.

And there's a tell. In maybe one image in a thousand the blob is missing, and when it's missing the image is wrecked — not slightly worse, severely corrupted. So the network doesn't merely tolerate the blob. It *relies* on it. Whatever the blob does, the rest of the generation is built on top of it, and pull it out and everything collapses.

So what could a single dominant blob *do* for the generator? It has to be doing some job that the generator can't do any other way given its architecture. Let me walk the architecture and ask where a network would be tempted to smuggle something.

The style mechanism is adaptive instance normalization. For each feature map in a layer, AdaIN takes the map, subtracts its mean and divides by its standard deviation — computed per map, per sample — and then re-scales and re-biases it by the current style: roughly `s_scale · (x_i − μ(x_i))/σ(x_i) + s_bias`. The point of that normalization is to make the *style* the thing that decides each map's magnitude; that's what lets me feed different styles to different layers and get clean, scale-specific control, the whole reason this generator is special.

But look at what the normalization throws away. It normalizes each feature map *separately*. After AdaIN, every map has been forced to unit standard deviation independently. So the *relative* magnitudes between maps — "this map should be twice as loud as that one" — are gone. The generator literally cannot express, through the activations alone, that one map matters more than another, because each one gets individually rescaled to the same statistics.

Now suppose the generator *does* need to express relative magnitudes across maps — and why wouldn't it; that's information. How could it get that past a per-map normalization that's about to wipe it? The thing I keep circling back to is that σ(x_i) is a single number summarizing the whole map, and it's dominated by the map's largest values. So if the generator plants one giant spike in a corner — one localized value far bigger than everything else — then σ(x_i) is essentially just *the size of that spike*, and when AdaIN divides the whole map by σ, it divides everything else by the spike too. Let me actually put numbers on that, because the argument only matters if the suppression is large. Take a small 8×8 map of roughly unit-variance content; its std is about 0.92. Now plant a single value of 50 in one cell. The std of the map jumps to about 6.3 — the spike has hijacked it almost entirely. After AdaIN subtracts the mean and divides by std, the *non-spike* content, which had a normalized std near 0.98 in the clean map, comes out at a normalized std of about 0.14 in the spiked map. The rest of the picture got quieter by a factor of ~6.8, and that factor is set by how big I made the spike. So the spike is a knob: by choosing its size the generator chooses the post-normalization loudness of the entire rest of the map. It has manufactured a private channel for "how loud is this map," and it pays for it with one big localized value.

So that localized value would *be* the blob — a spike the generator uses to set per-map scale by dominating its own statistics, smuggling signal-strength information *through* the very normalization meant to remove it. That story explains the things that bothered me: why it's in every map (every layer needs the scale channel), why it's stereotyped (it's the same trick), why the discriminator tolerates it (it's load-bearing), and why deleting it wrecks the image (you've cut the scale channel). The story also makes a falsifiable prediction I should hold onto: the spike only has a job *because* there's a per-map normalization to game. Remove the normalization and the spike should have no reason to exist — I'd want to confirm the blob actually vanishes once normalization is gone, not just assume it.

So: delete instance normalization, done? I wish. If I remove the normalization, I also remove the thing that makes the style scale dominate each map — and that's exactly what makes style mixing work. Styles in this generator can amplify a feature map by an order of magnitude or more. Without a per-sample normalization counteracting that amplification, when I feed a different w to a later layer the activations coming in are at wildly inconsistent magnitudes and the layer can't operate on them meaningfully. I'd kill the artifact and the FID would even tick up a bit, but I'd lose the scale-specific control that's the entire point. So removing normalization is a real option only if I'm willing to give up controllability, and I'm not.

I want the *effect* of the normalization — keep each map at a predictable scale so styles behave and mixing works — without the *mechanism* that the generator can game. What makes the mechanism gameable is that it's data-dependent: it computes σ from the *actual contents* of the map, so a spike in the contents can hijack it. What if I never look at the contents at all?

Before I chase that, let me clean up a couple of things in the block that are muddying my thinking, because they'll matter. Right now the bias and the per-pixel noise are added *inside* the style block, i.e. before/around the normalization-and-modulation. That means their absolute effect depends on the current style's magnitude — if the style is currently scaling this map way up, a fixed bias contributes proportionally less; if it's scaling down, the bias dominates. So the same bias does different things depending on the style. That's unpredictable. If I move bias and noise *outside* the block, so they act on already-normalized data, their effect becomes predictable and style-independent. And once I've done that, I notice I don't even need to normalize and modulate the mean anymore — operating on the standard deviation alone is enough. I can also drop the bias, noise, and normalization on the learned constant 4×4 input entirely; no observable harm. Good. Now the block is just: modulate, convolve, normalize-by-std. Three clean operations. That's the thing I want to redesign.

Now, the idea: base the normalization on the *expected* statistics of the signal, computed in closed form from the weights, instead of measuring the signal. If I can predict what the standard deviation of each output map *will be* given the style and the weights — under a mild assumption about the input — then I can divide by that predicted number. It's deterministic. There's no per-map content statistic to spike. The generator can't game an arithmetic constant.

Let me actually derive it. Start with the modulation. Modulation scales input feature map i by the style scalar s_i. I can push that scaling into the convolution weights instead of applying it to the activations: if the original weight connecting input map i to output map j with kernel tap k is w_{ijk}, then scaling input i by s_i is the same as using
```
w'_{ijk} = s_i · w_{ijk}.
```
Same output, but now the style lives in the weights.

Next the normalization. Its job was to undo the effect of s on the output statistics. Let me compute that effect directly. Assume the input activations are i.i.d. random variables with unit standard deviation — that's the same kind of assumption the He/Glorot initializers make when they pick weight scales to preserve variance, so it's not exotic. The output map j is `o_j = Σ_{i,k} w'_{ijk} · x_{ik}`. The x's are independent, mean-ish-zero, unit variance, so the variance of a sum of independent scaled terms is the sum of the variances:
```
Var(o_j) = Σ_{i,k} (w'_{ijk})² · Var(x_{ik}) = Σ_{i,k} (w'_{ijk})².
```
So the output standard deviation is
```
σ_j = sqrt( Σ_{i,k} (w'_{ijk})² ),
```
the L2 norm of the modulated weights feeding output map j. The modulation scaled the output up by exactly this much. To restore unit standard deviation I divide output map j by σ_j — I'll call this "demodulation." And just like the modulation, I can bake it into the weights rather than applying it to the activations:
```
w''_{ijk} = w'_{ijk} / sqrt( Σ_{i,k} (w'_{ijk})² + ε ).
```
The ε is just to keep me from dividing by zero when a map's weights are tiny. That's it. The entire style block — modulate, convolve, normalize — has collapsed into a single convolution whose weights I adjust with those two formulas before I run it.

I shouldn't trust a variance argument I haven't actually checked, so let me run the smallest version of it by hand. Take three input maps and two output maps, 1×1 kernel for now. Pick weights w = [[0.5,−1.0],[2.0,0.3],[−0.4,0.8]] and a style s = [1.5, 0.2, 3.0]. Modulate: w'_{ij} = s_i w_{ij}, which gives the first column [0.75, 0.4, −1.2] and the second [−1.5, 0.06, 2.4]. My formula predicts the output stds are the column L2 norms: σ₁ = √(0.75²+0.4²+1.2²) ≈ 1.471 and σ₂ = √(1.5²+0.06²+2.4²) ≈ 2.831. Now feed in two million i.i.d. unit-variance inputs and measure: the empirical output stds come out 1.470 and 2.829 — matching the predicted σ to three digits, so the closed-form really does capture the output std the modulation produces. Dividing each column by its σ and re-measuring gives empirical output stds of 1.000 and 0.9995. So demodulation does restore unit output std, and it does it from a divisor computed purely from s and w — it never touched the activations.

That last point is the crux, so let me be precise about how this differs from the instance norm I'm replacing. Instance norm divides by the *measured* std of the *actual* output, contents and all. Demodulation divides by the *expected* std predicted from the weights. It's strictly weaker: it doesn't force each realized map to exactly unit std (the numbers above only matched because I averaged over millions of samples), it only normalizes in expectation. But "weaker" is exactly what I want here, because the failure mode I'm trying to kill was *strength of a different kind* — a per-realization statistic the generator could spike. Here there is no realized statistic in the divisor for a spike to dominate; the divisor is a fixed function of s and w, the same for every spatial pattern the map could hold. So whatever the blob was buying — a way to set scale by hijacking σ — it buys nothing now. And the style scaling s is still sitting right there inside w', untouched, so the scale-specific control and style mixing are preserved. The trade-off I thought I was stuck with a moment ago — kill the blob *or* keep controllability — wasn't real once I stopped measuring the signal.

A couple of consequences I should respect. The unit-variance-input assumption has to actually hold as signal flows through, or my divisor is wrong by a layer-dependent factor that compounds. The nonlinearity changes variance — a leaky ReLU passes the positives and attenuates the negatives, so it doesn't preserve unit variance by default. Fix: scale the activation function so it retains the expected signal variance (for leaky-ReLU with slope 0.2 that's a gain of about √2 on the output). Then unit-variance-in stays roughly unit-variance-in at the next layer and the demodulation stays calibrated. Also the output-to-RGB layers: those produce a 3-channel image, not unit-variance features feeding another conv, so I should still modulate them (the style should affect color) but *not* demodulate them — there's no downstream variance to protect.

And a numerical reality: because s is per-sample, `w''` is *different for every sample in the minibatch*. I can't feed per-sample weights to a standard convolution that shares one weight tensor across the batch. The trick is grouped convolution — originally a way to split channels into independent groups each with their own weights. I reshape so that the convolution sees a *single* example with N groups instead of N examples with one group; then each "group" carries its own per-sample demodulated weights, and the reshape is just a view, it doesn't copy data. Efficient.

Where does this leave me? The arithmetic is the same flavor as weight normalization, which reparameterizes a weight as direction-times-magnitude and so also divides by the weight's L2 norm — except I'm using that division as a *replacement for a data-dependent normalization layer*, not as a reparameterization trick. The two checks I did say what I needed: the divisor restores unit output std (so styles still see well-scaled inputs and mixing should keep working), and it's computed only from s and w (so the spike has nothing left to dominate). What I can't settle on paper is whether the realized blob actually disappears in a trained net and what it does to FID — that's an empirical question. My expectation is that FID barely moves and that the precision/recall balance shifts toward recall, which would be fine, since recall can be traded back to precision with truncation but not vice versa — but I'd hold all of that as a prediction to test, not a result.

That's one artifact family. Now the other thing nagging me is quality measurement, because I can't optimize what I can't see, and FID has a blind spot. I keep finding pairs of generators with the *same* FID and the *same* precision/recall where one clearly looks better. Why would the metric be blind? FID and P&R both live in the feature space of an ImageNet-trained classifier, and ImageNet classifiers have been shown to decide mostly on *texture* while humans key on *shape*. So two generators with equally convincing textures can score identically even if one has worse shapes. The metric isn't wrong, its feature space just isn't watching the thing I care about.

What *does* track the quality difference? Perceptual path length. PPL was built to measure how smooth the latent→image map is: take a w, take a tiny step, measure the LPIPS perceptual distance between the two images, average. And empirically, lower PPL — a smoother map — goes with images that look better, exactly where FID is blind. Let me try to understand *why* smoothness would correlate with quality rather than just note the correlation. Picture training. The discriminator punishes broken images. The cheapest way for the generator to reduce its punishment isn't necessarily to make every region of latent space good; it's to *shrink* the latent regions that map to bad images and *stretch* the ones that map to good images. Stretch the good, squeeze the bad. In the short term average quality goes up. But squeezing the bad images into thin slivers of latent space means the map changes extremely fast there — huge image change per unit latent step — which is exactly high PPL, a badly conditioned mapping. Those regions of violent change distort the training dynamics and drag down the final quality. So a generator that's been encouraged to keep its mapping smooth has fewer of these pathological regions, and that's the mechanism behind "low PPL ↔ good images."

So I'd like to *regularize* for smoothness. But I have to be careful what "smooth" means, because the dumb version backfires: if I just minimize PPL — minimize how much the image moves when w moves — the optimum is a generator that ignores w and outputs the same image always. Zero PPL, zero recall, useless. I don't want "small image change." I want *uniform* image change: a fixed-size step in W should produce a fixed-magnitude change in the image, the same magnitude regardless of *where* in W I am and *which* direction I step. Not small — *constant*. That's a conditioning statement about the map, not a magnitude statement.

Let me formalize. At a point w, the local linearization of g: W→image is the Jacobian J_w = ∂g/∂w. A step δ in latent space produces an image change J_w δ. I want the *length* of that change to be the same for any unit-length direction. Now, rather than probing latent directions, it's cleaner to probe *image* directions: take a random image-space vector y and look at J_wᵀ y. "Image changes of equal length in every direction" is a statement about the singular values of J_w being equal, and J_wᵀ y is the natural quantity to measure that with a random y. So I want ‖J_wᵀ y‖₂ to be close to some constant a, for random y, regardless of w. That gives the regularizer
```
E_{w, y∼N(0,I)} ( ‖J_wᵀ y‖₂ − a )².
```
y is a random image with normal pixel intensities, w∼f(z). Penalize the deviation of the gradient length from a fixed value a.

Two things to nail down: how to compute J_wᵀ y without forming the Jacobian, and what a should be — and then I should check this prior actually does what I claimed.

Computation first. J_wᵀ y is a vector-Jacobian product, and that's exactly what backprop computes: the gradient with respect to w of the scalar g(w)·y is Σ_k (∂g_k/∂w) y_k = J_wᵀ y. Let me make sure I'm not fooling myself with index gymnastics by checking it on a toy map. Take g(w) = tanh(Aw + b) with a 4×2 matrix A, so W is 2-dimensional and the "image" is 4-dimensional. Its Jacobian is J_{kl} = (1 − tanh²(Aw+b)_k)·A_{kl}. At w = (0.3, −0.7) with a random image direction y, forming J explicitly and computing J_wᵀ y gives (−0.650, −0.352). Now compute ∇_w(g(w)·y) the dumb way, by central finite differences on the scalar g(w)·y: (−0.650, −0.352). They agree to five digits. So the identity holds — I get the whole vector-Jacobian product from one dot product and one backward pass, never forming J. (This is the analytic version of the finite-difference Jacobian-vector products in the Jacobian-clamping line of work, which probe J δ numerically; doing it analytically through the dot-product identity is cheaper and exact.)

Now a. In principle a is just a global scale — multiply all the singular values by a constant and the conditioning is unchanged, so I could fix a=1 and forget it. But in practice it matters during training. If I pin a to a value that doesn't match the scale the network happens to have at initialization, then the first, most critical training steps get spent dragging all the weight magnitudes toward my arbitrary a instead of doing the actual job of equalizing the spectrum — and that wastes the early dynamics and leaves the network in a worse internal state. So I shouldn't impose a scale; I should *adopt whatever scale already exists*. Set a to a long-running exponential moving average of the observed lengths ‖J_wᵀ y‖₂ themselves. Then the regularizer says "push every local Jacobian toward the *current global average* length," which equalizes conditioning without fighting the network over the absolute scale, and it removes the need to ever measure the right scale by hand. I'll initialize a to zero and let the EMA (decay around 0.99) find it.

Does the prior do what I want? I conjectured that "uniform image change in every direction" means equal singular values, but I'd better derive what this particular penalty is actually minimized by rather than assume it lines up with my intuition. Fix a w and look at the inner expectation over y:
```
L_w = E_y ( ‖J_wᵀ y‖₂ − a )².
```
Write the SVD J_wᵀ = U Σ̃ Vᵀ, where U is L×L orthogonal, V is M×M orthogonal (M = image dimension, L = latent dimension, M ≥ L), and Σ̃ = [Σ | 0] is an L×L diagonal of singular values padded with zeros. Two facts about the standard normal and the L2 norm: rotating a standard-normal vector by an orthogonal matrix leaves its distribution unchanged, and rotating any vector leaves its norm unchanged. So the leading U just rotates the vector inside the norm — drop it. And Vᵀ acting on y∼N(0,I) gives another N(0,I) — relabel it y. So
```
L_w = E_y ( ‖Σ̃ y‖₂ − a )².
```
The zero block of Σ̃ kills the components of y beyond the first L; marginalizing a standard normal over some coordinates leaves a standard normal on the rest, so I'm left with ỹ∼N(0,I) in R^L and
```
L_w = E_{ỹ} ( ‖Σ ỹ‖₂ − a )²,
```
which depends on J_w only through its singular values. (So every Jacobian with the same spectrum gives the same loss — the U and V really don't matter.)

Now minimize over the diagonal Σ. Write it as an integral against the standard-normal density and switch to polar coordinates ỹ = r φ, r the radius, φ a unit vector on the sphere S^{L-1}; the volume element picks up the Jacobian factor r^{L-1}:
```
L_w ∝ ∫_{S} ∫_0^∞ ( r ‖Σ φ‖₂ − a )² · r^{L-1} exp(−r²/2) dr dφ.
```
The radial part (2π)^{−L/2} r^{L-1} exp(−r²/2) is the standard-normal density written in radius. In high L this radial density is sharply concentrated: a Taylor argument turns it into a one-dimensional Gaussian in r centered at μ = √L with standard deviation σ = 1/√2. That's the well-known fact that an L-dimensional standard normal lives on a thin shell of radius √L. So almost all the mass is at r ≈ √L, and
```
L_w ≈ ∫_{S} ∫_0^∞ ( r ‖Σ φ‖₂ − a )² exp(−(r−√L)²/(2σ²)) dr dφ.
```
To minimize, I want the parabola ( r ‖Σ φ‖₂ − a )² to bottom out right where the mass is, at r = √L, and to do so for *every* direction φ. Its zero is at r = a / ‖Σ φ‖₂. Setting that equal to √L for all φ requires ‖Σ φ‖₂ to be the *same constant* for every unit φ — which forces Σ = (a/√L) I, all singular values equal. With that choice the integrand's zero coincides with the peak of the density in every direction simultaneously, minimizing the inner integral. So the minimizer is Σ ∝ I: equal singular values, i.e. J_w orthogonal up to a global scale, at every w — which does match the intuition I started from, though I needed the thin-shell argument to be sure the penalty actually rewards it.

Let me put one number on that, because the argument rode on a high-dimensional concentration fact and I want to see it bite at finite L. Take L = 64 and a = √L (the common-length the equal case predicts). Compare two spectra of *equal total energy* (same mean-square singular value): the flat one, all σ = 1, against a random uneven one rescaled to the same mean-square. Estimating L_w = E_y(‖Σy‖−a)² by Monte Carlo, the flat spectrum scores ≈0.50 and the uneven one ≈1.37. So at matched energy the equal-singular-value Jacobian really is the lower-loss configuration, by a wide margin — the penalty pays for *equalizing* the spectrum, not merely for shrinking or capping it. (And a really was only the global scale: it sets the common singular value a/√L and nothing else, which is why pinning it is harmless in theory and only matters for training dynamics.)

What does an everywhere-orthogonal-up-to-scale Jacobian *mean* for the map? Take any curve u(t) in latent space and push it through g to get a curve ũ = g∘u in image space. Its length is ∫ |ũ′(t)| dt = ∫ |J_g(u(t)) u′(t)| dt by the chain rule; if J_g is orthogonal it preserves the norm of u′(t), so the integral is just ∫ |u′(t)| dt — the length of the original latent curve. The map preserves curve lengths: it's a local isometry, embedding the flat latent space isometrically as the image manifold. The practical payoff: straight lines in W map to geodesics (shortest paths) on the image manifold, so a latent interpolation between two images is the minimal-change morph between them and can't take wild detours through distant images. Smoother interpolations, and — the part I care about operationally — a far better conditioned map, which should make *inverting* the generator (finding the w for a given image) reliable, because a well-conditioned forward map is one whose inverse optimization isn't fighting violent local stretching. I won't reach exact isometry in practice — it conflicts with the adversarial objective and the manifold may not even admit one — but a *pressure* toward it is enough to kill the detours.

One more thing to settle: spectral normalization is sitting right there and also touches singular values, so why not just use it? Because spectral norm constrains only the *largest* singular value — it caps the top of the spectrum and says nothing about the rest. My whole argument was about making *all* the singular values equal. Capping the max doesn't equalize anything; it can leave the map just as badly conditioned. So it's not a substitute. (In fact, with weight demodulation already dividing every conv weight by its norm, a per-layer spectral scaling is largely overridden anyway.)

Now I have two regularizers in play — the discriminator's R1 gradient penalty and this new path-length term — and they're both not cheap, especially mine, which needs an extra backward pass through g per step. Do I really have to evaluate them every iteration? They change slowly compared to the main adversarial loss. So compute them lazily: run the regularization in a separate pass only once every k steps. Empirically R1 every 16 minibatches does no harm; I'll do my path-length term every 8. This slashes the regularization compute and memory.

If I step the regularizer only every k iterations, I have to keep the optimizer honest. I'm sharing Adam's state between the main loss and the regularizer, and now over a window I do k main steps plus 1 regularizer step instead of k steps — so the effective hyperparameters drift. Correct for it with c = k/(k+1): scale the learning rate λ′ = cλ and the Adam momenta β₁′ = β₁^c, β₂′ = β₂^c, and multiply the regularization term by k so its accumulated gradient magnitude matches what it would have been applied every step. (And I can compute the path-length term on a *fraction* of the minibatch — half — to save memory, since it's only a regularizer.)

I should also fix the weighting of the path-length term so it's resolution-independent. The raw term has two implicit normalizations baked in: I divide the random image y by √(number of pixels) = √(r²), and I average ‖J_wᵀ y‖ over the affine layers, of which there are (2·log₂ r − 2). So an overall weight of 2 turns into an effective γ_pl = 2 / r² / (2·log₂ r − 2) = 1 / (r² (log₂ r − 1)). Converting the log base — log₂ r − 1 = (ln r − ln 2)/ln 2 — gives
```
γ_pl = ln 2 / ( r² (ln r − ln 2) ),
```
which I can just compute from the output resolution r and forget about. Good.

That's both artifact families fixed at the layer level. Now the *progressive-growing* family — the details that stick to fixed pixel positions and then jump, the busted shift invariance. I traced that to progressive growing: because each resolution serves momentarily as the network's *output* resolution while it's being faded in, it gets pushed to synthesize maximal-frequency detail at that resolution, which leaves the intermediate layers carrying excessively high frequencies and breaks shift invariance, producing the location preference. I want to keep what progressive growing is *good* at — the coarse-to-fine training schedule, where the network nails low-resolution structure first and only later commits to fine detail, which is what gives the stability — without ever changing the network topology during training, so no resolution is ever transiently "the output."

The toolbox for combining resolutions without growing is well stocked: skip connections, residual blocks, hierarchical pyramids. The MSG-GAN idea is the seed — connect matching resolutions of G and D with multiple skips, the generator emitting a stack of resolutions. Let me adapt it. In the generator, instead of one final to-RGB at the top, give *every* resolution block a to-RGB, upsample each lower-resolution RGB and *sum* them into a running image. Now the final image is an explicit sum of contributions from all resolutions; nothing is ever the lone output, every resolution contributes simultaneously and continuously. In the discriminator, mirror it: feed a downsampled copy of the input image into each resolution block. That's the "skip" version. I can also try a *residual* version in each, where each block adds a skip path through a 1×1 conv.

Which combination? Sweep the three generator styles (plain feedforward, skip, residual) against the three discriminator styles. Two clean trends fall out. Skip connections in the *generator* dramatically improve PPL across the board — the smoothness I was just fighting for, now partly architectural. A residual *discriminator* clearly helps FID — no surprise, the discriminator is basically a classifier and residual classifiers are known to be strong. But a residual *generator* was harmful (one LSUN-Car exception). So: **skip generator, residual discriminator**, no progressive growing. That's the configuration I'll carry forward, and it improves both FID and PPL.

A variance bookkeeping note for the residual blocks: adding two paths doubles the signal variance. In a classification ResNet batch norm quietly absorbs that; I have no batch norm here, so I have to cancel it explicitly — multiply the merged path by 1/√2. With unit-variance signals that's exactly the factor that restores unit variance after summing two independent unit-variance paths. It's small but it's load-bearing for keeping my demodulation calibration valid.

One payoff of the skip generator is that I can now *measure* coarse-to-fine behavior, because the final image is a literal sum of per-resolution RGB contributions. So I take the standard deviation of the pixel values each to-RGB layer contributes, over many random w, and normalize the contributions to sum to 100%. Early in training the low resolutions dominate — good, that's the coarse-to-fine schedule reproduced without growing. But late in training I expect the highest resolution to take over, and it doesn't: the top resolution stays under-weighted. Looking at the images by eye confirms it — they read like sharpened 512² images, not true 1024², missing the finest pixel-level detail. The mapping is leaving capacity on the table at the top. So I double the number of feature maps in the high-resolution layers (64² up to 1024²); the high-resolution contribution jumps to where I'd expect, and FID and recall improve markedly. That capacity bump is the last piece.

Let me also note where this smoother map pays off concretely: inversion. To invert, I fix a target image and optimize a single w (in the unextended W, so the recovered code is something the generator could really have produced — not a per-layer W+ that can match arbitrary images that have no honest latent) plus the per-layer noise maps, against an LPIPS image loss. I add ramped-down Gaussian noise to w during the search to explore, and I regularize the noise maps so they can't smuggle real signal — enforce that their one-pixel autocorrelations vanish at multiple scales, like genuine white noise, and renormalize them to zero mean and unit variance each step. With the path-length-regularized, well-conditioned generator this optimization actually converges to a faithful match, where with the un-regularized one it doesn't reliably — which is the direct fruit of the orthogonal-Jacobian pressure I derived above.

Let me put the load-bearing pieces into code, grounded in how the style block, the synthesis stack, the path-length term, and lazy regularization actually run.

```python
import numpy as np
import tensorflow as tf

# --- The redesigned style block: modulation + (analytic) demodulation, one conv ---
# Derivation: w'_ijk = s_i * w_ijk  (fold the style scale into the weights),
#             sigma_j = sqrt(sum_{i,k} w'_ijk^2)  (output std under unit-variance inputs),
#             w''_ijk = w'_ijk / sqrt(sum_{i,k} w'_ijk^2 + eps)  (divide it back out).
def modulated_conv2d_layer(x, w_latent, fmaps, kernel, up=False, demodulate=True,
                           resample_kernel=None, fused_modconv=True):
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps])
    ww = w[np.newaxis]                                   # [B,k,k,I,O]

    # Modulate: style s = affine(w_latent), bias initialized to 1 so s starts as a no-op.
    s = dense_layer(w_latent, fmaps=x.shape[1].value, weight_var='mod_weight')
    s = apply_bias_act(s, bias_var='mod_bias') + 1       # [B,I]
    ww *= tf.cast(s[:, np.newaxis, np.newaxis, :, np.newaxis], w.dtype)   # w'_ijk = s_i w_ijk

    # Demodulate: divide each output map by the predicted output std (the L2 norm of its weights).
    if demodulate:
        d = tf.rsqrt(tf.reduce_sum(tf.square(ww), axis=[1, 2, 3]) + 1e-8)  # [B,O]; sum over k,k,I
        ww *= d[:, np.newaxis, np.newaxis, np.newaxis, :]                  # w''_ijk = w'_ijk / sigma_j

    # Per-sample weights -> grouped convolution (one image with B groups). Reshapes are views.
    if fused_modconv:
        x = tf.reshape(x, [1, -1, x.shape[2], x.shape[3]])
        w = tf.reshape(tf.transpose(ww, [1, 2, 3, 0, 4]),
                       [ww.shape[1], ww.shape[2], ww.shape[3], -1])
    else:
        x *= tf.cast(s[:, :, np.newaxis, np.newaxis], x.dtype)            # equivalent: scale inputs

    if up:
        x = upsample_conv_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=resample_kernel)
    else:
        x = tf.nn.conv2d(x, tf.cast(w, x.dtype), data_format='NCHW',
                         strides=[1, 1, 1, 1], padding='SAME')

    if fused_modconv:
        x = tf.reshape(x, [-1, fmaps, x.shape[2], x.shape[3]])
    elif demodulate:
        x *= tf.cast(d[:, :, np.newaxis, np.newaxis], x.dtype)
    return x

# --- Synthesis: skip generator (per-resolution to-RGB summed), no progressive growing ---
def G_synthesis(dlatents_in, resolution=1024, num_channels=3,
                architecture='skip', resample_kernel=[1, 3, 3, 1]):
    res_log2 = int(np.log2(resolution))
    nf = lambda stage: int(np.clip(16 << 10 >> stage, 1, 512))   # double high-res maps -> capacity fix

    def layer(x, layer_idx, fmaps, kernel, up=False):     # modulated conv + bias + noise (outside block)
        x = modulated_conv2d_layer(x, dlatents_in[:, layer_idx], fmaps, kernel,
                                   up=up, resample_kernel=resample_kernel)
        noise = tf.random_normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3]], dtype=x.dtype)
        x += noise * tf.get_variable('noise_strength', shape=[],
                                     initializer=tf.initializers.zeros())
        return apply_bias_act(x, act='lrelu')             # lrelu gain set to preserve unit variance

    def torgb(x, y, res):                                 # modulate, do NOT demodulate; sum into running image
        t = apply_bias_act(modulated_conv2d_layer(x, dlatents_in[:, res*2-3],
                            fmaps=num_channels, kernel=1, demodulate=False))
        return t if y is None else y + t

    def block(x, res):
        x = layer(x, res*2-5, nf(res-1), 3, up=True)
        x = layer(x, res*2-4, nf(res-1), 3)
        return x

    # constant 4x4 input (no bias/noise/norm on it), then summed multi-resolution RGB.
    x = tf.tile(tf.cast(tf.get_variable('const', shape=[1, nf(1), 4, 4],
                initializer=tf.initializers.random_normal()), dlatents_in.dtype),
                [tf.shape(dlatents_in)[0], 1, 1, 1])
    x = layer(x, 0, nf(1), 3)
    y = torgb(x, None, 2)
    for res in range(3, res_log2 + 1):
        x = block(x, res)
        y = upsample_2d(y, k=resample_kernel)             # upsample running image, then add this res
        y = torgb(x, y, res)
    return tf.identity(y, name='images_out')

# --- Discriminator block: residual (1/sqrt(2) cancels the variance doubling of the merge) ---
def D_block(x, res, resample_kernel=[1, 3, 3, 1]):
    nf = lambda stage: int(np.clip(16 << 10 >> stage, 1, 512))
    t = x
    x = apply_bias_act(conv2d_layer(x, fmaps=nf(res-1), kernel=3), act='lrelu')
    x = apply_bias_act(conv2d_layer(x, fmaps=nf(res-2), kernel=3, down=True,
                       resample_kernel=resample_kernel), act='lrelu')
    t = conv2d_layer(t, fmaps=nf(res-2), kernel=1, down=True, resample_kernel=resample_kernel)
    return (x + t) * (1 / np.sqrt(2))

# --- Non-saturating G loss + path-length regularizer (J^T y via one backprop, dynamic target a) ---
def G_logistic_ns_pathreg(G, D, training_set, minibatch_size,
                          pl_minibatch_shrink=2, pl_decay=0.01, pl_weight=2.0):
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels  = training_set.get_random_labels_tf(minibatch_size)
    fake, dlat = G.get_output_for(latents, labels, is_training=True, return_dlatents=True)
    loss = tf.nn.softplus(-D.get_output_for(fake, labels, is_training=True))

    # Regularizer on a fraction of the minibatch (memory).
    pl_n = minibatch_size // pl_minibatch_shrink
    fake, dlat = G.get_output_for(tf.random_normal([pl_n] + G.input_shapes[0][1:]),
                                  training_set.get_random_labels_tf(pl_n),
                                  is_training=True, return_dlatents=True)
    # y ~ N(0,I)/sqrt(num_pixels);  J^T y = grad_w (g(w) . y)  -> one backward pass.
    y = tf.random_normal(tf.shape(fake)) / np.sqrt(np.prod(G.output_shape[2:]))
    pl_grads = tf.gradients(tf.reduce_sum(fake * y), [dlat])[0]
    pl_lengths = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(pl_grads), axis=2), axis=1))  # ||J^T y||

    # a = EMA of the observed lengths (adopt the existing scale instead of imposing one).
    pl_mean_var = tf.Variable(0.0, trainable=False, name='pl_mean')
    pl_mean = pl_mean_var + pl_decay * (tf.reduce_mean(pl_lengths) - pl_mean_var)
    with tf.control_dependencies([tf.assign(pl_mean_var, pl_mean)]):
        reg = tf.square(pl_lengths - pl_mean) * pl_weight    # (||J^T y|| - a)^2; pl_weight -> gamma_pl
    return loss, reg
# Lazy regularization (training loop): step `reg` every k iters in a separate pass sharing Adam state,
# with c = k/(k+1): lr' = c*lr, beta1' = beta1^c, beta2' = beta2^c, and reg scaled by k.
# k = 8 for G (path length), k = 16 for D (R1).
```

So the chain is: the blob is the generator gaming AdaIN's per-map normalization with a dominant spike to smuggle relative scale → kill the data-dependence by normalizing on the *expected* output std computed from the weights (fold style into the weights, divide by their L2 norm), which removes the blob while keeping style control → separately, perceived quality tracks a smooth, well-conditioned latent map, so regularize ‖J_wᵀ y‖ toward a running-average target, which provably equalizes the Jacobian's singular values (orthogonal up to scale, a local isometry) and as a bonus makes the generator invertible → make those regularizers cheap by running them lazily every k steps with the c-corrected optimizer → drop progressive growing (the source of the location-locked, shift-variant details) for a skip generator plus residual discriminator that keep the coarse-to-fine schedule without changing topology → and finally, since the skip generator lets me measure per-resolution usage, add capacity at the high resolutions that were under-used.
