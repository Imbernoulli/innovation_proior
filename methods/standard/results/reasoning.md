Let me start from what actually hurts. I want a generative model that throws off sharp, coherent 32×32 images — the kind of CIFAR samples that, until now, only adversarial models could produce — but I am tired of paying the adversarial tax. The moment the discriminator outpaces the generator the min-max diverges; mode collapse is a tuning séance; and when training stops I am holding no number that tells me how well the model actually fits the data. What I want instead is a single scalar loss that plain SGD pushes down, monotonically, no second player. I want it likelihood-based, so I have a real codelength to look at. And I want the definition to be simple enough that I can scale it without a new bag of tricks each time.

None of the roads in front of me give me all of that. The autoregressive models — the PixelCNN family — have beautiful likelihoods, but sampling crawls one pixel at a time over the whole image, thousands of sequential network calls, and the raster scan order is an inductive bias I inserted out of nowhere; nothing about an image says the top-left pixel comes first. Normalizing flows give exact likelihood and fast sampling, but invertibility plus a tractable Jacobian pins down how expressive each layer can be, and the latent has to match the data dimension exactly. VAEs are the cleanest to write — the ELBO falls out the instant you apply Jensen — but a single amortized Gaussian posterior keeps collapsing, the decoder learns to ignore the latent, and the samples come out smeared, because one latent layer is too short a bridge from a complicated image distribution to a simple prior. And the score-matching road, the noise-conditional score networks, gets samples that genuinely approach GAN quality — but its sampler's step sizes and per-scale noise are all dialed in by hand after training, there is no proper likelihood, and the training objective never directly optimizes the sampler I actually run.

So my question sharpens to something very concrete: can I have a latent-variable model whose inference path carries *no parameters at all* — nothing to collapse — trained by an honest variational bound, so it is likelihood-based and non-adversarial, and whose samples still land in the GAN tier?

The parameterized inference network is the root of the VAE's troubles. So what if I simply refuse to learn the inference side? Fix a process that grinds the data down into noise over many steps, and learn only how to run it backwards. There is a skeleton for exactly this in the nonequilibrium-thermodynamics line of Sohl-Dickstein and colleagues: define a fixed forward Markov chain q that adds a little noise at every step until, after enough steps, the structure is gone and the endpoint is a distribution I know completely; then learn a reverse chain p_θ that strips the noise one step at a time, and generate by running the reverse chain from that known endpoint. The crucial structural fact they leaned on: as long as each forward step adds a *small* amount of Gaussian noise, the conditional of the matching reverse step is itself approximately Gaussian — so a reverse model built out of Gaussian transitions is expressive enough to invert the chain. And because the forward chain is fixed, with no parameters, there is no encoder to collapse and no posterior to learn. That is precisely the shape I asked for. But it was never turned into high-fidelity images: on CIFAR it reached only about 5.40 bits per dimension and the samples were weak. So I have the skeleton and an empty interior — how to parameterize the reverse mean, what to use for the reverse variance, how to set the noise schedule, what the network is, how to weight the loss. Filling that interior is the whole job.

Let me write the object down. The generative model is

  p_θ(x_0) = ∫ p_θ(x_{0:T}) dx_{1:T},

with latents x_1,…,x_T all the same dimension as the data x_0 (these are noised images, same shape as the image). The reverse process is a Markov chain of learned Gaussian transitions, started from a fixed standard-normal prior p(x_T)=N(0,I):

  p_θ(x_{0:T}) = p(x_T) ∏_{t=1}^T p_θ(x_{t-1}|x_t),   p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), Σ_θ(x_t,t)).

The forward process I do *not* learn — I fix it as a string of noising steps. The most naive way to write a noising step is q(x_t|x_{t-1}) = N(x_t; x_{t-1}, β_t I), pure additive noise. But I can already see that going wrong: every step the variance creeps up, so x_t drifts to larger and larger scale, and the single shared network I am about to train will see inputs whose magnitude depends on t. That is exactly where the score-matching road stumbles — it adds noise without rescaling, so by the time the noise is large the perturbed data variance has blown up. I want the marginal variance roughly conserved along the entire chain. So while I add noise I shrink the surviving signal in proportion:

  q(x_t|x_{t-1}) = N(x_t; √(1−β_t) x_{t-1}, β_t I).

Check it. If x_{t-1} has unit coordinate variance, then Var(x_t) = (1−β_t)·1 + β_t = 1. The variance holds, step after step. That √(1−β_t) is not decoration — it is there so the marginal stays near unit scale and the one network always sees inputs of consistent scale, regardless of where in the chain it is.

Now I need to be able to sample for training. If, to get a training example at step t, I had to honestly walk t forward steps from x_0, that would be hopeless — t can be a thousand. Can I jump straight to an arbitrary t in one shot? Write α_t := 1−β_t, so x_t = √α_t x_{t-1} + √β_t ε_{t-1} with ε standard normal. Substitute x_{t-1} once:

  x_t = √α_t (√α_{t-1} x_{t-2} + √β_{t-1} ε_{t-2}) + √β_t ε_{t-1}
      = √(α_t α_{t-1}) x_{t-2} + √(α_t β_{t-1}) ε_{t-2} + √β_t ε_{t-1}.

The last two terms are independent zero-mean Gaussians, variances α_t(1−α_{t-1}) and (1−α_t). Independent Gaussians add by adding variances: α_t(1−α_{t-1}) + (1−α_t) = α_t − α_t α_{t-1} + 1 − α_t = 1 − α_t α_{t-1}. So

  x_t = √(α_t α_{t-1}) x_{t-2} + √(1 − α_t α_{t-1}) ε̄,

and the induction is immediate. Writing ᾱ_t := ∏_{s=1}^t α_s,

  q(x_t|x_0) = N(x_t; √ᾱ_t x_0, (1−ᾱ_t) I),  i.e.  x_t = √ᾱ_t x_0 + √(1−ᾱ_t) ε,  ε~N(0,I).

This is the engine. I can sample x_t at any t directly, in one shot, no chain walk, which means I can draw t at random and do SGD on a single random term of the bound — fast and cheap. And notice this is the reparameterization trick used as-is: "sample x_t" is written as a deterministic function of x_0 plus fixed noise ε scaled by constants, so everything stays differentiable with low-variance gradients.

Now the training objective. Like any latent-variable model, bound the negative log-likelihood:

  E[−log p_θ(x_0)] ≤ E_q[−log (p_θ(x_{0:T})/q(x_{1:T}|x_0))]
                  = E_q[ −log p(x_T) − Σ_{t≥1} log (p_θ(x_{t-1}|x_t)/q(x_t|x_{t-1})) ] =: L.

My first instinct is to optimize L exactly as written. But look at each term's denominator: q(x_t|x_{t-1}), a single forward noising step, not conditioned on the data. To estimate that ratio I would Monte-Carlo over the forward noise, and the estimator's variance is brutal, because each term compares two local noising steps without ever using the x_0 I actually have in hand during training. The bound is honest but the gradients would be too noisy to train on. Wall.

Stare at that denominator. q(x_t|x_{t-1}) — it forgets x_0 entirely. But during training x_0 is right there. What if I condition the forward posterior on x_0? Because the forward process is Markov, adding x_0 to the condition changes nothing — given x_{t-1}, the next state x_t is independent of x_0 — so q(x_t|x_{t-1}) = q(x_t|x_{t-1},x_0), and now Bayes lets me flip it:

  q(x_t|x_{t-1}) = q(x_t|x_{t-1},x_0) = q(x_{t-1}|x_t,x_0) · q(x_t|x_0) / q(x_{t-1}|x_0).

Substitute that into L. First peel the t=1 term off on its own:

  L = E_q[ −log p(x_T) − Σ_{t>1} log (p_θ(x_{t-1}|x_t)/q(x_t|x_{t-1})) − log (p_θ(x_0|x_1)/q(x_1|x_0)) ].

For each t>1 term, do the substitution:

  log (p_θ(x_{t-1}|x_t)/q(x_t|x_{t-1})) = log (p_θ(x_{t-1}|x_t)/q(x_{t-1}|x_t,x_0)) + log (q(x_{t-1}|x_0)/q(x_t|x_0)).

That second piece, log[q(x_{t-1}|x_0)/q(x_t|x_0)], telescopes when summed over t from 2 to T: head-to-tail cancellation leaves only log q(x_1|x_0) − log q(x_T|x_0). The whole sum sits behind a minus in L, so it contributes −log q(x_1|x_0) + log q(x_T|x_0). The −log q(x_1|x_0) cancels exactly against the +log q(x_1|x_0) buried in the t=1 term, and the leftover +log q(x_T|x_0) merges with −log p(x_T) into −log(p(x_T)/q(x_T|x_0)). Cleaning up,

  L = E_q[ KL(q(x_T|x_0)‖p(x_T)) + Σ_{t>1} KL(q(x_{t-1}|x_t,x_0)‖p_θ(x_{t-1}|x_t)) − log p_θ(x_0|x_1) ]
    =        L_T                  +  Σ L_{t-1}                                       +  L_0.

This is the payoff I was after. Every middle term L_{t-1} is now a KL between two *Gaussians*, which has a closed form — no Monte Carlo over the forward noise, no exploding variance. Conditioning the forward posterior on x_0 is what Rao-Blackwellized the whole bound. The variance I hit the wall on is gone.

So I had better actually compute q(x_{t-1}|x_t,x_0). It is proportional to q(x_t|x_{t-1})·q(x_{t-1}|x_0), both Gaussian in x_{t-1}, and the product of Gaussians is Gaussian — complete the square. Collect the quadratic coefficient in x_{t-1}: from q(x_t|x_{t-1}) = N(√α_t x_{t-1}, β_t) it is α_t/β_t, and from q(x_{t-1}|x_0) = N(√ᾱ_{t-1} x_0, 1−ᾱ_{t-1}) it is 1/(1−ᾱ_{t-1}). So the posterior precision is

  α_t/β_t + 1/(1−ᾱ_{t-1}) = [α_t(1−ᾱ_{t-1}) + β_t] / [β_t(1−ᾱ_{t-1})].

The numerator: α_t − α_t ᾱ_{t-1} + β_t = α_t − ᾱ_t + (1−α_t) = 1 − ᾱ_t (using β_t = 1−α_t and ᾱ_t = α_t ᾱ_{t-1}). So precision = (1−ᾱ_t)/[β_t(1−ᾱ_{t-1})], and the variance is its inverse,

  β̃_t = (1−ᾱ_{t-1})/(1−ᾱ_t) · β_t.

The mean is the linear-in-x_{t-1} term divided by the precision. The linear term picks up √ᾱ_{t-1}/(1−ᾱ_{t-1}) from the x_0 part and √α_t/β_t from the x_t part; multiplying through by β̃_t and simplifying,

  μ̃_t(x_t,x_0) = [√ᾱ_{t-1} β_t/(1−ᾱ_t)] x_0 + [√α_t (1−ᾱ_{t-1})/(1−ᾱ_t)] x_t.

So q(x_{t-1}|x_t,x_0) = N(x_{t-1}; μ̃_t(x_t,x_0), β̃_t I), a clean, explicitly computable Gaussian — the regression target for my reverse step.

Back to L_T. It is KL(q(x_T|x_0)‖N(0,I)), and there is not a single θ inside it — I fixed the forward q, and the β_t are constants — so during training it is a constant I throw away. That hands me a design decision in passing: the forward variances β_t *could*, like a VAE encoder, be learned by reparameterization, but I deliberately do not learn them. Fixing β_t makes q wholly parameterless — there is no such thing as posterior collapse here — and it turns L_T into a discarded constant, so all that is left to train is the reverse process. Simple, and it costs nothing, because the forward chain is only a tool for grinding the data down; I never sample from it at generation time.

The middle term is L_{t-1} = KL(N(μ̃_t, β̃_t I) ‖ N(μ_θ, Σ_θ)). I have to choose Σ_θ. Try the cheapest thing first: set Σ_θ to a non-learned, time-dependent constant σ_t² I. Which constant? Two candidates are natural and bracket the situation: σ_t² = β_t and σ_t² = β̃_t. They are the two extremes — if the data x_0 were itself standard normal, σ_t² = β_t is the optimal reverse variance; if x_0 were a single deterministic point, σ_t² = β̃_t is optimal. For coordinatewise unit variance these are the upper and lower bounds on the reverse step's entropy. I do not want to spend network capacity learning a per-pixel, per-timestep scale before it has even learned which *direction* to denoise; the mean is the part that has to carry the image structure. So I fix σ_t² to a schedule constant and learn only the mean. The framework can still keep a learned-variance branch as an alternative, because the Gaussian form permits it, but the clean path fixes variance and spends modeling capacity on the denoising direction.

With the reverse variance fixed, every variance and log-determinant piece of the Gaussian KL is θ-independent; the only θ-dependent part is the mean mismatch scaled by that fixed variance:

  L_{t-1} = E_q[ (1/(2σ_t²)) ‖μ̃_t(x_t,x_0) − μ_θ(x_t,t)‖² ] + C,

C independent of θ. The most direct parameterization screams at me: let μ_θ predict μ̃_t, the forward-posterior mean. That is a perfectly usable scheme. But the form of μ̃_t bothers me — it is a linear blend of x_0 and x_t, and the network already *has* x_t as its input, so this is asking it to recompute something it half knows. Let me pull μ̃_t apart further.

I have the reparameterization x_t = √ᾱ_t x_0 + √(1−ᾱ_t) ε, so x_0 = (x_t − √(1−ᾱ_t) ε)/√ᾱ_t. Substitute that for x_0 in μ̃_t and let the x_t and ε terms recombine. Using √ᾱ_{t-1}/√ᾱ_t = 1/√α_t, the coefficient of x_t becomes

  β_t/[(1−ᾱ_t)√α_t] + √α_t(1−ᾱ_{t-1})/(1−ᾱ_t).

Put both over √α_t(1−ᾱ_t): the numerator is β_t + α_t(1−ᾱ_{t-1}) = β_t + α_t − ᾱ_t = 1 − ᾱ_t, so the coefficient of x_t collapses to (1−ᾱ_t)/[√α_t(1−ᾱ_t)] = 1/√α_t. The coefficient of ε is −√ᾱ_{t-1}β_t√(1−ᾱ_t)/[(1−ᾱ_t)√α_t] = −β_t/[√α_t √(1−ᾱ_t)]. Together,

  μ̃_t = (1/√α_t)( x_t − (β_t/√(1−ᾱ_t)) ε ).

I stare at this. Written this way, the target μ̃_t splits into a piece in x_t — which the network already has, for free, as its input — and a piece in ε, which is the one thing the network genuinely does not know. So why make it predict the whole μ̃_t? Let it predict ε directly. Switch parameterizations:

  μ_θ(x_t,t) = (1/√α_t)( x_t − (β_t/√(1−ᾱ_t)) ε_θ(x_t,t) ),

where ε_θ is the network, now tasked with answering "which ε was mixed into x_t to make it." Plug this μ_θ back into the squared mean difference in L_{t-1}: the (1/√α_t)x_t terms are identical in μ̃_t and μ_θ and cancel, leaving only the ε difference times the coefficient (β_t/√(1−ᾱ_t))·(1/√α_t), squared:

  L_{t-1} − C = E_{x_0,ε}[ β_t²/(2σ_t² α_t (1−ᾱ_t)) · ‖ε − ε_θ(√ᾱ_t x_0 + √(1−ᾱ_t) ε, t)‖² ].

Wait. This thing — for each noise scale t, hand the network the noised image and have it regress out the noise that was added — that is denoising score matching. Vincent's identity says it cleanly: for a Gaussian corruption q_σ(x̃|x)=N(x̃; x, σ²I), the score ∇_{x̃} log q_σ = (x − x̃)/σ² is proportional to the negative of the added noise. On my side the marginal is q(x_t|x_0)=N(√ᾱ_t x_0, (1−ᾱ_t)I), so ∇_{x_t} log q(x_t|x_0) = −(x_t − √ᾱ_t x_0)/(1−ᾱ_t) = −ε/√(1−ᾱ_t). Predicting ε is, up to a positive scale, predicting the score of the noised data. Which means this ε-regression I just derived — summed over the noise scales t, dropping straight out of an ELBO — *is* the multi-scale denoising score matching the score-based road runs, except mine is the consequence of a proper variational bound rather than a hand-assembled objective.

And look at sampling. A single reverse draw from p_θ(x_{t-1}|x_t) is

  x_{t-1} = μ_θ(x_t,t) + σ_t z = (1/√α_t)( x_t − (β_t/√(1−ᾱ_t)) ε_θ(x_t,t) ) + σ_t z,  z~N(0,I).

That shape — step along a learned gradient ε_θ, then mix in a sliver of fresh noise σ_t z — is exactly Langevin dynamics, x ← x + (δ/2)∇ log p + √δ z, with ε_θ playing the learned (scaled) gradient of the data density. So the reverse chain *is* a Langevin sampler, but its step sizes and noise scales are fixed rigorously by the forward β_t, not dialed in by hand. Three views — fitting the ELBO, doing denoising score matching, training a Langevin sampler — turn out to be one objective. And this closes all four of the score-based road's gaps at once: the sampler coefficients come from β_t rather than a manual sweep; the √(1−β_t) scaling keeps the variance from blowing up; the forward chain genuinely destroys the signal down to N(0,I), so the prior matches the aggregate posterior and there is no start-distribution mismatch; and the whole sampler was trained directly, as a latent-variable model, by variational inference.

There is a third parameterization sitting right next to these: have the network predict x_0 directly and feed that into the posterior-mean formula. It is mathematically valid, but it makes the target shift its scale and meaning across t — at large t the network must hallucinate a clean image from near-pure noise, at small t it mostly copies its input — whereas predicting ε keeps the target standardized across all timesteps and is the thing that exposes the Langevin / score-matching structure. So ε-prediction is the cleaner design-time choice. I will keep the x_0 and x_{t-1} branches in the code because they are natural parameterizations of the same Gaussian reverse step, but train the standard path with ε.

I still have the last term, L_0 = −log p_θ(x_0|x_1). Image data are integers in {0,…,255}, which I scale linearly to [−1,1] so the reverse network — running all the way down from the standard-normal prior p(x_T) — sees a consistent input scale throughout, the same scale the forward chain conserved. But x_0 is *discrete*, while my final reverse step is a continuous Gaussian N(x_0; μ_θ(x_1,1), σ_1² I), and laying a continuous density over discrete data gives no proper codelength. After the scaling, adjacent integer pixel values sit 2/255 apart, so the bin around an interior value x has half-width 1/255. Borrow the trick from the autoregressive and improved-VAE decoders: integrate the continuous density over that bin to get a discrete probability,

  p_θ(x_0|x_1) = ∏_i ∫_{δ_-(x_0^i)}^{δ_+(x_0^i)} N(x; μ_θ^i(x_1,1), σ_1²) dx,

with δ_+(x) = x + 1/255 and δ_-(x) = x − 1/255 for interior values, and the limits taken to ±∞ at the two ends ±1 to absorb the edges. Now the bound is a genuine lossless codelength on the 8-bit data, with no dequantization noise added and no scaling Jacobian smuggled into the likelihood. At the very end of sampling I just display μ_θ(x_1,1) noiselessly, without the final σ_1 z kick.

Now assemble the full bound — L_0 plus the t>1 ε-regression terms — and it is differentiable in θ, trainable as is. But I keep frowning at the weight in front of L_{t-1}: β_t²/(2σ_t² α_t (1−ᾱ_t)). It varies with t, and it leans hardest on the small-t, almost-no-noise terms. Those small-t terms teach the network to remove a *tiny* amount of noise — a task close to the identity map, easy. Yet the true bound spends disproportionate attention there, steering capacity toward dull near-trivial denoising and away from the high-noise steps where the network has to actually invent structure. That feels backwards for a model whose goal is sample quality.

So I drop the weight entirely — set it to 1:

  L_simple(θ) = E_{t,x_0,ε}[ ‖ε − ε_θ(√ᾱ_t x_0 + √(1−ᾱ_t) ε, t)‖² ],   t ~ Uniform{1,…,T}.

Why this helps, stated carefully: the true variational weighting over-emphasizes the small-t terms relative to the high-noise ones, and those small-t terms are the easy near-identity denoising. A uniform weight removes that relative over-emphasis, so the network spends less capacity fussing over imperceptible corrections and more on the genuinely hard high-noise denoising at large t — exactly the steps that decide whether a generated image has coherent global structure. It is still a *re*-weighted variational bound, in the spirit of emphasizing different reconstruction scales as needed, so it changes the codelength emphasis rather than abandoning likelihood. And it lines up with the score-based road's own choice — that uniform weight is exactly what its λ(σ)-weighted denoising score matching uses when the per-scale weight is set to a constant. For the t=1 case I let this same ε-surrogate cover the first step (it equals the discretized L_0 with the bin integral approximated by the density times the bin width, ignoring σ_1² and the edges); L_T never appears, because β_t is fixed. For exact likelihood accounting I keep the discretized L_0 path around, but for the sample-quality training objective it is plain mean-squared error on ε. The implementation could not be simpler.

The math leaves three implementation choices exposed: T, the β schedule, and the network.

How large should T be? Take T = 1000, so the number of network forward passes during sampling lines up with the prior chain-based and score-based work. How to set β? I want them *small*. Small β_t keeps the reverse step's functional form close to the forward step's, so the "reverse conditional is approximately Gaussian" assumption stays valid; and small steps keep the signal-to-noise ratio at x_T as low as possible. With β_1 = 10^{-4} ramping linearly up to β_T = 0.02, the leftover L_T = KL(q(x_T|x_0)‖N(0,I)) works out to about 10^{-5} bits per dimension, essentially zero — meaning the forward chain truly destroys the signal, the prior N(0,I) matches the aggregate posterior, and there is no distribution drift when I start sampling from pure noise. Constant or quadratic schedules can also be constrained to make L_T nearly vanish, but the linear ramp is the simplest small-step schedule and these values are small relative to data in [−1,1]. And note the chain length need not equal the data dimension — 1000 is far below 32·32·3 — so I can shorten it for faster sampling or lengthen it for more expressiveness, unlike an autoregressive model pinned to one step per coordinate.

The network. The core of the reverse process is a denoiser whose input and output are both same-resolution images, so a U-Net is the natural backbone: an encoder–decoder with skip connections, where the skips let high-frequency detail bypass the bottleneck straight to the output. That matters here — a denoiser has to *restore* fine detail, and a pure bottleneck would crush exactly the detail it needs to reconstruct. I will shape it like the unmasked PixelCNN++ backbone, a U-Net over Wide-ResNet blocks.

I want a single network to handle *all* t, not T separate networks, because ᾱ_t makes the denoising task a smoothly t-varying family; sharing parameters across t is far cheaper than training a thousand networks, and it lets the network interpolate across neighboring noise levels — the noise-conditioning idea from the score-based road. How do I tell the network which t it is on? Encode the integer t with a Transformer-style sinusoidal embedding, push it through a small MLP, and add the result into *every* residual block — not, as one score-network variant did, only into the normalization layers, nor, as another did, only at the output. Sinusoidal because it gives t a smooth multi-frequency code, so nearby t map to nearby embeddings and the network generalizes across the schedule; injected everywhere so every layer can self-modulate to the current noise scale rather than relying on a single early or late conditioning point.

Where to put self-attention? Attention is O(n²) in the number of spatial positions, so at the full 32×32 resolution (1024 positions) it is expensive, and at the coarsest 4×4 it has little global structure left to coordinate. The sweet spot is the 16×16 feature resolution: coarse enough that the quadratic cost is affordable, fine enough that there is real global layout and symmetry to coordinate across the image. So I place a self-attention block at 16×16, between the convolutional residual blocks, and leave the local texture at the finer resolutions to convolutions. Concretely on a 32×32 image that means four feature resolutions, 32→16→8→4, with attention only at the second one. For normalization I use group normalization in place of weight normalization: it is independent of batch size and of the per-sample noise scale, so it normalizes consistently across all t and across the small batches used at higher resolution, and it is simpler to implement. On CIFAR I add dropout 0.1 inside the residual blocks, because without it the same-resolution image backbone overfits with the kind of artifacts an unregularized PixelCNN++ shows.

Nothing exotic on the optimizer side: Adam with standard hyperparameters, learning rate 2×10^{-4} (lowered to 2×10^{-5} at 256×256, where the larger spatial tensors make optimization more sensitive), an exponential moving average of the parameters with decay 0.9999 for the weights I actually sample from, batch size 128 on CIFAR (64 at high resolution), and random horizontal flips where the data's semantics allow.

The training loop is now short. Training: repeat { sample x_0 ~ data; sample t ~ Uniform{1,…,T}; sample ε ~ N(0,I); take a gradient step on ‖ε − ε_θ(√ᾱ_t x_0 + √(1−ᾱ_t) ε, t)‖² }. Sampling is the reverse loop: x_T ~ N(0,I); for t = T,…,1, draw z ~ N(0,I) if t>1 else z=0, set x_{t-1} = (1/√α_t)( x_t − ((1−α_t)/√(1−ᾱ_t)) ε_θ(x_t,t) ) + σ_t z; return x_0.

Two connections I noticed while deriving this, both rather pretty, both staying inside what I already have. First, rewrite the bound in the equivalent form L = KL(q(x_T)‖p(x_T)) + Σ_{t≥1} E_q KL(q(x_{t-1}|x_t)‖p_θ(x_{t-1}|x_t)) + H(x_0). It is not directly estimable, but it is suggestive: set the chain length T equal to the data dimension, define the forward step as "mask the t-th coordinate of x_0," put all of p(x_T)'s mass on a single blank image, and let p_θ be a fully expressive conditional. Then KL(q(x_T)‖p(x_T))=0 and minimizing each KL trains p_θ to copy the already-revealed coordinates and predict the next — which is precisely an autoregressive model. So my Gaussian diffusion is an autoregressive model with a *generalized* bit ordering, one that cannot be obtained by permuting coordinates; and Gaussian noise may be a more natural ordering for images than masking, while the chain length is freed from equaling the data dimension. Second, treat L_1+…+L_T as rate and L_0 as distortion: the reverse process becomes a progressive decoder, with x̂_0 = (x_t − √(1−ᾱ_t) ε_θ(x_t)) / √ᾱ_t the running estimate of the clean image at any moment along the chain. Since the high-noise early steps can only carry broad structure, the chain should allocate coarse layout before fine detail, and I would expect a large share of the codelength to go to visually small residual detail — which would make the model a natural lossy compressor. That is a direction I would want to validate, not a conclusion I am asserting.

Now the code has to preserve every branch I actually used. Precompute and cache all the schedule-derived coefficients up front — β_t, ᾱ_t, ᾱ_{t-1}, the square-root combinations that appear in q_sample and in the ε-to-x_0 inversion, the posterior variance β̃_t and its log, the two posterior-mean coefficients, and the fixed-large reverse-variance log branch. In the arrays, index 0 is the first one-based reverse step t=1, where the posterior variance is zero; for finite logs, the clipped posterior log variance reuses posterior_variance[1] at index 0, and the fixed-large log-variance uses that same first entry followed by betas[1:]. Use an extract helper to gather a coefficient at the right t and broadcast it to image shape. The closed-form forward sample, the forward-posterior mean and variance, the ε-to-x_0 inversion, the three prediction branches (x_{t-1} / x_0 / ε), the KL-versus-MSE loss switch, the discretized-Gaussian decoder for the t=0 term, the no-noise mask at the final sampling step, and the full reverse loop are exactly the formulas above. The backbone is the same-shape image network I designed: a U-Net with a sinusoidal time embedding, additive time projections inside each residual block, dropout, skip connections, and self-attention at the 16×16 resolution.

```python
import copy
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam


def get_step_variance_schedule(schedule, *, start, end, steps):
    if schedule == "linear":
        return torch.linspace(start, end, steps, dtype=torch.float64)
    if schedule == "quad":
        return torch.linspace(start ** 0.5, end ** 0.5, steps, dtype=torch.float64) ** 2
    if schedule == "const":
        return torch.full((steps,), end, dtype=torch.float64)
    raise NotImplementedError(schedule)


def extract(a, t, x_shape):
    out = a.gather(0, t)
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


def mean_flat(x):
    return x.mean(dim=tuple(range(1, x.ndim)))


def normal_kl(mean1, logvar1, mean2, logvar2):
    # KL between two diagonal Gaussians, per element
    return 0.5 * (-1.0 + logvar2 - logvar1 +
                  torch.exp(logvar1 - logvar2) +
                  (mean1 - mean2).pow(2) * torch.exp(-logvar2))


def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    # log prob of 8-bit x in [-1,1] under N(means, exp(log_scales)^2), bin half-width 1/255
    centered = x - means
    inv_std = torch.exp(-log_scales)
    cdf_plus = approx_standard_normal_cdf(inv_std * (centered + 1.0 / 255.0))
    cdf_min = approx_standard_normal_cdf(inv_std * (centered - 1.0 / 255.0))
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    return torch.where(
        x < -0.999, log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )


class ImageLatentGenerator(nn.Module):
    def __init__(self, backbone, image_size, latent_steps=1000, schedule="linear",
                 variance_start=1e-4, variance_end=0.02, prediction_type="eps",
                 variance_type="fixedlarge", loss_type="mse"):
        super().__init__()
        if prediction_type not in {"xprev", "xstart", "eps"}:
            raise ValueError("prediction_type must be 'xprev', 'xstart', or 'eps'")
        if variance_type not in {"learned", "fixedsmall", "fixedlarge"}:
            raise ValueError("variance_type must be 'learned', 'fixedsmall', or 'fixedlarge'")
        if loss_type not in {"kl", "mse"}:
            raise ValueError("loss_type must be 'kl' or 'mse'")
        self.backbone = backbone
        self.image_size = image_size
        self.latent_steps = latent_steps
        self.channels = backbone.channels
        self.prediction_type = prediction_type
        self.variance_type = variance_type
        self.loss_type = loss_type

        # forward variances beta_t (small, linearly ramped) and the cumulative products
        betas = get_step_variance_schedule(
            schedule, start=variance_start, end=variance_end, steps=latent_steps).float()
        if not ((betas > 0).all() and (betas <= 1).all()):
            raise ValueError("all step variances must be in (0, 1]")
        alphas = 1. - betas                                  # alpha_t = 1 - beta_t
        alphas_cumprod = torch.cumprod(alphas, dim=0)        # bar(alpha)_t
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)  # bar(alpha)_{t-1}
        reg = lambda n, v: self.register_buffer(n, v.float())

        reg("betas", betas)
        reg("alphas_cumprod", alphas_cumprod)
        reg("alphas_cumprod_prev", alphas_cumprod_prev)
        # coefficients for q(x_t|x_0): x_t = sqrt(bar_a) x0 + sqrt(1-bar_a) eps
        reg("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        reg("sqrt_one_minus_alphas_cumprod", torch.sqrt(1. - alphas_cumprod))
        reg("log_one_minus_alphas_cumprod", torch.log(1. - alphas_cumprod))
        # coefficients to invert eps -> x_0
        reg("sqrt_recip_alphas_cumprod", torch.sqrt(1. / alphas_cumprod))
        reg("sqrt_recipm1_alphas_cumprod", torch.sqrt(1. / alphas_cumprod - 1))

        # forward posterior q(x_{t-1}|x_t,x_0) = N(mu_tilde, beta_tilde I)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)  # beta_tilde_t
        # index 0 is paper step t=1; its posterior variance is zero, so reuse the next entry for logs
        posterior_log_variance_clipped = torch.log(
            torch.cat([posterior_variance[1:2], posterior_variance[1:]]))
        # fixedlarge log variance: same clipped first entry, then beta_t for all later indices
        fixedlarge_log_variance = torch.log(torch.cat([posterior_variance[1:2], betas[1:]]))
        reg("posterior_variance", posterior_variance)
        reg("posterior_log_variance_clipped", posterior_log_variance_clipped)
        reg("fixedlarge_log_variance", fixedlarge_log_variance)
        # mu_tilde_t = coef1 * x0 + coef2 * x_t
        reg("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        reg("posterior_mean_coef2", (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def q_sample(self, x0, t, noise):
        # x_t = sqrt(bar_a_t) x0 + sqrt(1-bar_a_t) eps   -- jump straight to step t
        return (extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    def q_posterior_mean_variance(self, x0, x_t, t):
        mean = (extract(self.posterior_mean_coef1, t, x_t.shape) * x0 +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        var = extract(self.posterior_variance, t, x_t.shape)
        log_var = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var

    def predict_start_from_noise(self, x_t, t, noise):
        # x_0 = (x_t - sqrt(1-bar_a) eps) / sqrt(bar_a)  -- invert the reparameterization
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)

    def predict_start_from_xprev(self, x_t, t, xprev):
        return (extract(1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
                extract(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape) * x_t)

    def p_mean_variance(self, x_t, t, clip_denoised=True, return_pred_xstart=False):
        model_output = self.backbone(x_t, t)
        if self.variance_type == "learned":
            model_output, model_log_variance = model_output.chunk(2, dim=1)
            model_variance = torch.exp(model_log_variance)
        elif self.variance_type == "fixedsmall":            # sigma_t^2 = beta_tilde_t
            model_variance = extract(self.posterior_variance, t, x_t.shape).expand_as(x_t)
            model_log_variance = extract(self.posterior_log_variance_clipped, t, x_t.shape).expand_as(x_t)
        else:                                               # fixedlarge: sigma_t^2 = beta_t
            model_variance = extract(self.betas, t, x_t.shape).expand_as(x_t)
            model_log_variance = extract(self.fixedlarge_log_variance, t, x_t.shape).expand_as(x_t)

        maybe_clip = (lambda y: y.clamp(-1., 1.)) if clip_denoised else (lambda y: y)
        if self.prediction_type == "xprev":
            pred_xstart = maybe_clip(self.predict_start_from_xprev(x_t, t, model_output))
            model_mean = model_output
        elif self.prediction_type == "xstart":
            pred_xstart = maybe_clip(model_output)
            model_mean, _, _ = self.q_posterior_mean_variance(pred_xstart, x_t, t)
        else:                                               # eps: the parameterization we train
            pred_xstart = maybe_clip(self.predict_start_from_noise(x_t, t, model_output))
            model_mean, _, _ = self.q_posterior_mean_variance(pred_xstart, x_t, t)

        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, pred_xstart
        return model_mean, model_variance, model_log_variance

    @torch.no_grad()
    def p_sample(self, x_t, t_int, clip_denoised=True, return_pred_xstart=False):
        t = torch.full((x_t.shape[0],), t_int, device=x_t.device, dtype=torch.long)
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            x_t, t, clip_denoised=clip_denoised, return_pred_xstart=True)
        noise = torch.randn_like(x_t)
        # no noise at the final step t=0 (display the mean noiselessly)
        nonzero_mask = (t != 0).float().reshape(x_t.shape[0], *((1,) * (x_t.ndim - 1)))
        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        return (sample, pred_xstart) if return_pred_xstart else sample

    @torch.no_grad()
    def sample(self, batch_size=16, device=None):
        device = device or self.betas.device
        img = torch.randn(batch_size, self.channels, self.image_size, self.image_size, device=device)
        for t_int in reversed(range(self.latent_steps)):    # reverse Langevin loop, T-1 .. 0
            img = self.p_sample(img, t_int)
        return img

    def vb_terms_bpd(self, x0, x_t, t, clip_denoised=True, return_pred_xstart=False):
        # exact variational term in bits/dim: Gaussian KL for t>0, discretized decoder for t=0
        true_mean, _, true_log_variance = self.q_posterior_mean_variance(x0, x_t, t)
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            x_t, t, clip_denoised=clip_denoised, return_pred_xstart=True)
        kl = mean_flat(normal_kl(true_mean, true_log_variance, model_mean, model_log_variance)) / math.log(2.)
        decoder_nll = -discretized_gaussian_log_likelihood(
            x0, means=model_mean, log_scales=0.5 * model_log_variance)
        decoder_nll = mean_flat(decoder_nll) / math.log(2.)
        out = torch.where(t == 0, decoder_nll, kl)
        return (out, pred_xstart) if return_pred_xstart else out

    def training_losses(self, x0, t=None, noise=None):
        if t is None:
            t = torch.randint(0, self.latent_steps, (x0.shape[0],), device=x0.device).long()
        if noise is None:
            noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        if self.loss_type == "kl":                          # exact variational bound
            return self.vb_terms_bpd(x0, x_t, t, clip_denoised=False)
        if self.variance_type == "learned":
            raise ValueError("the simplified MSE objective uses a fixed-variance branch")
        target = {                                          # L_simple: unit-weight MSE
            "xprev": self.q_posterior_mean_variance(x0, x_t, t)[0],
            "xstart": x0,
            "eps": noise,                                   # predict the noise we added
        }[self.prediction_type]
        model_output = self.backbone(x_t, t)
        return mean_flat((target - model_output).pow(2))

    def training_loss(self, x0, t=None, noise=None):
        return self.training_losses(x0, t=t, noise=noise).mean()

    def forward(self, x0):
        return self.training_loss(x0)


class TimeEmbedding(nn.Module):
    """Transformer sinusoidal embedding of the integer timestep."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000) / (half - 1)))
        emb = t[:, None].float() * freqs[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


def group_norm(channels, max_groups=32):
    groups = min(max_groups, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class ResnetBlock(nn.Module):
    """Wide-ResNet block with the time embedding added in."""
    def __init__(self, dim_in, dim_out, time_dim, dropout):
        super().__init__()
        self.norm1 = group_norm(dim_in)
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, dim_out)
        self.norm2 = group_norm(dim_out)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = zero_module(nn.Conv2d(dim_out, dim_out, 3, padding=1))
        self.act = nn.SiLU()
        self.res = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(self.act(t_emb))[:, :, None, None]   # inject t into every block
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return h + self.res(x)


class AttentionBlock(nn.Module):
    """Self-attention over spatial positions (used at 16x16)."""
    def __init__(self, channels):
        super().__init__()
        self.norm = group_norm(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = zero_module(nn.Conv2d(channels, channels, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        hn = self.norm(x)
        q = self.q(hn).reshape(b, c, h * w)
        k = self.k(hn).reshape(b, c, h * w)
        v = self.v(hn).reshape(b, c, h * w)
        attn = torch.softmax(torch.einsum("bcn,bcm->bnm", q * (c ** -0.5), k), dim=-1)
        out = torch.einsum("bnm,bcm->bcn", attn, v).reshape(b, c, h, w)
        return x + self.proj(out)


class ImageBackbone(nn.Module):
    """Time-conditioned U-Net: same-shape image -> image (predicts eps), attention at 16x16."""
    def __init__(self, image_size=32, channels=3, base_channels=128, out_channels=None,
                 channel_mult=(1, 2, 2, 2), num_res_blocks=2,
                 attn_resolutions=(16,), dropout=0.1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        time_dim = base_channels * 4
        self.init_conv = nn.Conv2d(channels, base_channels, 3, padding=1)
        self.time_mlp = nn.Sequential(TimeEmbedding(base_channels),
                                      nn.Linear(base_channels, time_dim), nn.SiLU(),
                                      nn.Linear(time_dim, time_dim))

        self.downs = nn.ModuleList()
        hs_channels = [base_channels]
        current_channels = base_channels
        current_res = image_size
        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock(current_channels, out_ch, time_dim, dropout))
                current_channels = out_ch
                attns.append(AttentionBlock(current_channels)
                             if current_res in attn_resolutions else nn.Identity())
                hs_channels.append(current_channels)
            down = Downsample(current_channels) if level != len(channel_mult) - 1 else nn.Identity()
            if level != len(channel_mult) - 1:
                hs_channels.append(current_channels)
                current_res //= 2
            self.downs.append(nn.ModuleList([blocks, attns, down]))

        self.mid1 = ResnetBlock(current_channels, current_channels, time_dim, dropout)
        self.mid_attn = AttentionBlock(current_channels)
        self.mid2 = ResnetBlock(current_channels, current_channels, time_dim, dropout)

        self.ups = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                skip_ch = hs_channels.pop()
                blocks.append(ResnetBlock(current_channels + skip_ch, out_ch, time_dim, dropout))
                current_channels = out_ch
                attns.append(AttentionBlock(current_channels)
                             if current_res in attn_resolutions else nn.Identity())
            up = Upsample(current_channels) if level != 0 else nn.Identity()
            if level != 0:
                current_res *= 2
            self.ups.append(nn.ModuleList([blocks, attns, up]))

        self.final_norm = group_norm(current_channels)
        self.final_conv = zero_module(nn.Conv2d(current_channels, self.out_channels, 3, padding=1))

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h = self.init_conv(x)
        skips = [h]
        for level, (blocks, attns, down) in enumerate(self.downs):
            for block, attn in zip(blocks, attns):
                h = attn(block(h, t_emb))
                skips.append(h)
            if level != len(self.downs) - 1:
                h = down(h)
                skips.append(h)

        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)

        for level, (blocks, attns, up) in enumerate(self.ups):
            for block, attn in zip(blocks, attns):
                h = torch.cat((h, skips.pop()), dim=1)       # skip connection
                h = attn(block(h, t_emb))
            h = up(h)
        return self.final_conv(F.silu(self.final_norm(h)))


class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        online = model.state_dict()
        for name, value in self.ema_model.state_dict().items():
            src = online[name].detach()
            if value.is_floating_point():
                value.mul_(self.decay).add_(src, alpha=1.0 - self.decay)
            else:
                value.copy_(src)


def training_step(diffusion, x0, opt, ema=None):
    loss = diffusion.training_loss(x0)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if ema is not None:
        ema.update(diffusion)
    return loss.detach()


if __name__ == "__main__":
    backbone = ImageBackbone(image_size=32, base_channels=16, channel_mult=(1, 2),
                             num_res_blocks=1, attn_resolutions=(16,), dropout=0.1)
    diffusion = ImageLatentGenerator(backbone, image_size=32, latent_steps=8,
                                     prediction_type="eps", variance_type="fixedlarge",
                                     loss_type="mse")
    opt = Adam(diffusion.parameters(), lr=2e-4)
    ema = ModelEMA(diffusion)
    x0 = torch.rand(2, 3, 32, 32) * 2 - 1
    loss = training_step(diffusion, x0, opt, ema)
    samples = ema.ema_model.sample(batch_size=2)
    print(float(loss), samples.shape)
```

Tracing the chain back: I refused the adversarial game and the collapsing parameterized inference, so I fixed the inference side as a Gaussian chain that grinds the image into noise and learned only how to run it backwards; the √(1−β_t) scaling kept the variance conserved so one network sees a consistent input scale, and induction gave a closed-form q(x_t|x_0) that lets me jump to any t directly; optimizing the ELBO naively had unworkable variance, so I conditioned the forward posterior on x_0 and telescoped, turning the bound into a string of closed-form Gaussian KLs; completing the square produced μ̃_t and β̃_t; fixing the reverse variance and learning only the mean reduced each KL to an MSE on the mean; substituting the reparameterized x_0 showed the network only needs to predict the noise ε — which is exactly multi-scale denoising score matching, with the reverse step a Langevin update whose coefficients are pinned by β_t; a discretized-Gaussian decoder pinned down L_0; and disliking how the true weight overloads the easy low-noise terms, I dropped it to the uniform-MSE L_simple, freeing the network for the harder high-noise denoising — which, with T=1000, a small linear β schedule, and a shared U-Net carrying a sinusoidal time embedding and self-attention at the 16×16 resolution, closes the whole construction.
