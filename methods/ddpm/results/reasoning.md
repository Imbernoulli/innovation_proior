OK, let me think this through from scratch. I want an image generative model that samples at GAN quality, but I've had enough of adversarial training — the moment the discriminator collapses the whole min-max diverges, mode collapse is a matter of black-magic tuning, and when training finishes I'm holding no measurable likelihood. What I want is: a single scalar objective that plain SGD can push down; ideally it's likelihood-based too, so I have some hard evidence about "how well does the model actually fit"; and the definition should be simple.

None of the roads on the table satisfy me. Autoregressive models (the PixelCNN family) have excellent likelihood, but sampling walks one pixel at a time over O(D) steps, and they force a coordinate scan order that is an inductive bias inserted out of nowhere. Flow models have exact likelihood and fast sampling, but invertibility and a tractable Jacobian pin down the expressiveness, and the latent dimension has to equal the data dimension. VAEs are clean — the ELBO drops out the moment you write it — but a single-layer amortized posterior keeps collapsing, and the samples come out blurry: one Gaussian latent layer trying to bridge a complex data distribution to a simple prior in one jump is too short a bridge. The score-matching / EBM road (NCSN) gets sample quality approaching GANs, but its sampler step sizes and noise scales are all hand-tuned after the fact, there's no real likelihood, and the training objective doesn't directly optimize the sampler itself.

So my question turns very concrete: is there a latent-variable model whose **inference path carries no parameters at all** (nothing to collapse), trained by a genuine variational bound (likelihood-based, non-adversarial), whose sample quality still reaches the GAN tier?

A parameterized inference network is the root of the trouble. So… what if I just don't learn the inference side? I fix a process that slowly grinds the data into noise, and I only learn how to run it backwards. Sohl-Dickstein and colleagues, in their 2015 nonequilibrium-thermodynamics work, give exactly this skeleton: define a fixed forward Markov chain q that adds noise to the data step by step, over many steps, until the structure is wiped out and the endpoint becomes a simple distribution I know completely; then **learn** a reverse chain p_θ that removes the noise one step at a time, and running that reverse chain from the simple endpoint generates data. They pointed out one crucial fact: as long as each forward step adds a small enough amount of Gaussian noise, the conditional of the corresponding reverse step is also approximately Gaussian — so a Gaussian reverse transition is expressive enough. And because the forward process is fixed and parameterless, there's no encoder to collapse, no posterior to learn. That fits exactly what I want. But they never turned it into high-quality images — on CIFAR the likelihood only reached 5.40 bits/dim, and the samples were weak. So the skeleton exists, but how to parameterize the reverse mean, what to pick for the reverse variance, how to set the noise schedule, what the network looks like, how to weight the objective — all of that is empty. Filling it is exactly my job.

Let me write the object down first. The generative model is
p_θ(x_0) = ∫ p_θ(x_{0:T}) dx_{1:T},
with latents x_1,…,x_T the same dimension as the data x_0. The reverse process is a Markov chain of learned Gaussian transitions, started from a fixed prior p(x_T)=N(0,I):
p_θ(x_{0:T}) = p(x_T) ∏_{t=1}^T p_θ(x_{t-1}|x_t),  p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), Σ_θ(x_t,t)).

The forward process I **do not** learn; I fix it as a string of noising steps. The most naive way to write it is q(x_t|x_{t-1}) = N(x_t; x_{t-1}, β_t I), pure additive noise. But that has a problem: every step the variance grows a bit, so x_t drifts to larger and larger scale, and the input scale fed to the one shared network becomes inconsistent — this is exactly where NCSN stumbles (it doesn't rescale, so once the noise scale is large the data variance blows up). I want the variance roughly conserved along the whole chain. So while I add noise, I also scale the signal down proportionally:
q(x_t|x_{t-1}) = N(x_t; √(1−β_t) x_{t-1}, β_t I).
Check it: let x_{t-1} have unit variance, then Var(x_t) = (1−β_t)·1 + β_t = 1, the variance holds. That √(1−β_t) isn't decoration — it's there so the marginal variance of the whole forward chain stays near 1, giving the network a consistent input scale.

Next I want to sample for training. If every time I had to walk t honest forward steps from x_0 just to get x_t, that's too slow. Can I jump straight to an arbitrary t in one shot? Write α_t := 1−β_t, so x_t = √α_t x_{t-1} + √β_t ε_{t-1}, with ε standard normal. Expand x_{t-1} once more:
x_t = √α_t (√α_{t-1} x_{t-2} + √β_{t-1} ε_{t-2}) + √β_t ε_{t-1}
    = √(α_t α_{t-1}) x_{t-2} + √(α_t β_{t-1}) ε_{t-2} + √β_t ε_{t-1}.
The last two terms are two independent Gaussians, with variances α_t(1−α_{t-1}) and (1−α_t). Two independent zero-mean Gaussians add by adding variances: α_t(1−α_{t-1}) + (1−α_t) = α_t − α_t α_{t-1} + 1 − α_t = 1 − α_t α_{t-1}. Clean, so
x_t = √(α_t α_{t-1}) x_{t-2} + √(1 − α_t α_{t-1}) ε̄.
The structure induces immediately. Write ᾱ_t := ∏_{s=1}^t α_s, and
q(x_t|x_0) = N(x_t; √ᾱ_t x_0, (1−ᾱ_t) I),
i.e. x_t = √ᾱ_t x_0 + √(1−ᾱ_t) ε, ε~N(0,I). This is the engine: I can directly sample x_t for any t in one shot, no chain walk needed. Which means I can sample t at random and do SGD on a random term of the ELBO — fast and cheap. Note this reuses the VAE reparameterization trick directly: writing "sample x_t" as a deterministic function of parameters (here the fixed ᾱ_t) plus fixed noise ε, so everything is differentiable in θ with low-variance gradients.

Now the training objective. Like any latent-variable model, bound the negative log-likelihood by a variational bound:
E[−log p_θ(x_0)] ≤ E_q[−log p_θ(x_{0:T})/q(x_{1:T}|x_0)]
= E_q[ −log p(x_T) − Σ_{t≥1} log p_θ(x_{t-1}|x_t)/q(x_t|x_{t-1}) ] =: L.

My naive instinct is to optimize this L directly. Every term contains q(x_t|x_{t-1}); to estimate it I'd Monte-Carlo over the forward noise, but the single-step transition q(x_t|x_{t-1}) has large variance, so the estimated gradient comes out absurdly noisy. Wall. Stare at that denominator — q(x_t|x_{t-1}), a single forward step, not conditioned on x_0. But I actually know x_0 (during training it's in my hand). What if I condition the forward posterior on x_0 — does it become easier?

Flip the single forward step with Bayes. The forward process is Markov, so
q(x_t|x_{t-1}) = q(x_t|x_{t-1}, x_0)  (adding x_0 changes nothing, since given x_{t-1} the variable x_t is independent of x_0)
            = q(x_{t-1}|x_t, x_0) · q(x_t|x_0) / q(x_{t-1}|x_0).
Substitute that back into L. First pull out the t=1 term on its own:
L = E_q[ −log p(x_T) − Σ_{t>1} log p_θ(x_{t-1}|x_t)/q(x_t|x_{t-1}) − log p_θ(x_0|x_1)/q(x_1|x_0) ].
For the t>1 terms do the substitution:
log p_θ(x_{t-1}|x_t)/q(x_t|x_{t-1}) = log p_θ(x_{t-1}|x_t)/q(x_{t-1}|x_t,x_0) + log q(x_{t-1}|x_0)/q(x_t|x_0).
That string of log[q(x_{t-1}|x_0)/q(x_t|x_0)] telescopes when summed over t from 2 to T — adjacent terms cancel head-to-tail, leaving only log q(x_1|x_0) − log q(x_T|x_0). The log q(x_1|x_0) cancels precisely against the −log q(x_1|x_0) inside the t=1 term, and the −log q(x_T|x_0) merges with −log p(x_T) into −log p(x_T)/q(x_T|x_0). Tidying up:
L = E_q[ −log p(x_T)/q(x_T|x_0) − Σ_{t>1} log p_θ(x_{t-1}|x_t)/q(x_{t-1}|x_t,x_0) − log p_θ(x_0|x_1) ]
  = E_q[ KL(q(x_T|x_0)‖p(x_T)) + Σ_{t>1} KL(q(x_{t-1}|x_t,x_0)‖p_θ(x_{t-1}|x_t)) − log p_θ(x_0|x_1) ].

Name the three kinds of term: the first is L_T = KL(q(x_T|x_0)‖p(x_T)); the middle pile is L_{t-1} = KL(q(x_{t-1}|x_t,x_0)‖p_θ(x_{t-1}|x_t)); the last is L_0 = −log p_θ(x_0|x_1). The payoff of this step is concrete: every L_{t-1} is now a **KL between two Gaussians**, with a closed form, no high-variance Monte Carlo — and that's exactly what conditioning the forward posterior on x_0 bought me. Rao-Blackwellized, variance cut away.

So I need to actually compute q(x_{t-1}|x_t,x_0). It is proportional to q(x_t|x_{t-1}) q(x_{t-1}|x_0), both Gaussian in x_{t-1}, and the product is again Gaussian — complete the square. Collect the quadratic coefficient in x_{t-1}: from q(x_t|x_{t-1})=N(√α_t x_{t-1}, β_t) it's α_t/β_t, and from q(x_{t-1}|x_0)=N(√ᾱ_{t-1}x_0, 1−ᾱ_{t-1}) it's 1/(1−ᾱ_{t-1}). So the posterior precision = α_t/β_t + 1/(1−ᾱ_{t-1}). Over a common denominator:
= [α_t(1−ᾱ_{t-1}) + β_t] / [β_t(1−ᾱ_{t-1})] = [α_t − α_t ᾱ_{t-1} + 1 − α_t] / [β_t(1−ᾱ_{t-1})] = (1−ᾱ_t)/[β_t(1−ᾱ_{t-1})].
Invert it to get the variance
β̃_t = (1−ᾱ_{t-1})/(1−ᾱ_t) · β_t.
The mean is the linear term divided by the precision. In the linear term the coefficient of x_0 comes from √ᾱ_{t-1}/(1−ᾱ_{t-1}) and the coefficient of x_t from √α_t/β_t. Multiply through by the variance β̃_t and simplify:
μ̃_t(x_t,x_0) = [√ᾱ_{t-1} β_t/(1−ᾱ_t)] x_0 + [√α_t (1−ᾱ_{t-1})/(1−ᾱ_t)] x_t.
So q(x_{t-1}|x_t,x_0) = N(x_{t-1}; μ̃_t(x_t,x_0), β̃_t I). A clean, explicitly computable Gaussian — exactly my regression target.

Back to L_T. It is KL(q(x_T|x_0)‖N(0,I)), and there isn't **a single θ inside it** — because I fixed the forward q and the β_t are constants. So during training it's just a constant I can throw away. That hands me my first design decision in passing: the forward variances β_t could, like in a VAE, be learned by reparameterization, but I **don't** learn them. Fixing β_t means q is wholly parameterless (no such thing as posterior collapse), L_T degenerates to a discarded constant, and all that's left to train is the reverse. Simple, and at no cost — the forward is only a tool for grinding the data down.

The middle term L_{t-1} is KL(N(μ̃_t, β̃_t I) ‖ N(μ_θ, Σ_θ)). I have to set Σ_θ. Try the cheapest thing first: set Σ_θ to a non-learned, t-dependent constant σ_t² I. Which constant? Two natural candidates: σ_t²=β_t, and σ_t²=β̃_t. Think about what each means — if the data x_0 were itself N(0,I), then σ_t²=β_t is optimal; if x_0 were a single deterministic point, then σ_t²=β̃_t is optimal. These are the two extremes, the upper and lower bounds on the entropy of the reverse step (for unit coordinate variance). I don't need the network to spend its capacity choosing a diagonal scale at every pixel and timestep before it has learned the direction of the reverse move; the mean is the part that has to carry image structure. So I keep σ_t² fixed and learn only the mean.

With the reverse variance fixed, all variance and log-determinant pieces are independent of θ; the θ-dependent part of the Gaussian KL is just the mean mismatch scaled by that fixed reverse variance. So
L_{t-1} = E_q[ (1/2σ_t²) ‖μ̃_t(x_t,x_0) − μ_θ(x_t,t)‖² ] + C,
with C independent of θ. The most direct parameterization suggests itself: let μ_θ predict μ̃_t, the forward-posterior mean. Fine, that's a usable scheme. But the form of μ̃_t bothers me — it's a linear combination of x_0 and x_t, and the model already has x_t, so this asks the model to recompute something it partly already knows. Let me pull it apart further.

I have the reparameterization x_t = √ᾱ_t x_0 + √(1−ᾱ_t) ε. Solve it for x_0 = (x_t − √(1−ᾱ_t) ε)/√ᾱ_t and substitute into μ̃_t. After substitution the x_0 and x_t terms recombine. Work it out: μ̃_t = [√ᾱ_{t-1}β_t/(1−ᾱ_t)]·(x_t−√(1−ᾱ_t)ε)/√ᾱ_t + [√α_t(1−ᾱ_{t-1})/(1−ᾱ_t)] x_t. Note √ᾱ_{t-1}/√ᾱ_t = 1/√α_t. The total coefficient of x_t: β_t/[(1−ᾱ_t)√α_t] + √α_t(1−ᾱ_{t-1})/(1−ᾱ_t). Write √α_t(1−ᾱ_{t-1}) as (1−ᾱ_{t-1})·α_t/√α_t, put everything over √α_t(1−ᾱ_t); the numerator = β_t + α_t(1−ᾱ_{t-1}) = β_t + α_t − ᾱ_t = 1 − ᾱ_t (using β_t+α_t=1). So the coefficient of x_t = (1−ᾱ_t)/[√α_t(1−ᾱ_t)] = 1/√α_t. The coefficient of ε: −√ᾱ_{t-1}β_t√(1−ᾱ_t)/[(1−ᾱ_t)√α_t] = −β_t/[√α_t √(1−ᾱ_t)]. Together:
μ̃_t = (1/√α_t)( x_t − (β_t/√(1−ᾱ_t)) ε ).

I stare at this. The target μ_θ has to approximate, once written as a combination of x_t (which the model has) and ε, has an x_t part the model needn't learn at all — it's the input. The one thing the model doesn't know is ε. So why make the network predict the whole μ̃_t? Let it predict ε directly. Switch to a new parameterization:
μ_θ(x_t,t) = (1/√α_t)( x_t − (β_t/√(1−ᾱ_t)) ε_θ(x_t,t) ),
where ε_θ is a function approximator that predicts "which ε was mixed into x_t". Plug this back into the squared mean difference in L_{t-1}; the x_t terms cancel, leaving only the difference of ε, and the coefficient comes along:
L_{t-1} − C = E_{x_0,ε}[ β_t²/(2σ_t² α_t (1−ᾱ_t)) · ‖ε − ε_θ(√ᾱ_t x_0 + √(1−ᾱ_t) ε, t)‖² ].

Wait — this thing — for each noise scale t, having the network regress out the noise it was given from the noised sample — that is denoising score matching. Vincent's 2011 identity: for a Gaussian corruption kernel q_σ(x̃|x)=N(x̃; x, σ²I), ∇_{x̃} log q_σ = (x − x̃)/σ², proportional to the negative of the noise that was added. Apply it on my side: ∇_{x_t} log q(x_t|x_0) = −(x_t − √ᾱ_t x_0)/(1−ᾱ_t) = −ε/√(1−ᾱ_t). So predicting ε is, up to a positive scale, predicting the score of the noised data. Meaning this ε-regression I derived — summed over scales t, dropping out of an ELBO — **is precisely NCSN-style multi-scale denoising score matching**, except mine falls out of a proper variational bound rather than being assembled by hand.

Now look at sampling. One step drawn from p_θ(x_{t-1}|x_t) is
x_{t-1} = μ_θ(x_t,t) + σ_t z = (1/√α_t)( x_t − (β_t/√(1−ᾱ_t)) ε_θ(x_t,t) ) + σ_t z,  z~N(0,I).
That shape — step along a learned gradient ε_θ, then mix in a bit of fresh noise σ_t z — is exactly the shape of Langevin dynamics x ← x + (δ/2)∇log p + √δ z, with ε_θ playing the learned (scaled) gradient of the data density. So the reverse chain is a Langevin sampler, but its step sizes and noise scales are all derived rigorously from the forward β_t, not hand-tuned the way NCSN's are. Three views — fitting the ELBO, doing denoising score matching, training a Langevin sampler — are one objective. And this closes NCSN's four gaps at once: (1) the sampler coefficients are determined by β_t, not hand-tuned; (2) I have the √(1−β_t) scaling, so the variance doesn't blow up; (3) my forward really destroys the signal down to N(0,I), so the prior and the aggregate posterior match; (4) I trained this sampler directly as a latent-variable model via variational inference.

There is a third parameterization sitting right next to this one: have the network predict x_0 directly, then plug that prediction into the posterior mean formula. It is mathematically valid, but it makes the target change its scale and semantic content across t: at large t the network must hallucinate a clean image from almost pure noise, while at small t it mostly copies the input. Predicting ε keeps the target distribution standardized across timesteps and exposes the Langevin / score-matching form, so it is the cleaner design-time choice.

I still haven't handled the last term L_0 = −log p_θ(x_0|x_1). Image data are integers in {0,…,255}; I scale them linearly to [−1,1] — that way the reverse network, running all the way down from the standard-normal prior p(x_T), sees a consistent input scale throughout. But x_0 is discrete, while my final reverse step is a continuous Gaussian N(x_0; μ_θ(x_1,1), σ_1² I), and a continuous density laid over discrete data gives no proper codelength. After the scaling, adjacent integer pixel values are 2/255 apart, so the interior bin around a value x has width 2/255 and half-width 1/255. Borrow the trick from the autoregressive / improved-VAE decoders: integrate the continuous density over that bin to get a discrete probability:
p_θ(x_0|x_1) = ∏_i ∫_{δ_-(x_0^i)}^{δ_+(x_0^i)} N(x; μ_θ^i(x_1,1), σ_1²) dx,
with δ_+(x)=x+1/255 and δ_-(x)=x−1/255 for interior values, and the boundaries taken to ±∞ at the extremes ±1 (absorbing the two ends). Now the bound is a lossless codelength on the discrete data, with no need to add dequantization noise and no scaling Jacobian smuggled into the likelihood. At the very end of sampling, I just display μ_θ(x_1,1) noiselessly.

Now assemble the full bound — L_0 plus the t>1 ε-regression terms — it's differentiable in θ and directly trainable. But I keep frowning at that weight in front of L_{t-1}, β_t²/(2σ_t² α_t (1−ᾱ_t)). It varies with t, and the small-t, almost-no-noise terms get much more emphasis than the high-noise terms. Those small-t terms teach the network to remove **tiny** amounts of noise — an easy task, nearly the identity map. Yet the bound spends disproportionate attention on them, pushing network capacity toward this dull, near-trivial denoising. That feels wrong.

So I just drop the weight entirely, set it to 1:
L_simple(θ) = E_{t,x_0,ε}[ ‖ε − ε_θ(√ᾱ_t x_0 + √(1−ᾱ_t) ε, t)‖² ],
with t sampled uniformly over 1 to T. For exact likelihood accounting I still use the discretized L_0 path; for the sample-quality training objective I let the same ε-regression surrogate cover the first step too. The t>1 terms are the weight-stripped ε-terms — which is exactly what NCSN's λ(σ)-weighted denoising score matching does when the weight is set to a uniform scale. L_T does not appear, because β_t is fixed.

Why does dropping the weight help? The true variational weighting emphasizes the small-t terms much more than the high-noise terms — those small-t terms teach the network to remove a tiny sliver of noise, an easy near-identity job. A uniform weight removes that relative over-emphasis, so the network spends less capacity fussing over tiny corrections and more capacity on the genuinely hard high-noise denoising at large t. This is a deliberate reweighting of variational terms, in the β-VAE spirit of emphasizing different reconstruction scales as needed. The likely cost is worse codelength, but the objective now matches the image-synthesis goal more directly, and the implementation is simpler: it's just a mean-squared error.

What's left is to fill out the engine: T, the β schedule, the network.

How large should T be? Take 1000, so the number of network forward passes at sampling time lines up with prior work. How to set β? I want them **small** — small β_t guarantees the reverse has approximately the same functional form as the forward (so the Gaussian reverse is valid, the small-step argument of Sohl-Dickstein holds), and it keeps the signal-to-noise ratio at x_T as low as possible, making L_T = KL(q(x_T|x_0)‖N(0,I)) ≈ 10^{-5} bits/dim, essentially 0. That means the forward truly destroys the signal, the prior N(0,I) matches the aggregate posterior, and there's no distribution drift at sampling time. Among constant, linear, and quadratic schedules (all constrained to L_T≈0), I pick linear: β_1=10^{-4} rising linearly to β_T=0.02. These are all small relative to the [−1,1] data. And note T need not equal the data dimension — shorten it for speed, lengthen it for expressiveness.

The network. The core of the reverse process is a denoiser whose input and output are both same-resolution images, so a U-Net is the natural choice: encoder–decoder with skip connections, where the skips let high-frequency detail bypass the bottleneck straight to the output — a denoiser has to reconstruct detail, and a pure bottleneck would destroy it. Model the backbone on PixelCNN++ (a U-Net built over Wide-ResNet blocks).

I want one network to handle **all** t, not T separate networks, because ᾱ_t turns the task into a smoothly t-varying family; sharing parameters is far cheaper than T networks and lets the network interpolate across neighboring noise scales (NCSN's noise-conditioning idea). How to tell the network which t it's on? Encode the integer t with a Transformer-style sinusoidal embedding, push it through a small MLP, and add it into **every** residual block — rather than, like NCSNv1, only into the normalization layers, or, like v2, only at the output. Why sinusoidal: it gives t a smooth multi-frequency code so that nearby t map to nearby embeddings, generalizing well; and injecting it everywhere lets every layer self-modulate to the current noise scale.

Place self-attention at the 16×16 feature resolution. Why 16×16 specifically? Attention is O(n²) in the number of positions, affordable only at a coarse resolution; 16×16 is the sweet spot where global layout and symmetry — that kind of long-range structure — can be coordinated without the quadratic cost exploding at fine resolutions; local texture at fine resolutions is left to convolutions. Use group norm in place of weight norm: it's independent of batch size and of the per-sample noise scale, so it normalizes consistently across all t and across the small batches used at high resolution, and it's simpler to implement. On CIFAR add dropout 0.1 as a guard against PixelCNN++-style overfitting in the same-resolution image backbone.

Nothing novel on the optimizer side: Adam, standard hyperparameters, learning rate 2×10^{-4} (dropped to 2×10^{-5} at 256×256 because the larger spatial tensors make optimization more sensitive), an EMA of the parameters with decay 0.9999 for evaluation, batch 128 (64 at high resolution), random horizontal flips. Sweep on CIFAR, then transfer.

At this point the whole thing closes, in two algorithms:

Training: repeat { sample x_0~data; sample t~Uniform{1,…,T}; sample ε~N(0,I); take a gradient step on ‖ε − ε_θ(√ᾱ_t x_0 + √(1−ᾱ_t) ε, t)‖² }.

Sampling: x_T~N(0,I); for t=T,…,1: z~N(0,I) (if t>1, else z=0); x_{t-1} = (1/√α_t)( x_t − ((1−α_t)/√(1−ᾱ_t)) ε_θ(x_t,t) ) + σ_t z; return x_0.

Two more connections I noticed along the way, both rather pretty. Rewrite the bound in an equivalent form (not estimable, but suggestive):
L = KL(q(x_T)‖p(x_T)) + Σ_{t≥1} E_q KL(q(x_{t-1}|x_t)‖p_θ(x_{t-1}|x_t)) + H(x_0).
Now imagine setting the chain length T to the data dimension, defining the forward as "q(x_t|x_0) masks the first t coordinates of x_0" (one coordinate masked per step), putting all of p(x_T)'s mass on a single blank image, and assuming p_θ is a fully expressive conditional. Then KL(q(x_T)‖p(x_T))=0, and minimizing each KL(q(x_{t-1}|x_t)‖p_θ) trains p_θ to copy coordinates t+1,…,T verbatim and predict the t-th — which is just an autoregressive model. So my Gaussian diffusion model can be read as an autoregressive model with a "generalized bit ordering" that cannot be obtained by permuting coordinates. And Gaussian noise as that ordering may be more natural for images than masking, while the chain length need not equal the dimension.

Another: treat L_1+…+L_T as rate and L_0 as distortion, and the reverse process becomes a progressive decoder, with x̂_0 = (x_t − √(1−ᾱ_t) ε_θ(x_t)) / √ᾱ_t the running estimate of x_0 at any moment. The chain should allocate coarse structure before fine detail because high-noise steps can only carry broad information. I'd guess much of the codelength may go to visually small residual detail — making the model a natural candidate for lossy compression — which is a direction I'd want to validate, not a conclusion.

Finally, down to real code. Precompute and cache all the schedule-derived coefficients first, then use extract to gather them per t and broadcast to image shape; the closed-form forward q_sample, the reverse posterior mean, the predict_start_from_noise that inverts ε back to x_0, the single-step p_sample, the full-chain sample loop, and the training loss are exactly the formulas above. The network is a compact runnable U-Net with sinusoidal time embedding, residual blocks, skip connections, and an attention block; the large image configuration just scales the same wiring.

```python
import copy
import math
import torch
import torch.nn.functional as F
from torch import nn

def linear_beta_schedule(timesteps):
    return torch.linspace(1e-4, 0.02, timesteps)

def extract(a, t, x_shape):
    out = a.gather(0, t)
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

class GaussianDiffusion(nn.Module):
    def __init__(self, model, image_size, timesteps=1000, model_var_type="fixedlarge"):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.num_timesteps = timesteps
        self.channels = model.channels

        betas = linear_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        reg = lambda n, v: self.register_buffer(n, v.float())

        reg("betas", betas)
        reg("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        reg("sqrt_one_minus_alphas_cumprod", torch.sqrt(1. - alphas_cumprod))
        reg("sqrt_recip_alphas_cumprod", torch.sqrt(1. / alphas_cumprod))
        reg("sqrt_recipm1_alphas_cumprod", torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_log_variance = torch.log(posterior_variance.clamp(min=1e-20))
        fixedlarge_variance = torch.cat([posterior_variance[1:2], betas[1:]])
        if model_var_type == "fixedsmall":
            model_log_variance = posterior_log_variance
        elif model_var_type == "fixedlarge":
            model_log_variance = torch.log(fixedlarge_variance.clamp(min=1e-20))
        else:
            raise ValueError("model_var_type must be 'fixedsmall' or 'fixedlarge'")

        reg("model_log_variance", model_log_variance)
        reg("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        reg("posterior_mean_coef2", (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def q_sample(self, x0, t, noise):
        return (extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    def predict_start_from_noise(self, x_t, t, noise):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)

    def q_posterior_mean(self, x0, x_t, t):
        return (extract(self.posterior_mean_coef1, t, x_t.shape) * x0 +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)

    def p_losses(self, x0, t):
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        return F.mse_loss(self.model(x_t, t), noise)

    def forward(self, x0):
        b = x0.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=x0.device).long()
        return self.p_losses(x0, t)

    @torch.no_grad()
    def p_sample(self, x_t, t_int):
        b = x_t.shape[0]
        t = torch.full((b,), t_int, device=x_t.device, dtype=torch.long)
        eps = self.model(x_t, t)
        x0 = self.predict_start_from_noise(x_t, t, eps).clamp(-1., 1.)
        mean = self.q_posterior_mean(x0, x_t, t)
        log_var = extract(self.model_log_variance, t, x_t.shape)
        noise = torch.randn_like(x_t) if t_int > 0 else 0.
        return mean + (0.5 * log_var).exp() * noise

    @torch.no_grad()
    def sample(self, batch_size=16):
        img = torch.randn(batch_size, self.channels, self.image_size, self.image_size, device=self.betas.device)
        for t_int in reversed(range(self.num_timesteps)):
            img = self.p_sample(img, t_int)
        return img

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000) / (half - 1)))
        emb = t[:, None].float() * freqs[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

def group_norm(channels, max_groups=8):
    groups = min(max_groups, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))

class ResnetBlock(nn.Module):
    def __init__(self, dim_in, dim_out, time_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, dim_out * 2))
        self.norm1 = group_norm(dim_in)
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, padding=1)
        self.norm2 = group_norm(dim_out)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, padding=1)
        self.act = nn.SiLU()
        self.res = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(self.act(self.norm1(x)))
        scale, shift = self.mlp(t_emb)[:, :, None, None].chunk(2, dim=1)
        h = h * (scale + 1) + shift
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.res(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels, max_heads=4):
        super().__init__()
        heads = min(max_heads, channels)
        while channels % heads:
            heads -= 1
        self.heads = heads
        self.norm = group_norm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        scale = (c // self.heads) ** -0.5
        attn = torch.softmax(torch.einsum("bhdn,bhdm->bhnm", q * scale, k), dim=-1)
        out = torch.einsum("bhnm,bhdm->bhdn", attn, v).reshape(b, c, h, w)
        return x + self.proj(out)

class Unet(nn.Module):
    def __init__(self, dim=64, channels=3, dim_mults=(1, 2, 4)):
        super().__init__()
        self.channels = channels
        time_dim = dim * 4
        self.init_conv = nn.Conv2d(channels, dim, 3, padding=1)
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(dim),
                                      nn.Linear(dim, time_dim), nn.GELU(),
                                      nn.Linear(time_dim, time_dim))
        dims = [dim, *[dim * m for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = nn.ModuleList()
        skip_dims = []
        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i == len(in_out) - 1
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_dim),
                ResnetBlock(dim_out, dim_out, time_dim),
                Downsample(dim_out) if not is_last else nn.Identity(),
            ]))
            skip_dims.append(dim_out)

        mid_dim = dims[-1]
        self.mid1 = ResnetBlock(mid_dim, mid_dim, time_dim)
        self.mid_attn = AttentionBlock(mid_dim)
        self.mid2 = ResnetBlock(mid_dim, mid_dim, time_dim)

        self.ups = nn.ModuleList()
        cur_dim = mid_dim
        for i, skip_dim in enumerate(reversed(skip_dims)):
            is_last = i == len(skip_dims) - 1
            self.ups.append(nn.ModuleList([
                ResnetBlock(cur_dim + skip_dim, skip_dim, time_dim),
                ResnetBlock(skip_dim, skip_dim, time_dim),
                AttentionBlock(skip_dim) if i == 1 else nn.Identity(),
                Upsample(skip_dim) if not is_last else nn.Identity(),
            ]))
            cur_dim = skip_dim

        self.final_block = ResnetBlock(cur_dim, dim, time_dim)
        self.final_conv = nn.Conv2d(dim, channels, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = self.init_conv(x)
        skips = []
        for block1, block2, down in self.downs:
            x = block1(x, t_emb)
            x = block2(x, t_emb)
            skips.append(x)
            x = down(x)

        x = self.mid1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid2(x, t_emb)

        for block1, block2, attn, up in self.ups:
            x = torch.cat((x, skips.pop()), dim=1)
            x = block1(x, t_emb)
            x = block2(x, t_emb)
            x = attn(x)
            x = up(x)
        return self.final_conv(self.final_block(x, t_emb))

class EMA:
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
    loss = diffusion(x0)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if ema is not None:
        ema.update(diffusion)
    return loss.detach()

if __name__ == "__main__":
    model = Unet(dim=16, channels=3, dim_mults=(1, 2, 4))
    diffusion = GaussianDiffusion(model, image_size=32, timesteps=8)
    opt = torch.optim.Adam(diffusion.parameters(), lr=2e-4)
    ema = EMA(diffusion)
    x0 = torch.rand(2, 3, 32, 32) * 2 - 1
    loss = training_step(diffusion, x0, opt, ema)
    samples = ema.ema_model.sample(batch_size=2)
    print(float(loss), samples.shape)
```

Tracing the causal chain back: I disliked that parameterized inference collapses and adversarial training is unstable, so I fixed the inference side as a Gaussian chain that grinds the data into noise and only learned how to run it backwards; for a consistent input scale I used the √(1−β_t) scaling, and induction gave a closed-form q(x_t|x_0) that lets me sample any t directly; optimizing the ELBO naively blew up the variance, so I conditioned the forward posterior on x_0 and telescoped, turning the bound into a string of closed-form Gaussian KLs; completing the square produced μ̃_t and β̃_t; fixing the reverse variance and learning only the mean reduced the bound to an MSE on the mean; substituting the reparameterized x_0 revealed that the model only needs to predict the noise ε, which happens to be exactly multi-scale denoising score matching, with the reverse step exactly a Langevin update whose coefficients are pinned by β_t; a discrete decoder pinned down L_0; finally, disliking how the true weight overloads small t, I dropped the weight to get the uniform-MSE L_simple, freeing the network to focus on the harder high-noise denoising — and with large T, a small linear β, and a shared U-Net carrying a sinusoidal time embedding and 16×16 self-attention, the whole thing closes.
