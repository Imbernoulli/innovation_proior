Okay, let me think about this from the start, because something is bugging me and I want to name it precisely before I reach for any machinery. Supervised deep learning is working absurdly well right now. You take a big high-dimensional input — an image, a spectrogram — map it to a label, and with backprop, dropout, and these piecewise-linear units, ReLU and maxout, the thing just trains. The gradients are well behaved, nothing vanishes through a dozen layers, you stack capacity and it works. Then I look at deep *generative* models and it's a graveyard. Why is generation so much harder than discrimination? That asymmetry shouldn't be this severe — generating and recognizing feel like they should be roughly dual. So let me find the exact place where the pain lives, because if I can name it I can try to route around it instead of paying it.

Here is the thing I keep circling back to. To *learn* a generative model the default tool is maximum likelihood, and maximum likelihood demands that I have a probability density `p(x; θ)` I can actually evaluate — or at least evaluate up to something I can differentiate. The whole field is organized, whether people say it out loud or not, around how you get that handle on `p(x)`. Let me lay the options out as a tree, because I think the tree is the diagnosis. If I insist on an *explicit, exactly tractable* density, I can factor it autoregressively, `p(x) = ∏_i p(x_i | x_{<i})`, a fully-visible belief net — every conditional a little tractable model. That genuinely works for likelihood, but sampling is inherently sequential, one coordinate at a time, and there's no latent code to manipulate; for a big image that's hundreds of forward passes per sample. If instead I keep the density explicit but only tractable *up to something*, I land in the Boltzmann family — `p(x) = exp(-E(x))/Z` — where the price is `Z`, or I land in the variational world, where the price is a bound and an inference network. And if I refuse all of that and only keep the ability to *draw a sample* — an implicit model — then I've escaped the density entirely, but now I have no obvious thing to maximize. That third branch is the empty one. Nobody has made it train cleanly with pure backprop and no Markov chain. So that's where I want to push.

Let me make the Boltzmann pain concrete because it's the cleanest illustration of "the tax." For `p(x) = exp(-E(x))/Z` with `Z = Σ_x exp(-E(x))`, the maximum-likelihood gradient is `∂ log p(x)/∂θ = -E_{data}[∂E/∂θ] + E_{model}[∂E/∂θ]`. The first term, the positive phase, is fine — it's an expectation over my dataset. The second term, the negative phase, is an expectation under the *model itself*, and that's exactly the object `Z` makes intractable. So I reach for MCMC — contrastive divergence, persistent CD — to approximate the model expectation by samples from a Markov chain. And then the real killer shows up: mixing. The chain's transition operators are local, and to get an honest model sample it has to wander across regions of low probability that sit between modes, and in any reasonable number of steps it simply doesn't. So my negative-phase samples are biased, the gradient is wrong, learning is slow and fragile. Notice the structure of this complaint: every bit of the pain is downstream of having written `p(x)` down and needing `Z`. The tax is for the density. Deep belief nets don't escape it — they're hybrids, an undirected RBM layer on top of directed layers, fast greedy layerwise training, but by being a hybrid they inherit the headaches of *both* the directed and the undirected world. Not the exit.

Can I dodge the likelihood while keeping an explicit model? There's score matching: instead of matching `p`, match the gradient `∇_x log p` between model and data. Beautiful trick, because `∇_x log p = ∇_x log p̃ - ∇_x log Z`, and `Z` doesn't depend on `x`, so the `Z` term vanishes — no partition function at all. Denoising autoencoders and contractive autoencoders end up with learning rules that look a lot like score matching applied to an RBM, which is suggestive. But the catch — and it's the same catch that's going to keep recurring — is that score matching still needs me to *write the unnormalized density `p̃(x)` analytically*, because I need `∇_x log p̃`. For a model with several layers of latent variables, a DBN or a DBM, I can't even derive a tractable unnormalized density. So score matching doesn't apply to the deep models I actually care about. Same wall, slightly repainted.

Now let me sit with noise-contrastive estimation, because something about it feels closer than everything else and I want to understand exactly why. NCE says: don't compute the likelihood at all, turn density estimation into *logistic regression*. Take your data, take samples from some fixed noise distribution, and train a classifier to tell the two apart — where the classifier is built out of your model's own log-density `log p̃(x; θ)` fed through a sigmoid, and you treat the normalizer `Z` as just one more parameter to fit. Let me actually sit with the implication of that. A *discriminative classification task* — "is this point data or noise?" — drives the fitting of a *generative model*. The model improves precisely by making itself hard to distinguish from data. That is a genuinely different shape from "maximize likelihood," and it's the first thing in this whole survey that doesn't immediately pay the `Z` tax in the same way. So why isn't NCE the answer? Two reasons, and the second one is the interesting one. First: it *still* needs the unnormalized density specified analytically — `log p̃(x; θ)` has to be a thing I can write — so deep latent-variable models are still out. Same wall again. Second, and subtler: the noise distribution is *fixed*. Walk through training. At the start, telling data from a fixed broad noise is a real task and there's signal. But the instant my model becomes even approximately right on some small, low-dimensional subset of the variables — the moment it captures a couple of easy features — distinguishing data from that fixed noise becomes trivial. The classifier saturates, it's confident everywhere, and there's essentially no gradient left to push the model the rest of the way. The learning signal dies because the opponent was a pushover and the model only had to beat the pushover.

Stay on that, because the failure is a clue, not just a flaw. The discriminative *idea* in NCE is great — a classifier supplying the learning signal. What's broken is that the thing being contrasted against never gets harder. It's a fixed punching bag, so once you're mediocre the contrast is solved and there's nothing more to learn. So the question forms: what if the noise weren't fixed? What if the thing I'm classifying against kept getting better in lockstep, so the classification problem never goes slack? Hold that.

Let me check the other escape route people are taking — give up the explicit density and just train a *machine* that emits samples. This is attractive on its face: a sampler can be a neural net, a neural net trains by backprop, and backprop is the thing that works. Generative stochastic networks, generalized denoising autoencoders — they parameterize *one step* of a generative Markov chain, learn that transition, and you sample by running the chain to (approximate) equilibrium. Backprop-friendly, yes. But to get a sample I have to run a Markov chain again, so I've dragged mixing right back into the picture — the exact thing I was trying to escape from the Boltzmann side. And there's a second, sneakier cost: the feedback loop in the recurrence is hostile to piecewise-linear units, because an unbounded activation like a rectifier, fed back through a loop, can blow up. So precisely the units that gave me clean gradients in the feedforward discriminator become liabilities inside the recurrent sampler. I lose my best ingredients. Not clean.

And then there's the variational route — auto-encoding variational Bayes, stochastic backpropagation, happening right now, and this one is genuinely close to what I want. The move is to write the sample (or its latent) as a *deterministic differentiable function of injected noise*, `x = G(z)`, `z ~ p_z` — the reparameterization trick — so gradients flow straight through the sampling step into the parameters. No Markov chain anywhere. Then they maximize a variational lower bound on the log-likelihood, the ELBO, using a learned approximate-inference network — an encoder — regularized so its marginal matches the prior on the latents. So: differentiable generator, trained by pure backprop, no chain. That's three of my four wishes. But it's still anchored to an explicit likelihood — a *bound* on it — it still needs the inference network during training, and the explicit per-pixel reconstruction term in the ELBO is the kind of objective that pushes toward blurry samples, because averaging over plausible reconstructions is itself a low-loss thing to do. So it's close, but it's still tethered to an (approximate) likelihood and to an encoder.

Let me collect what the survey actually taught me, because a shape is forming and I want it sharp. The thing that keeps killing me is *the explicit probability*: the partition function `Z`, or the requirement to write the unnormalized density analytically, or the variational bound. Every method that insists on an explicit `p(x)` pays the tax somewhere — intractable `Z`, MCMC mixing, an inference net plus a bound. Meanwhile two ideas survived contact: from NCE, that a *classifier's success can be the learning signal for a generative model* (it died only because its contrast was a fixed pushover); and from the variational/stochastic-backprop line, that a *differentiable generator `x = G(z)` trained by pure backprop with no chain* is the right machine. And separately I have the cleanest tools I own just sitting there: a deep net built from maxout and dropout with lovely gradients, and an automatic-differentiation framework that will hand me the gradient of *any* differentiable expression with respect to *any* parameters I name.

The hole in "train a generative machine" was: what's the downstream loss, if not a likelihood? The flaw in NCE was: the contrast distribution is a fixed pushover. Both holes are filled by the same move. Use *the generator itself* as the contrast distribution, and use a *separate, learned* classifier net to do the telling-apart — and never write down a density at all. The generator is implicit, so by construction there's no `Z` and no analytic density to specify; I've chosen the implicit branch of the tree on purpose, exactly so I never owe the tax. And because the contrast is a *learned generator that keeps improving*, the classification task can't go slack the way NCE's did — the opponent is being trained to win, so it never stays a pushover.

Let me make the two players concrete. `G`: takes noise `z ~ p_z`, outputs `G(z)`. This implicitly defines a distribution `p_g` over `x` — the pushforward of `p_z` through `G` — but I will never need its formula, only the ability to sample by one forward pass. `D`: takes an `x`, outputs a single scalar `D(x) ∈ (0,1)`, read as the probability that `x` came from the real data rather than from `G`. Train `D` to be a good classifier — high on real, low on fake. And — the move that repairs NCE — train `G` to make `D` *wrong*, i.e. to push `D`'s output on the fakes up toward "real." As `G` improves, the classification problem gets *harder*, not easier. Counterfeiters and police: the counterfeiters improve, which forces the police to improve, which forces the counterfeiters to improve. The opponent is never a pushover because the opponent is being optimized against me.

Why a *separately learned* classifier and not, say, a fixed hand-chosen statistic between sample sets? A fixed statistic has the same disease as NCE's fixed noise: it's a static measuring stick, and once `G` matches the data on whatever features the statistic is sensitive to, the signal flattens, even if the distributions still differ on features the statistic can't see. A *learned* `D` is an adaptive measuring stick — it actively searches for *whatever* feature currently separates fake from real, and keeps re-sharpening as `G` closes each gap. That adaptivity is the whole point, and it's why the opponent has to be trained, not fixed. And why a *classifier* rather than something that directly outputs the density ratio `p_data/p_g`? Because that raw ratio ranges over `(0, ∞)` — it's tiny in places, enormous in others, numerically miserable, you'd have to clip it. A classifier with a sigmoid output naturally produces `p_data/(p_data + p_g)`, the ratio squashed into `(0,1)` — same information, bounded and stable. The discriminator is, in effect, estimating the (transformed) density ratio between two distributions I can only sample from, neither of which I can write down. That's the trick that lets a classifier stand in for two intractable densities.

Now write the objective. `D` wants to assign high probability to the correct label: on real data `x ~ p_data` it wants `log D(x)` high; on a fake `G(z)` it wants `log(1 - D(G(z)))` high, since `1 - D` is "probability it's fake." So `D` maximizes

    E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))].

That expression is exactly the (negative) binary cross-entropy of a classifier with label `1` for data and `0` for fakes — which is the natural, well-understood loss for the well-behaved units I want to use, no new machinery. And `G` wants the opposite: it wants `D` to fail on the fakes, i.e. it wants that second term *small*. So `G` minimizes the very thing `D` maximizes. One value function, two players pulling opposite ways:

    min_G max_D V(D, G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))].

A minimax game. Let me smell-test the corners. If `G` is perfect, `p_g = p_data`, no classifier can beat chance, `D` should sit at `1/2` everywhere, both terms are `log(1/2)`, and `V = -log 4`. If `G` is terrible, `D` nails it: `D(x) ≈ 1` on data, `D(G(z)) ≈ 0` on fakes, both logs go to `0`, so `V → 0`, which is the max — the discriminator is maximally happy and the generator is maximally exposed. The signs line up. Good.

But "the corners smell right" is not "this recovers the data distribution." I want a guarantee, at least in the idealized case where `D` and `G` have unlimited capacity — so let me think nonparametrically, in the space of densities and functions, and forget parameters for a moment. Solve the inner problem first: for a *fixed* `G`, what is the best `D`? Write `V` as an integral. First term: `∫_x p_data(x) log D(x) dx`. Second term is over `z`: `∫_z p_z(z) log(1 - D(G(z))) dz`. I want both expressed over `x`. The second is an expectation under `p_g` by the law of the unconscious statistician — pushing `z` through `G` produces exactly samples distributed as `p_g` — so `∫_z p_z(z) log(1 - D(G(z))) dz = ∫_x p_g(x) log(1 - D(x)) dx`. That step is the entire payoff of defining `p_g` as the pushforward: I never need its formula, I only need that this expectation transforms. So

    V(G, D) = ∫_x [ p_data(x) log D(x) + p_g(x) log(1 - D(x)) ] dx.

Now maximize over `D` *pointwise*. For each fixed `x`, `D(x)` is just a number `y ∈ [0,1]`, and I'm maximizing `a log y + b log(1 - y)` with `a = p_data(x)`, `b = p_g(x)`. Differentiate: `a/y - b/(1 - y) = 0 ⇒ a(1 - y) = b y ⇒ a = (a + b) y ⇒ y = a/(a + b)`. And it's a maximum, not a saddle — the second derivative `-a/y² - b/(1-y)²` is negative, so `a log y + b log(1-y)` is concave on `(0,1)` for `a, b ≥ 0`. Hence

    D*_G(x) = p_data(x) / (p_data(x) + p_g(x)).

Clean. And it's exactly Bayes-optimal classification between the two classes with equal priors — the optimal `D` *is* the (squashed) density ratio I argued the classifier should estimate. The connection is literal, not a metaphor. One edge case to note: I only need `D` defined on the support of `p_data ∪ p_g`; where both densities vanish the integrand is irrelevant, so `D` is free there. And I notice the inner objective for `D` is exactly maximizing the conditional log-likelihood of the label `Y` ("came from data," `y=1`, vs "from `p_g`," `y=0`) given `x` — the inner loop is literally fitting a logistic posterior `P(Y=1|x)`. Same skeleton as NCE, but the negative class is the *learned* `p_g`, not fixed noise. The ancestry I wanted is real and now precise.

Now substitute `D*_G` back and see what game `G` is *really* playing. Define `C(G) = max_D V(G, D) = V(G, D*_G)`:

    C(G) = E_{x~p_data}[log D*_G(x)] + E_{x~p_g}[log(1 - D*_G(x))]
         = E_{x~p_data}[ log( p_data/(p_data + p_g) ) ] + E_{x~p_g}[ log( p_g/(p_data + p_g) ) ],

using `1 - D*_G = p_g/(p_data + p_g)`. I want the `G` that minimizes `C(G)`, and I'll guess the optimum is `p_g = p_data` and then prove it's the *global* min. Plug `p_g = p_data` into `D*_G`: it gives `p_data/(2 p_data) = 1/2`, so `C(G) = log(1/2) + log(1/2) = -log 4`. So my candidate optimal value is `-log 4`; now I must show `C(G) ≥ -log 4` always, with equality iff `p_g = p_data`.

The move is to subtract `-log 4` from `C(G)` and read off what's left. Note `-log 4 = E_{x~p_data}[-log 2] + E_{x~p_g}[-log 2]`, because the expectation of a constant is the constant. So

    C(G) - (-log 4) = E_{p_data}[ log(p_data/(p_data + p_g)) + log 2 ] + E_{p_g}[ log(p_g/(p_data + p_g)) + log 2 ].

Fold the `log 2` inside each log: `log(p_data/(p_data + p_g)) + log 2 = log( 2 p_data/(p_data + p_g) ) = log( p_data / ((p_data + p_g)/2) )`. So the first expectation is `E_{p_data}[ log( p_data / ((p_data + p_g)/2) ) ]`, which is exactly `KL( p_data ‖ (p_data + p_g)/2 )`. And symmetrically the second is `KL( p_g ‖ (p_data + p_g)/2 )`. Therefore

    C(G) = -log 4 + KL( p_data ‖ (p_data + p_g)/2 ) + KL( p_g ‖ (p_data + p_g)/2 ).

The sum of those two KLs, each to the *mixture* `m = (p_data + p_g)/2`, is by definition `2 · JSD(p_data ‖ p_g)`. So

    C(G) = -log 4 + 2 · JSD( p_data ‖ p_g ).

And now I'm smiling, because the JSD is exactly the right object and I didn't engineer it. KL is always `≥ 0`, so JSD `≥ 0`, hence `C(G) ≥ -log 4` always; and JSD is zero *iff* the two distributions are equal. So the global minimum of `C(G)` is `-log 4`, attained uniquely at `p_g = p_data`. That is the whole guarantee: the generator that wins this game is the one whose distribution *is* the data distribution. I never wrote down "minimize JSD" — I wrote down "fool a Bayes-optimal classifier," solved the inner max, and the JSD fell out. The discriminator, at its optimum, was secretly handing the generator a symmetric divergence to descend.

And it matters *which* divergence fell out. The game treats `p_data` and `p_g` symmetrically — real-vs-fake is a symmetric two-class problem — so I get the symmetric JSD, not an asymmetric KL with its mode-covering-vs-mode-seeking baggage, and unlike KL the JSD is finite even where the two supports don't overlap (the mixture `m` always has support wherever either does, so the logs never blow up). That symmetry and boundedness are precisely the properties a sane training signal wants, and they came for free from the structure of the game rather than from a design choice I had to defend. Good.

So far everything is conditioned on "if `D` has reached its optimum for the current `G`." Does the ideal alternating picture converge if I really solve that inner problem before moving `G`? Let me think of it as optimization over `p_g` directly, in function space. Define `U(p_g, D) = V(G, D)` viewed as a function of `p_g`. For fixed `D`, `p_g` enters `V` only through `∫ p_g(x) log(1 - D(x)) dx`, which is *linear* in `p_g` — hence convex (linear functions are convex). Now `C(p_g) = sup_D U(p_g, D)` is a supremum of functions each convex in `p_g`, and a sup of convex functions is convex. So `C` is convex in `p_g`. Here is the subgradient fact I need: if `f(x) = sup_α f_α(x)` with each `f_α` convex, then the subgradient of the particular member `f_β` attained at the supremum is a subgradient of `f` — formally `∂f_β(x) ⊂ ∂f(x)` when `β = arg sup_α f_α(x)`. In words: to take a downhill step on `C` at the current `p_g`, I evaluate the gradient of `U(p_g, D)` at `D = D*_G`, the maximizing discriminator, and that's a valid subgradient of `C`. So computing the gradient of the value function at the *optimal* `D` and stepping `p_g` downhill is genuinely subgradient descent on `C`. And `C` is convex with a unique global optimum at `p_g = p_data` — just proved. Therefore, with small enough steps, the ideal distribution-space update with exact `D*` converges to `p_data`. That proof tells me the target and the descent direction are right; it does not prove that a finite neural net with an approximate `D` is globally safe.

One honest caveat I should write down so I don't fool myself. In practice I don't optimize `p_g` directly; I optimize `θ_g` through an MLP, which represents only a *limited family* of distributions, and an MLP introduces many critical points in parameter space. So the convexity argument lives in distribution space, not in parameter space — it tells me the *target* is right and the *idealized* dynamics converge, not that gradient descent on `θ_g` can't get stuck. My justification for using MLPs anyway is just the brute empirical fact that MLPs optimize beautifully in practice despite the lack of a parameter-space guarantee — the same leap of faith every deep net already rests on.

Now make this actually run. Two practical problems.

First problem: I cannot optimize `D` to completion in the inner loop at every step. It's computationally prohibitive, and on a finite dataset a fully-optimized `D` would just overfit — memorize which points are real — and stop giving useful gradients about the *distribution*. But the proof above wanted `D*` exactly. The practical patch: don't solve `D` to optimality; take `k` gradient steps on `D`, then one step on `G`, and repeat. If `G` changes slowly, then `D` can track the moving optimum from one outer iteration to the next — I never re-burn-in from scratch, I just keep the inner state warm. This is exactly the persistent-chain trick from training Boltzmann machines, SML/PCD: you carry the negative-phase Markov state across learning steps rather than restarting it each time, because the model moved only a little. Here the carried state is `D`'s parameters. In practice `k = 1` — one `D` step per `G` step — is the cheapest version of that tracking idea. The theorem still belongs to the exact-`D*` idealization; the schedule is the engineering approximation that tries to stay close to it.

Second problem, the gradient one. The generator's term in the minimax game is `log(1 - D(G(z)))`. Early in training `G` is garbage, the fakes are obviously fake, so `D` confidently rejects them: `D(G(z)) ≈ 0`. I have to look at the gradient in the discriminator's *logit* space, not only at the derivative with respect to the already-squashed probability. Write `a` for the discriminator logit on a fake sample, so `D = σ(a)`. Then

    d/da log(1 - σ(a)) = -σ(a) = -D.

That is the saturation. When `D(G(z)) ≈ 0`, the derivative reaching the fake-sample logit is also near zero, so the chain back through `a(G(z))` and into `θ_g` nearly disappears. The derivative `d/dD log(1 - D)` is not the right place to diagnose this; it is about `-1` near `D = 0`, but the sigmoid factor `dD/da = D(1-D)` has already killed the signal. So precisely when `G` is losing badly — exactly when it most needs a strong push to improve — the logit-space gradient flowing back into `θ_g` is tiny, and `G` barely moves. That's NCE's saturating-classifier disease sneaking back in through the generator's own loss. Wall.

How do I fix it without changing the game's equilibrium? I don't actually need `G` to *literally minimize* `log(1 - D(G(z)))`. I need a signal that points the fake samples toward larger discriminator logits — "make `D` think the fakes are real" — but that has strong gradient when `G` is losing. So flip the objective: instead of minimizing `log(1 - D(G(z)))`, let `G` minimize `-log D(G(z))`, equivalently maximize `log D(G(z))`. In the same logit notation,

    d/da[-log σ(a)] = -(1 - σ(a)) = -(1 - D).

Now when `D(G(z)) ≈ 0`, the derivative is about `-1`: not infinite, not dead, just a clean strong push to increase the fake logit. It vanishes only when the fake is already being called real. This is the non-saturating loss. It is no longer the minimax generator term; it shares the same fixed point of the two-player game while having a healthy gradient precisely where the minimax term saturates. I'll keep the minimax form for the clean theory and use the non-saturating form to run.

One more failure mode I should name now so I watch for it. `D` and `G` have to stay synchronized. If I let `G` train hard against a *stale* `D`, `G` can cheat: collapse many different `z` onto the same few outputs `x` that happen to fool the current `D`. It wins the local battle — `D` is fooled — but it has thrown away diversity, so `p_g` no longer covers `p_data`; call it the collapse, the "Helvetica" failure where everything maps to one blob. The defense is the same balance the schedule already imposes from the other side: don't let `G` run away from `D`, keep `D` fresh — which is, once again, the same logic as keeping a Boltzmann machine's negative chains up to date between learning steps. So the `k`-step schedule is doing double duty: it tries to keep `D` tracking the optimum the theory asks for, and it keeps `G` from outrunning `D` into collapse.

Let me also account for `z` and the differentiability of `G`, because the whole gradient story depends on them. The only stochasticity in the generator is the noise `z ~ p_z`; everything after it is the deterministic map `G(·; θ_g)`. That deterministic-function-of-noise structure is exactly what lets backprop carry `D`'s gradient through `G`'s outputs and into `θ_g` — the same reparameterization logic the variational line uses. If the randomness were *inside* `G` as a stochastic unit instead of an injected input, I couldn't backprop a clean gradient through the sampling. The prior `p_z` itself is arbitrary as long as I can sample it and `G` is rich enough to reshape it — Gaussian, uniform, spherical, doesn't much matter — because `G` is doing all the work of bending `p_z` into `p_g`. And while the theory permits injecting noise at *every* layer of `G`, in practice it's enough to feed noise only at the bottom layer and let the deterministic stack do the rest. No feedback loop anywhere in generation means I can use the piecewise-linear units freely in `G`, the very thing the GSN recurrence couldn't afford.

Before I commit, let me tally what I've bought and what I've paid, because this departs so far from everything else. Paid: there is no explicit `p_g(x)` — I can sample but I cannot evaluate the density, so any likelihood-style evaluation has to be indirect (fit a Gaussian Parzen window to a batch of samples, pick the bandwidth `σ` by cross-validation, report test log-likelihood under that — noisy, weak in high dimensions, but it's the available yardstick); and `D` must be kept synchronized with `G` or I get collapse. Bought: no Markov chain *anywhere*, in training or in sampling — a sample is one forward pass through `G`; no inference network during learning, unlike the variational route; the gradient is pure backprop through `D` into `G`; piecewise-linear units usable freely because there's no generation-time feedback loop. And a subtler statistical perk — `G` never touches a data example directly, it only ever sees the data *through the gradient that flows back through `D`*, so it can't trivially copy training points into its parameters; it can only "overfit" if `D` overfits, and `D`'s overfitting is the easy thing to control. It can even represent very sharp, near-degenerate distributions, which MCMC methods structurally cannot, because chains need the distribution blurry enough to mix between modes. That's a trade I'm glad to take.

Now land it as actual code. In the framework I'm building on — Theano plus Pylearn2 — this fills the empty sampler, learned-signal, cost, and split-optimizer slots from the scaffold: a `Generator` wraps the noise-driven MLP, an `AdversaryPair` holds the generator and discriminator, `AdversaryCost2` produces separate scalar losses for the two parameter groups, and a small SGD variant runs discriminator updates separately from generator updates. The generator maps noise to data (rectifier-plus-sigmoid units in the setup I have in mind); the discriminator is maxout with dropout — maxout because its piecewise-linear pieces give the cleanest gradients for the classifier, dropout because `D` is a powerful net being trained on a finite dataset against a moving target and I want to keep it from overfitting the current `G`. The detail that lets me write the whole game with one loss: the discriminator's last-layer binary cross-entropy with target `1` is just `-log D(x)`, and with target `0` is just `-log(1 - D(x))`. So `d_obj` labels real as `1` and fake as `0`; and `g_obj`, non-saturating, labels the *fakes* as `1`, so that *minimizing* its BCE is the same as *maximizing* `log D(G(z))`. Then `T.grad` takes the two gradients against the two disjoint parameter sets independently — which is the one thing automatic differentiation is perfect for. The core code shape is:

```python
# Pylearn2 Cost over an AdversaryPair(generator G, discriminator D).
# Targets are supplied by the cost itself, not the dataset.
def get_samples_and_objectives(self, model, data):
    g, d = model.generator, model.discriminator
    X = data                                   # minibatch x ~ p_data
    m = X.shape[0]
    y1 = T.alloc(1, m, 1)                       # label "real"
    y0 = T.alloc(0, m, 1)                       # label "fake"
    S, z, _ = g.sample_and_noise(
        m,
        default_input_include_prob=self.generator_default_input_include_prob,
        default_input_scale=self.generator_default_input_scale,
        all_g_layers=False,
    )                                          # S = G(z), z ~ p_z; one forward pass, no chain

    y_hat1 = d.dropout_fprop(
        X,
        self.discriminator_default_input_include_prob,
        self.discriminator_input_include_probs,
        self.discriminator_default_input_scale,
        self.discriminator_input_scales,
    )                                          # D(x), with discriminator dropout
    y_hat0 = d.dropout_fprop(
        S,
        self.discriminator_default_input_include_prob,
        self.discriminator_input_include_probs,
        self.discriminator_default_input_scale,
        self.discriminator_input_scales,
    )                                          # D(G(z)), same dropout settings

    # d_obj is minimized; minimizing BCE is the same as ascending
    # E[log D(x)] + E[log(1 - D(G(z)))].
    d_obj = 0.5 * (d.layers[-1].cost(y1, y_hat1)    # cost(target=1, .) = -log D(x)
                 + d.layers[-1].cost(y0, y_hat0))   # cost(target=0, .) = -log(1 - D(G(z)))

    if self.no_drop_in_d_for_g:
        y_hat0_for_g = d.dropout_fprop(S)
    else:
        y_hat0_for_g = y_hat0

    # generator: non-saturating. Label fakes as real (1) so minimizing BCE
    # maximizes log D(G(z)) -> strong gradient when D(G(z)) is near 0.
    g_obj = d.layers[-1].cost(y1, y_hat0_for_g)
    return S, d_obj, g_obj, 0

def get_gradients(self, model, data):
    S, d_obj, g_obj, _ = self.get_samples_and_objectives(model, data)
    g_params = model.generator.get_params()
    d_params = model.discriminator.get_params()
    for param in g_params:
        assert param not in d_params
    for param in d_params:
        assert param not in g_params

    d_grads = T.grad(d_obj, d_params)
    g_grads = T.grad(g_obj, g_params)

    if self.scale_grads:
        S_grad = T.grad(g_obj, S)
        scale = T.maximum(1., self.target_scale / T.sqrt(T.sqr(S_grad).sum()))
        g_grads = [g_grad * scale for g_grad in g_grads]

    rval = OrderedDict()
    rval.update(OrderedDict(safe_zip(
        d_params, [self.now_train_discriminator * dg for dg in d_grads]
    )))
    rval.update(OrderedDict(safe_zip(
        g_params, [self.now_train_generator * gg for gg in g_grads]
    )))
    return rval, OrderedDict()

# The SGD loop compiles separate update functions and runs k discriminator steps per generator step.
for batch in iterator:
    d_func(*batch)
    i += 1
    if i == discriminator_steps:               # default 1
        g_func(*batch)
        i = 0
```

That is enough to run in the framework. The model definitions are ordinary MLPs: for MNIST-scale data, uniform noise of dimension `100`; a generator with `100 -> 1200 -> 1200 -> 784`, rectified hidden layers, and a sigmoid output for `[0,1]` pixels; a discriminator with two maxout layers of `240` units and `5` pieces, dropout on its inputs and first hidden layer, and a sigmoid output; batch size `100`; SGD learning rate `.1`; momentum starting at `.5` and rising to `.7`. The crucial thing is not a new optimizer. It is the target assignment inside the cost, especially giving generated samples target `1` for the generator update, so minimizing the generator's binary cross-entropy is the non-saturating `-log D(G(z))` objective.

So the whole causal chain. I started annoyed that deep generation lags deep discrimination, and traced the lag to one root cause: the explicit-probability tax — the intractable `Z` and MCMC mixing in the Boltzmann family, the need for an analytic unnormalized density in score matching, the variational bound plus inference net in the variational route, the reintroduced Markov chain in the generative-machine route. Two survivors from the survey pointed the way: NCE showed a *classifier can be the learning signal for a generative model* but died because its contrast was a fixed pushover, and the reparameterized differentiable generator showed how to train a sampler by pure backprop with no chain. Filling both holes with one move — make the contrast a *learned generator* and the classifier a *separate learned discriminator*, and stay implicit so there is never a density to normalize — gives a minimax value function that is just the optimal classifier's cross-entropy. Solving the inner max gives `D* = p_data/(p_data + p_g)`, the squashed density ratio; substituting it collapses the game to `-log 4 + 2·JSD(p_data ‖ p_g)`, minimized exactly and only at `p_g = p_data`; exact inner optimization turns the distribution-space update into subgradient descent on a convex functional, while the practical `k`-step schedule tries to track that ideal and also prevents collapse; and because the minimax `log(1 - D)` generator term has logit derivative `-D` when `G` is losing, I swap `G`'s objective to maximize `log D(G(z))`, sharing the fixed point but giving a strong early logit-space gradient. The result is a generative model trained by pure backprop, sampled by one forward pass, with no Markov chain and no inference network anywhere.
