I keep coming back to the same nuisance. I have a time series, and I want to fit an autoregression to it, and the coefficients are the easy part — least squares hands me those the moment I fix the order. The order is what won't sit still. Should it be AR(2)? AR(5)? AR(12)? Every textbook answer reduces to "look at the residual variance," and the residual variance is a liar: it goes down, monotonically, every time I add a coefficient, because AR(k) contains AR(k−1) as the special case where the last coefficient is zero, so the best fit at order k can only beat the best fit at order k−1. If I trust the fitted likelihood, it tells me to take the largest order I'm willing to write down, and that model has memorized the noise. The data, scored by their own likelihood, vote for the worst model. So maximum likelihood, which I love for the coefficients, is structurally blind to the one question I actually care about.

I could reach for Akaike's trick. He noticed exactly this — that the maximized log-likelihood is an optimistic estimate of how the model will do on fresh data, optimistic by about k, the number of parameters — and so he says: don't minimize −2 log P(D|θ̂), minimize −2 log P(D|θ̂) + 2k. If I divide by two to put it on the same −log scale as a code length, that is −log P(D|θ̂) + k: one fixed unit per parameter. That's the first time anyone made complexity *cost* something on the same axis as fit, and it works far better than raw likelihood. But the more I sit with that constant toll the less I believe it. It is the same whether I have a hundred observations or a hundred thousand. In a nested family, an unnecessary extra coefficient can still buy an O(1) likelihood-ratio gain just by fitting noise; a fixed O(1) charge leaves a nonzero chance that the noise gain beats the charge, no matter how much data I pour in. The penalty has to grow with the sample size if it is going to drive that overfitting probability down, and AIC's penalty doesn't. Worse, when I ask where the penalty comes from, the answer is an expected-Kullback-Leibler bias calculation that assumes there's a true distribution out there and computes how badly the log-likelihood estimates the distance to it. I don't want to assume a true distribution. In real modeling I never believe my Gaussian is "true"; I'm using it as a convenient description. A criterion built on the fiction of a true model rests on a fiction I'd rather not sign.

So let me throw out the borrowed machinery and ask what I actually want, in the plainest possible terms. I want the model that best captures the *regularity* in this data. And I think I finally know, from a completely different corner of the world, what "captures the regularity" means quantitatively. Kolmogorov, Chaitin, Martin-Löf — the algorithmic randomness people — say a string is random exactly when it has no description shorter than itself, and it is regular to the extent that it *can* be described more briefly than by writing it out. Solomonoff said the same thing facing forward: the best explanation of a sequence is the shortest program that produces it. Regularity is compressibility. A pattern in the data is precisely a way to say the data in fewer symbols than the data take to say themselves.

That reframes everything. Choosing a model is not "estimating parameters of the truth"; it's choosing a *language* in which to write the data down, and the best language is the one in which the data come out shortest. The model is good to the exact extent that, equipped with it, I can compress the data. Learning *is* compression. I want to make that the whole criterion: among all the ways I could describe this data, take the shortest.

Now, I can't take Solomonoff at face value, because his shortest-program-over-all-programs is uncomputable — there's no algorithm that finds the shortest program for arbitrary data — and for the short series I actually hold, the invariance theorem's "up to a constant" is useless, the constant is bigger than everything. The general-purpose programming language is too rich. But I don't need a general-purpose language. I need to scale his idea down: fix a *restricted* description method — a parametric model class, say all AR models — and ask for the shortest description of the data *within that restricted language*. Inside a parametric family the shortest description is a finite, computable thing. I lose the universality, I lose the guarantee that I'll find every regularity there is, but I gain a usable criterion. That's the trade I want.

So: shortest description of the data, using a model from my family. What does a "description of the data using a model" actually consist of? Here's the thing I keep almost saying and need to say carefully. If I hand you the data described "with the help of" a fitted model, you can't decode it unless you *also* know the model. The model isn't free — to use it to shorten the data, I first have to tell you which model I'm using. So a complete, decodable description has two parts: first I write down the model, then I write down the data as seen through that model. The total bill is

  L(model) + L(data | model).

And immediately this has the shape Occam's razor always wanted and never had. The second part, L(data | model), is the fit: a model that fits well lets me describe the data cheaply, because I only have to encode the discrepancies it fails to predict. The first part, L(model), is the complexity: a richer model takes more bits to specify. A model so complex it nails the data exactly drives L(data | model) toward zero but pays a huge L(model); the trivial model is cheap to state but leaves a long residual to encode. The total is minimized somewhere in between — at a model that is simple enough to be cheap to describe yet rich enough to compress the data. The trade-off isn't something I impose with a tuning constant. It falls out of insisting the whole description be decodable. That's what I've been missing: complexity is charged automatically, in the same currency as fit, because *you can't use a model you haven't transmitted.*

But "in the same currency" is doing enormous work, and I have to make sure it's real. L(data | model) — is that even commensurate with L(model)? What forces them into the same unit? This is where Shannon saves me, and it's the cleanest part of the whole picture. Take any probability distribution P over the possible outcomes. If I allow ideal code lengths, ℓ(z)=−log₂ P(z) satisfies Kraft exactly, because ∑z 2^−ℓ(z)=∑z P(z)=1. If I need actual integer prefix codewords, I use ⌈−log₂ P(z)⌉ and lose less than one bit per item, or I code blocks and push the overhead into an O(1) term. Conversely, any prefix code with lengths ℓ(z) satisfies ∑z 2^−ℓ(z)≤1, so q(z)=2^−ℓ(z) is a subprobability and completing it only changes lengths by a constant. Code length and negative log-probability are not two separate ideas; they match up to the ordinary integer-code constant that I am already going to absorb into O(1). Large probability is short code, small probability is long code, and in bits the ideal conversion is ℓ = −log₂ probability.

Let me convince myself this is forced and not a convention I'm free to break. Suppose I have an alphabet and I want a prefix code. There are at most 2 strings of length 1, at most 4 of length 2, at most 2^m of length ≤ m — so a code is a budget: spending a short codeword on one outcome uses up a big slice of the budget, and the slices have to sum to ≤ 1 if I write them as 2^(−length). That sum-to-one is exactly the constraint a probability distribution satisfies. So every prefix code *is* a distribution wearing different clothes, and assigning outcome z the ideal length −log P(z) is the matched way to spend the budget according to P. And it's optimal in the sense I care about: if the data really do come out with frequencies near some P, then the code with lengths −log P gives the shortest total — by the information inequality, E_P[−log Q(X)] ≥ E_P[−log P(X)] with equality only at Q = P, so using the matched code is best and any mismatch costs you. Good.

Now apply it to the second part. L(data | model) is the length of the data encoded with the help of a model — and a model, in my family, *is* a probability distribution P(·|θ). The ideal matched code spends −log P(data|θ) bits on the data, with the integer-code overhead absorbed into the constant. Which means

  L(data | model) = −log P(D | θ) + O(1).

Stare at that. The fit term in my description-length criterion is the negative log-likelihood, with only the coding constant suppressed. The thing I was already minimizing when I did maximum likelihood is exactly the cost, in bits if I use log₂, of transmitting the data once the model is known. My new criterion doesn't throw away likelihood — it *recovers* likelihood as the data-encoding cost, and then adds the one thing likelihood was missing: the cost of transmitting the model itself, L(model). The whole story is: minimize negative log-likelihood (fit) plus the bits to describe the model (complexity), in one unit, automatically traded off. That's the criterion. That's the answer to the order-selection problem — minimizing L(model) + L(data|model) over both the order and the coefficients gives me the integer structure and the reals together, because they're just different parts of the one description being shortened.

Except I've been glib about L(model), and now it bites. What is the cost of writing down the model? The order k is a small integer — I can encode it in some fixed cheap way, a code for the integers, a handful of bits, negligible. The real problem is the k continuous coefficients θ. A real number has infinitely many digits. To put θ into a decodable message I have to round it to finite precision, and *the precision is a free choice that controls the whole complexity term.* If I describe θ very coarsely — few bits — L(model) is small but my rounded θ fits the data worse, so L(data|θ) goes up. If I describe θ very finely — many bits — L(data|θ) is as small as it can be but L(model) blows up, every extra digit of every coefficient costing another bit. So there's a precision that minimizes the sum, and finding it *is* finding the true cost of the model. This is the crux. The complexity penalty isn't something I'll assert; it's whatever the optimal precision turns out to cost.

Let me actually do it. I keep one log base throughout; if I want literal bits it is log₂, and if I use natural logs every length is just rescaled by a constant. Say I quantize each coordinate of θ to a grid of spacing δ. To name which grid cell θ̂ falls in, over a parameter range of order 1, costs about log(1/δ) code units per coordinate, so

  L(model) ≈ k · log(1/δ)   (plus the cheap cost of k itself).

Finer grid (small δ) ⇒ more bits here. Now the fit. I'll be encoding the data with the *rounded* parameter θ rather than the maximum-likelihood θ̂, so I pay −log P(D|θ) instead of −log P(D|θ̂), and the excess is how much worse the rounded value fits. How big is that excess as a function of how far θ sits from θ̂? Expand the negative log-likelihood around its minimum θ̂. The first-order term vanishes because θ̂ is the minimizer. The second-order term is

  −log P(D|θ) ≈ −log P(D|θ̂) + ½ (θ−θ̂)ᵀ H (θ−θ̂),

where H is the curvature of the negative log-likelihood at θ̂ — the observed information. And here is the fact that sets the scale of the whole problem: H grows with the sample size. The negative log-likelihood is a sum of n per-observation terms, so its second derivative is a sum of n terms too; H ≈ n·I, with I the per-observation Fisher information, an O(1) thing. The curvature accumulates linearly in n. So if I round a coordinate of θ by an amount of order δ, the fit penalty I incur is of order

  ½ · (n·I) · δ² .

Now I can see the tension as a single one-dimensional minimization. Per coordinate, the description costs me about log(1/δ) units to *say* the parameter, and about ½ n I δ² extra units to *use* the rounded parameter. Total per coordinate:

  f(δ) = log(1/δ) + ½ n I δ² = −log δ + ½ n I δ².

Differentiate and set to zero. d/dδ [−log δ + ½ n I δ²] = −1/δ + n I δ = 0, so n I δ² = 1, so

  δ* = 1/√(n I) ,  i.e. δ* is of order 1/√n.

There it is, and it's beautiful, because the 1/√n isn't pulled out of anywhere — it's exactly the width over which the data can't distinguish θ from θ̂. The likelihood near its peak looks Gaussian with standard deviation ~1/√(nI); two parameter values closer than that produce essentially the same fit, so spending bits to tell them apart is waste, and values farther apart fit visibly worse, so coarsening past that point hurts. The optimal precision is exactly the resolution at which the data can still tell parameters apart, and that resolution sharpens as 1/√n as I collect more data. Round to finer than the data can see and you pay in model bits for distinctions nobody can verify; round to coarser and you pay in fit. The sweet spot is the statistical resolution itself.

Put δ* back to get the cost. The fit penalty at the optimum is ½ n I δ*² = ½ n I /(nI) = ½ — a constant per coordinate, folding into the O(1). The naming cost per coordinate is log(1/δ*) = log √(nI) = ½ log n + ½ log I. The dominant, n-dependent piece is ½ log n per coordinate. In k dimensions the same calculation can be done after diagonalizing the local Hessian; the determinant and parameter-volume factors live in O(1), while the n-dependent term is k copies of ½ log n. Summing,

  L(model) ≈ k · ½ log n = (k/2) log n,

up to an O(1) constant that absorbs the constant fit penalties, the log-Fisher terms, and the cheap cost of encoding k. And so the total shortest description length, the criterion I've been chasing, is

  L = −log P(D | θ̂) + (k/2) log n + O(1).

Let me make sure I believe every factor. The −log P(D|θ̂): that's the fit, the negative log-likelihood at the maximum-likelihood estimate, exactly the term ordinary estimation already gives me, now reinterpreted as the bits to transmit the data through the model. The (k/2) log n: each of the k real parameters needs ½ log n bits to be pinned down to the data's own resolution of 1/√n. Each additional parameter therefore costs +½ log n in model bits, and is only worth adding if it shortens the data encoding −log P(D|θ̂) by *more* than ½ log n. That's a stopping rule with no tuning knob: keep adding structure exactly as long as each new coefficient pays for its half-log-n of overhead by buying more than that in fit. The criterion selects the order on its own.

And now compare it back to where I started, because the contrasts are sharp and they reassure me I've got the right object. Against Akaike: his penalty is 2k on the −2 log scale, or k on this −log code-length scale, and it is constant in n. Mine is (k/2) log n on the −log scale, or k log n on the −2 log scale, and it grows with n. That difference is exactly the defect I distrusted in AIC — a fixed toll leaves a non-vanishing chance that an unnecessary nested parameter pays for itself by fitting noise, whereas a toll that grows like log n eventually dominates those O(1) spurious gains. The reason mine grows with n is not a patch; it's forced, it's the ½ log n cost of resolving a parameter to precision 1/√n, and the 1/√n is forced by the curvature growing like n. The n-dependence of the penalty and the n-dependence of the statistical resolution are the same fact. Akaike's 2k came from an expected-KL bias against a presumed truth; my (k/2) log n came from counting bits to write the model down, no truth assumed — I only ever talk about describing *this* data, which is the data-only stance I wanted.

The Bayesian evidence calculation explains the same leading scale from the other side. Around θ̂, the likelihood is locally Gaussian:

  P(D|θ) ≈ P(D|θ̂) exp(−½(θ−θ̂)ᵀ nI (θ−θ̂)).

Then a marginal likelihood with a smooth prior w has the Laplace approximation

  ∫ P(D|θ)w(θ)dθ ≈ P(D|θ̂) w(θ̂) (2π)^(k/2) |nI|^(−1/2).

Taking −log gives −log P(D|θ̂) + (k/2)log n plus constants from w(θ̂), 2π, and |I|. On the conventional −2 log scale, that is −2 log P(D|θ̂) + k log n, the Schwarz/BIC form. The route is different — posterior/evidence integration there, quantized two-part coding here — and the equality is only leading-order, because priors, parameter volume, Fisher determinant, and other model-shape terms sit in the O(1). But the shared (k/2)log n is no accident; it is the local volume of a k-dimensional likelihood peak whose width shrinks like 1/√n in each direction.

One more worry, because it's the one that, unattended, would let me cheat the whole thing. The decoder cannot rely on private knowledge of the observed sequence. If I am allowed to choose a data-specific code and not pay to describe that choice, I can assign the one sequence I observed a one-bit codeword and "compress" anything to nothing, which is meaningless. Any tailoring I do after seeing the data has to be transmitted. That's why L(model) is non-negotiable and why it has to include the precision: the receiver can't decode the data part without knowing exactly which quantized θ I used, so I genuinely must spend those (k/2) log n bits transmitting it. The two-part structure isn't bookkeeping; it's what stops me from fooling myself. And it's why this isn't the Bayesian two-part message of Wallace and Boulton, even though it has the same L(H)+L(D|H) silhouette: their L(H) is −log of a prior, a degree of belief; mine is a code length forced by Kraft and a precision forced by the data's own resolution. No prior, no truth, just: write the data as briefly as a chosen language allows, and let the language pay for itself.

So the minimum description length principle, landed: model the data by the description that is shortest. Concretely, for a family indexed by an integer order k with k real parameters, score each order by the two-part code length

  L(D, k) = −log P(D | θ̂ₖ) + (k/2) log n + O(1),

where θ̂ₖ is the maximum-likelihood fit at order k, the first term is the bits to encode the data through the fitted model (the negative log-likelihood, by Kraft–Shannon), and the second is the bits to encode the k parameters at their optimal precision 1/√n (each costing ½ log n because the likelihood's curvature grows like n). Pick the k, and the θ̂ₖ, that minimize it. Complexity is penalized automatically because an unused-richness model still has to be transmitted; the penalty grows with n at exactly the rate the data's resolving power grows; and the criterion delivers the integer structure and the real coefficients at once, having assumed nothing about a true distribution — it just asks, of all the ways to say the data, which is shortest.

Let me write it as the procedure it is.

```python
import numpy as np

def description_length_bits(data, k, fit_mle, neg_log_likelihood_bits):
    """Two-part code length of `data` under the order-k model, in bits.

    L(D,k) = -log2 P(D | theta_hat_k)  +  (k/2) log2 n
              \--------- fit --------/    \--- model cost ---/
    Part 1 is the data encoded through the fitted model: by Kraft-Shannon,
    the ideal cost in bits of an outcome under a distribution is its -log2 probability,
    so the data-given-model term is exactly the negative log-likelihood at the MLE.
    Part 2 is the cost of transmitting the k real parameters. Each is encoded to
    precision ~1/sqrt(n) -- the resolution at which the data can still tell two
    parameter values apart, since the log-likelihood's curvature grows like n --
    which costs (1/2) log2 n bits per parameter. Finer wastes bits on distinctions
    the data can't verify; coarser worsens the fit by more than it saves.
    """
    n = len(data)
    theta_hat = fit_mle(data, k)                 # the reals: MLE at fixed order k
    fit_bits = neg_log_likelihood_bits(data, theta_hat, k)  # -log2 P(D | theta_hat)
    model_bits = 0.5 * k * np.log2(n)            # part 2: (k/2) log2 n
    return fit_bits + model_bits                 # (O(1) terms dropped)

def select_order(data, k_max, fit_mle, neg_log_likelihood_bits):
    """Choose the order by shortest total description.
    Unlike maximum likelihood (which decreases fit_bits monotonically and so
    always prefers k_max), the (k/2) log2 n term makes adding a parameter worth
    it only if it shortens the data encoding by more than half a log2 n -- a
    self-contained stopping rule, no tuning constant, no assumed true model.
    """
    scores = {k: description_length_bits(data, k, fit_mle, neg_log_likelihood_bits)
              for k in range(k_max + 1)}
    k_star = min(scores, key=scores.get)
    return k_star, fit_mle(data, k_star)         # the integer structure AND the reals
```

The chain, start to finish: maximum likelihood can't choose model order because fitted likelihood only ever improves with more parameters; Occam says prefer the simpler model but never said in what units or by how much; Kolmogorov and Solomonoff supply the unit — regularity is compressibility, the best model is the one that describes the data shortest — but their universal version is uncomputable, so I restrict the description language to a parametric family; a decodable description must transmit the model *and then* the data given it, so the criterion is L(model)+L(data|model); Kraft and Shannon make the two parts commensurate in bits and reveal L(data|model) = −log P(D|θ) + O(1) is just the negative log-likelihood up to coding constants, so likelihood reappears as the fit term; the model term forces me to quantize the k real parameters, and minimizing model-bits-plus-fit-penalty over the precision gives optimal spacing 1/√n (because the likelihood curvature grows like n), costing (k/2) log n; the total −log P(D|θ̂) + (k/2) log n + O(1) penalizes complexity automatically, grows with n at the rate the data's resolution grows, and selects the order and the coefficients together with no assumed truth — Occam's razor, finally a number.
