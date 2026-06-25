OK, let me think this through from scratch. The thing I actually care about is wall-clock training time. The networks that matter take days to converge, and the easy answer — throw more machines at it — keeps disappointing me: splitting the batch or the model across machines costs communication and the returns fall off fast once I parallelize past a point. So I want the other lever. Not "compute the same gradients faster on more hardware" but "change the forward computation so that each gradient step is worth more." That means making the optimization landscape itself easier to walk.

Where does the difficulty come from? Stare at one layer. Neuron i in layer l takes the previous layer's output h and forms a summed input a_i = w_i^T h, then fires f(a_i + b_i). When I update the weights of the layers below, the distribution of h changes, so the distribution of every a_i in this layer changes too — and it changes in a correlated way across the layer, because all these a_i are driven by the same shifting h. With ReLU below, an output can swing by a lot, so the swing in the a_i is large. Every time the input distribution to a layer lurches, the gradient this layer wants is computed against a moving target. That's what forces me to use tiny learning rates and to baby the initialization. If I could pin the distribution of the summed inputs — keep its mean and variance roughly fixed as the layers below churn — the layer above would see a stable input and I could push the learning rate up.

There's a recipe that does exactly this for feed-forward nets. For each neuron, standardize its summed input across the data: subtract the mean, divide by the standard deviation, then put back a learned gain and bias before the nonlinearity,

  abar_i = (g_i / sigma_i)(a_i - mu_i),

with mu_i = E_x[a_i] and sigma_i = sqrt(E_x[(a_i - mu_i)^2]). The expectation is over the data distribution, which I can't actually evaluate — it would mean a forward pass over the whole dataset at the current weights — so in practice I estimate mu_i and sigma_i from the current mini-batch. And it works: plain SGD converges much faster, and the jitter from estimating the statistics on a random batch even regularizes a little for free.

But look at what the estimate is. mu_i and sigma_i are per-neuron, averaged across the batch dimension. That single fact is the source of everything that's about to go wrong. First, the estimate is only as good as the batch is big — shrink the batch and the statistics get noisy; go to batch size 1, the pure online setting, and the variance across a one-element batch is meaningless. Second, at training time I normalize with batch statistics, but at test time there's no batch to speak of, so I have to keep running averages during training and swap them in at test — now training and inference compute different functions and I'm carrying extra bookkeeping. Third, and this is the one that really bites me, recurrent networks.

Let me try to actually apply this to an RNN and watch it break. The recurrent layer reuses the same weights at every step: a^t = W_hh h^{t-1} + W_xh x^t. If I want to normalize a^t I need its mean and variance — but the distribution of a^t is different at every time-step t, because the hidden state evolves. So "per-neuron across the batch" is now "per-neuron across the batch, separately for each t." I have to store a separate mu, sigma for t = 1, 2, 3, ... And sequences have different lengths. What happens at test time when a sequence runs longer than anything I saw in training? There's no stored statistic for t = 51 if my longest training sequence was 50. I have nothing to normalize with. The whole scheme assumes a fixed, enumerable set of "slots" to keep statistics for, and a recurrent net of unbounded length doesn't give me that. People have pushed on this — keeping independent statistics per time-step does turn out to be the best recurrent variant — but they also report that it only behaves if the recurrent gain is initialized tiny, around 0.1, and falls apart otherwise. That fragility is a tell: I'm forcing a batch-statistics method into a place it doesn't fit.

So let me write down what I'd need from a fix, precisely: a normalization whose statistics (a) don't depend on the batch at all, so batch size 1 and online learning are fine; (b) are identical at training and test, so no running averages; (c) are defined the same way at every time-step regardless of sequence length. The recurring offender in all three is the same: I am reducing over the batch dimension.

What else could I reduce over?

The summed inputs to a layer form a matrix: rows are training cases in the batch, columns are neurons. The batch recipe computes a mean and variance down each column — one statistic per neuron, pooled over the cases. The batch dimension is the one that's causing me grief. What if I just turn the computation ninety degrees and reduce across the *other* axis — across the neurons, within a single row, for one training case at a time? Then the mean and variance are properties of *this example's layer*, computed from the H units that are present right now. No batch. Nothing to store. Nothing that knows or cares what time-step it is.

But wait — does normalizing across neurons even address the original disease? The disease was: when the layer below shifts, all the a_i in this layer shift together, in a correlated way. Let me reason about what form that shift actually takes. If the drift were *uncorrelated* across units — each a_i wandering independently — then averaging over the units would just give me noise, and re-centering by that average would do nothing useful for any individual unit. But the drift isn't independent: every a_i is driven through the same shifting h, so a common component of the change is shared across the units. To first order, if h moves by Δh, then a_i moves by w_i^T Δh, and while that's a different number per unit, the part of Δh that's large and common pushes the whole layer's pre-activations in a consistent direction — it shows up as a change in the *layer's* mean and spread, which are exactly the two quantities I'd compute across the neurons. So re-centering and re-scaling by the layer's own mean and variance should subtract off the bulk of that correlated drift. I can't be fully sure of the magnitude without measuring it on a real network, but the mechanism lines up: the more correlated the shift, the more of it lands in the mean and is removed. And I don't need the batch to see this drift; I can see it within a single example, because it's a property of how that example's pre-activations sit relative to each other.

So: compute, for one training case,

  mu = (1/H) sum_{i=1}^H a_i,   sigma = sqrt((1/H) sum_{i=1}^H (a_i - mu)^2),

over the H units of the layer, and normalize every unit of that layer with this shared mu and sigma. Contrast it cleanly with the batch recipe: there, all training cases shared a neuron's statistics; here, all neurons in a layer share an example's statistics, and different examples get different mu, sigma. Symmetric transpose. And every one of my three requirements falls out for free — the statistics use only the current example's units, so batch size is irrelevant (size 1 is fine), training and test do literally the same computation, and in an RNN I just compute mu^t, sigma^t from the H summed inputs at step t, with one shared gain and bias across all steps, never caring how long the sequence is.

The full normalized unit, then, with a per-neuron gain g_i and bias b_i restored after standardizing:

  h_i = f( (g_i / sigma)(a_i - mu) + b_i ).

Why keep the gain and bias at all? Because standardizing fixes the normalized vector to a centered, unit-scale coordinate system, and the network might genuinely want each neuron to sit at a different operating point going into the nonlinearity. The learned g_i and b_i hand back a per-neuron scale and offset after the shared standardization. They do not make this a simple reparameterization that can undo every per-example mean and variance change; they restore the local degrees of freedom the nonlinearity needs while keeping the common drift removed. And the affine restore goes on the summed input, after normalization and before the nonlinearity, so the nonlinearity always sees a controlled range — standardizing after the nonlinearity instead would fight saturating units and miss the point.

The plain recurrent step is a^t = W_hh h^{t-1} + W_xh x^t, and the normalized one is

  h^t = f[ (g / sigma^t) ⊙ (a^t - mu^t) + b ],

with mu^t, sigma^t the mean and standard deviation over the H components of a^t, and a single g, b shared over time. In a vanilla RNN the average magnitude of the summed inputs tends to either grow or shrink a bit at every step — the recurrence compounds, and over a long sequence that compounding becomes exploding or vanishing gradients. But sigma^t is, by construction, the scale of a^t. If I divide a^t by its own scale before feeding it forward, then multiplying every summed input by a positive common factor should leave the normalized vector unchanged: scale a^t by c > 0 and both (a^t - mu^t) and sigma^t scale by c, so the ratio is identical. Quick sanity check on a random a^t with H = 6 and c = 3.7: standardizing c·a^t versus a^t and taking the max component difference gives 4.4e-5, which is the eps inside the sqrt talking, not a real difference — so the step is invariant to re-scaling the whole vector of summed inputs. That's exactly the degree of freedom that was compounding into explosion or vanishing, and I've quotiented it out. The hidden-to-hidden dynamics stop drifting in overall magnitude; they're held at a fixed scale every step. For long sequences with tiny batches — where the batch recipe is useless anyway — this stability is the whole game.

Let me also notice something about how this differs in kind from the alternatives, because it changes what I can prove. There's a reparameterization approach floating around: write each weight vector as w = g · v/||v||, length times direction; that's the same as normalizing the summed input with mu = 0, sigma = ||w||. And the batch recipe, if I imagine using its true expected statistics, is also just a rescaling of the original network's pre-activations. Both of those are *reparameterizations* — they describe the same family of functions the original network already had, in new coordinates. Mine is not. My mu and sigma depend on the actual summed inputs of this example through the data, not on a fixed function of the weights, so re-centering by the layer's own per-example mean is something the original network simply can't reproduce by relabeling its weights. That means my scheme has its own invariances, genuinely different from the others', and I should work them out, because the invariances are what tell me how learning will behave.

Let me put all three on the same footing first so the comparison is honest. Each of them normalizes a summed input a_i through two scalars and then restores a gain and bias:

  h_i = f( (g_i / sigma_i)(a_i - mu_i) + b_i ),

where for the batch and layer recipes mu, sigma are their respective means and standard deviations, and for the weight reparameterization mu_i = 0, sigma_i = ||w_i||. Now I ask, for each, which transformations of the weights or the data leave the output untouched.

Start with positively rescaling a single neuron's incoming weight vector, w_i -> delta w_i with delta > 0. Under the batch recipe and the weight reparameterization, the statistics for neuron i are functions of w_i alone (its pre-activation across the batch, or its norm), so a_i, mu_i and sigma_i all scale by delta together, and (a_i - mu_i)/sigma_i is unchanged. But under my layer recipe, mu and sigma are pooled over *all* the units of the layer, so scaling one row's weights changes that row's a_i while the shared mu and sigma are also affected by the other rows; the ratio shifts. So I am *not* invariant to rescaling a single weight vector. That sounds like a loss, but watch what I am invariant to instead. Take the whole weight matrix and both positively scale it and shift every incoming weight by the same vector: W' = delta W + 1 gamma^T, i.e. every row becomes delta w_i + gamma. Then for input x,

  a' = W' x = delta W x + 1 (gamma^T x),

so every component a'_i = delta a_i + (gamma^T x), where gamma^T x is the *same scalar added to every unit*. The layer mean is mu' = (1/H) sum_i (delta a_i + gamma^T x) = delta mu + gamma^T x, and the centered values are a'_i - mu' = delta(a_i - mu) — the common additive term gamma^T x should cancel because it's identical across units, and it drops out of the difference from the mean. The spread is sigma' = delta sigma. So

  (a'_i - mu') / sigma' = delta(a_i - mu) / (delta sigma) = (a_i - mu)/sigma,

and the output should be identical: h' = h. This is the kind of thing I keep getting wrong by dropping a factor, so let me actually instantiate it before I trust it. Take H = 5 units, D = 4 inputs, random W, x, gain g and bias b, and compute h = LN(Wx) with the bare standardize-then-affine. Now form W' = delta·W + 1·gamma^T with delta = 2.3 and a random shift gamma, and recompute h' = LN(W'x). The largest component-wise discrepancy max_i |h'_i - h_i| comes out at 9.5e-7 — floating-point zero. So the invariance is real, not an algebra slip. As a control I scale only the *first* row, W_0 -> delta·W_0, leaving the rest; now max_i |h_i - h_i^orig| = 0.48, decidedly nonzero — confirming the asymmetry I claimed: the whole-matrix move is invariant, the single-row move is not. So layer normalization is invariant to scaling the entire weight matrix and to shifting all incoming weights by a constant — invariances the other two don't have (re-centering the weights in particular). And note this only works because I normalize *after* the weights; if I normalized the input before multiplying by the weights, none of this would hold.

Now the data side. Positively rescale the whole dataset by delta. Under the batch and layer recipes, every a_i scales by delta and the data-dependent statistics scale by delta too, so the ratio cancels. Under the weight reparameterization it does not cancel, because sigma_i = ||w_i|| stays fixed while w_i^T x scales. But here's one only the layer recipe gets. Rescale a *single training case*, x' = delta x with delta > 0. Since my mu and sigma for this example are computed from this example's own summed inputs, they scale with it:

  h_i' = f( (g_i/sigma')(w_i^T (delta x) - mu') + b_i ) = f( (g_i/(delta sigma))(delta w_i^T x - delta mu) + b_i ) = h_i.

so per-example rescaling should wash out completely. I'll check it in the same little setup: take the same W, x, g, b and recompute LN(W·(delta x)) with delta = 2.3. The discrepancy from the original is again 9.5e-7 — gone. Now the contrasting case I claimed *fails*: re-center a single case, x -> x + c for a random constant vector c. Here mu and sigma shift by w_i^T c terms that differ per unit and don't share a common scale, so I'd expect a real change; numerically max_i |h_i - h_i^orig| = 0.56, nonzero, as predicted. So the per-example rescaling invariance holds and the per-example recentering does not — the two experiments separate exactly along the line the algebra drew. The batch recipe can't get the rescaling invariance because its statistics are pooled over data, and the weight recipe can't because its normalizer is tied to ||w_i|| rather than to this case's activation scale. Conversely, re-centering the whole dataset (shift x by a constant c) is invariant under the batch recipe — it subtracts the data mean, so a per-neuron shift w_i^T c gets removed — but not under mine, for the same reason the per-example recenter just failed. So the invariance tables genuinely differ: layer normalization trades single-weight-vector and dataset-recentering invariance for whole-matrix-recentering and per-example-rescaling invariance. That per-example invariance is a real asset for the inner hidden layers, where I don't want a particular example's overall input magnitude to matter.

Knowing which functions are equal isn't enough, though. Two parameterizations can express the same function and yet *train* completely differently, because learning lives in the geometry of the parameter space, not just the function it represents. So I want the local metric: how much does a small parameter step actually move the model's output? For a model whose output is a distribution P(y|x;theta), the honest way to measure "how much did the output move" is the KL divergence between the old and new output distributions, and that makes parameter space a Riemannian manifold. To second order this distance is a quadratic form,

  ds^2 = KL[P(y|x;theta) || P(y|x;theta+delta)] ≈ (1/2) delta^T F(theta) delta,
  F(theta) = E_{x,y}[ (∂ log P/∂theta)(∂ log P/∂theta)^T ],

with F the Fisher information matrix — the local metric tensor. F tells me, for a unit-size parameter step in a given direction, how far the function actually travels. If I can read off how normalization reshapes F, I'll know how it reshapes learning.

To get something tractable, work with a generalized linear model, since a single neuron is essentially one, and a layer of them with a block-diagonal approximation to F (one block per neuron) extends the conclusion to deep nets. A GLM with summed input a = w^T x and bias b has log-likelihood

  log P(y|x; w, b) = ((a + b) y - eta(a + b))/phi + c(y, phi),

with mean E[y|x] = f(a + b) and variance Var[y|x] = phi f'(a + b), where f is the transfer function (the analog of the nonlinearity), eta the log-partition term, phi a dispersion constant. Stack H GLM outputs into a vector output Y = [y_1,...,y_H] with weight matrix W (rows w_i) and bias vector b, parameters theta = vec([W, b]^T), and allow their conditional covariance to be the full matrix Cov[Y|x]. Working through the second derivative of the log-likelihood, the Fisher matrix is the expected Kronecker product of the input features with the output covariance,

  F(theta) = E_x[ Cov[Y|x]/phi^2 ⊗ [[x x^T, x], [x^T, 1]] ].

Now build the *normalized* GLM by passing each summed input through the standardizing scalars mu_i, sigma_i and adding a learned gain, so the parameters become theta = vec([W, b, g]^T). I need the derivative with respect to w_i carefully, because this is where the normalization can hide a dropped factor. Let z_i = (a_i - mu_i)/sigma_i. A small move in w_i changes z_i by

  dz_i = (1/sigma_i) chi_i^T dw_i,
  chi_i = x - ∂mu_i/∂w_i - ((a_i - mu_i)/sigma_i) ∂sigma_i/∂w_i.

The raw input x is replaced by chi_i because w_i affects the normalized input in three ways at once: directly through a_i, through the mean, and through the standard deviation. With coordinates ordered as (w_i, b_i, g_i) against (w_j, b_j, g_j), the (i,j) Fisher block is

  F-bar_ij = E_x[ Cov[y_i, y_j | x]/phi^2 ·
    [[(g_i g_j)/(sigma_i sigma_j) chi_i chi_j^T,
      (g_i/sigma_i) chi_i,
      (g_i (a_j - mu_j))/(sigma_i sigma_j) chi_i],
     [(g_j/sigma_j) chi_j^T,
      1,
      (a_j - mu_j)/sigma_j],
     [(g_j (a_i - mu_i))/(sigma_i sigma_j) chi_j^T,
      (a_i - mu_i)/sigma_i,
      ((a_i - mu_i)(a_j - mu_j))/(sigma_i sigma_j)]] ].

Now the scaling effect has to be stated at the right level. The derivative of the normalized activation with respect to w_i carries one factor g_i/sigma_i. If an invariant positive rescaling doubles the relevant weight scale, sigma_i doubles too. A one-sided sensitivity involving that w_i direction halves; a mixed Fisher block involving one scaled side also halves; the pure w_i-w_i diagonal quadratic term for a fixed absolute dw_i has two such factors and drops to one quarter, provided chi_i itself is unchanged by the invariant rescaling. Either way, the conclusion is the same and the square matters: as the norm grows, the same Euclidean parameter update moves the function less. The norm has quietly become a per-direction step-size control. It is harder to rotate a large-norm weight vector because a fixed optimizer step is a smaller angular change, and the normalized metric damps the output movement as the scale grows. The batch and weight recipes get this through invariance to scaling w_i; the layer recipe gets the corresponding effect through invariance to scaling the whole matrix.

The zero directions should show up in this same metric. For the batch recipe, the rescaling direction gives w_i^T chi_i = a_i - mu_i - ((a_i - mu_i)/sigma_i) w_i^T(∂sigma_i/∂w_i). For this to vanish I need w_i^T(∂sigma_i/∂w_i) = sigma_i, which is the claim that sigma_i is homogeneous of degree one in w_i (Euler's theorem). That's worth checking rather than assuming: sigma_i is sqrt(E_x[(w_i^T x - mu_i)^2]), and scaling w_i by t scales the centered pre-activation by t, hence sigma_i by t — degree one. Numerically, with a random w over 200 sampled inputs, autodiff gives w_i^T(∂sigma_i/∂w_i) = 1.0439 against sigma_i = 1.0439, agreeing to five digits, so w_i^T chi_i = (a_i - mu_i) - (a_i - mu_i) = 0. For the weight reparameterization, chi_i = x - (a_i/sigma_i^2) w_i with sigma_i = ||w_i||, so w_i^T chi_i = w_i^T x - (a_i/||w_i||^2)(w_i^T w_i) = a_i - a_i = 0; plugging random w, x in gives -1.5e-7, zero to floating point. For the layer recipe, the single-row version is not a zero direction, which matches the invariance table, but summing the blocks over the whole-matrix scaling direction vec([W,0,0]^T) gives zero, and over the pure recentering direction vec([1 gamma^T,0,0]^T) gives zero too — the same two directions whose function-level invariance I just measured at 1e-6 above. So the metric's flat directions line up with the function-level invariances; the two pictures are consistent.

The gain direction is the explicit handle on the magnitude of the incoming weights, so I want to know whether learning that magnitude is well-behaved. Project a pure gain update delta_g, with no change to W or b, into the metric. Under the batch recipe mu_i and sigma_i are data-distribution constants, so the standardized activations (a_i - mu_i)/sigma_i are zero-mean and unit-variance over the data and the gain block of F-bar collapses to

  ds^2 = (1/2) delta_g^T E_x[ Cov[Y|x]/phi^2 ] delta_g,

which depends *only* on the output covariance -- the prediction error -- and not at all on the scale of the input or of the weights. Under the layer recipe the same projection gives

  ds^2 = (1/2) delta_g^T (1/phi^2) E_x[ {Cov(y_i, y_j | x) · (a_i - mu)(a_j - mu)/sigma^2}_{ij} ] delta_g,

i.e. the same prediction-error covariance but weighted by the normalized activations (a_i - mu)/sigma, scale-free quantities with unit layer variance. For the weight reparameterization, the explicit gain coordinate instead sees the raw projection divided by the weight norm,

  ds^2 = (1/2) delta_g^T (1/phi^2) E_x[ {Cov(y_i, y_j | x) · a_i a_j/(||w_i|| ||w_j||)}_{ij} ] delta_g,

so it is stable to rescaling w_i but still carries the input scale. In the *un-normalized* model the equivalent magnitude move is not an explicit gain coordinate at all; I have to push the weight along its own direction, projecting the gain update as delta_gi · w_i/||w_i||, and the metric comes out

  ds^2 = (delta_gi delta_gj)/(2 phi^2) · E_x[ Cov(y_i, y_j | x) · a_i a_j/(||w_i|| ||w_j||) ],

which depends on a_i a_j = (w_i^T x)(w_j^T x) -- that is, on the raw input scale and the current weights. So in the plain model, how far a magnitude update moves the function depends on how big the inputs and weights happen to be; in the batch recipe it depends on prediction error alone, and in the layer recipe it depends on prediction error times scale-free normalized activations. Learning the magnitude of the incoming weights is therefore far more robust to input and parameter scaling in the batch and layer recipes; the geometry of magnitude learning is stabilized, which is the other half of why these schemes train faster and more reliably.

Now let me make this real as code, and pin down the few remaining choices by what would break otherwise. The core operator standardizes a vector along the trailing feature axes and restores a per-feature affine. I'll fold the variance computation as E[z^2] - E[z]^2 for one pass, and put eps inside the square root so a near-constant layer (variance near zero) doesn't blow up. I'll initialize gain to 1 and bias to 0, so at the start the affine is the identity on the standardized signal and normalization doesn't disrupt the initial dynamics; this is also why I don't need the tiny 0.1 gain that the recurrent batch scheme demanded.

```python
from typing import List, Union

import torch
from torch import Size, nn


class LayerNorm(nn.Module):
    # Standardize over the trailing feature axes of a single example; restore
    # a learned per-feature gain and bias when elementwise_affine is enabled.
    def __init__(self, normalized_shape: Union[int, List[int], Size], *,
                 eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = torch.Size([normalized_shape])
        elif isinstance(normalized_shape, list):
            normalized_shape = torch.Size(normalized_shape)
        assert isinstance(normalized_shape, torch.Size)

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.gain = nn.Parameter(torch.ones(normalized_shape))   # init 1
            self.bias = nn.Parameter(torch.zeros(normalized_shape))  # init 0

    def forward(self, x: torch.Tensor):
        assert self.normalized_shape == x.shape[-len(self.normalized_shape):]
        dims = [-(i + 1) for i in range(len(self.normalized_shape))]
        mean = x.mean(dim=dims, keepdim=True)
        mean_x2 = (x ** 2).mean(dim=dims, keepdim=True)
        var = mean_x2 - mean ** 2
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            x_norm = self.gain * x_norm + self.bias
        return x_norm
```

Let me trace this on a concrete input to check the three properties I'm relying on, rather than just trusting the code. Feed a batch of two rows with H = 4: row [1, 2, 3, 4] and row [10, 10, 10, 10], gain = 1, bias = 0. Row one standardizes to [-1.3416, -0.4472, 0.4472, 1.3416]; its mean is 0.0 and its population std is 1.0 to five digits — so the operator does produce a zero-mean, unit-variance row, as intended. Row two is constant, so its variance is zero; the eps inside the sqrt is what saves it, and the output is [0, 0, 0, 0] rather than a NaN — exactly the degenerate-layer case I put eps there to handle. Now the property the whole motivation rested on: run the *same* first row alone as a batch of one. The output matches the row-from-the-batch-of-two to floating point — no batch statistic entered, so batch size 1 and batch size 2 compute the identical function on that example, which is the train/test consistency I claimed and couldn't get from the batch recipe. As a last check I compare against `torch.nn.LayerNorm(4)` on the same input; the outputs agree to 1e-5, so this is the standard operator and not an idiosyncratic variant.

For the recurrent cell I won't normalize the summed input a^t = W_h h^{t-1} + W_x x^t as one lump. The recurrent contribution W_h h^{t-1} and the input contribution W_x x^t generally live at different scales and distributions, and if I standardize their sum jointly the larger one dominates the shared mu, sigma and drowns out the smaller. Standardizing each contribution on its own keeps both at a controlled scale before they meet, so I apply a separate LayerNorm to each and only then add the bias:

```python
class LayerNormLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_h = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.W_x = nn.Linear(input_size, 4 * hidden_size, bias=False)
        self.b = nn.Parameter(torch.zeros(4 * hidden_size))
        self.ln_h = LayerNorm(4 * hidden_size)   # normalize recurrent contribution
        self.ln_x = LayerNorm(4 * hidden_size)   # normalize input contribution
        self.ln_c = LayerNorm(hidden_size)       # normalize cell before output tanh

    def forward(self, x_t, state):
        h_prev, c_prev = state
        # separate LN on each contribution, then add bias -> stable scale per step
        gates = self.ln_h(self.W_h(h_prev)) + self.ln_x(self.W_x(x_t)) + self.b
        f, i, o, g = gates.chunk(4, dim=-1)
        c_t = torch.sigmoid(f) * c_prev + torch.sigmoid(i) * torch.tanh(g)
        h_t = torch.sigmoid(o) * torch.tanh(self.ln_c(c_t))   # invariance to rescaling a^t -> stable dynamics
        return h_t, (h_t, c_t)
```

One more boundary the geometry tells me to respect. The output layer that produces logits before a softmax should be left alone — there, the *scale* of the logits is the prediction confidence, and layer normalization is by construction scale-invariant, so applying it to the logits would erase exactly the information the classifier needs. So in a feed-forward classifier I normalize the hidden layers and skip the final softmax layer. And one place I'd expect trouble: convolutional layers. My whole justification rested on the units of a layer contributing similarly so that a single layer-wide mean and variance summarize them; in a conv layer the units near the image boundary are rarely active and carry statistics unlike the rest, so a single layer-wide statistic mixes very different populations — the assumption that makes the layer-wide moment meaningful fails there, and I'd want fully-connected layers, where it holds, to be the natural home.

So the causal chain, end to end: I want faster optimization by stabilizing each layer's input distribution; the batch recipe does that but pools statistics over the batch dimension, which makes it batch-size dependent, train/test-inconsistent, and ill-defined per time-step in RNNs; the correlated, layer-wide nature of the covariate shift means I can capture the same drift by reducing over the units of a layer for a single example instead of over the batch; that transpose removes the batch from the statistics entirely, so it's identical at train and test, works at batch size 1, and applies unchanged at every recurrent step; for RNNs it makes the step invariant to positive rescaling of the whole summed-input vector, killing the magnitude drift that causes exploding/vanishing dynamics; the invariance analysis shows this is not a reparameterization and has its own invariances (whole-matrix rescale/recenter, per-example rescale); and the Fisher geometry shows sigma growing with an invariant weight scale damps the Jacobian and the Fisher quadratic, while magnitude learning depends on prediction error and normalized activations rather than raw input scale -- so it trains faster and more stably; finally, gain=1/bias=0 initialization, separate normalization of the recurrent and input contributions, and leaving the logit layer un-normalized are the choices that keep all of this intact.
