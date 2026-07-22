OK, let me think this through from scratch. I want one unsupervised recipe that pulls high-level structure out of high-dimensional sequences — speech waveforms, images, text, streams of agent frames — with no labels, and I want it generic enough that the same mechanism works across all of them. The thing I keep coming back to as the unsupervised signal is prediction: make the model predict the future, or the missing part, or the surrounding context, and to do that well it has to discover the regularities in the data, which is exactly a representation. Predictive coding has said this since the 1950s — Elias, Atal and Schroeder predicting the next speech sample for compression — and word2vec is a modern echo of it, predicting neighboring words. So prediction is the lever. The question is what, exactly, to predict, and how to score the prediction.

The naive instantiation is: build a conditional generative model of the future observation, p(x_{t+k} | c_t), where c_t is some summary of the past, and maximize the likelihood of the real future. That's what WaveNet does for audio, PixelCNN/PixelRNN for images, skip-thought for sentences with an LSTM decoder. And the moment I write it down I feel the problem. To put a likelihood on x_{t+k} I have to generate x_{t+k} — every sample of the waveform, every pixel, every word. But a high-dimensional observation carries a huge number of bits, and the part I actually want is tiny: the phoneme, the object class, the sentiment. An image is thousands of bits; "which of 1024 classes" is ten bits. If my loss scores the reconstruction of x_{t+k}, then almost all of the loss — and almost all of the model's capacity — goes into the thousands of bits of texture and noise that I don't care about. Worse, a powerful conditional generative model can model x_{t+k} so well on its own that it learns to mostly ignore c_t; the context, the thing whose representation I'm trying to learn, becomes a vestigial input. And unimodal losses like MSE or per-pixel cross-entropy are a bad fit when the true conditional is multimodal, so I'd be pushed toward an even heavier decoder. Modeling p(x|c) directly is just the wrong objective for the thing I want.

So back up. What do I actually want from the pair (context, future)? Not the future itself. I want what they *share* — the part of the future that is *predictable* from the present, because that shared part is precisely the slowly-varying global factor I'm after. The local detail isn't shared (it's noise, it's not predictable far ahead); the phoneme, the object, the topic — those persist, those are shared. So I should be maximizing the information that c_t carries about x_{t+k}. That's mutual information:

  I(x; c) = Σ_{x,c} p(x,c) log [ p(x|c) / p(x) ].

Now stare at the term inside the log. It is not a density. It is a *ratio* of densities, p(x|c)/p(x). The thing that measures shared information is a density ratio, and a ratio is a much smaller, much less detailed object than either density on its own. p(x) and p(x|c) each have to account for all the texture of x; their ratio cancels everything that's common between "x in general" and "x given this context" and keeps only what the context changes. So if I could learn the ratio without ever learning p(x) or p(x|c) separately, I'd be capturing the shared information and paying nothing for the low-level detail — that seems like the object worth chasing. There is a free bonus: if I encode x into a representation, the MI between encoded representations is upper-bounded by the MI between the raw signals, so squeezing MI through a compact encoding is a principled way to keep the shared bits and drop the rest.

Let me set up the architecture around that. Encode each observation into a latent, z_t = g_enc(x_t) — a non-linear encoder, and it can run at lower temporal resolution than the input, which is good, it's already throwing away local detail. Summarize the past into a context with an autoregressive model, c_t = g_ar(z_{≤t}); a GRU is the natural choice, it rolls the prefix up into one vector. And now, instead of a generative head p(x_{t+k}|c_t), I want a head that models the density ratio:

  f_k(x_{t+k}, c_t) ∝ p(x_{t+k} | c_t) / p(x_{t+k}).

The "∝" is doing real work — I'm explicitly allowing f to be *unnormalized*; it doesn't have to integrate to one, because a ratio that's right up to a multiplicative constant is fine for everything I'll need. So f can be any positive score. What's the simplest score that takes the future and the context and returns a positive number? Encode the future into z_{t+k} with the same encoder, and take a log-bilinear form: f_k = exp(z_{t+k}^T W_k c_t), with a separate matrix W_k for each look-ahead step k, since "predict 1 step" and "predict 12 steps" are different relations. Exponential keeps it positive; the bilinear inside is a learned compatibility between the future latent and a linear projection of the context. Notice what I've avoided: I encode x_{t+k} into z_{t+k}, I never reconstruct it. The model is relieved of the entire burden of generating high-dimensional output.

But I've moved the difficulty, not removed it. I can't evaluate p(x) or p(x|c) — that's the whole point, they're intractable — so how do I train f to become the ratio without ever touching those densities? I can't normalize f either. This is exactly the wall that noise-contrastive estimation and importance sampling hit for language models, and the way they got around it is the clue: don't compute the normalization, *sample* against it. NCE (Gutmann and Hyvärinen) fits an unnormalized model by training a classifier to discriminate real data from noise; the partition function never appears, it's replaced by a discrimination task against samples. Importance sampling (Bengio and Senécal) approximates the softmax denominator over a huge vocabulary with a few samples from a proposal. word2vec's negative sampling is the same trick: score the true neighbor against random negatives. The common shape: *compare the real target to randomly drawn alternatives.* I have samples even though I don't have densities — I have real futures (from the joint p(x,c)) and I have random observations (from the marginal p(x)). So make the task: among a set of candidates, one of which is the genuine future and the rest are random draws, pick the real one.

So build a set X = {x_1, …, x_N} containing exactly one positive sample drawn from p(x_{t+k}|c_t) — the actual future paired with this context — and N−1 negatives drawn from the proposal p(x_{t+k}), the marginal. Score every element with f_k and ask the model to point at the positive. The natural loss is the categorical cross-entropy of getting that classification right, with the softmax over scores as the model's guess:

  L_N = − E_X [ log ( f_k(x_{t+k}, c_t) / Σ_{x_j ∈ X} f_k(x_j, c_t) ) ].

This is the loss. No densities, no normalization constant, just a softmax over a positive and N−1 negatives, trained end to end through the encoder and the GRU. I'll call it InfoNCE because it's NCE in service of an information objective.

The loss is only useful if its minimizer is the ratio I need. It is the cross-entropy of identifying the positive slot, so its minimizer is the model whose softmax equals the *true* posterior probability that each slot is the positive one. Let me compute that true posterior. Write the generative story of how X was made: a slot is chosen to be the positive (uniformly), that slot's element is drawn from p(·|c), and every other slot's element is drawn from the marginal p(·). The probability that the configuration of values came about with slot i being the positive is then proportional to the likelihood of that assignment:

  p(d=i | X, c) = [ p(x_i|c) ∏_{l≠i} p(x_l) ] / [ Σ_{j=1}^N p(x_j|c) ∏_{l≠j} p(x_l) ].

Every term in the numerator and in each summand of the denominator contains the product of marginals; divide top and bottom by ∏_{l} p(x_l). In the numerator, p(x_i|c) ∏_{l≠i} p(x_l) divided by ∏_l p(x_l) leaves p(x_i|c)/p(x_i). Same for each denominator term. So

  p(d=i | X, c) = [ p(x_i|c)/p(x_i) ] / Σ_{j=1}^N [ p(x_j|c)/p(x_j) ].

So the true posterior is a softmax over the density ratios p(x_j|c)/p(x_j). My model's softmax is over f_k(x_j, c). Cross-entropy is minimized when the model's posterior equals the true posterior, so the optimum is f_k(x, c) ∝ p(x|c)/p(x) — the ratio I was after — and it fell out *independent of N*, the number of negatives. Ratio estimation is the exact optimum of the contrastive task, not a heuristic surrogate for it, and that holds for any number of negatives.

I should pin this down on something I can actually compute, because the algebra has been frictionless and that always makes me suspicious. Take a tiny toy where every density is known: c and x each range over {0,1,2,3}, c is uniform, and the future agrees with the context, x=c, with probability a=0.7 and is otherwise uniform over the other three values. By symmetry p(x) is uniform (0.25 each), and the exact ratio is r(x,c)=p(x|c)/p(x). The true mutual information of this joint, summed by hand over the 16 cells, is 0.4459 nats. That number is my ground truth to check the whole construction against.

The other thing I need is the mutual-information direction: lowering L_N should raise something that cannot exceed I(x;c). Plug the optimal f back into the loss. At the optimum f = r := p(x|c)/p(x). Split the set X into the one positive and the negatives X_neg. The loss at the optimum is

  L_N^opt = − E_X log [ r_pos / ( r_pos + Σ_{x_j ∈ X_neg} r_j ) ],

where r_pos = p(x_{t+k}|c)/p(x_{t+k}) is the ratio for the true future. Flip the fraction inside the log:

  L_N^opt = E_X log [ ( r_pos + Σ_{neg} r_j ) / r_pos ] = E_X log [ 1 + (1/r_pos) Σ_{neg} r_j ]
          = E_X log [ 1 + ( p(x_{t+k}) / p(x_{t+k}|c) ) Σ_{x_j ∈ X_neg} r_j ].

I need to handle Σ_{neg} r_j. The negatives are drawn from the marginal p(x), so look at the expectation of one of them:

  E_{x_j ∼ p(x)} [ r_j ] = E_{x_j ∼ p(x)} [ p(x_j|c)/p(x_j) ] = Σ_x p(x) · p(x|c)/p(x) = Σ_x p(x|c) = 1.

On the toy this is the easiest thing to check first: for each of the four contexts c, Σ_x p(x) r(x,c) comes out to 1.000000, and a half-million marginal draws for c=0 average r to 1.0005. So each negative's ratio really does average to exactly 1, and the sum over N−1 negatives is approximately (N−1):

  L_N^opt ≈ E_X log [ 1 + ( p(x_{t+k}) / p(x_{t+k}|c) ) (N−1) ].

That expectation check tells me why a large negative set behaves like N−1, but a tempting shortcut breaks: a particular joint sample can have p(x|c) smaller than p(x). Mutual information is an average log-ratio, not a guarantee that every paired x is more likely under its own context. I need to keep the random denominator and compare log N − L_N^opt directly to I:

  log N − L_N^opt
    = E_X log [ N r_pos / (r_pos + Σ_{neg} r_j) ]
    = E_{(x,c)} log r(x,c)
      − E_X log [ (1/N)(r_pos + Σ_{neg} r_j) ]
    = I(x_{t+k}; c_t)
      − E_X log [ (1/N)(r_pos + Σ_{neg} r_j) ].

Now the last expectation has the right sign. Let m(X,c)=p(c)∏_{j=1}^N p(x_j) be the distribution where all candidates are marginal samples, and let

  q(X,c)=p(c) (1/N) Σ_{i=1}^N p(x_i|c) ∏_{j≠i} p(x_j)

be the symmetric distribution where one uniformly chosen slot is drawn from p(x|c). The average ratio in the denominator is exactly q(X,c)/m(X,c). The function log[(1/N)Σ_j r_j] is symmetric in the candidate slots, so taking its expectation under "slot 1 is positive" is the same as taking it under the uniform-positive mixture q. Therefore

  E_X log [ (1/N)(r_pos + Σ_{neg} r_j) ]
    = E_q log [ q(X,c) / m(X,c) ]
    = KL(q || m) ≥ 0.

So the denominator term can only subtract a nonnegative amount from I, and the bound follows:

  log N − L_N^opt ≤ I(x_{t+k}; c_t),
  hence I(x_{t+k}; c_t) ≥ log N − L_N^opt.

A suboptimal f only makes the cross-entropy larger than L_N^opt, so log N − L_N is even smaller and remains a valid lower bound during training. The ceiling is log N nats because the training task asks for the identity of one correct slot among N candidates; if the true shared information is larger than that, the classifier can saturate and the bound cannot certify the extra bits. Larger N should raise that ceiling and make the sampled denominator behave more like its marginal expectation, so the batch size is not just an engineering detail — it ought to control how much mutual information this objective can expose.

I'd rather not take "ought to" on faith, so I run the toy through the actual contrastive sampling with the optimal critic f=r and measure L_N^opt by Monte Carlo: draw c, draw the positive from p(·|c), draw N−1 negatives from p(x), and average −log[r_pos/(r_pos+Σ r_neg)]. Then compare log N − L_N^opt against the known I=0.4459:

  N= 2   L_N^opt=0.502   log N − L_N^opt = 0.191
  N= 4   L_N^opt=1.076   log N − L_N^opt = 0.310
  N= 8   L_N^opt=1.701   log N − L_N^opt = 0.378
  N=16   L_N^opt=2.361   log N − L_N^opt = 0.412

Three things land here. First, log N − L_N^opt ≤ I = 0.4459 at every N — the bound holds, never crossing the true MI. Second, it climbs monotonically toward I as N grows (0.19 → 0.31 → 0.38 → 0.41), so more negatives really do tighten the estimate, exactly as the KL term predicts. Third, the loosely-binding constraint here is *not* the log N ceiling: even at N=2 the ceiling log 2 = 0.69 already sits above I, yet the bound is only 0.19, so the slack is entirely the nonnegative KL(q‖m) term, which shrinks as N rises. That matches the derivation cell for cell, and it tells me the practical knob is N: push it up and the gap to the true MI closes.

Let me sanity-check this against the variational MI estimators floating around, because if it disagrees with them something is wrong. Write f(x,c) = e^{F(x,c)} for a critic F. Then

  E_X [ log ( f(x,c) / Σ_{x_j ∈ X} f(x_j,c) ) ]
    = E_{(x,c)}[F(x,c)] − E_{(x,c)}[ log Σ_{x_j ∈ X} e^{F(x_j,c)} ]
    = E_{(x,c)}[F(x,c)] − E_{(x,c)}[ log ( e^{F(x,c)} + Σ_{x_j ∈ X_neg} e^{F(x_j,c)} ) ].

Drop the positive term e^{F(x,c)} inside the log — that only decreases the argument, so the whole expression goes *up*:

    ≤ E_{(x,c)}[F(x,c)] − E_c[ log Σ_{x_j ∈ X_neg} e^{F(x_j,c)} ]
    = E_{(x,c)}[F(x,c)] − E_c[ log ( (1/(N−1)) Σ_{neg} e^{F(x_j,c)} ) + log(N−1) ].

The right-hand side is, up to the additive log(N−1), the Donsker–Varadhan / MINE estimator: E_{(x,c)}[F] minus the log of the average of e^F over the marginal. So InfoNCE is a *lower bound* on the MINE estimator — InfoNCE keeps the positive term inside the log-sum-exp, MINE does not. And that extra positive term is exactly what tames the variance: when the target is trivially predictable from the context (one step ahead, target overlapping the context), MINE's bare log-mean-exp over the marginal blows up and training goes unstable, while InfoNCE's denominator always contains the positive and stays bounded. So I'm trading a bit of tightness for a lot of stability — the right trade when the easy cases are the ones that wreck you.

One design point the bound makes concrete: predicting *many* steps ahead matters, and it's not optional. One step ahead, local smoothness already gives you the answer, the ratio is near-trivial, the task is too easy, log N − L_N certifies almost nothing about global structure. Pushing k out forces the model to find what's *actually* shared across a long horizon — the slow features, phoneme, speaker, object, storyline — because that's all that survives the distance. So I'll predict a whole block of future steps k = 1..K with the per-step W_k, and apply the contrastive loss at each.

Now the negatives. Where do the N−1 draws from the marginal p(x_{t+k}) come from, in practice? I don't want a separate sampler. The cleanest source is the minibatch itself: within a batch, every other example's true future is, relative to *this* context, a sample from the marginal. So I compute, for look-ahead k, the score between this context's prediction and every batch element's encoded future, and the diagonal of that score matrix is the positives while the off-diagonal entries are the in-batch negatives. The number of negatives N is then just the batch size — free, and it ties "use more negatives to tighten the bound" directly to "use a bigger batch."

Now the implementation can stay almost embarrassingly small. Encoder: a stack of strided 1-D convolutions straight on the 16 kHz waveform — five layers, strides [5,4,2,2,2], kernels [10,8,4,4,4], 512 channels, total downsampling 160, so one z vector per 10 ms, matching the phone-label rate. Autoregressive summary: a single-layer GRU over the prefix of z's, hidden size 256, whose output at time t is c_t. Predictions: for each step k a linear map W_k from the 256-dim context to the 512-dim latent space, W_k c_t, which is the model's predicted z_{t+k}. The score f_k = exp(z_{t+k}^T W_k c_t) is realized as a dot product that becomes the logit; the softmax-cross-entropy over the batch gives InfoNCE. I keep the implementation's Kaiming initialization for convolutional, linear, and GRU weights, with batch-normalization scale set to one and bias to zero.

Before I trust the in-batch shortcut I want to watch the index bookkeeping on a concrete tensor, because off-by-one in which axis is the positive would silently train garbage. Take batch B=3, latent dim 4. Stack the encoded true futures into a (3×4) matrix and the predictions W_k c_t into another (3×4) matrix, and form total = futures · predictions^T, a (3×3) score grid where total[i,j] is example i's real future dotted with example j's prediction. The positive for column j — prediction j — lives in row j, the diagonal; every off-diagonal entry in column j is some *other* example's future scored against prediction j, which is exactly a marginal draw, an in-batch negative. So the loss should take a softmax *down each column* (over the future index, dim 0) and read off the diagonal. I do that on a random-init grid and the diagonal of the column-wise log-softmax comes out as roughly [−0.81, −2.63, −0.99] — finite, no NaN, and the argmax down each column is [2,2,2] rather than [0,1,2], i.e. the untrained model gets every slot wrong, which is the right behaviour before any learning. Then I replace the grid with a strong diagonal (10·I) to simulate a trained critic that nails its own future: the InfoNCE term collapses to 0.0001 and the per-column argmax becomes correct, while log B = log 3 = 1.0986 is the ceiling it is pushing down from. The axis is right, the positive is on the diagonal, and the loss moves in the direction it should.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SequenceRepresentation(nn.Module):
    def __init__(self, timestep, batch_size, seq_len):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.timestep = timestep

        # g_enc: strided conv stack on raw waveform -> z_t, downsample factor 160
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=3, bias=False),
            nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512), nn.ReLU(inplace=True),
        )
        # g_ar: autoregressive summary of z_{<=t} -> context c_t
        self.gru = nn.GRU(512, 256, num_layers=1, bidirectional=False, batch_first=True)
        # one linear "predictor" W_k per look-ahead step: c_t (256) -> predicted z (512)
        self.Wk = nn.ModuleList([nn.Linear(256, 512) for _ in range(timestep)])
        self.softmax = nn.Softmax(dim=0)
        self.lsoftmax = nn.LogSoftmax(dim=0)
        self._init_recurrent_weights()
        self.apply(self._weights_init)

    def _init_recurrent_weights(self):
        for names in self.gru._all_weights:
            for name in names:
                if "weight" in name:
                    nn.init.kaiming_normal_(
                        getattr(self.gru, name),
                        mode="fan_out",
                        nonlinearity="relu",
                    )

    @staticmethod
    def _weights_init(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def init_hidden(self, batch_size, device=None):
        return torch.zeros(1, batch_size, 256, device=device)

    def forward(self, x, hidden):
        batch = x.size(0)
        # pick a random anchor time t such that t+K still fits in the sequence
        encoded_steps = self.seq_len // 160
        t = torch.randint(encoded_steps - self.timestep, size=(1,), device=x.device).item()

        z = self.encoder(x)            # (B, 512, L)
        z = z.transpose(1, 2)          # (B, L, 512)

        # the true futures z_{t+1..t+K}, one row per look-ahead step
        encode_samples = x.new_empty((self.timestep, batch, 512))
        for k in np.arange(1, self.timestep + 1):
            encode_samples[k - 1] = z[:, t + k, :].view(batch, 512)

        # context: run the GRU over the prefix z_{<=t}, take its output at t
        forward_seq = z[:, :t + 1, :]
        output, hidden = self.gru(forward_seq, hidden)
        c_t = output[:, t, :].view(batch, 256)

        # predicted future latents W_k c_t for each step
        pred = x.new_empty((self.timestep, batch, 512))
        for k in np.arange(self.timestep):
            pred[k] = self.Wk[k](c_t)

        nce = 0.0
        correct = None
        for k in np.arange(self.timestep):
            # score matrix: row i (true future of example i) vs col j (prediction j).
            # f_k = exp(z_{t+k}^T W_k c_t); 'total' holds the logits z^T (W_k c_t).
            total = torch.mm(encode_samples[k], pred[k].t())   # (B, B)
            # diagonal = positive (real future paired with its own context);
            # off-diagonal = in-batch negatives, i.e. draws from the marginal.
            # InfoNCE = -mean over the diagonal of the log-softmax down each column.
            correct = torch.sum(
                torch.eq(torch.argmax(self.softmax(total), dim=0),
                         torch.arange(0, batch, device=x.device))
            )
            nce += torch.sum(torch.diag(self.lsoftmax(total)))

        nce = -nce / (batch * self.timestep)   # average over negatives-set and over K
        # certified information is bounded by log(batch) nats: I >= log N - L_N
        # diagnostic accuracy is for the farthest predicted step
        accuracy = correct.float() / batch
        return accuracy, nce, hidden

    def extract(self, x, hidden):
        # downstream use: freeze and read off c_t (or z_t) as the representation
        z = self.encoder(x).transpose(1, 2)
        output, hidden = self.gru(z, hidden)
        return output, hidden


class LinearProbe(nn.Module):
    def __init__(self, num_classes, dim=256):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        return F.log_softmax(self.fc(x), dim=-1)


def train_step(model, batch, optimizer, hidden):
    optimizer.zero_grad()
    _, loss, hidden = model(batch, hidden)
    loss.backward()
    optimizer.step()
    return loss, hidden
```
