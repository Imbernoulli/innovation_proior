Let me start from what actually goes wrong when I forecast one of these long streams. I fit a recurrent net — a GRU, say — on years of air-quality or stock-factor data, it does fine on the stretch it trained on, and then on the held-out future it quietly gets worse. The cell itself is not the problem; a GRU finds nonlinear temporal dependence beautifully. The problem is the assumption baked into how I trained it: I minimized the average prediction error over the whole training stream as if the whole stream were one fixed distribution, and the future I am predicting is not drawn from that distribution. The mean, the variance, the correlations among the input variables drift as the series goes on — regimes, shocks, structural shifts — so the test stretch at the end has an input marginal unlike the bulk of what I trained on. Every classical model and every recurrent model I have shares exactly this I.I.D. assumption, and the drift violates it head on. So the whole game is: out of one long stream whose distribution changes over time, and no labels telling me when it changes, manufacture a forecaster that survives a future regime I have never seen.

Before I reach for an architecture let me get the structure of the problem straight, because if I name it precisely the tools I need might fall out of the name. What exactly is shifting? Write the joint law of inputs and target as P(x, y). Over the series the marginal P(x) changes with time. Is there a part that tends to persist? Decompose P(x, y) = P(y | x) P(x). If anything survives a regime change it has to be the conditional P(y | x), the law that maps a configuration of inputs to its outcome — because that is the thing I would actually call "the regularity I am trying to learn," as opposed to "which inputs happen to show up." In markets the factors P(x) swing wildly with the regime, but the economic regularity turning factors into returns is plausibly far more stable. In air quality the conditions P(x) vary by season and weather, but the physics turning conditions into pollutant level does not. So provisionally the thing worth carrying from past to future is P(y | x); the thing that betrays me is P(x). That is exactly the shape Shimodaira called covariate shift back in 2000 — P_train(x) != P_test(x) while P_train(y|x) = P_test(y|x) — and the classic fix there is to reweight the training loss by the density ratio P_test(x)/P_train(x). But that fix is dead on arrival for me twice over: it needs the test density, and my test is a genuinely unseen future regime I have no density for; and covariate shift as posed is for ordinary non-sequential data with a single fixed train/test pair, with no notion of *when* inside a stream the distribution turns over. My stream is not one shift, it is a sequence of shifts at unknown times.

So let me extend the covariate-shift picture to a stream. Suppose the training stream secretly decomposes into K consecutive periods D_1, ..., D_K, where within a period the distribution is constant, but across periods the input marginal differs while the conditional stays put: for i != j, P_{D_i}(x) != P_{D_j}(x) and P_{D_i}(y|x) = P_{D_j}(y|x). The unseen test period is one more such period — different P(x), same P(y|x). That is a covariate shift indexed by time rather than by a single train/test pair. If that picture is right, it tells me what to learn: across all these periods the conditional P(y|x) is shared, so the part of the data worth keeping is precisely that shared conditional, and the way to learn it robustly is to stop letting the differing marginals fool the model — to force the model to represent the periods by what is invariant across them. Two concrete obstacles, though, and I should not paper over either. First, nobody handed me the periods: I do not know K, I do not know the boundaries, the non-stationarity is unlabeled in time, and the space of possible splits is enormous. Second, even once I have periods, I need a training procedure that actually uses them to build a model robust to a future period — and recovering the periods badly will poison everything downstream. Let me take these one at a time.

Discovering the periods first. I have a single stream and I want to chop it into K segments along time. What objective should the chop optimize? My instinct, coming from clustering, is to make each segment internally homogeneous — group similar stretches together — so the segments are as *similar* to each other as possible after merging like with like. Let me follow that instinct and see if it serves the downstream goal... and it does the opposite of what I need. The downstream goal is a model robust to an unseen future distribution. If I split the stream into segments that are all distributionally *similar*, I have hidden the drift from the model — I have manufactured a nearly-stationary view of a non-stationary world, and a model trained on it learns nothing about coping with a regime gap. That is the wall. I want the chop to *surface* the drift, not bury it.

Turn it around then: split so the periods are as *dissimilar* as possible. Is there a principle that says that is the right thing, or is it just a hunch? There is, and it is the principle of maximum entropy. Jaynes' point is that when you are ignorant about something, you should commit to the least-committal hypothesis consistent with what you know — assume no structure beyond your constraints, do not pretend to knowledge you lack. I am ignorant about the test distribution; it is a future regime, unseen. The least-committal stance toward an unknown future is to refuse to bet that it resembles any particular past stretch — to instead train under the most diverse spread of distributions I can extract from the training stream, the *worst case* of cross-period divergence. A model that has been forced to find what is common across the most dissimilar periods has been trained for the hardest version of the transfer problem, and that is exactly the model most likely to generalize to a new regime. The non-I.I.D. generalization theory for sequences, Kuznetsov and Mohri, points the same way: diversity across the training distributions is not noise, it enters the bound — more diversity, handled, is a stronger guarantee. So the splitting objective is to *maximize* the average pairwise distribution distance among the periods, while also choosing how many periods to allow. Let me write it down. Pick a distance d between distributions; choose K under a cap K_0 and choose the boundary points to

  max_{0 < K <= K_0} max_{n_1, ..., n_K} (1/K) sum_{1 <= i != j <= K} d(D_i, D_j)

subject to each period being neither too short nor too long, Delta_1 < |D_i| < Delta_2, and the pieces covering the whole stream, sum_i |D_i| = n. The length constraints matter: a period of two points has a meaningless distribution, and one giant period plus slivers is a trivial cheat that maximizes nothing useful — Delta_1 and Delta_2 fence those off.

Now, can I solve this? In general, no — it is a combinatorial search over K and over all boundary placements, intractable, with no closed form. With a well-behaved metric I could in principle run dynamic programming over boundary positions, but on a stream with hundreds of thousands of points and a distribution distance to evaluate at every candidate cut, even DP is too slow at the scale I care about. So I drop to a greedy construction, which is the honest scalability compromise. Coarsen first: evenly pre-split the stream into N = 10 minimal units, so the only legal cut points are the 9 internal boundaries — that bounds the search and also enforces a minimum period length for free. Then consider K over a small set, {2, 3, 5, 7, 10}. For a given K, build the split greedily. Call the stream's start A and end B. For K = 2 I need one cut C among the 9 candidates; pick the C that maximizes d(S_AC, S_CB), the distance between the two halves. Freeze C. For K = 3, with C fixed, scan the remaining candidates for the next point D, again choosing the one that most increases the total pairwise distance among the now-three pieces. Same move for larger K: at each step, among the currently selected boundaries, sum the pairwise distances of the sub-segments they induce, and greedily add the candidate cut that pushes that sum highest, then remove it from the pool. If K is being tuned, repeat this construction over the candidate K values and select by the same diversity objective under the validation protocol.

Before I trust the "maximize" direction I want to actually watch it on a stream where I know the answer, because the whole TDC stage rests on the claim that the diverse split is the *useful* one. Let me build a toy: ten units, the first five drawn around mean 0 and the last five around mean 3 in two dimensions, so there is exactly one true regime boundary, at unit 5. Use the cheapest distance — Euclidean between segment means — and for K = 2 sweep all nine internal cuts, scoring each by the cross-period mean distance. The cut that *maximizes* that distance comes out at unit 5 (score 4.09), exactly the true regime boundary; the cut that *minimizes* it comes out at unit 9 (score 2.29), shaving off one tail unit and lumping the rest together so the shift is buried inside one giant period. That is the concrete payoff of the maximum-entropy argument: maximizing surfaces the real boundary and hands the matching stage two genuinely different regimes to reconcile, while minimizing hides the drift behind a near-stationary partition — precisely the failure I argued against. So the direction is right for the reason I thought, not just by assertion. The distance d here can be anything sensible — Euclidean, dynamic time warping, or a proper distribution distance — and I will come back to which.

Periods in hand, now the model. Start from the obvious objective and find where it breaks. I have K periods; train the recurrent net to minimize prediction error on all of them,

  L_pred(theta) = (1/K) sum_{j=1}^{K} (1/|D_j|) sum_{i=1}^{|D_j|} ell(y_i^j, M(x_i^j; theta)),

with ell the MSE for regression. Fine, but this learns only the predictive mapping inside each period; it does absolutely nothing to reduce the divergence *between* periods. The model can happily fit each regime in its own way and still fall apart on a new one. I need a term that pulls the periods' representations together, so the net is pushed to represent them by something invariant — the shared conditional. The domain-adaptation literature already built such terms: take a distribution distance d and add it as a regularizer that drives two domains' representations to look alike. The standard choices are all available to me. Maximum mean discrepancy maps each distribution to its mean embedding in a reproducing-kernel Hilbert space and measures the gap, MMD(P,Q) = ||mu_P - mu_Q||_H, with the empirical squared form

  (1/n^2) sum_{i,j} k(h_i^s, h_j^s) + (1/m^2) sum_{i,j} k(h_i^t, h_j^t) - (2/nm) sum_{i,j} k(h_i^s, h_j^t).

I want to know how cheaply I can evaluate this, because I will be calling it at every hidden state of every period pair. With a linear kernel k(a,b) = <a,b>, the double sum (1/n^2) sum_{i,j} <h_i, h_j> equals < (1/n) sum_i h_i, (1/n) sum_j h_j > = ||mean(H)||^2 by linearity of the inner product, and the cross term collapses the same way, so the whole estimator should reduce to ||mean(H^s) - mean(H^t)||^2 — just a distance of means, no Gram matrix. Let me confirm that is not wishful: take H^s of 5 rows and H^t of 7 rows in three dimensions, drawn with a 0.5 mean offset. Computing the full double-sum estimator gives 0.378620787..., and ||mean(H^s) - mean(H^t)||^2 gives 0.378620787... — they agree to machine precision. Good, so the linear case is free, while a universal RBF kernel keeps the full Gram form and senses all the moments. Or CORAL, aligning the covariances, d_coral = (1/(4 q^2)) ||C_s - C_t||_F^2. Or the domain-adversarial route, a small discriminator D trained to tell the periods apart while the features are trained to fool it via a gradient-reversal layer, with discrepancy d_adv = -(E[log D(h_s)] + E[log(1 - D(h_t))]). Or just cosine distance on the mean feature vectors. They are interchangeable plug-ins; the framework should not care which.

So the naive thing: for each pair of periods (D_i, D_j), run them through the RNN, take the final hidden state, and penalize d(h_i^V, h_j^V). Add it up over pairs, add it to L_pred. Let me actually picture what this does inside the recurrent net... and it throws away the very thing the recurrent net is for. The RNN does not produce one representation of a sequence; it produces a *trajectory* of hidden states h^1, h^2, ..., h^V, one per timestep, each carrying partial information about the sequence so far. The distribution shift between two periods is not a single fact about their endpoints; it plays out across the whole trajectory — early states, late states, all of them carry distributional information and all of them drift, possibly by different amounts at different points in the sequence. Matching only h^V aligns the summary and leaves every intermediate state unaligned, and the intermediate states are exactly the temporal structure I cared about. That is the wall: an output-only alignment, the kind the image-domain-adaptation methods use because a CNN has no per-step trajectory, leaves the recurrent net's distinctive feature unconstrained. I need to match the distributions at *every* hidden state, not just the last.

So match all of them. For a period pair, sum the distance over the V states:

  L_tdm(D_i, D_j; theta) = sum_{t=1}^{V} alpha_{i,j}^t d(h_i^t, h_j^t; theta).

But now I have written alpha_{i,j}^t in front, and I should say why, because the moment I sum over all states a question appears: should every state count equally? No — and I can see it without measuring anything. The states differ in how much they have drifted and in how much aligning them actually helps the final prediction; an early state dominated by the input embedding and a late state that has integrated the whole sequence are not equally informative about the cross-period gap. If I weight them all by 1/V I am asserting they are equally important, which is just the equal-weight prior I have no reason to hold. So introduce an importance vector alpha in R^V over the states, one such vector per period pair, that learns the relative importance of each state in the matching. That is what the alpha_{i,j}^t are: the per-state, per-pair weights, which I will normalize to sum to one so they are a clean convex weighting and the overall regularizer strength stays controlled by a single trade-off knob. Folding this together with the prediction loss, the objective for one RNN layer is

  L(theta, alpha) = L_pred(theta) + lambda * (2/(K(K-1))) sum_{1 <= i < j <= K} L_tdm(D_i, D_j; theta, alpha),

where the 2/(K(K-1)) should be the average over the unordered period pairs, and lambda trades off matching against fitting. Let me make sure that prefactor is right and not off by a factor of two, since the sum runs over i < j: the number of unordered pairs of K periods is K(K-1)/2, so the averaging weight has to be its reciprocal, 2/(K(K-1)). Checking across the K values I sweep: K = 2 gives 1 pair and weight 1; K = 3 gives 3 pairs and weight 1/3; K = 5 gives 10 pairs and weight 1/10; K = 7 gives 21 and 1/21; K = 10 gives 45 and 1/45 — and 2/(K(K-1)) reproduces 1, 0.333, 0.100, 0.0476, 0.0222 in turn, matching the pair counts exactly. So lambda's effective scale stays constant as K changes, which is what I need if K is to be tuned without silently re-scaling the whole regularizer. And since the RNN can be stacked, I do this per layer — each layer has its own hidden trajectory and its own importance vector — to match distributions at every depth.

Now the real question: how do I learn alpha? My first reflex is to make it a learnable function of the data: feed the two periods' stacked hidden states into a little network with weights W_{i,j}, push through an activation and a softmax, out comes alpha for that pair. Clean, end-to-end, differentiable. Let me think about whether it actually trains... and it does not, for two reasons that compound. First, alpha and theta are deeply coupled, and at the start of training the hidden representations that theta produces are meaningless — the GRU has not learned anything yet — so the alpha-network is being asked to judge the importance of states that carry no real distributional information. It learns garbage early and the garbage steers theta. Second, a separate W_{i,j} for *every* period pair is expensive — the number of pairs is quadratic in K, and each is a matrix mapping the doubled hidden width to V — and it scales badly exactly when I want more periods. So the learned-network route is a wall: coupling makes it fail and cost makes it ugly.

Two fixes, and they slot together. The coupling problem is a chicken-and-egg between alpha and theta: alpha needs good hidden states, good hidden states come from a trained theta. So break the egg first — *pre-train* theta on the prediction loss alone, with lambda = 0, across all the periods, to get meaningful hidden representations theta_0 before I ever try to learn alpha. Now the states the importance vector is judging actually mean something. That handles coupling. For the cost-and-stability problem, I do not need a parametric network for alpha at all if I can find a rule that updates alpha directly from a signal I already compute. And I do compute one every step: the per-state cross-period distance d^t_{i,j} itself. Here is where boosting earns its place. Schapire's boosting concentrates effort on the parts of the problem currently handled worst and reweights toward them iteratively. Translate that to my matching: a state t whose cross-period distance grows from the previous epoch to the current one is a state my model is failing to align — it is the hard case — so I should *increase* its importance to make the matching loss push harder there. A state that is already aligning (distance shrinking or flat within numerical tolerance) does not need more weight. So watch each state's distance across epochs. If the new distance is larger than the old distance, multiply its alpha up; otherwise leave it. Concretely,

  alpha_new^t = alpha_old^t * G(d_new^t, d_old^t)   if d_new^t > d_old^t,
  alpha_new^t = alpha_old^t                         otherwise,

and I need the multiplier G to be greater than one so importance only ever ratchets up on the stalling states, and bounded and smooth so a single noisy epoch does not blow a weight up. The sigmoid of the distance increase does both: set

  G(d_new^t, d_old^t) = 1 + sigma(d_new^t - d_old^t),

where sigma is the logistic sigmoid. Since sigma maps into (0,1), G should land in (1, 2): always above one, so the selected weights increase, and capped at two, so a single epoch cannot run away — and the bigger the distance *grew*, the closer sigma gets to one and the harder the push, which is the boosting instinct of leaning into the worst-handled case. After updating, re-normalize the vector by its L1 norm, alpha_new^t <- alpha_new^t / sum_t alpha_new^t, so it stays a weighting summing to one and lambda alone controls the matching strength.

Let me trace one update by hand on three states to make sure this behaves the way I am claiming, rather than just asserting the range and the renormalization. Start uniform, alpha = (1/3, 1/3, 1/3). Suppose last epoch's per-state distances were d_old = (0.20, 0.50, 0.10) and this epoch's are d_new = (0.35, 0.45, 0.10): state 0 grew, state 1 shrank, state 2 is flat. The grow-mask picks out only state 0. The multipliers are G = 1 + sigma(d_new - d_old) = (1 + sigma(0.15), 1 + sigma(-0.05), 1 + sigma(0.0)) = (1.537, 1.488, 1.500), every one inside (1, 2) as intended. But only state 0's mask is true, so after the ratchet alpha = (0.333 * 1.537, 0.333, 0.333) = (0.512, 0.333, 0.333), unnormalized. L1-normalize by the new sum 1.179: alpha = (0.435, 0.283, 0.283), and the three add back to 1.000. The thing I had to check is the side effect of renormalization — state 1 and state 2 never grew, yet their weights dropped from 0.333 to 0.283. That is correct and is what I want: a state that is *already aligning* should lose relative importance when some other state is stalling, because alpha is a budget over states, not independent dials. So the rule does exactly two things at once — push hard on the state whose cross-period distance is growing, and let the well-behaved states recede — with the convex constraint enforced by the renorm and the whole strength still gated by lambda. Initialize uniform, alpha = {1/V}^V, the honest equal prior before any distances are seen, and let the boosting reshape it. This needs no extra parameters, costs nothing beyond the distances I already compute, and dodges both failures of the learned network.

Let me assemble the whole procedure so I can see it end to end. First, run the period discovery — maximize cross-period diversity, greedily, sweeping K — to turn the one stream into K periods. Second, pre-train theta by minimizing the prediction loss alone (lambda = 0) to get sensible hidden states. Third, iterate: for each epoch, run period pairs through the stacked GRU, compute the prediction loss on each period plus the importance-weighted, summed-over-states matching loss between pairs, update theta by gradient descent on L(theta, alpha), and update alpha by the boosting rule from the per-state distances measured this epoch versus last. Return the best theta and alpha. At inference time none of the matching machinery is needed — I just run the GRU and the head, one forward pass, L_pred only — so prediction costs no more than a vanilla recurrent net; all the adaptation is a training-time cost.

Now let me pin the distance computations down concretely, because the framework is supposed to be agnostic and I want each one implemented exactly. Cosine on the period means: reduce each period's hidden batch to its mean vector and take d = 1 - cos(mean(h_s), mean(h_t)) = 1 - <h_s_bar, h_t_bar>/(||h_s_bar|| ||h_t_bar||); cheap, and the cheapest choice is the sane default on the largest data. Linear MMD: with the linear kernel the squared MMD is just the squared distance of the means, so delta = mean(h_s) - mean(h_t) and the loss is delta . delta — note this is the linear kernel collapse of the general MMD estimator I wrote above, not a different object. RBF MMD when I want sensitivity to all moments: build the full Gram matrix over the concatenation of the two samples, set the base bandwidth from the average off-diagonal squared distance, expand it into a small geometric ladder, then take the within-source plus within-target minus the two cross terms. CORAL: form each side's covariance C = (H^T H - (1^T H)^T (1^T H)/n)/(n-1), and the loss is ||C_s - C_t||_F^2 /(4 q^2). Adversarial: a two-layer discriminator on top of a gradient-reversal layer (identity forward, gradient negated on the backward pass, so one backprop trains the discriminator to separate the periods while the features are pushed to confuse it), with the loss the binary cross-entropy of calling source 1 and target 0. The same family of distances serves both the period discovery and the matching; technically the choice can even differ between the two stages.

Let me write the network as I will actually build it. The backbone is a ModuleList of single-layer GRUs stacked, so I can grab the per-layer hidden trajectory; an optional bottleneck of two linear layers with batch-norm and ReLU before the head (useful on the wide financial-factor inputs); and a linear head reading the last hidden state. The algorithmic pre-training idea is prediction-only warmup for theta, but the implementation also uses a tiny per-layer gate during the warmup path to produce an initial state-weight vector from the concatenated source/target hidden states — a linear map from the flattened doubled-width sequence to V, batch-normalized, then softmax. Once I cross into the boosting phase those gate weights hand off to the boosting-updated weight matrix. The warmup forward path computes the head output and a gate-weighted matching loss with an optional local window len_win around each state (so state t can be matched to a small neighborhood of states in the other period, not only the exact index); the boosting forward path computes the head output, the matching loss weighted by the current weight matrix, and the per-state distance matrix that the boosting rule will consume to update the weights for next epoch.

```python
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader


def cosine(source, target):                                    # 1 - cos on period means
    source, target = source.mean(0), target.mean(0)
    return nn.CosineSimilarity(dim=0)(source, target).mean()


def CORAL(source, target):                                     # covariance alignment
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)
    tmp_s = torch.ones((1, ns), device=source.device) @ source
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)
    tmp_t = torch.ones((1, nt), device=target.device) @ target
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)
    return (cs - ct).pow(2).sum() / (4 * d * d)


class MMD_loss(nn.Module):                                     # linear / rbf MMD
    def __init__(self, kernel_type="linear", kernel_mul=2.0, kernel_num=5):
        super().__init__()
        self.kernel_type, self.kernel_mul, self.kernel_num = kernel_type, kernel_mul, kernel_num
        self.fix_sigma = None

    def guassian_kernel(self, source, target):
        n = int(source.size(0)) + int(target.size(0))
        total = torch.cat([source, target], dim=0)
        t0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        t1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2 = ((t0 - t1) ** 2).sum(2)
        bandwidth = torch.sum(L2.data) / (n ** 2 - n)          # average off-diagonal squared distance
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bws = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        return sum(torch.exp(-L2 / b) for b in bws)            # ladder of bandwidths

    def linear_mmd(self, X, Y):                                 # linear kernel => ||meanX - meanY||^2
        delta = X.mean(0) - Y.mean(0)
        return delta.dot(delta.T)

    def forward(self, source, target):
        if self.kernel_type == "linear":
            return self.linear_mmd(source, target)
        kernels = self.guassian_kernel(source, target)
        b = int(source.size(0))
        with torch.no_grad():
            XX = kernels[:b, :b].mean(); YY = kernels[b:, b:].mean()
            XY = kernels[:b, b:].mean(); YX = kernels[b:, :b].mean()
            return torch.mean(XX + YY - XY - YX)


class ReverseLayerF(Function):                                 # gradient-reversal for adversarial d
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class Discriminator(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256):
        super().__init__()
        self.dis1 = nn.Linear(input_dim, hidden_dim)
        self.dis2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.dis2(F.relu(self.dis1(x))))


def adv(source, target, device, input_dim=256, hidden_dim=512):
    bce = nn.BCELoss()
    net = Discriminator(input_dim, hidden_dim).to(device)
    src = torch.ones(len(source), 1, device=device)
    tar = torch.zeros(len(target), 1, device=device)
    p_s = net(ReverseLayerF.apply(source, 1))
    p_t = net(ReverseLayerF.apply(target, 1))
    return bce(p_s, src) + bce(p_t, tar)


class TransferLoss:                                            # the agnostic distance selector
    def __init__(self, loss_type="cosine", input_dim=512, GPU=0):
        self.loss_type, self.input_dim = loss_type, input_dim
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")

    def compute(self, X, Y):
        if self.loss_type in ("mmd_lin", "mmd"):
            return MMD_loss(kernel_type="linear")(X, Y)
        if self.loss_type == "mmd_rbf":
            return MMD_loss(kernel_type="rbf")(X, Y)
        if self.loss_type == "coral":
            return CORAL(X, Y)
        if self.loss_type in ("cosine", "cos"):
            return 1 - cosine(X, Y)
        if self.loss_type == "adv":
            return adv(X, Y, self.device, input_dim=self.input_dim, hidden_dim=32)


class AdaRNN(nn.Module):
    def __init__(self, use_bottleneck=False, bottleneck_width=256, n_input=128,
                 n_hiddens=[64, 64], n_output=6, dropout=0.0, len_seq=9,
                 model_type="AdaRNN", trans_loss="mmd", GPU=0):
        super().__init__()
        self.use_bottleneck, self.n_input = use_bottleneck, n_input
        self.num_layers, self.hiddens = len(n_hiddens), n_hiddens
        self.n_output, self.model_type = n_output, model_type
        self.trans_loss, self.len_seq = trans_loss, len_seq
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")

        in_size = n_input
        features = nn.ModuleList()
        for hidden in n_hiddens:                               # stacked single-layer GRUs => per-layer trajectory
            features.append(nn.GRU(input_size=in_size, num_layers=1, hidden_size=hidden,
                                   batch_first=True, dropout=dropout))
            in_size = hidden
        self.features = nn.Sequential(*features)

        if use_bottleneck:                                     # wide-input (finance) bottleneck
            self.bottleneck = nn.Sequential(
                nn.Linear(n_hiddens[-1], bottleneck_width),
                nn.Linear(bottleneck_width, bottleneck_width),
                nn.BatchNorm1d(bottleneck_width), nn.ReLU(), nn.Dropout())
            self.bottleneck[0].weight.data.normal_(0, 0.005); self.bottleneck[0].bias.data.fill_(0.1)
            self.bottleneck[1].weight.data.normal_(0, 0.005); self.bottleneck[1].bias.data.fill_(0.1)
            self.fc = nn.Linear(bottleneck_width, n_output)
            torch.nn.init.xavier_normal_(self.fc.weight)
        else:
            self.fc_out = nn.Linear(n_hiddens[-1], self.n_output)

        if self.model_type == "AdaRNN":                        # per-layer gate => initial state weights (pre-train)
            self.gate = nn.ModuleList(
                [nn.Linear(len_seq * self.hiddens[i] * 2, len_seq) for i in range(len(n_hiddens))])
            self.bn_lst = nn.ModuleList([nn.BatchNorm1d(len_seq) for _ in range(len(n_hiddens))])
            self.softmax = torch.nn.Softmax(dim=0)
            self.init_layers()

    def init_layers(self):
        for i in range(len(self.hiddens)):
            self.gate[i].weight.data.normal_(0, 0.05); self.gate[i].bias.data.fill_(0.0)

    def gru_features(self, x, predict=False):                  # run stacked GRU; collect per-layer states
        x_input, out, out_lis = x, None, []
        out_weight_list = [] if self.model_type == "AdaRNN" else None
        for i in range(self.num_layers):
            out, _ = self.features[i](x_input.float())
            x_input = out
            out_lis.append(out)
            if self.model_type == "AdaRNN" and not predict:
                out_weight_list.append(self.process_gate_weight(x_input, i))
        return out, out_lis, out_weight_list

    def process_gate_weight(self, out, index):                 # gate weights from concat[src,tar] hidden states
        x_s = out[0: out.shape[0] // 2]
        x_t = out[out.shape[0] // 2: out.shape[0]]
        x_all = torch.cat((x_s, x_t), 2)
        x_all = x_all.view(x_all.shape[0], -1)
        weight = torch.sigmoid(self.bn_lst[index](self.gate[index](x_all.float())))
        return self.softmax(torch.mean(weight, dim=0))

    @staticmethod
    def get_features(output_list):                             # split each layer's batch into source/target halves
        src, tar = [], []
        for fea in output_list:
            src.append(fea[0: fea.size(0) // 2]); tar.append(fea[fea.size(0) // 2:])
        return src, tar

    def forward_pre_train(self, x, len_win=0):                 # phase 1: gate-weighted matching, local window
        out = self.gru_features(x)
        fea = out[0]
        fc_out = (self.fc(self.bottleneck(fea[:, -1, :])) if self.use_bottleneck
                  else self.fc_out(fea[:, -1, :])).squeeze()
        out_list_all, out_weight_list = out[1], out[2]
        out_list_s, out_list_t = self.get_features(out_list_all)
        loss_transfer = torch.zeros((1,)).to(self.device)
        for i, n in enumerate(out_list_s):
            criterion = TransferLoss(loss_type=self.trans_loss, input_dim=n.shape[2])
            for j in range(self.len_seq):                      # sum matching over all states t=1..V
                i_start = max(j - len_win, 0)
                i_end = min(j + len_win, self.len_seq - 1)
                for k in range(i_start, i_end + 1):            # match state j to a small neighborhood
                    weight = (out_weight_list[i][j] if self.model_type == "AdaRNN"
                              else 1 / (self.len_seq) * (2 * len_win + 1))
                    loss_transfer = loss_transfer + weight * criterion.compute(
                        n[:, j, :], out_list_t[i][:, k, :])
        return fc_out, loss_transfer, out_weight_list

    def forward_Boosting(self, x, weight_mat=None):            # phase 2: boosting-weighted matching + dist matrix
        out = self.gru_features(x)
        fea = out[0]
        fc_out = (self.fc(self.bottleneck(fea[:, -1, :])) if self.use_bottleneck
                  else self.fc_out(fea[:, -1, :])).squeeze()
        out_list_s, out_list_t = self.get_features(out[1])
        loss_transfer = torch.zeros((1,)).to(self.device)
        weight = (1.0 / self.len_seq * torch.ones(self.num_layers, self.len_seq).to(self.device)
                  if weight_mat is None else weight_mat)
        dist_mat = torch.zeros(self.num_layers, self.len_seq).to(self.device)
        for i, n in enumerate(out_list_s):
            criterion = TransferLoss(loss_type=self.trans_loss, input_dim=n.shape[2])
            for j in range(self.len_seq):
                loss_trans = criterion.compute(n[:, j, :], out_list_t[i][:, j, :])
                loss_transfer = loss_transfer + weight[i, j] * loss_trans
                dist_mat[i, j] = loss_trans                    # record per-state distance for boosting
        return fc_out, loss_transfer, dist_mat, weight

    def update_weight_Boosting(self, weight_mat, dist_old, dist_new):  # the boosting rule on alpha
        epsilon = 1e-5
        dist_old, dist_new = dist_old.detach(), dist_new.detach()
        ind = dist_new > dist_old + epsilon                    # states whose distance grew
        weight_mat[ind] = weight_mat[ind] * (1 + torch.sigmoid(dist_new[ind] - dist_old[ind]))  # G>1
        weight_norm = torch.norm(weight_mat, dim=1, p=1)       # L1-normalize each layer's row -> convex
        return weight_mat / weight_norm.t().unsqueeze(1).repeat(1, self.len_seq)

    def predict(self, x):                                      # inference: one forward pass, no matching
        out = self.gru_features(x, predict=True)
        fea = out[0]
        return (self.fc(self.bottleneck(fea[:, -1, :])) if self.use_bottleneck
                else self.fc_out(fea[:, -1, :])).squeeze(-1)
```

Stepping back to make sure the causal chain holds. The pain is that a recurrent forecaster trained on one long stream silently assumes the stream is one distribution, and the future is not; the persistent piece is the conditional P(y|x) and the betraying piece is the marginal P(x), which is temporal covariate shift. Because the periods are unlabeled and I have no prior on the unseen future, maximum entropy says train under the most diverse split, so I discover periods by greedily maximizing cross-period distribution distance. Because the conditional is shared, I learn it by adding a distribution-matching regularizer that forces the periods' representations together — but an endpoint match wastes the RNN's hidden trajectory, so I match at every state, and because the states are not equally important I weight them, and because a learned weight-network fails from alpha-theta coupling and per-pair cost, I pre-train the representations and then ratchet the weights up only on the states whose distance grows, the boosting move. The result is a recurrent net that costs a vanilla forward pass at inference and, at training time, has been forced to represent the worst-case spread of regimes by what is common across them.
