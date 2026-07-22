Let me start from what actually hurts. The data I want to train on lives on people's phones — everything they type, every photo they take — and it is exactly the data that I must not haul into a data center, because it is private and there is a lot of it. So I am stuck with a constraint that no ordinary training setup has: the examples never leave the device. Each device can run computation on its own data and send me messages, but the messages have to be model-sized at most, never raw data. And the link is terrible — a phone is online for a few minutes when it happens to be charging on wifi, uploading at maybe a megabyte a second, willing to talk to me a handful of times a day. Computing on the device, on the other hand, is basically free: the local dataset is tiny and the processor is fast. That inversion is the whole game. In a cluster I'd be counting FLOPs and fighting for GPU time; here the only currency that matters is *rounds of communication*. If I can reach a good model in ten rounds instead of a thousand, I win, even if each device burns a hundred times more local compute per round. So the objective I'm really optimizing is: minimize the number of communication rounds to hit a target accuracy, and I'm allowed — encouraged — to spend the free on-device compute to get there.

What am I optimizing, concretely? The usual finite-sum, f(w) = (1/n) Σ_{i=1}^n f_i(w) with f_i the loss on example i. The twist is that the n examples are scattered across K clients, P_k the indices on client k, n_k = |P_k|. So group the sum by client: f(w) = Σ_{k=1}^K (n_k/n) F_k(w) where F_k(w) = (1/n_k) Σ_{i∈P_k} f_i(w) is client k's own average loss. The global objective is *literally* the n_k-weighted average of the per-client objectives. I want to remember that, because it's going to tell me how to weight things later. Now, if the data had been scattered uniformly at random, each client would be a random sample of the population and E[F_k(w)] = f(w) — that's the IID assumption all the distributed-optimization machinery quietly relies on. But my data isn't scattered that way. A phone's data is one person's usage. So F_k can be a wildly biased view of f — someone who only texts in one language, a camera roll that's all dogs. That non-IID-ness, plus the fact that some users have a thousand times more data than others, plus there being far more clients than examples-per-client, plus the brutal communication budget — that's the setting, and every tool I know was built without one or more of those.

Deep learning runs on SGD, and basically every architecture trick is in service of making the loss nicer for gradient methods, so whatever I build has to be SGD at its core. Let me write down the most obvious thing and see where it breaks. Naive distributed SGD: each round, pick some clients, each computes the gradient of its loss on its data, the server adds them up and takes one step. Make it precise with C the fraction of clients I select. Take C = 1 and a fixed learning rate η for the cleanest case. Client k computes g_k = ∇F_k(w_t), the average gradient on its local data at the current global model w_t. The server wants a step on f, and since f = Σ(n_k/n)F_k, its gradient is ∇f(w_t) = Σ(n_k/n) ∇F_k(w_t) = Σ(n_k/n) g_k. So the server applies

  w_{t+1} ← w_t − η Σ_{k=1}^K (n_k/n) g_k.

This is just full-batch gradient descent on f, distributed. Selecting only a C-fraction makes it stochastic — C controls the effective global batch size, and even though I'm picking whole clients rather than individual examples, the expected batch gradient is still ∇f(w) (a uniformly random client k has weight n_k/n in f, and the n_k-weighted gradient over a sampled subset is an unbiased estimate of Σ(n_k/n)∇F_k = ∇f), so it's a legitimate stochastic gradient step. Chen and colleagues showed in the data center that this synchronous style, with a couple of backup workers so one slow machine doesn't stall the round, beats asynchronous training — asynchrony injects stale gradients that hurt, and synchronizing avoids that. Fine. So I have a defensible baseline; call it FedSGD. And it works. The problem is the count. It takes *one gradient step per communication round*. Training a CNN on MNIST to convergence is tens of thousands of SGD steps even with every modern trick; here every one of those steps is a separate, expensive, rare communication round. Tens of thousands of rounds when a phone gives me a handful a day is hopeless. The arithmetic of FedSGD is the wall: one step of progress per round, and rounds are the scarce thing.

The asynchronous route doesn't save me either. DistBelief's Downpour SGD has many replicas, each fetching parameters, computing a minibatch gradient, pushing it back to a parameter server, over and over, asynchronously. That's a beautiful way to saturate a thousand-core cluster, but it's the *same disease amplified*: every minibatch is a round-trip to the server, so the number of messages is astronomical, and on top of that the staleness that a fast cluster network shrugs off becomes lethal when clients are sporadic and high-latency. More communication, not less. Wrong direction.

So the real lever has to be: do *more* work on each client between communications, since local work is free. How do I get more than one gradient step of progress out of a single round? Let me stare at the FedSGD update again, because I have a feeling it can be rewritten. The server does w_{t+1} = w_t − η Σ(n_k/n) g_k. What if, instead of the clients shipping gradients and the server stepping, each client takes the step *itself* and ships the result? Client k computes w^k = w_t − η g_k — one local gradient step from the shared starting point w_t — and the server averages the stepped models with the same weights: w_{t+1} = Σ(n_k/n) w^k. Expand it: Σ(n_k/n)(w_t − η g_k) = (Σ n_k/n) w_t − η Σ(n_k/n) g_k = w_t − η Σ(n_k/n) g_k, since the weights sum to one. That's *identical* to the FedSGD update. Let me put a number on it so I'm not just trusting the symbol-pushing: take w0 = 0, four clients with quadratic objectives whose optima are c = [0.6, −0.4, 1.0, −0.2], η = 0.3, equal weights. Each client's one-step model is w0 − η(w0 − c_k) = 0.3·c_k, and the average of those is 0.3·mean(c) = 0.3·0.25 = 0.075. The direct FedSGD step is w0 − η·∇f(w0) = 0 − 0.3·(w0 − mean(c)) = 0.3·0.25 = 0.075. Same number, 0.075000 either way. The two are the same algorithm written two ways: "every client sends a gradient, server steps" equals "every client takes one step, server averages the stepped models."

That equivalence is the crack I needed, because the second form has slack the first doesn't. Once each client is *taking steps and shipping a model* rather than shipping a single gradient, nothing stops a client from taking more than one step before it ships. Let it sweep its whole local dataset, in minibatches of size B, for E epochs — w^k ← w^k − η ∇(local minibatch loss), iterated — and only *then* send w^k to the server to be averaged. The number of local updates a client does per round is u_k = E · n_k / B; I've turned one rigid gradient step per round into a knob I can crank. Set E = 1 and B = ∞ (the whole local set as one batch) and I'm back to exactly one local gradient step — FedSGD falls out as the boundary case, which is reassuring: I haven't invented a different algorithm, I've found the family that FedSGD sits at one corner of, parameterized by how much local computation (E, B) and how much parallelism (C) I use. Now I can spend the free compute: more local epochs means more progress per round, which should mean fewer rounds. That's the trade I wanted.

Except — wait. I just waved my hand over the most dangerous step. When E = 1 and one step, the average of the stepped models equalled a gradient step on f; the algebra carried me, and the 0.075 above checks it. But the moment each client takes *many* steps, w^k = w_t − η g_k − η ∇F_k(w^k after one step) − ... is no longer w_t minus η times client k's gradient at w_t. I can see the equivalence break on the same toy: run those four clients for E = 5 steps each instead of one and average, and the result is 0.2080, whereas the single FedSGD step is still 0.0750 — not equal. The clean "average = one gradient step on f" identity is gone the instant E > 1. Each client has wandered off, descending its *own* F_k, which for non-IID data points somewhere quite different from where f wants to go. The clients drift apart, and at the end I'm averaging models that have moved to different places in a non-convex landscape. And averaging parameters of different neural networks is, in general, a catastrophe. Two networks can both be excellent and their parameter-space midpoint can be garbage — the loss surface is non-convex, the midpoint can sit on top of a hill between two valleys. So why would averaging E-epochs-drifted client models give me anything but a worse model? This is the thing that has to actually work, and I have no right to assume it does. Let me try to break it and see.

I can't simulate a deep net in my head, but I can build the smallest caricature that has the feature I'm worried about and actually compute it. The fear is the non-convex hump, so let me first reproduce the hump in a one-line objective and check that averaging really does fail there. Take a symmetric double-well, L(w) = (w² − 1)², which has two equally-good minima at w = −1 and w = +1, both with L = 0, and a barrier at w = 0. Call those two minima the parents wA = −1, wB = +1. L(wA) = 0, L(wB) = 0, and their average is (−1 + 1)/2 = 0, with L(0) = (0 − 1)² = 1. So the average scores 1 while both parents score 0 — strictly worse than either. Walking the segment confirms a clean hump: at the five points t ∈ {0, 0.25, 0.5, 0.75, 1} the loss runs 0 → 0.5625 → 1 → 0.5625 → 0, a symmetric bump peaking exactly at the midpoint. So yes, averaging two good-but-distinct solutions of a non-convex loss can land me on the barrier between them. The fear is real and I've now seen it cost a full unit of loss.

Why does that happen? The two minima are genuinely different basins separated by a ridge, and the straight line between them must cross the ridge. With real networks the same thing happens for a structural reason: two networks trained from *different* random initializations break symmetry differently — hidden units get permuted, features land in different coordinates — so even when they represent equally good functions they sit in different basins, and the segment between them climbs out of one and into the other. Different-init averaging is averaging across a ridge. So the question that decides whether my algorithm is sane is: are the models I average sitting in the *same* basin or different ones?

Look at what the algorithm actually does. Every round the server broadcasts *one* model w_t to all selected clients, and that w_t is the shared starting point for everybody this round. The clients don't start from independent random seeds — they start from the *same* w_t, then walk in somewhat different directions over a few epochs. They share their symmetry-breaking, inherited from w_t. So this is the opposite regime from the double well: same starting point, modest drift. Let me build the caricature of *that* and compute it too, because "should be fine" is not the same as "is fine." Model the round as: a shared init w0 = 0; four clients with non-IID local objectives F_k(w) = ½(w − c_k)², local optima c = [0.6, −0.4, 1.0, −0.2] (so they pull in different directions, and the global optimum is the mean, 0.25); each client takes E = 5 plain GD steps from w0 with η = 0.3; then I average the four stepped models (equal n_k here). The global loss is f = mean_k F_k.

Running it: the four stepped models come out at [0.499, −0.333, 0.832, −0.166] — spread out, each pulled toward its own c_k. Their average is 0.208, close to the true global optimum 0.25. Now the loss numbers, which are the whole point. f at the shared init w0 is 0.1950; f at the four parents is [0.1948, 0.3336, 0.3331, 0.2504]; f at the averaged model is **0.1646** — and f at the global optimum is 0.1638. So the average scores *below every parent* and below the init, almost reaching the optimum. This is the reverse of the double-well outcome: there the average jumped to L = 1 above both L = 0 parents; here it sinks to 0.165 under parents that range 0.195–0.334. The mechanism is visible in the numbers — the clients' idiosyncratic drifts (one pulled to +0.83, another to −0.17) largely cancel when averaged, leaving the component they agree on, which is descent toward the shared global minimum. Shared init turns "averaging across a ridge" into "averaging within one well," and within the well the average can sit lower than any single drifted parent.

So the two computed cases bracket it: different basins (double well) → average strictly worse; shared init, modest drift (the four-client quadratic) → average strictly better, near the global optimum. That tells me the shared per-round initialization is not a detail — it is the thing that keeps every averaging operation an in-basin interpolation. If I let clients keep their own models across rounds and never re-synchronized to a common w_t, I'd drift toward the double-well regime and the averaging would blow up. Re-broadcasting w_t at the *start of every round* is what holds the algorithm inside the second regime. This is the lineage behind the linear-interpolation diagnostic people use on real nets: along the line from a network's init to its trained solution the loss is smooth and roughly convex, and two MNIST classifiers trained on different subsets from a *common* seed interpolate to a model with lower full-set loss than either parent, while from *different* seeds they interpolate over a tall barrier — the same split my two toy computations show. The dangerous step survives, and I now know *why* it does rather than hoping it does: not by assertion, by the two numbers 1.0 and 0.165.

This wasn't a sure thing historically, and I shouldn't pretend it was. Iterative parameter mixing for the structured perceptron — train a local model per shard, after each epoch send the weight vectors to a server, average, redistribute — does exactly the iterate-then-average-then-resync dance, and it was shown there that one-shot mixing (average only at the very end) can fail while repeated averaging-and-redistribution converges. That's the right structural lineage, but it was a convex, linear model on balanced data-center shards. And the deep-network attempts were shakier: averaging DNN parameters periodically, every minute or two, and redistributing — the people who tried it reported flatly that by itself it doesn't work well, and had to swap in natural gradient to rescue it. So "just average the parameters of deep networks trained for many local SGD steps" was, by reputation, unreliable. The safe lesson is narrower: don't let the local walks become an uncontrolled second training process. Keep every round anchored to a shared w_t, weight the returned models by their sample counts, and keep E and B as explicit controls over how far a client is allowed to drift before the next average.

Now the other endpoint, to make sure I understand the family I'm in. Push E all the way up — let each client train its local model essentially to convergence on its own data — and average once. That's one-shot parallelized averaging: every machine runs SGD to the end on its shard, average the results, done, one round of communication. The contraction-mapping analysis of that scheme says that with a small fixed learning rate each machine's parameters settle into an asymptotically-normal cloud around the optimum; averaging k independent parameter vectors reduces variance by about 1/k, or standard deviation by 1/√k. But it reduces only the independent noise — it doesn't remove bias from each machine's local empirical problem or finite local run, and the guarantee leans on convexity, IID data, and equal shard sizes. Worst case, in the convex IID setting, the single averaged model is no better than training on one machine alone. So the E → ∞, one-round corner is the over-committed extreme: each client can move so far toward its own non-IID slice that the average may stop representing the global objective. There must be a sweet spot for E: enough local work to make rounds count, not so much that clients overfit their local data and drift out of agreement. If that drift starts to dominate late in training, the natural control is to *decay* the local computation, smaller E or larger B, the same way one decays a learning rate. EASGD lives near here too — it couples each worker to a center variable by an elastic spring and loosens the spring to let workers explore between syncs — but it assumes every worker can sample the whole dataset and explicitly sidesteps the data-partitioned case, which is the only case I have. I'm in a family with FedSGD at one corner (E=1, B=∞: one full-client batch step, max rounds) and one-shot averaging at the other (E=∞, one round), and I want to live in the interior.

Now the averaging weights, which I've been writing as n_k/n but should justify, not assume. The global objective is f = Σ(n_k/n)F_k — it is *itself* the n_k-weighted combination of the client objectives. So if I want the average of the stepped models to behave like a step on f, the weights have to be n_k/n; that's what made the one-step algebra collapse exactly onto the FedSGD/gradient-of-f update. Weighting every client equally instead would implicitly optimize the *unweighted* mean of the F_k, over-counting a client with three examples as much as one with three thousand, biasing the global model toward tiny clients. Sample-count weighting is the unbiased choice, falling straight out of the structure of f.

There's a subtlety in the weighting that I have to get right when I only use a fraction C of clients per round. I don't see all K clients; I see a random selected set S_t of m = max(C·K, 1) of them. If I form Σ_{k∈S_t} (n_k/n) w^k, those weights don't sum to one — they sum to (Σ_{k∈S_t} n_k)/n, which is well below 1 because S_t is a small subset — and the result would be a tiny fraction of a model, nonsense. Let me check this on an unbalanced instance, since unbalanced is exactly when it bites. Four clients with n = [3, 3000, 50, 1], and suppose this round I select S = {client 1, client 2} with models [0.83, −0.33]. Dividing by the global n = 3054: the two weights are 3000/3054 and 50/3054, which sum to 0.9987 — close to 1 only by luck here because the giant client happens to be in the round; with a different draw it could be far from 1, and the combined model lands at 0.8099, a shrunk-down version of nothing in particular. Dividing instead by m_t = 3000 + 50 = 3050: the weights are 3000/3050 and 50/3050, summing to exactly 1.0000, and the result is 0.8110, which sits between the two selected models 0.83 and −0.33 — a genuine convex combination, as a model average must be. So the fix is to normalize over the clients I *actually have* this round: let m_t = Σ_{k∈S_t} n_k and average with weights n_k/m_t. So w_{t+1} = Σ_{k∈S_t} (n_k/m_t) w^k_{t+1}. Dividing by the global n conflates "weight within this round's sample" with "fraction of all data," and only the former gives a proper convex combination of the models I'm averaging — normalize by m_t, the total samples among the selected clients.

So the algorithm assembles itself. The server holds w_0. Each round t: form m = max(C·K, 1); pick a random set S_t of m clients; broadcast w_t to each; each selected client runs ClientUpdate — split its data P_k into size-B batches, and for E local epochs over those batches, w ← w − η ∇loss(w; batch) — and returns its w^k_{t+1}; the server sets m_t = Σ_{k∈S_t} n_k and forms the weighted average w_{t+1} = Σ_{k∈S_t} (n_k/m_t) w^k_{t+1}; repeat. Three knobs stay explicit: C controls how many clients contribute to a round, E controls how many passes each selected client makes over its data, and B controls the local minibatch size. Local SGD stays *plain* SGD on purpose: no momentum, no adaptive rates locally. Not because those couldn't help, but because the contribution I'm making is the aggregation structure — iterate locally, average with the right weights anchored to a shared per-round init — and bolting on a fancier local optimizer would tangle that up. Momentum, AdaGrad, Adam locally are a separate axis to explore later; keep this clean.

One correctness point I have to be careful about in the averaging is the normalizer. The natural impulse is to write the server combine as a sum over *all* K clients, w_{t+1} = Σ_{k=1}^K (n_k/n) w^k. That's wrong whenever C < 1: I only received models from S_t, not from all K, so summing over K is undefined for the clients I didn't hear from, and even formally it would normalize by the wrong total. The combine must be over the *selected* set with the *selected-set* normalizer: w_{t+1} = Σ_{k∈S_t} (n_k/m_t) w^k_{t+1}, m_t = Σ_{k∈S_t} n_k. Same weighting idea, but normalized to the clients actually present this round.

Let me write the whole thing as pseudocode so the structure is unambiguous:

  Server:
    initialize w_0
    for each round t = 1, 2, ...:
      m   ← max(C·K, 1)
      S_t ← random set of m clients
      for each client k ∈ S_t in parallel:
        w^k_{t+1} ← ClientUpdate(k, w_t)
      m_t ← Σ_{k∈S_t} n_k
      w_{t+1} ← Σ_{k∈S_t} (n_k / m_t) w^k_{t+1}

  ClientUpdate(k, w):
    B ← split P_k into batches of size B
    for each local epoch i = 1..E:
      for batch b ∈ B:
        w ← w − η ∇loss(w; b)
    return w to server

That's it. The only state is the global model; clients are stateless across rounds (they receive w_t, train, return, forget), which is exactly what I want for devices that come and go — no client has to remember anything between the rare times it participates. Memory and bandwidth per round are one model each way. And because the current global model touches a client's local optimization *only through the initialization* w_t, as E grows the initialization matters less and less and a client moves toward minimizing its own F_k regardless of where it started; if that local pull becomes too strong, decaying E or growing B late in training is the natural way to bring the round back toward the shared global objective.

Now let me drop this into the actual harness, where the server side is a strategy object exposing the local trainer, the combine rule, and client selection. The local trainer is plain SGD for E epochs returning the trained model, its sample count, and its loss. The combine rule is the n_k-weighted average over the received updates, normalized by the total samples among them — and I have to handle one practical wrinkle the math glosses over: a model's state_dict contains non-floating-point buffers (integer counters, things like batch-norm's tracked-batches count) that it makes no sense to average; for those I just carry one client's value through unchanged, and weight-average only the floating-point tensors. Aggregation happens on CPU in float32 for numerical safety, then casts back to each parameter's original dtype. Client selection is a uniform random sample of the available clients.

```python
import random
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader


def _client_sgd(model, loader, loss_fn, local_epochs, local_lr, device):
    # ClientUpdate: E epochs of plain SGD on this client's local data.
    opt = torch.optim.SGD(model.parameters(), lr=local_lr)
    total_loss, total_n = 0.0, 0
    for _ in range(local_epochs):                      # E local epochs
        for inputs, targets in loader:                 # minibatches of size B
            inputs, targets = inputs.to(device), targets.to(device)
            opt.zero_grad()
            outputs = model(inputs)
            if outputs.dim() == 3:                     # seq models: flatten time
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)
            loss = loss_fn(outputs, targets)
            loss.backward()
            opt.step()                                 # w <- w - eta * grad(loss; b)
            total_loss += loss.item() * inputs.size(0)
            total_n += inputs.size(0)
    return total_loss / max(total_n, 1)


class Strategy:
    """FedAvg: local SGD on each client + sample-count-weighted average of the
    returned models, with every round re-anchored to the broadcast global model."""

    def __init__(self, global_model, args):
        self.args = args                               # stateless across rounds

    def client_local_train(self, global_state_dict, client_dataset, model_fn,
                           loss_fn, local_epochs, local_lr, local_batch_size,
                           device, client_idx):
        model = model_fn()
        model.load_state_dict(global_state_dict)       # start from the shared w_t
        model.to(device)
        model.train()
        loader = DataLoader(client_dataset, batch_size=local_batch_size,
                            shuffle=True, drop_last=False, num_workers=0)
        avg_loss = _client_sgd(model, loader, loss_fn,
                               local_epochs, local_lr, device)
        local_state = OrderedDict(
            (key, value.detach().cpu().clone())
            for key, value in model.state_dict().items()
        )
        # return (local model w_k, n_k, loss) for the weighted average
        return local_state, len(client_dataset), avg_loss

    def aggregate(self, global_state_dict, client_updates, round_num):
        # w_{t+1} = sum_{k in S_t} (n_k / m_t) * w^k,  m_t = sum_{k in S_t} n_k
        if not client_updates:
            return OrderedDict((k, v.detach().clone())
                               for k, v in global_state_dict.items())
        total_samples = sum(n for _, n, _ in client_updates)            # m_t
        if total_samples <= 0:
            raise ValueError("FedAvg aggregation requires positive sample counts")
        new_state = OrderedDict()
        for key, ref in global_state_dict.items():
            if not ref.is_floating_point():            # integer buffers: don't average
                new_state[key] = client_updates[0][0][key].detach().clone()
                continue
            acc = torch.zeros_like(ref, device="cpu", dtype=torch.float32)
            for st, n, _ in client_updates:            # n_k-weighted average
                acc += st[key].detach().cpu().float() * (n / total_samples)
            new_state[key] = acc.to(ref.dtype)
        return new_state

    def select_clients(self, num_available, num_to_select, round_num):
        # random set S_t of m = max(C*K, 1) clients
        return random.sample(range(num_available),
                             min(num_to_select, num_available))
```
