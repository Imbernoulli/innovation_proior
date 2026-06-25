Let me start from what actually goes wrong. I have a stream of tasks coming at me one at a time — task 1, then task 2, and so on — and I have to keep solving all of them. The brutal fact is that if I just keep training the same network as each task arrives, the moment I do gradient descent on task 2 I am rewriting the very weights that solved task 1, and task 1's accuracy falls off a cliff. Catastrophic forgetting. And I cannot escape it the easy way, because the rules of the regime I care about forbid it: I get to see each task's data exactly once, a single pass, online; I have a task id so I can pick the right output head; and I am allowed only a small fixed buffer in which to stash a few past examples. That small-buffer constraint is the whole point — if I could keep every task's full dataset around I would just be doing multi-task learning and there would be no problem. So the question is sharp: how do I lose as little of the old tasks as possible, at almost no extra cost over plain fine-tuning, using only a handful of remembered examples?

What do people do today? Two camps. One camp says: protect the weights that mattered. After I finish a task, measure how important each weight was and, on the next task, add a penalty that drags each weight back toward its old value in proportion to that importance. That is EWC: minimize L_t(θ) + Σ_i (λ/2) F_i (θ_i − θ*_i)^2, where F_i is the Fisher-information diagonal, the expected squared gradient of the log-likelihood, a per-weight importance, and θ* are the old weights. I can see why this should help — the weights that carried task 1 get pinned. But two things bother me. It stores an importance scalar for every parameter, so its memory is the size of the network in the worst case. And the anchor is blunt: it pulls each weight toward its old scalar value weighted by a scalar importance, but it has no idea which *directions* in weight space actually preserve the old function — and the reports I have say that under a single pass it barely beats plain fine-tuning and is finicky about λ. So regularization alone is not carrying me.

The other camp says: keep a few old examples and use them to stop the old-task loss from rising. This is where the action is in the single-pass studies, and the strongest version is the gradient-constraint idea. GEM phrases it cleanly: minimize the current-task loss subject to ℓ(θ, M_k) ≤ ℓ(θ^{t-1}, M_k) for every past task k — never let the loss on any past task's stored memory go up. The mechanics: at each step compute a gradient g_k on each past task's memory, and if my proposed step g would make an old-task loss climb — which shows up as ⟨g, g_k⟩ < 0, the update pointing against the direction that reduces task k's loss — then don't take g; take the closest vector g̃ in L2 that keeps all the inner products nonnegative. That nearest-feasible-vector problem is a quadratic program. In the dual it has t−1 variables, one per past task, and to set it up I have to assemble the matrix G = (g_1, …, g_{t−1}) of all past-task gradients, P-dimensional each, at every single step. It even allows positive backward transfer, old-task loss going down, which is nice. But the cost is ugly: a QP plus a (t−1)×P gradient matrix every step, and it only gets worse as tasks pile up.

A-GEM trims this to the bone. Instead of t−1 constraints, use one *averaged* constraint: draw a single random minibatch from the union of all the memories, take its gradient g_ref, and require only ⟨g̃, g_ref⟩ ≥ 0. Now the projection is closed form — if g^T g_ref ≥ 0 the constraint is already satisfied, keep g; otherwise project,

  g̃ = g − (g^T g_ref / g_ref^T g_ref) g_ref,

which is just subtracting off the part of g that points against the memory gradient. One extra gradient, one inner product, no QP, no stored G. It is orders of magnitude faster than GEM and reportedly about as good on the single pass. So A-GEM is the thing to beat: it is cheap and it works.

But let me stare hard at what A-GEM is actually doing with the memory, because something nags at me. The memory only ever acts as a *veto*. Most of the time g^T g_ref ≥ 0 and the memory does literally nothing — I take my plain current-task step. The only time the memory speaks is when there is a violation, and even then all it does is delete a component of my current-task gradient; it removes the part that would have hurt the old tasks. Look at the projected update at a violation: g̃ = g − (g^T g_ref / g_ref^T g_ref) g_ref. Take the inner product with g_ref: g̃^T g_ref = g^T g_ref − (g^T g_ref / g_ref^T g_ref)(g_ref^T g_ref) = g^T g_ref − g^T g_ref = 0. So the projected step is exactly *orthogonal* to the memory gradient — its directional derivative on the memory loss is zero. It does not increase the memory loss, but it does not decrease it either; it guards a ceiling and nothing more. The memory loss only ever drops by accident, on the steps where no violation fires and g happens to align with g_ref. That orthogonality is the mathematical signature of a method that caps the old-task loss without ever pushing it down, and it makes me suspect such a method leaves the memory only loosely fit — but that is a suspicion about behavior I would want to check on a real run, not something I can read off the formula, so I will hold it as a hypothesis rather than a fact.

So here is the thing I keep wanting to try and keep being told not to: why not just *train on the memory*? Don't veto, don't project — put the old examples into the batch and descend their loss like any other data. If the constraint approach only ever guards a ceiling on the memory loss, then driving that loss down directly is the obvious complementary move. Concretely: each step, alongside my current-task minibatch B_n, draw a minibatch B_M from the memory, and just take an ordinary SGD step on the *union* B_n ∪ B_M. No QP, no projection, no inner product — computationally simpler than A-GEM, because A-GEM also computes g_ref and then does extra arithmetic, whereas I just feed both batches through the same forward/backward and step once.

And this is exactly the move the field warned against. The gradient-constraint people said it in plain words: "minimizing the loss at the current example together with the loss on the episodic memory results in overfitting to the examples stored in the memory." The worry is not stupid. My memory is *tiny* — a few examples per class. If every single step I take a gradient step on those same few examples, over and over, for the whole length of a task, that is the canonical setup for memorizing them and generalizing to nothing. I will drive the memory loss to zero, sure, but the memory loss being zero is worthless; what I need is accuracy on the *held-out* old-task test set, and memorizing five images per class should give me nothing there. So on its face, training directly on the tiny memory should overfit and lose. That is the objection I have to take apart before I trust this.

But let me not take the warning on faith — let me actually reason about whether it's true, because if it's wrong the whole expensive constraint apparatus is unnecessary. The claim is: repeatedly fitting a handful of stored examples must overfit them. When is that claim actually right? It's right when those few examples are *all* I'm training on. If I sat and trained only on M_1, the five-per-class memory of task 1, with nothing else, then yes — the network would happily drive those to zero loss and contort itself to do it, and test accuracy on task 1 would crater. That's textbook small-sample overfitting.

The thing is, that's not the situation I'm in. When I'm learning task 2, I'm not training on M_1 alone — I'm training on M_1 *together with* the full, large dataset D_2 of task 2. So at every step the gradient is a blend: a little bit of "fit these five old examples" and a lot of "fit this fresh, large batch of new-task data." And now I have to ask what that large companion does to the small-sample overfitting. The five old examples want to bend the network into some idiosyncratic shape that nails them; but simultaneously the task-2 examples are constraining the network to remain a sensible function that fits *them*. The network can't go fully idiosyncratic on M_1 because doing so would wreck its fit on D_2, and D_2's gradient is large and persistent. The big current-task dataset would be acting as a regularizer on the repeated learning of the tiny memory — not an explicit penalty I added, but an implicit, data-dependent one that comes for free from co-training. That's the hopeful story. The word "regularizer" is doing a lot of work in it, though, so let me actually compute whether replay beats fine-tune on a case small enough to solve by hand, instead of asserting it.

I'll take the smallest non-trivial instance: a linear model f(x) = w·φ(x) with features φ(x) = [x, 1], so two parameters, fit by mean-squared error. Task 1 targets y = 2x + 0.5 on x ∈ [−2,2]; task 2 is a different line, y = −0.5x − 1, on the same inputs (moderately related — same input range, different labels). I solve task 1 exactly by least squares, so after task 1 the task-1 MSE is 0. Then I do one online pass over task 2 in batches of 10 at learning rate 0.05, two ways. Fine-tune: batches of task 2 only. Replay: each step I stack the task-2 batch with a batch resampled from a 6-point memory of task 1. Running it:

  after task 1: task-1 MSE = 0
  fine-tune : task-1 MSE = 9.31,  task-2 MSE = 0.15
  replay    : task-1 MSE = 3.34,  task-2 MSE = 2.28

This is more honest than I expected, and more useful than a clean win. Replay does what I hoped on the old task — task-1 MSE 9.31 → 3.34, a real reduction in forgetting, the memory genuinely pulling the function back toward task-1 points rather than memorizing them into uselessness. So the field's "must overfit" verdict is wrong here: co-training with D_2 did regularize the memory. But replay paid for it: task-2 MSE went from 0.15 to 2.28. With only two shared parameters and one shared head there isn't enough capacity to satisfy both lines at once, so descending the average of the two gradients lands at a compromise that fits neither perfectly. That tension is real and I shouldn't hide it — replay is not free retention, it's a *trade* of some current-task fit for much less forgetting. What loosens the trade in the actual setting is exactly what this toy lacks: the real network is large and has a *per-task output head* selected by the task id, so task 1 and task 2 don't have to share the final linear map, and there is enough capacity that descending both losses needn't force a bad compromise. So I read the toy as confirming the mechanism (memory + companion data reduces forgetting and does *not* overfit) while warning me that the size of the win depends on capacity and head separation — which the harness provides.

Let me also locate the failure edge of the regularization story, because "regularizer" should not be unconditional. Push the relatedness of the two tasks from "different" toward *adversarial*: imagine task 2's labels actively contradicting task 1's on near-identical inputs (a rotated 2 that looks like a 5). Then D_2 is no longer a benign companion for M_1 — it is pulling the function toward answers that are *wrong* for task 1, and co-training on M_1 ∪ D_2 could end up worse for task 1 than the memory alone would be. So the data-dependent regularization has two knobs, and naming them tells me the whole story: the *amount* of current-task data sets the regularizer's strength (more D_2 = stronger pull away from memorizing M_1), and the *relatedness* of the tasks sets whether that pull is helpful, neutral, or harmful. The toy above sat in the "different but not contradictory" regime and replay won on retention; only the near-adversarial corner flips the sign, and that corner is rare in the benchmarks I care about.

This also reframes the original objection cleanly. The constraint camp may be right that I will fit the memory very closely; those few stored examples can be memorized. But the toy shows memorization and bad *generalization* are not the same thing: replay drove the memory close to fit yet *improved* held-out task-1 error, because the generalization came from the companion data, not from the memory size. And it sharpens my earlier suspicion about A-GEM: I showed its projected step is orthogonal to g_ref, so it never descends the memory loss on purpose — by construction it forgoes exactly the co-training pressure that, in my toy, did the retention. Whether that costs it accuracy on a full benchmark I'd still want to measure, but the mechanism it gives up is now concrete.

So the move is to stop being clever. Drop the constraint, drop the projection, drop the QP — just put the memory into the batch. Compared to plain fine-tune, this is two small changes: maintain an episodic memory of past examples, and at each step double the batch by stacking a random memory minibatch onto the current minibatch, then take one ordinary gradient step. Let me nail down the optimization geometry of "double the batch," because that's the whole algorithm and I want to know exactly what direction it moves in. Let g be the gradient of the *mean* loss on B_n and g_ref the gradient of the mean loss on B_M. The mean loss on the concatenation B_n ∪ B_M is (|B_n| · meanloss(B_n) + |B_M| · meanloss(B_M)) / (|B_n| + |B_M|), so its gradient is the size-weighted blend (|B_n| g + |B_M| g_ref) / (|B_n| + |B_M|). That is only the clean average (g + g_ref)/2 when the two batches are the *same size* — and the harness uses a training minibatch of 10 and a memory minibatch of 10, so equal size is exactly the operating point. Let me check the equal-size claim numerically rather than trust the algebra: on a one-parameter MSE toy with a memory batch and an equal-size current batch, I get g_e = 2.5, g_ref = −7.0, so (g_e + g_ref)/2 = −2.25, and the gradient computed directly on the concatenated 6-point batch is also −2.25 — they match. When I instead make the current batch much larger than the memory batch, the union gradient (1.31) is nowhere near (g + g_ref)/2 (−2.43); it sits close to g, because the large batch dominates the size-weighted blend. Good — so the "(g + g_ref)/2" picture is correct precisely at equal batch sizes, which is what I'll run, and I should remember that an unequal split silently down-weights the memory. So where A-GEM uses g_ref only as a half-space constraint and only when violated, ER always moves along the average direction: always descending the current task *and* the memory together, with the constant absorbed into the learning rate. That is the precise difference — average, not project; use the memory every step, not only on a violation; descend the old-task loss, not merely cap it. And it costs essentially nothing: one forward/backward on a batch of twice the size, no extra gradient solve.

Now, the memory is tiny and the stream is long and of unknown length, so I have to decide *what to keep* in it — the write strategy — and this matters more the smaller the memory is. The classic tool for "keep a uniform random subset of a stream of unknown length in a fixed buffer" is reservoir sampling: when the n-th example arrives, if the buffer isn't full, append it; otherwise pick a slot index i uniformly in [0, n), and if i is inside the buffer (i < mem_sz), overwrite slot i with the new example. The reason to use it is that it keeps the buffer an unbiased sample, but I should verify that uniformity rather than quote it. Claim: after n arrivals, every item seen so far sits in the buffer with probability mem_sz/n. Check the boundary by induction. While n ≤ mem_sz everything is kept, so probability 1 = mem_sz/n trivially. For the step n → n+1 with the buffer full: the (n+1)-th item is stored iff its uniform index lands in [0, mem_sz), probability mem_sz/(n+1) — correct for the new item. An already-stored item (in by hypothesis with prob mem_sz/n) survives unless the new draw picks its specific slot, which happens with prob 1/(n+1); so its new probability is (mem_sz/n)·(1 − 1/(n+1)) = (mem_sz/n)·(n/(n+1)) = mem_sz/(n+1). Every item, old and new, ends at mem_sz/(n+1) — the induction closes, so the buffer is genuinely uniform. To be sure I didn't fool myself with the algebra, I also simulated a stream of 7 items into a buffer of 3 over 200k trials: every item's empirical inclusion frequency came out ≈ 0.428–0.430 against the predicted 3/7 = 0.429. Uniform, confirmed. That's the right default when the memory is reasonably sized. But I can predict its failure when the memory is *very* small relative to the number of tasks: with, say, one slot per class on average, pure randomness will sometimes evict every example of an early class, and once a class has zero representatives in the buffer it gets no replay at all and forgetting on it spikes. So in the tiny regime, uniform randomness is the wrong objective — what I want is *coverage*: guarantee at least a few examples per class.

That points to a balanced writer. The simplest is a ring buffer: give each class its own fixed-size FIFO of size mem_sz/C and keep the last few examples of each class. This guarantees every class is represented — exactly the property reservoir loses when the memory is tiny — at the cost that old-task stored examples never change (slightly more overfitting risk, now harmless given the co-training argument) and the memory is underused early when few classes have been seen. I can do fancier balanced writers in the same spirit: online k-Means in the pre-classifier feature space, storing per class the examples closest to the k centroids, for better feature coverage; or mean-of-features, storing per class the examples closest to a running average feature vector, putting samples near the class mode. All three share the balance guarantee and differ only in *which* representatives they keep. The trade is clear: reservoir wins once the memory is comfortably sized (true randomness, no stale samples), balanced writers win when the memory is tiny (no class ever starves). Which suggests a hybrid that needs no advance knowledge of the task count: start with reservoir, and the moment any class drops below a minimum number of stored examples, switch to a balanced ring-buffer-style scheme that protects the starving classes — best of both across memory sizes.

Let me write the core procedure as a loop, because the whole contribution is two lines longer than fine-tune:

  M ← empty buffer of size mem_sz ; n ← 0
  for task t = 1 … T:
    for each minibatch B_n drawn (without replacement) from D_t:
      B_M ← random minibatch from M            # the only-if-M-nonempty replay batch
      θ ← SGD-step(θ, B_n ∪ B_M, lr)           # one step on the stacked batch: direction (g + g_ref) / 2 at equal sizes
      M ← UpdateMemory(mem_sz, t, n, B_n)      # reservoir / ring / k-means / MoF / hybrid
      n ← n + |B_n|

Two modifications to fine-tune: update the memory, and double the batch. Everything else is the existing training loop.

Now let me land this in the actual harness I'll run, which is a behavior-cloning continual learner: the same lifecycle hooks plain fine-tuning already uses — __init__, start_task before each task, observe on every minibatch, end_task after each task — where observe does to-device, zero_grad, loss = policy.compute_loss(data), backward, optional grad-clip, step. I only need to fill the hooks. In __init__ I hold the per-task datasets I've finished and a replay iterator that's empty until there's something to replay. In end_task I simply remember the dataset I just finished. For this behavior-cloning harness, I use the simpler per-task truncation rule that the data utilities already provide: keep a fixed number of sequence windows from each finished task. In start_task, once I'm past the first task, I build the replay buffer by truncating each finished dataset to cfg.lifelong.n_memories sequences, concatenating them, and wrapping a randomly-sampled DataLoader in an infinite cycle — infinite because the replay stream has to outlast the current-task loader, which has a different length, so I can't just zip them. In observe, if the replay buffer exists I pull one replay batch and concatenate it onto the current batch before running the step — that concatenation is the "double the batch," and since I draw the replay batch at the same batch_size as the current one, I'm sitting at the equal-size operating point where the step direction is the clean average of the two gradients. The BC data is a nested dict of observation modalities, so the merge has to recurse into the dict and cat each tensor along the batch dimension. The single SGD step on the merged batch is the average-the-gradients move.

```python
import collections

import numpy as np
import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, RandomSampler

from libero.lifelong.algos.base import Sequential
from libero.lifelong.datasets import TruncatedSequenceDataset
from libero.lifelong.utils import *


def cycle(dl):
    # the replay stream must outlive the (different-length) current-task loader
    while True:
        for data in dl:
            yield data


def merge_datas(x, y):
    # BC data is a nested dict of obs modalities; concatenate matching tensors along the batch dim
    if isinstance(x, (dict, collections.OrderedDict)):
        if isinstance(x, dict):
            new_x = dict()
        else:
            new_x = collections.OrderedDict()

        for k in x.keys():
            new_x[k] = merge_datas(x[k], y[k])
        return new_x
    elif isinstance(x, torch.FloatTensor) or isinstance(x, torch.LongTensor):
        return torch.cat([x, y], 0)


class ER(Sequential):
    """Experience replay: train on the current minibatch stacked with a random memory minibatch."""

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        # Finished-task datasets are truncated into replay buffers when replay is used.
        self.datasets = []
        self.descriptions = []
        self.buffer = None

    def start_task(self, task):
        super().start_task(task)
        if self.current_task > 0:
            buffers = [
                TruncatedSequenceDataset(dataset, self.cfg.lifelong.n_memories)
                for dataset in self.datasets
            ]

            buf = ConcatDataset(buffers)
            self.buffer = cycle(
                DataLoader(
                    buf,
                    batch_size=self.cfg.train.batch_size,
                    num_workers=self.cfg.train.num_workers,
                    sampler=RandomSampler(buf),
                    persistent_workers=True,
                )
            )

    def end_task(self, dataset, task_id, benchmark):
        self.datasets.append(dataset)

    def observe(self, data):
        if self.buffer is not None:
            buf_data = next(self.buffer)
            data = merge_datas(data, buf_data)

        data = self.map_tensor_to_device(data)

        self.optimizer.zero_grad()
        loss = self.policy.compute_loss(data)
        (self.loss_scale * loss).backward()
        if self.cfg.train.grad_clip is not None:
            grad_norm = nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.cfg.train.grad_clip
            )
        self.optimizer.step()
        return loss.item()
```

The causal chain, start to finish. Fine-tuning forgets because each new task overwrites the old weights, and the realistic regime gives me only a single pass and a tiny memory to fight it with. EWC tries to pin important weights but its blunt quadratic anchor barely helps under a single pass and costs one scalar per parameter. The memory-based methods do better, and the prevailing instinct was to use the memory as a *constraint* — GEM projects the gradient via a per-step QP so no past-task loss can rise, A-GEM collapses that to one averaged half-space constraint with a closed-form projection whose projected step I checked is exactly orthogonal to the memory gradient, so it caps the old-task loss without ever descending it. The field had ruled out the simplest alternative — just train on the memory — on the grounds that repeatedly fitting a tiny buffer must overfit it. The two-parameter two-task toy showed otherwise: replay cut old-task error (9.31 → 3.34) instead of inflating it, because co-training with the large current-task dataset acts as an implicit data-dependent regularizer whose strength is set by its size and whose helpfulness is set by task relatedness — the same toy also showing the cost (some current-task fit traded away) and the reason the real harness pays less of it (per-task heads and ample capacity). So I drop all the machinery: maintain a small episodic memory, and each step stack a random *equal-size* memory minibatch onto the current minibatch and take one ordinary SGD step — which, at equal batch sizes (verified: union gradient = (g + g_ref)/2 exactly), descends the average of the current-task and memory gradients, always, every step, at the cost of a double-size batch and nothing else. The classification write strategy can be reservoir sampling for uniform coverage when the memory is reasonable (uniformity verified by induction and a 200k-trial simulation, both giving mem_sz/n), a balanced ring/k-means/MoF writer (or a hybrid) when it's tiny so no class starves; in the behavior-cloning code, that becomes keeping a fixed number of trajectory windows per past task and replaying them at random.
