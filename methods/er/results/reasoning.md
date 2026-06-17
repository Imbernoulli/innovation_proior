Let me start from what actually goes wrong. I have a stream of tasks coming at me one at a time — task 1, then task 2, and so on — and I have to keep solving all of them. The brutal fact is that if I just keep training the same network as each task arrives, the moment I do gradient descent on task 2 I am rewriting the very weights that solved task 1, and task 1's accuracy falls off a cliff. Catastrophic forgetting. And I cannot escape it the easy way, because the rules of the regime I care about forbid it: I get to see each task's data exactly once, a single pass, online; I have a task id so I can pick the right output head; and I am allowed only a small fixed buffer in which to stash a few past examples. That small-buffer constraint is the whole point — if I could keep every task's full dataset around I would just be doing multi-task learning and there would be no problem. So the question is sharp: how do I lose as little of the old tasks as possible, at almost no extra cost over plain fine-tuning, using only a handful of remembered examples?

What do people do today? Two camps. One camp says: protect the weights that mattered. After I finish a task, measure how important each weight was and, on the next task, add a penalty that drags each weight back toward its old value in proportion to that importance. That is EWC: minimize L_t(θ) + Σ_i (λ/2) F_i (θ_i − θ*_i)^2, where F_i is the Fisher-information diagonal, the expected squared gradient of the log-likelihood, a per-weight importance, and θ* are the old weights. I can see why this should help — the weights that carried task 1 get pinned. But two things bother me. It stores an importance scalar for every parameter, so its memory is the size of the network in the worst case. And the anchor is blunt: it pulls each weight toward its old scalar value weighted by a scalar importance, but it has no idea which *directions* in weight space actually preserve the old function — and empirically, under a single pass, it barely beats plain fine-tuning and it is finicky about λ. So regularization alone is not carrying me.

The other camp says: keep a few old examples and use them to stop the old-task loss from rising. This is where the action is in the single-pass studies, and the strongest version is the gradient-constraint idea. GEM phrases it cleanly: minimize the current-task loss subject to ℓ(θ, M_k) ≤ ℓ(θ^{t-1}, M_k) for every past task k — never let the loss on any past task's stored memory go up. The mechanics: at each step compute a gradient g_k on each past task's memory, and if my proposed step g would make an old-task loss climb — which shows up as ⟨g, g_k⟩ < 0, the update pointing against the direction that reduces task k's loss — then don't take g; take the closest vector g̃ in L2 that keeps all the inner products nonnegative. That nearest-feasible-vector problem is a quadratic program. In the dual it has t−1 variables, one per past task, and to set it up I have to assemble the matrix G = (g_1, …, g_{t−1}) of all past-task gradients, P-dimensional each, at every single step. It even allows positive backward transfer, old-task loss going down, which is nice. But the cost is ugly: a QP plus a (t−1)×P gradient matrix every step, and it only gets worse as tasks pile up.

A-GEM trims this to the bone. Instead of t−1 constraints, use one *averaged* constraint: draw a single random minibatch from the union of all the memories, take its gradient g_ref, and require only ⟨g̃, g_ref⟩ ≥ 0. Now the projection is closed form — if g^T g_ref ≥ 0 the constraint is already satisfied, keep g; otherwise project,

  g̃ = g − (g^T g_ref / g_ref^T g_ref) g_ref,

which is just subtracting off the part of g that points against the memory gradient. One extra gradient, one inner product, no QP, no stored G. It is orders of magnitude faster than GEM and about as good on the single pass. So A-GEM is the thing to beat: it is cheap and it works.

But let me stare hard at what A-GEM is actually doing with the memory, because something nags at me. The memory only ever acts as a *veto*. Most of the time g^T g_ref ≥ 0 and the memory does literally nothing — I take my plain current-task step. The only time the memory speaks is when there is a violation, and even then all it does is delete a component of my current-task gradient; it removes the part that would have hurt the old tasks. It never actually *pushes the old-task loss down*. It guards a ceiling. And there is a documented symptom that fits this exactly: a method that uses the memory only as a constraint underfits. Even after a thousand steps, A-GEM's accuracy on the stored memory examples does not reach 100%, and its current-task training fit also remains short of full direct fitting of the available data. The constrained update is so cautious it doesn't fully fit either the new task or the memory. That underfitting is leaving performance on the table.

So here is the obvious thing I keep wanting to try and keep being told not to: why not just *train on the memory*? Don't veto, don't project — put the old examples into the batch and descend their loss like any other data. If A-GEM underfits because it only guards a ceiling on the memory loss, then driving that loss down directly should fix the underfitting. Concretely: each step, alongside my current-task minibatch B_n, draw a minibatch B_M from the memory, and just take an ordinary SGD step on the *union* B_n ∪ B_M. That's it. No QP, no projection, no inner product — computationally simpler than A-GEM, because A-GEM also computes g_ref and then does extra arithmetic, whereas I just feed both batches through the same forward/backward and step once.

I hit a wall — this is exactly the move the field warned against. The gradient-constraint people said it in plain words: "minimizing the loss at the current example together with the loss on the episodic memory results in overfitting to the examples stored in the memory." And the worry is not stupid. My memory is *tiny* — a few examples per class. If every single step I take a gradient step on those same few examples, over and over, for the whole length of a task, that is the canonical setup for memorizing them and generalizing to nothing. I will drive the memory loss to zero, sure, but the memory loss being zero is worthless; what I need is accuracy on the *held-out* old-task test set, and memorizing five images per class should give me nothing there. So on its face, training directly on the tiny memory should overfit and lose. That is precisely why everyone built constraint machinery instead of just adding the examples to the batch.

But let me not take the warning on faith — let me actually reason about whether it's true, because if it's wrong the whole expensive constraint apparatus is unnecessary. The claim is: repeatedly fitting a handful of stored examples must overfit them. When is that claim actually right? It's right when those few examples are *all* I'm training on. If I sat and trained only on M_1, the five-per-class memory of task 1, with nothing else, then yes — the network would happily drive those to zero loss and contort itself to do it, and test accuracy on task 1 would crater. I believe that; that's textbook small-sample overfitting.

The thing is, that's not the situation I'm in. When I'm learning task 2, I'm not training on M_1 alone — I'm training on M_1 *together with* the full, large dataset D_2 of task 2. So at every step the gradient is a blend: a little bit of "fit these five old examples" and a lot of "fit this fresh, large batch of new-task data." And now I have to ask what that large companion does to the small-sample overfitting. The five old examples want to bend the network into some idiosyncratic shape that nails them; but simultaneously the thousands of task-2 examples are constraining the network to remain a sensible function that fits *them*. The network can't go fully idiosyncratic on M_1 because doing so would wreck its fit on D_2, and D_2's gradient is large and persistent. The big current-task dataset is acting as a regularizer on the repeated learning of the tiny memory — not an explicit penalty I added, but an implicit, data-dependent one that comes for free from co-training.

Let me pressure-test that with a thought experiment I can reason all the way through, because "regularizer" is a hopeful word and I want to know when it holds and when it breaks. Take the cleanest case: two tasks, T_1 and T_2, and let me dial the *relatedness* between them continuously — say T_2 is T_1's images rotated by some angle, so 10° is nearly the same task and 90° is a very different one. Memory M_1 is a few T_1 examples. Three ways to train and what each should do to T_1's test accuracy.

Train on M_1 alone: the five-per-class case. Overfit, test accuracy on T_1 collapses — from whatever I had after task 1 down to something poor. Expected.

Train on D_2 alone, no memory at all — this is just the fine-tune baseline. What happens to T_1? It depends on relatedness. If T_2 is nearly T_1 (small rotation), then fitting D_2 *is* roughly fitting T_1, so T_1's accuracy can even go *up* — positive transfer, no memory needed. But if T_2 is unrelated (large rotation), fitting D_2 drags the network away from the T_1 solution and T_1 is forgotten. So D_2-alone is good when tasks are similar, bad when they're different — which is just catastrophic forgetting restated.

Now train on M_1 ∪ D_2, my proposed replay step. What should happen? The D_2 part supplies the data-dependent regularization; the M_1 part keeps pulling the function back toward fitting actual T_1 points. When the tasks are similar, D_2 already helps T_1, and M_1 can only sharpen that — I should beat fine-tune. When the tasks are different, D_2 alone would forget T_1, but now M_1 is in every batch insisting on T_1 points while D_2 stops M_1 from being memorized into uselessness — so I should beat fine-tune here too. The prediction is that direct replay beats the fine-tune baseline on T_1 *regardless* of relatedness, and that's the crux: the memory, far from overfitting, is rescued from overfitting by its large companion, and in turn rescues the old task from being forgotten.

There has to be a failure edge, and I can locate it by pushing the relatedness past zero into *adversarial*. Picture rotating a 2 until it looks like a 5 — now T_2's labels actively contradict T_1's on similar-looking inputs. Then D_2 is no longer a benign regularizer on M_1; it is pulling the function toward answers that are *wrong* for T_1. In that regime the "regularizer" can hurt — co-training on M_1 ∪ D_2 could end up worse for T_1 than fitting M_1 alone, because the companion data is dragging it the wrong way. So the data-dependent regularization has two knobs, and naming them tells me the whole story: the *amount* of current-task data sets the regularizer's strength (more D_2 = stronger pull away from memorizing M_1), and the *relatedness* of the tasks sets whether that pull is helpful, neutral, or harmful. Helpful and neutral cover the normal case; only the near-adversarial corner is bad, and that corner is rare.

This also dissolves the original objection cleanly. The constraint camp may be right that I will fit the memory very closely; those few stored examples can be memorized. But they are wrong that memorization has to imply bad *generalization*. Fitting the memory and generalizing to the old task are not in conflict, because the generalization is supplied by the companion data, not by the memory size. And now I can see why A-GEM, which deliberately avoids training directly on the memory, leaves accuracy on the table: by only vetoing gradients it never treats reducing the memory loss as its own goal, and its constrained steps underfit the new task too, which means it never fully exposes itself to the beneficial co-training regularization in the first place. The cautious method is throwing away the exact mechanism that makes the reckless method work.

So the move is to stop being clever. Drop the constraint, drop the projection, drop the QP — just put the memory into the batch. Compared to plain fine-tune, this is two small changes: maintain an episodic memory of past examples, and at each step double the batch by stacking a random memory minibatch onto the current minibatch, then take one ordinary gradient step. Let me make sure I understand the optimization geometry of "double the batch," because that's the whole algorithm. If B_n and B_M have the same size, and g and g_ref are the gradients of the mean losses on those two minibatches, then the mean loss on the concatenation has gradient (g + g_ref) / 2. If the implementation sums losses instead, the scale is g + g_ref; either way the descent direction is the average of the current-task and memory directions, with the constant absorbed into the learning rate. So where A-GEM uses g_ref only as a half-space constraint and only when violated, ER always moves along the average direction: always descending the current task *and* the memory together. That is the precise difference — average, not project; use the memory every step, not only on a violation; descend the old-task loss, not merely cap it. And it costs essentially nothing: one forward/backward on a batch of twice the size, no extra gradient solve.

Now, the memory is tiny and the stream is long and of unknown length, so I have to decide *what to keep* in it — the write strategy — and this matters more the smaller the memory is. The classic tool for "keep a uniform random subset of a stream of unknown length in a fixed buffer" is reservoir sampling: when the n-th example arrives, if the buffer isn't full, append it; otherwise pick a slot index i uniformly in [0, n), and if i is inside the buffer, overwrite slot i with the new example. A quick check that this gives uniformity: after n items, each stored item is present with probability mem_sz/n, so the buffer is an unbiased random sample of everything seen. That's the right default when the memory is reasonably sized. But I can predict its failure when the memory is *very* small relative to the number of tasks: with, say, one slot per class on average, pure randomness will sometimes evict every example of an early class, and once a class has zero representatives in the buffer it gets no replay at all and forgetting on it spikes. So in the tiny regime, uniform randomness is the wrong objective — what I want is *coverage*: guarantee at least a few examples per class.

That points to a balanced writer. The simplest is a ring buffer: give each class its own fixed-size FIFO of size mem_sz/C and keep the last few examples of each class. This guarantees every class is represented — exactly the property reservoir loses when the memory is tiny — at the cost that old-task stored examples never change (slightly more overfitting risk, now harmless given the co-training argument) and the memory is underused early when few classes have been seen. I can do fancier balanced writers in the same spirit: online k-Means in the pre-classifier feature space, storing per class the examples closest to the k centroids, for better feature coverage; or mean-of-features, storing per class the examples closest to a running average feature vector, putting samples near the class mode. All three share the balance guarantee and differ only in *which* representatives they keep. The trade is clear: reservoir wins once the memory is comfortably sized (true randomness, no stale samples), balanced writers win when the memory is tiny (no class ever starves). Which suggests a hybrid that needs no advance knowledge of the task count: start with reservoir, and the moment any class drops below a minimum number of stored examples, switch to a balanced ring-buffer-style scheme that protects the starving classes — best of both across memory sizes.

Let me write the core procedure as a loop, because the whole contribution is two lines longer than fine-tune:

  M ← empty buffer of size mem_sz ; n ← 0
  for task t = 1 … T:
    for each minibatch B_n drawn (without replacement) from D_t:
      B_M ← random minibatch from M            # the only-if-M-nonempty replay batch
      θ ← SGD-step(θ, B_n ∪ B_M, lr)           # one step on the stacked batch: direction (g + g_ref) / 2
      M ← UpdateMemory(mem_sz, t, n, B_n)      # reservoir / ring / k-means / MoF / hybrid
      n ← n + |B_n|

Two modifications to fine-tune: update the memory, and double the batch. Everything else is the existing training loop.

Now let me land this in the actual harness I'll run, which is a behavior-cloning continual learner: the same lifecycle hooks plain fine-tuning already uses — __init__, start_task before each task, observe on every minibatch, end_task after each task — where observe does to-device, zero_grad, loss = policy.compute_loss(data), backward, optional grad-clip, step. I only need to fill the hooks. In __init__ I hold the per-task datasets I've finished and a replay iterator that's empty until there's something to replay. In end_task I simply remember the dataset I just finished. For this behavior-cloning harness, I use the simpler per-task truncation rule that the data utilities already provide: keep a fixed number of sequence windows from each finished task. In start_task, once I'm past the first task, I build the replay buffer by truncating each finished dataset to cfg.lifelong.n_memories sequences, concatenating them, and wrapping a randomly-sampled DataLoader in an infinite cycle — infinite because the replay stream has to outlast the current-task loader, which has a different length, so I can't just zip them. In observe, if the replay buffer exists I pull one replay batch and concatenate it onto the current batch before running the step — that concatenation is the "double the batch," and the BC data is a nested dict of observation modalities, so the merge has to recurse into the dict and cat each tensor along the batch dimension. The single SGD step on the merged batch is the average-the-gradients move.

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

The causal chain, start to finish. Fine-tuning forgets because each new task overwrites the old weights, and the realistic regime gives me only a single pass and a tiny memory to fight it with. EWC tries to pin important weights but its blunt quadratic anchor barely helps under a single pass and costs one scalar per parameter. The memory-based methods do better, and the prevailing instinct was to use the memory as a *constraint* — GEM projects the gradient via a per-step QP so no past-task loss can rise, A-GEM collapses that to one averaged half-space constraint with a closed-form projection. But the constraint only ever vetoes; it caps the old-task loss without driving it down, and that caution makes A-GEM underfit both the memory and the new task. The field had ruled out the simplest alternative — just train on the memory — on the grounds that repeatedly fitting a tiny buffer must overfit it. Reasoning through the two-task rotation thought experiment showed that's only true when the memory is trained on *alone*: co-training with the large current-task dataset turns that dataset into an implicit data-dependent regularizer whose strength is set by its size and whose helpfulness is set by task relatedness, so memorizing the memory and generalizing to the old task stop being in conflict (the only bad corner is near-adversarial tasks). So I drop all the machinery: maintain a small episodic memory, and each step stack a random memory minibatch onto the current minibatch and take one ordinary SGD step — which descends the average of the current-task and memory gradients, always, every step, at the cost of a double-size batch and nothing else. The classification write strategy can be reservoir sampling for uniform coverage when the memory is reasonable, a balanced ring/k-means/MoF writer (or a hybrid) when it's tiny so no class starves; in the behavior-cloning code, that becomes keeping a fixed number of trajectory windows per past task and replaying them at random.
