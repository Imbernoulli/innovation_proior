A-GEM cleared EWC, and the numbers say exactly how and where it stalled. avg_final_success climbed from $0.045$ to $0.22$ and auc from $0.0487$ to $0.2166$ — four to five times the floor, confirming that putting real old data into the update beats a decayed weight-spring — and fwt rose sharply, $0.21 \to 0.68$, so the constrained learner now fits each incoming task well. But two things in the row stop me. avg_final $0.22$ is still low — almost four out of five rollouts on the early tasks fail. And the tell: nbt is $0.573$, *far higher* than EWC's $0.174$, which looks paradoxical for a method whose whole job is to stop the average old loss from rising — until I remember what A-GEM actually does. The memory acts only as a *veto*: on the common step where $\langle g, g_{\text{ref}}\rangle \geq 0$ it does nothing and I take the plain current-task gradient; only on a violation does it shave off a component. It never *pushes the old-task loss down*; it guards a ceiling. So the high nbt and the high fwt fit together — the method fits the new task aggressively and merely caps, rather than reduces, the damage to old tasks. This is precisely the underfitting-the-memory failure I predicted: the veto is leaving old-task performance on the table.

I propose **Experience Replay** (ER). Stop vetoing, and just *train on* the old examples: do not measure a reference direction and project against it — put the old examples *into the batch* and descend their loss like any other data. Each step, alongside the current-task minibatch, draw a minibatch from the memory and take one ordinary BC step on the *union*. That is the whole algorithm: no projection, no inner product, no two-backward sequencing — and it is computationally *simpler* than A-GEM, which still computes $g_{\text{ref}}$ and then does the projection arithmetic; here both batches go through one forward/backward and one step. The optimization geometry of "double the batch" is the entire mechanism: if the current and memory minibatches have gradients $g$ and $g_{\text{ref}}$, the gradient of the loss on the concatenation is the average direction $(g + g_{\text{ref}})$ up to a constant absorbed into the learning rate. Where A-GEM used $g_{\text{ref}}$ only as a half-space constraint and only on a violation, ER *always* moves along the sum of the current-task and memory directions — always descending the old loss, every step, not merely capping it.

This is the move the gradient-constraint line warned against in plain words — minimizing the loss on the current example together with the loss on the episodic memory results in overfitting to the stored examples — so I have to actually reason about whether that warning is true before committing, because if it is wrong the whole projection apparatus was unnecessary. My memory is a thousand sequence windows per finished task, a small fixed slice. If every step I take a gradient step on those same stored windows for fifty epochs, that is the canonical setup for memorizing them and generalizing to nothing — I would drive the memory loss to zero, but memory loss being zero is worthless; what I need is success on the *held-out* old-task rollouts. The claim against me is: repeatedly fitting a small stored set must overfit it. When is that actually right? It is right when those stored windows are *all* I am training on. But that is not my situation: when I learn task $k$ I am not training on the memory alone — I am training on the memory *together with* the full, large dataset of task $k$. So every step's gradient is a blend, a little "fit these old windows" and a lot "fit this fresh, large batch of new-task demonstrations," and the large companion changes everything. The old windows want to bend the policy into an idiosyncratic shape that nails them, but the thousands of task-$k$ windows simultaneously constrain it to remain a sensible function that fits *them*; it cannot go fully idiosyncratic on the memory without wrecking its fit on task $k$, whose gradient is large and persistent. The big current-task dataset acts as an implicit, data-dependent regularizer on the repeated learning of the small memory — not a penalty I add, but one that comes for free from co-training.

Pressure-test that, because "regularizer" is a hopeful word. Three ways to train an early task's rollout success. Train on the memory alone: overfit, success collapses — the small-sample case. Train on the new task alone, no memory: that is plain fine-tuning, and on LIBERO-Spatial the ten tasks share the *same* manipulation skill and differ only in spatial layout, so they are quite related — which is why fine-tuning's fwt was never the problem; fitting a new layout partly helps the old ones, and the failure was retention. Now train on memory $\cup$ new-task: the new-task part supplies the data-dependent regularization, the memory part keeps pulling the policy back toward fitting old-layout windows, and because the tasks are related the new data already helps the old skill while the memory sharpens it. So I should beat fine-tuning *and* beat the veto, because I am now reducing the old loss rather than capping it. The only regime where the companion data hurts is the near-adversarial corner where new-task labels contradict old ones on similar inputs, and LIBERO-Spatial is the opposite of that corner. This also dissolves the original objection cleanly: the constraint camp may be right that I will fit the memory closely — those windows can be memorized — but they are wrong that memorization implies bad *generalization*, because the generalization is supplied by the companion data, not the memory size. And it explains A-GEM's $0.573$ nbt directly: by only vetoing gradients it never treats reducing the memory loss as a goal, and its cautious steps underfit the new task too, so it never even exposes itself to the beneficial co-training. The cautious method threw away the exact mechanism that makes the reckless one work.

So I drop all the machinery — no constraint, no projection, no flat-gradient buffers, no two-backward dance — and make two changes from plain fine-tuning: maintain the episodic memory, and each step stack a random memory minibatch onto the current minibatch and take one ordinary step. Landing it in *this* harness, the implementation does none of the *write*-strategy care of the general ER recipe (reservoir sampling, balanced ring or k-means writers): `end_task` simply appends the whole finished dataset to a list, and `start_task` past task 0 truncates each finished dataset to its first `n_memories = 1000` sequence windows with `TruncatedSequenceDataset`, `ConcatDataset`s them, and wraps a `RandomSampler` `DataLoader` in `cycle(...)`. So the buffer is "the first 1000 windows of every past task, sampled at random" — a fixed, balanced-by-construction per-task slice, no reservoir, no eviction. The `cycle` is needed because the replay stream must outlast the current-task loader, which has a different length, so I cannot zip them. `observe` is the minimal change: if the buffer exists, pull one replay batch with `next(self.buffer)` and `merge_datas(data, buf_data)` — the BC data is a nested dict of observation modalities, so the merge recurses and concatenates each tensor along the batch dimension — then the ordinary to-device, zero_grad, `compute_loss`, `(loss_scale*loss).backward()`, grad-clip, step. The single step on the merged batch *is* the average-the-gradients move; that one concatenation is the entire contribution. On task 0 the buffer is `None`, so it is plain BC, and `persistent_workers=(num_workers > 0)` is the one robustness guard so the DataLoader does not error when `num_workers = 0` (which the eval config uses).

The delta from A-GEM is the whole point: where A-GEM vetoed steps pointing against the average old direction and left the old loss merely capped (nbt $0.573$, avg_final $0.22$), ER puts the old windows in the batch and descends their loss every step, with the large current-task data preventing the small memory from being memorized into uselessness — the co-training A-GEM's caution never reached. I expect avg_final to jump well past $0.22$, into the high-$0.4$s or beyond, since I am now actively recovering old-task success rather than capping its decay, with auc following up from $0.2166$ and nbt falling substantially below $0.573$ because every step reduces the old loss instead of only vetoing increases. I would not be surprised if fwt dips slightly from $0.68$, since half of every batch is now old-task data so the per-step pull toward the new task is diluted — but that is the correct trade, a little forward speed for a lot of retention, and on a metric averaged over all ten tasks after the last one, retention is what dominates.

```python
# EDITABLE region of LIBERO/libero/lifelong/algos/custom.py (lines 37-65) — step 3: ER
class Custom(Sequential):
    """ER (Experience Replay) lifelong learning algorithm."""

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        self.n_memories = 1000
        self.datasets = []
        self.buffer = None

    def start_task(self, task):
        super().start_task(task)
        if self.current_task > 0:
            buffers = [
                TruncatedSequenceDataset(dataset, self.n_memories)
                for dataset in self.datasets
            ]
            buf = ConcatDataset(buffers)
            self.buffer = cycle(
                DataLoader(
                    buf,
                    batch_size=self.cfg.train.batch_size,
                    num_workers=self.cfg.train.num_workers,
                    sampler=RandomSampler(buf),
                    persistent_workers=(self.cfg.train.num_workers > 0),
                )
            )

    def observe(self, data):
        if self.buffer is not None:
            buf_data = next(self.buffer)
            data = merge_datas(data, buf_data)

        data = self.map_tensor_to_device(data)

        self.optimizer.zero_grad()
        loss = self.policy.compute_loss(data)
        (self.loss_scale * loss).backward()
        if self.cfg.train.grad_clip is not None:
            nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.cfg.train.grad_clip
            )
        self.optimizer.step()
        return loss.item()

    def end_task(self, dataset, task_id, benchmark, env=None):
        self.datasets.append(dataset)
```
