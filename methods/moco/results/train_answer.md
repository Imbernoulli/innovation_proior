The problem I want to solve is the asymmetry between language and vision. In natural language, unsupervised pretraining simply works — word2vec, then GPT, then BERT — you pretrain on raw text and fine-tune and you win; in vision, supervised ImageNet pretraining is still king and the unsupervised methods trail. Stare at the signal and the reason emerges. Language is discrete: words and sub-word units come with a finite vocabulary, a *dictionary* sitting right there, so the unsupervised objective is essentially "predict the next or masked token over the vocabulary" — a clean softmax over a known set. Vision is continuous, high-dimensional, unstructured pixels with no pre-existing dictionary. If I want an analogous objective for images, I do not get a dictionary for free; I must *build* one out of the data itself. What I need from such an objective is threefold: it must require no human labels, it must produce a *standard* backbone with no task-specific architectural surgery so the features drop straight into detectors and segmenters, and it must scale even to uncurated, billion-image collections.

The useful reframing is that a whole family of recent self-supervised methods — instance discrimination, contrastive predictive coding, deep InfoMax, contrastive multiview coding — tell different stories on paper but do the same thing mechanically: there is an encoded *query* $q$, a collection of encoded *keys* $\{k_0, k_1, \dots\}$, exactly one of which (another augmented view of the same image) is the right match, and the encoder is trained so the query is close to its matching key and far from all the others. That is a *dictionary look-up*. So the question "what is a good objective?" becomes "what is a good dictionary?", and two properties pull against each other. The dictionary should be **large**: the keys are negatives sampling the continuous visual space, and a handful of negatives is too easy and uninformative while a huge set forces the representation to separate this instance from a dense sample of everything else. This is not a hunch — the InfoNCE objective proves it. Writing the loss over a query with context $c$, one positive $x_+ \sim p(x|c)$ and $N-1$ negatives $x_j \sim p(x)$, $L_N = -\mathbb{E}\big[\log \tfrac{f(x_+,c)}{\sum_{x_j\in X} f(x_j,c)}\big]$, the optimal classifier posterior collapses, after dividing through by $\prod_l p(x_l)$, to $p(d=i\mid X,c) = \tfrac{p(x_i|c)/p(x_i)}{\sum_j p(x_j|c)/p(x_j)}$, so the optimum is the density ratio $f(x,c)\propto p(x|c)/p(x)$. Substituting it back and applying Jensen to the resulting log-sum-exp gap (being careful that the valid inequality is in expectation, not the tempting pointwise $1+a(N-1)\ge Na$) gives the lower bound

$$I(x_+, c) \ge \log N - L_N,$$

which **tightens as $N$ grows**: more negatives literally means a tighter handle on the information the representation captures. The dictionary should also be **consistent**: the query is compared to every key by the same dot product and the loss treats those comparisons as commensurable, so if different keys were produced by very different encoder states, $q\cdot k_a$ and $q\cdot k_b$ no longer measure the same thing — inconsistent keys are noise injected straight into the target the query chases.

The existing mechanisms each buy one property and pay with the other. The **end-to-end** route back-propagates through both encoders and uses the other samples in the current minibatch as keys, so consistency is perfect — every key came out of the current encoder this step — but the dictionary size *equals the batch size*, so making $N$ large means large-batch optimization, which needs the linear learning-rate scaling rule plus warmup just to converge, loses a couple points of accuracy at batch 1024 without it, and may not extrapolate; the side door of enlarging the comparison set with many spatial positions mutilates the backbone and ruins transfer. The **memory bank** from instance discrimination keeps a lovely pretext task — every image is its own class, a non-parametric softmax over all $n\approx1.28\text{M}$ images approximated by NCE — and supplies negatives by sampling rows from a table holding one stored feature per dataset image, with a per-sample write-back $v_i \leftarrow (1-\lambda)v_i + \lambda f_\theta(x_i)$. Its dictionary is effectively the whole dataset, so size is solved, but each sampled row was last written the last time *that specific image* was drawn — up to an entire epoch ago — so the negatives are a grab-bag of features from encoder states scattered across the past epoch: maximally inconsistent. Crucially the bank's momentum is on the *stored feature of a sample*, not on the *encoder*, so it does nothing to make two rows written at two different times agree; and storing all $n$ features does not scale. End-to-end is consistent but small; the memory bank is large but inconsistent.

I propose **Momentum Contrast (MoCo)**, which engineers a dictionary that is large *and* consistent at once. The first piece attacks size. The reason the end-to-end dictionary cannot exceed the batch is that every key is in the computation graph, so all keys must live in memory and be encoded this step. But the keys are negatives, and a negative encoded a few steps ago is still a perfectly good negative as long as it is not too stale — so I simply *keep* the encoded keys from previous minibatches and reuse them. Concretely, maintain a **queue** of $K$ encoded keys: each step enqueue the current minibatch's keys and dequeue the oldest. Now $K$ is a free hyperparameter *decoupled from the batch size* — batch 256 with a queue of 65536 — and large-batch optimization is never touched. A FIFO queue specifically, rather than a random reservoir, is the right choice because the encoder drifts over time, so the oldest keys were made by the most outdated encoder; evicting oldest-first throws out exactly the least-consistent keys and keeps the freshest $K$.

That, however, reintroduces a milder version of the consistency problem: the queue spans the encoder states of the last several minibatches, and if the key encoder changes fast those states differ meaningfully. The naive key encoder copies the query encoder's weights every step (I cannot back-propagate into the queued keys — the gradient would have to flow to all $K$ keys across many past steps, the very intractability I was avoiding). But $f_q$ takes an SGD step every iteration and can move a lot, so a per-step copy makes $f_k$ lurch every iteration and the queue becomes inconsistent again. What I actually want is a key encoder that **tracks** the query encoder but *smoothly* — slowly enough that all $K$ keys in the queue come from a nearly identical encoder. A smoothed running version of a changing quantity is an exponential moving average, so I update the key encoder's parameters as

$$\theta_k \leftarrow m\,\theta_k + (1-m)\,\theta_q,$$

with only $\theta_q$ trained by back-propagation and $m$ close to 1 (default $m=0.999$). Each step $\theta_k$ moves only a fraction $1-m$ of the gap toward $\theta_q$. It takes about $K/N$ steps to cycle the queue — with $K=65536$, $N=256$ that is roughly 256 steps — and over those 256 steps with per-step motion scaled by $0.001$, $\theta_k$ barely budges, so every key currently in the queue was encoded by a $\theta_k$ almost identical to the current one. The slowness *is* the mechanism: $m=0$ degenerates to copying every step and the loss oscillates; $m=0.9$ still moves the key encoder $10\%$ of the gap each step and representation quality suffers; only $m$ genuinely near 1 works. The queue delivers large; the momentum-averaged key encoder keeps it consistent — neither end-to-end nor the memory bank could manage both.

The objective is InfoNCE in its operational form. For a query $q$, its single positive key $k_+$, and the $K$ keys in the dictionary,

$$L_q = -\log \frac{\exp(q\cdot k_+ / \tau)}{\sum_{i=0}^{K} \exp(q\cdot k_i/\tau)},$$

which is exactly the log-loss of a $(1+K)$-way softmax classifier trying to pick $k_+$ out of one positive and $K$ negatives. So I implement nothing exotic: concatenate the one positive similarity with the $K$ negative similarities, divide by $\tau$, and feed it to a standard cross-entropy loss whose target index is the position of the positive — and if the positive is always placed first, the label is just $0$ for every example. The pretext task is instance discrimination in two-views form: take an image, apply random data augmentation twice (random resized crop to $224\times224$, color jitter, horizontal flip, grayscale — strong enough to force learning content rather than low-level cues), send one view through the query encoder and the other through the key encoder; they are a positive pair, every other queued key is a negative. The encoder is a *standard* ResNet whose final fully-connected layer outputs a fixed 128-D vector that I L2-normalize, putting every feature on the unit sphere so $q\cdot k$ is cosine similarity in $[-1,1]$ and the temperature $\tau=0.07$ controls how sharply the softmax concentrates. Keeping the backbone standard — no patchifying, no custom receptive fields — is deliberate: I may leave a little linear-probe accuracy on the table versus a backbone tortured to maximize the pretext task, but I gain a backbone that drops straight into detectors and segmenters, and transferability is the goal.

One subtlety nearly breaks it: Batch Normalization. BN couples the samples within a batch through the batch statistics, and the mean and variance of a batch are a signature of *which samples are in it*. If a query and its positive key are processed in the same batch, both encoders see statistics encoding "the positive partner is here," and the network can exploit that intra-batch leakage to identify the positive without learning anything about content — the telltale sign being pretext training accuracy shooting past $99.9\%$ almost immediately while kNN validation accuracy drops. The leak is that $q$ and its positive $k$ see the *same* BN statistics, so I make them see *different* ones. Training is on multiple GPUs with BN computed per-GPU, so before feeding the key view through $f_k$ I **shuffle the sample order across GPUs**, encode, then unshuffle to restore correspondence: the statistics that touch a given query's positive key now come from a different sub-batch than those touching the query, the signature is destroyed, and the network is forced to actually learn. The query branch's order is left untouched. (Mechanically on a distributed setup: all-gather the keys, draw a random permutation on rank 0 and broadcast it so every rank agrees, index this rank's slice in the shuffled order, encode, then scatter back via the argsort of the permutation.) The memory bank never needed this because its positive came from a past batch, not the current one.

The forward pass assembles cleanly. Encode the query view $q = \text{normalize}(f_q(\text{im}_q))$ — only this branch is learned. Then under no-grad: first run the momentum update so the keys come from the freshly-nudged key encoder, shuffle the key batch across GPUs, encode and normalize $k$, unshuffle, and detach. Build the logits: the positive logit is the per-example dot product of $q$ with its own $k$, an $N\times1$ column; the negative logits are $q$ against every column of the queue, an $N\times K$ matrix; concatenate to $N\times(1+K)$, divide by $\tau$, cross-entropy against the all-zeros label. Back-propagation updates only $f_q$. Finally enqueue this batch's $k$ and dequeue the oldest. The key encoder is initialized as an exact copy of the query encoder and the queue with random normalized vectors. The whole chain holds together: build a dictionary out of image data; view contrastive learning as look-up; the bound $I\ge\log N - L$ demands a large dictionary and the dot-product demands a consistent one; a FIFO queue decouples size from the batch while respecting freshness; a momentum EMA key encoder keeps the queued keys consistent; the objective is a $(1+K)$-way cross-entropy with the positive at index 0; and shuffling BN closes the last shortcut.

```python
import torch
import torch.nn as nn


class MoCo(nn.Module):
    """Query encoder + momentum key encoder + a queue dictionary."""
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07):
        super().__init__()
        self.K, self.m, self.T = K, m, T

        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data.copy_(p_q.data)      # key encoder = copy of query encoder
            p_k.requires_grad = False     # key encoder is not updated by gradient

        self.register_buffer("queue", nn.functional.normalize(torch.randn(dim, K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data = p_k.data * self.m + p_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_ptr[0] = (ptr + batch_size) % self.K

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // x.shape[0]
        idx_shuffle = torch.randperm(batch_size_all, device=x.device)
        torch.distributed.broadcast(idx_shuffle, src=0)
        idx_unshuffle = torch.argsort(idx_shuffle)
        idx_this = idx_shuffle.view(num_gpus, -1)[torch.distributed.get_rank()]
        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        x_gather = concat_all_gather(x)
        num_gpus = x_gather.shape[0] // x.shape[0]
        idx_this = idx_unshuffle.view(num_gpus, -1)[torch.distributed.get_rank()]
        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        q = nn.functional.normalize(self.encoder_q(im_q), dim=1)        # N x C

        with torch.no_grad():
            self._momentum_update_key_encoder()
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k = nn.functional.normalize(self.encoder_k(im_k), dim=1)    # N x C
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)               # N x 1
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])  # N x K
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T                   # N x (1+K)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)  # positive = 0

        self._dequeue_and_enqueue(k)
        return logits, labels


@torch.no_grad()
def concat_all_gather(tensor):
    gathered = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered, tensor, async_op=False)
    return torch.cat(gathered, dim=0)
```

```python
import torchvision.transforms as T
import torchvision.models as models


class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]


aug = T.Compose([
    T.RandomResizedCrop(224, scale=(0.2, 1.)),
    T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    T.RandomGrayscale(p=0.2),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_transform = TwoCropsTransform(aug)

model = MoCo(models.resnet50).cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)

for view_q, view_k in loader:                 # each image -> two augmented views
    logits, labels = model(im_q=view_q.cuda(), im_k=view_k.cuda())
    loss = criterion(logits, labels)          # InfoNCE
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
