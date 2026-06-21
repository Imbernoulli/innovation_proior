The problem is to train a dense retriever without any labeled query-document pairs while still matching or beating BM25, the unsupervised lexical baseline. Lexical retrievers like BM25 are robust and need no labels, but they fail when a query and a relevant document use different words or scripts. Supervised dense retrievers such as DPR and ANCE close that lexical gap, but they require large human-labeled datasets and transfer poorly out of domain. In non-English languages there may be no large retrieval labels at all, so the supervised recipe cannot even be run. The only reliable supervision left in raw text is identity: two spans sampled from the same document should be closer in embedding space than spans sampled from different documents.

Earlier self-supervised attempts such as the Inverse Cloze Task build a positive pair from a sentence and the surrounding context, but they perform below BM25 zero-shot. A likely reason is that ICT deliberately removes the query tokens from the key, so the model never sees exact lexical overlap between positive views. That discards the very signal that makes BM25 strong. It also makes the query and key follow very different length distributions, which is unlike real retrieval where a query and a passage can share any wording. A better augmentation is independent random cropping: sample two contiguous spans from the same document, optionally delete a small fraction of tokens, and declare them a positive pair. Because the spans are sampled independently, they sometimes overlap, teaching the model to reward shared rare terms, and sometimes do not, forcing it to learn semantic paraphrase. This keeps both the lexical and the semantic pathways open.

The proposed method is Contriever. It treats unsupervised dense retrieval as an instance-discrimination problem. From each unlabeled document, two independent contiguous token spans are sampled as a positive pair. The query and key are encoded by a single shared BERT-base encoder, and each sequence is collapsed to one vector by mean pooling over the last-layer hidden states. Mean pooling treats every token's contribution evenly, which is a better inductive bias for retrieval than relying on a single [CLS] token when any part of a long passage might match the query. A shared encoder, rather than separate query and document towers, forces queries and documents into the same space symmetrically and transfers more robustly across domains and languages.

The model is trained to score the positive key above a large pool of negative keys using the InfoNCE loss. To obtain many negatives without using an infeasibly large batch, Contriever uses a momentum encoder: a slow exponential moving average of the query encoder maintains a queue of recent key embeddings. The queue provides tens of thousands of consistent negatives, while the momentum update keeps those negatives from drifting across training steps. Gradients flow only through the query side; the keys are detached. The loss is implemented as cross-entropy against a fixed label of zero, with the positive logit placed first and the queued negatives following. A small temperature sharpens the discrimination, and optional L2 normalization turns the dot product into cosine similarity for stable scaling.

Training is straightforward: AdamW with a small learning rate, a batch size in the low thousands, and hundreds of thousands of steps on a mixture of raw corpora such as Wikipedia and CCNet. No labeled pairs are used at any point. A multilingual variant initializes from multilingual BERT, samples languages uniformly, and adds a warmup-then-linear-decay schedule for stability. After training, document embeddings are pre-computed once and searched with FAISS, giving a scalable bi-encoder retriever that can be pointed at any domain or language without collecting annotations.

```python
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers


class Contriever(transformers.BertModel):
    """Bi-encoder tower: BERT -> one vector via mean pooling. Shared by query and key."""
    def __init__(self, config):
        super().__init__(config, add_pooling_layer=False)

    def forward(self, input_ids, attention_mask, normalize=False):
        out = super().forward(input_ids=input_ids, attention_mask=attention_mask)
        last = out["last_hidden_state"]
        last = last.masked_fill(~attention_mask[..., None].bool(), 0.0)
        emb = last.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        if normalize:
            emb = F.normalize(emb, dim=-1)
        return emb


def token_delete(tokens, p=0.10):
    if p <= 0.0 or len(tokens) <= 1:
        return tokens
    kept = [tok for tok in tokens if random.random() > p]
    return kept if kept else [tokens[random.randrange(len(tokens))]]


def build_positive_pair(tokens, low=0.05, high=0.5, delete_prob=0.10):
    """Two independent crops of one document = a label-free positive pair."""
    n = len(tokens)
    if n == 0:
        return [], []

    def crop():
        length = max(1, min(n, int(round(n * random.uniform(low, high)))))
        start = random.randint(0, n - length)
        return token_delete(tokens[start:start + length], delete_prob)
    return crop(), crop()


class MoCo(nn.Module):
    """InfoNCE with a momentum-encoder queue."""
    def __init__(self, opt):
        super().__init__()
        self.temperature = opt.temperature
        self.momentum = opt.momentum
        self.queue_size = opt.queue_size
        self.label_smoothing = opt.label_smoothing
        self.norm = opt.normalize
        self.encoder_q = Contriever.from_pretrained(opt.model_id)
        self.encoder_k = copy.deepcopy(self.encoder_q)
        for p in self.encoder_k.parameters():
            p.requires_grad = False
        self.register_buffer("queue", F.normalize(torch.randn(opt.dim, self.queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update(self):
        for pq, pk in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            pk.data = pk.data * self.momentum + pq.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        bsz = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % bsz == 0
        self.queue[:, ptr:ptr + bsz] = keys.T
        self.queue_ptr[0] = (ptr + bsz) % self.queue_size

    def forward(self, q_tokens, q_mask, k_tokens, k_mask):
        q = self.encoder_q(q_tokens, q_mask, normalize=self.norm)
        with torch.no_grad():
            self._momentum_update()
            k = self.encoder_k(k_tokens, k_mask, normalize=self.norm)
        l_pos = torch.einsum("nc,nc->n", q, k).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", q, self.queue.clone().detach())
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
        loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        self._dequeue_and_enqueue(k)
        return loss


def train(trainer, loader, opt):
    optim = torch.optim.AdamW((p for p in trainer.parameters() if p.requires_grad), lr=opt.lr)
    for batch in loader:
        loss = trainer(batch["q_tokens"], batch["q_mask"], batch["k_tokens"], batch["k_mask"])
        loss.backward(); optim.step(); optim.zero_grad()
```
