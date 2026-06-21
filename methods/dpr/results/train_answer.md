Open-domain question answering is usually split into retrieval followed by reading. The reader is already strong when it receives the right passage, so the entire end-to-end exact-match score is capped by whether the retriever can find that passage. Sparse lexical methods like BM25 are fast and precise on rare keywords, but they fail when questions and passages use different words for the same meaning: "bad guy" and "villain" share no tokens, so the correct passage is never returned. Dense retrieval could fix this by learning semantic similarity, but prior work believed that beating BM25 required expensive self-supervised retriever pretraining such as the Inverse Cloze Task, and often froze the passage encoder after that pretraining. The real question is whether a simple dual-encoder can surpass BM25 using only ordinary question-passage supervision.

I propose Dense Passage Retrieval, or DPR. It trains two independent BERT encoders, one for questions and one for passages, each producing a single [CLS] vector of dimension 768. Similarity between a question and a passage is the plain inner product of their vectors. Because the two encoders do not interact, every passage vector can be computed once offline and stored in a maximum inner-product search index. At query time the system encodes the question and runs a sub-linear MIPS lookup, so serving remains fast even over tens of millions of passages. The architecture is intentionally simple; the modeling effort is directed entirely at learning a good embedding space from limited supervised data.

The training objective is a softmax negative log-likelihood that pushes the positive passage to score higher than the negatives. For a batch of B questions, each with its positive passage, the question vectors and passage vectors are stacked into matrices Q and P. The score matrix S = Q P^T contains every question scored against every passage in the batch. The diagonal entries are positives, and every off-diagonal entry is a free "gold" negative: another question's positive passage, already encoded at no extra cost. In addition, one BM25 hard negative is appended per question: a passage that BM25 ranks highly and that matches many question tokens, but that does not contain the answer. This hard negative teaches the model to separate topically similar but incorrect passages, which random negatives cannot do. The loss is then a row-wise log-softmax cross-entropy where each row's target is its own positive column. The model is fine-tuned end-to-end with Adam, learning rate 1e-5, linear warmup, dropout 0.1, and no auxiliary pretraining.

After training, all passages are encoded with the passage encoder and loaded into a FAISS HNSW index configured for inner-product search. A small extractive reader is applied only to the top-k retrieved passages, where cross-attention is affordable. The reader predicts answer-span start and end logits as well as a passage-selection score, and the final answer is taken from the passage with the highest selection score. This keeps the heavy cross-attention computation bounded by k rather than the corpus size.

```python
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import BertModel
import faiss

class BertEncoder(nn.Module):
    def __init__(self, pretrained="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained)

    def forward(self, input_ids, attn_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attn_mask)
        return out.last_hidden_state[:, 0, :]  # [CLS] -> (B, 768)

def dot_product_scores(q_vecs, p_vecs):
    return torch.matmul(q_vecs, p_vecs.transpose(0, 1))  # Q P^T

def biencoder_nll(q_vecs, pos_vecs, hard_vecs=None):
    # positives in columns 0..B-1; BM25 hard negatives append after them
    p_vecs = pos_vecs if hard_vecs is None else torch.cat([pos_vecs, hard_vecs], dim=0)
    target = torch.arange(q_vecs.size(0), device=q_vecs.device)
    scores = dot_product_scores(q_vecs, p_vecs)  # (B, B + #hard)
    log_p = F.log_softmax(scores, dim=1)
    return F.nll_loss(log_p, target)  # diagonal positives

def train_step(encoder_q, encoder_p, q_ids, q_mask, pos_ids, pos_mask,
               hard_ids, hard_mask, opt):
    q = encoder_q(q_ids, q_mask)
    pos = encoder_p(pos_ids, pos_mask)
    hard = encoder_p(hard_ids, hard_mask)
    loss = biencoder_nll(q, pos, hard)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()

class PassageIndex:
    def __init__(self, dim):
        self.index = faiss.IndexHNSWFlat(dim, 512, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = 200
        self.index.hnsw.efSearch = 128

    def add(self, vecs):
        self.index.add(vecs)  # all passages encoded offline

    def search(self, q, k):
        return self.index.search(q, k)  # top-k MIPS

class Reader(nn.Module):
    def __init__(self, pretrained="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained)
        h = self.bert.config.hidden_size
        self.qa_outputs = nn.Linear(h, 2)     # start/end logits
        self.qa_classifier = nn.Linear(h, 1)  # passage selection

    def forward(self, pair_ids, pair_mask):
        # each row: [CLS] question [SEP] passage [SEP]
        seq = self.bert(input_ids=pair_ids, attention_mask=pair_mask).last_hidden_state
        start, end = self.qa_outputs(seq).split(1, dim=-1)
        select = self.qa_classifier(seq[:, 0, :])
        return start.squeeze(-1), end.squeeze(-1), select.squeeze(-1)
```
