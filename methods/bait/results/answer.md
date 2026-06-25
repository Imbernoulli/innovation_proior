# BAIT, Distilled

BAIT (Batch Active learning via Information maTrices) is a batch active-learning rule that selects
points by minimizing a Fisher-information estimate of downstream error. It takes the trace
objective from convex MLE theory, restricts it to the neural network's last layer, and makes the
greedy selection tractable with low-rank Fisher factors, Woodbury updates, and a forward/backward
greedy pass.

## Objective

For a labeled candidate batch `S`, the target objective is the weighted trace

```text
argmin_{S subset U, |S| <= B} tr((sum_{x in S} I(x; theta))^{-1} I_U(theta)),
```

where `I(x; theta) = E_y [nabla^2 ell(x,y;theta)]` under the negative-log-likelihood convention and
`I_U(theta)` is the pool Fisher. Constant normalizations of `I_U` do not change the argmin; the
reference implementation averages the pool Fisher.

Two derivations point to the same trace:

- Bayesian linear regression gives `BayesRisk(S) = sigma^2 tr(Lambda_S^{-1} Sigma)`, with
  `Lambda_S = sum_{x in S} xx^T + lambda sigma^2 I`.
- Chaudhuri et al.'s active-MLE analysis gives leading error
  `tr(I_Gamma(theta*)^{-1} I_U(theta*))/m`, with matching lower/upper bounds up to lower-order
  terms.

This is why BAIT uses a trace rather than a determinant. A determinant cannot carry the pool metric
in the same way: `det(I(x;theta)^{-1} I(theta)) = det(I(x;theta)^{-1}) det(I(theta))`, and the
second factor is constant in `x`.

## Neural Adaptation

BAIT makes three changes to the ideal objective:

1. Use only the last-layer Fisher `I(x; theta^L)`, so the parameter dimension is `emb_dim * n_class`.
2. Recompute the Fisher every active-learning round, because the representation changes after
   retraining.
3. Replace the SDP/combinatorial selection with greedy selection.

The trace objective is not submodular, so BAIT first greedily selects `2B` points, then greedily
removes `B` points to return a batch of size `B`.

## Fisher Factors

For multiclass logistic regression, with penultimate representation `x^L` and softmax vector `pi`,

```text
I(x; theta^L) = x^L (x^L)^T otimes (diag(pi) - pi pi^T).
```

Define `V_x` as a `dk x k` matrix whose column for possible class `c` is
`sqrt(pi_c) ((e_c - pi) otimes x^L)`. This is the sign-flipped cross-entropy gradient under class
`c`; the sign is irrelevant because BAIT uses outer products. Then

```text
V_x V_x^T = I(x; theta^L).
```

The reference code stores the transpose: `X[i]` has shape `[k, dk]` and equals `V_x.T`, so a point's
Fisher contribution is `X[i].T @ X[i]`.

## Efficient Score

For adding a candidate with factor `V_x`,

```text
(M + V_x V_x^T)^{-1}
  = M^{-1} - M^{-1} V_x (I + V_x^T M^{-1} V_x)^{-1} V_x^T M^{-1}.
```

Dropping the constant term, the greedy add score is

```text
tr(V_x^T M^{-1} I(theta) M^{-1} V_x (I + V_x^T M^{-1} V_x)^{-1}),
```

so the best add is the largest score. In the backward pass the update uses
`-I + V_x^T M^{-1} V_x`; with that signed inverse, the reference code removes
`argmin(-traceEst)`, equivalently the point whose removal least increases the objective.

## Regression

For `k`-output Gaussian regression under the same negative-log-likelihood convention,

```text
I(x; W) = xx^T otimes Sigma^{-1}.
```

The covariance cancels in the trace objective:

```text
tr((sum_{x in S} I(x;theta))^{-1} I_U(theta))
  = k * tr((sum_{x in S} x^L (x^L)^T)^{-1} (sum_{x in U} x^L (x^L)^T)).
```

Thus regression uses rank-one `x^L (x^L)^T` factors. BADGE has no analogous regression path because
its embedding depends on a most-likely class label.

## Reference-Faithful Core

The classification code is in `methods/bait/code/bait_sampling.py` and
`methods/bait/code/strategy.py`. Its essential logic is:

```python
import gc
import numpy as np
import torch
from torch.nn import functional as F


def get_exp_grad_embedding(model, loader, n_pool, n_labels, emb_dim, probs=None):
    """Return X with shape [n_pool, n_labels, emb_dim * n_labels] = V_x.T."""
    model.eval()
    embedding = np.zeros([n_pool, n_labels, emb_dim * n_labels])
    with torch.no_grad():
        for assumed in range(n_labels):
            for x, y, idxs in loader:
                x = x.cuda()
                logits, features = model(x)
                features = features.data.cpu().numpy()
                batch_probs = F.softmax(logits, dim=1).data.cpu().numpy()
                for j in range(len(y)):
                    idx = int(idxs[j])
                    for c in range(n_labels):
                        start, end = emb_dim * c, emb_dim * (c + 1)
                        if c == assumed:
                            block = features[j] * (1.0 - batch_probs[j][c])
                        else:
                            block = features[j] * (-batch_probs[j][c])
                        embedding[idx][assumed][start:end] = block
                    p = probs[idx][assumed] if probs is not None else batch_probs[j][assumed]
                    embedding[idx][assumed] *= np.sqrt(p)
    return torch.Tensor(embedding)


def select(X, K, fisher, iterates, lamb=1, nLabeled=0):
    """X stores V_x.T with shape [n_candidates, rank, dim]."""
    indsAll = []
    dim = X.shape[-1]
    rank = X.shape[-2]

    currentInv = torch.inverse(
        lamb * torch.eye(dim).cuda() + iterates.cuda() * nLabeled / (nLabeled + K)
    )
    X = X * np.sqrt(K / (nLabeled + K))
    fisher = fisher.cuda()

    over_sample = 2
    for _ in range(int(over_sample * K)):
        xt_ = X.cuda()
        innerInv = torch.inverse(
            torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)
        ).detach()
        bad = torch.where(torch.isinf(innerInv))
        innerInv[bad] = torch.sign(innerInv[bad]) * np.finfo("float32").max
        traceEst = torch.diagonal(
            xt_ @ currentInv @ fisher @ currentInv @ xt_.transpose(1, 2) @ innerInv,
            dim1=-2,
            dim2=-1,
        ).sum(-1)

        traceEst = traceEst.detach().cpu().numpy()
        for j in np.argsort(traceEst)[::-1]:
            if j not in indsAll:
                ind = j
                break
        indsAll.append(ind)

        xt_ = X[ind].unsqueeze(0).cuda()
        innerInv = torch.inverse(
            torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)
        ).detach()
        currentInv = (
            currentInv - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv
        ).detach()[0]

    for _ in range(len(indsAll) - K):
        xt_ = X[indsAll].cuda()
        innerInv = torch.inverse(
            -1 * torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)
        ).detach()
        traceEst = torch.diagonal(
            xt_ @ currentInv @ fisher @ currentInv @ xt_.transpose(1, 2) @ innerInv,
            dim1=-2,
            dim2=-1,
        ).sum(-1)
        delInd = torch.argmin(-1 * traceEst).item()

        xt_ = X[indsAll[delInd]].unsqueeze(0).cuda()
        innerInv = torch.inverse(
            -1 * torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)
        ).detach()
        currentInv = (
            currentInv - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv
        ).detach()[0]
        del indsAll[delInd]

    del xt_, innerInv, currentInv
    torch.cuda.empty_cache()
    gc.collect()
    return indsAll


def query_bait(strategy, n):
    idxs_unlabeled = np.arange(strategy.n_pool)[~strategy.idxs_lb]
    xt = strategy.get_exp_grad_embedding(strategy.X, strategy.Y)

    batchSize = 1000
    fisher = torch.zeros(xt.shape[-1], xt.shape[-1])
    for i in range(int(np.ceil(len(strategy.X) / batchSize))):
        xt_ = xt[i * batchSize:(i + 1) * batchSize].cuda()
        fisher += torch.sum(torch.matmul(xt_.transpose(1, 2), xt_) / len(xt), 0).detach().cpu()
        del xt_
        torch.cuda.empty_cache()
        gc.collect()

    init = torch.zeros(xt.shape[-1], xt.shape[-1])
    xt2 = xt[strategy.idxs_lb]
    for i in range(int(np.ceil(len(xt2) / batchSize))):
        xt_ = xt2[i * batchSize:(i + 1) * batchSize].cuda()
        init += torch.sum(torch.matmul(xt_.transpose(1, 2), xt_) / len(xt2), 0).detach().cpu()
        del xt_
        torch.cuda.empty_cache()
        gc.collect()

    chosen = select(
        xt[idxs_unlabeled],
        n,
        fisher,
        init,
        lamb=strategy.lamb,
        nLabeled=np.sum(strategy.idxs_lb),
    )
    return idxs_unlabeled[chosen]
```

Defaults used in the reported experiments and reference code path: seed with about 100 random labels, retrain from scratch
each round, use `lambda = 1` except `0.01` for CIFAR-10, and use forward oversampling factor `2`.
