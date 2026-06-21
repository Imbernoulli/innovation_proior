Pool-based active learning for neural networks is stuck between two unsatisfying extremes. Uncertainty-only methods such as least-confidence pick points the model finds ambiguous, but they score examples in isolation and can return batches of near-duplicate boundary points, so one label would have resolved most of the rest. Diversity-only coreset methods cover the representation space, yet they ignore whether the model is actually unsure about those regions and rely on the learned embedding already being good. BADGE, the strongest deep-learning baseline at the time, fuses the two ideas by taking hallucinated last-layer gradients and spreading them in embedding space, but its determinant objective is only loosely connected to estimation error, it discards most of the per-point second-order structure, and it has no regression analogue because it depends on a most-likely class label.

The method I would use is BAIT, short for Batch Active learning via Information maTrices. The starting point is to view the network as a probability model and minimize the actual estimation error of the maximum-likelihood estimate after labeling a batch. Both Bayesian linear regression and the classical MLE active-learning analysis lead to the same objective: a weighted trace tr((Σ_{x∈S} I(x;θ))^{-1} I_U(θ)), where I(x;θ) is the per-example Fisher information and I_U(θ) is the pool Fisher. This is A-optimality with the pool metric baked in, not a determinant. A determinant cannot represent the pool term because det(I(x;θ)^{-1} I_U(θ)) factors into a constant times a per-point term, so it is structurally blind to which directions matter under the pool distribution. The trace, by contrast, is dominated by the weakest information directions weighted by the pool, which is exactly where labels should be spent.

To make this tractable at neural scale, BAIT restricts everything to the network's last layer, recomputes the Fisher every round from the freshly retrained model, and replaces the combinatorial SDP with greedy selection. Because the trace objective is not submodular, plain forward greedy can make early commitments it later regrets, so BAIT first greedily adds 2B points and then greedily prunes back to B. The key efficiency move is that each point's last-layer Fisher is low rank: for multiclass logistic regression I(x;θ^L) = V_x V_x^T, where the columns of V_x are sqrt(π_c) times the sign-flipped last-layer gradient under class c. Using the Woodbury identity and a cyclic trace rotation, each candidate's score reduces to a small k×k inner product, and all candidates can be scored in one batched matrix multiply after precomputing M^{-1} I_U M^{-1} once per step. The same framework also covers regression, where the Fisher is rank one and the noise covariance cancels, which BADGE cannot do at all.

```python
import gc
import numpy as np
import torch
import torch.nn.functional as F


def get_exp_grad_embedding(model, loader, n_pool, n_labels, emb_dim, probs=None):
    """Build per-point Fisher factors V_x^T of shape [n_pool, n_labels, emb_dim*n_labels]."""
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
    """Greedy forward/backward selection on Fisher factors X [n_candidates, rank, dim]."""
    indsAll = []
    dim = X.shape[-1]
    rank = X.shape[-2]

    currentInv = torch.inverse(
        lamb * torch.eye(dim).cuda()
        + iterates.cuda() * nLabeled / (nLabeled + K)
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
            dim1=-2, dim2=-1,
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
            currentInv
            - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv
        ).detach()[0]

    for _ in range(len(indsAll) - K):
        xt_ = X[indsAll].cuda()
        innerInv = torch.inverse(
            -1 * torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)
        ).detach()
        traceEst = torch.diagonal(
            xt_ @ currentInv @ fisher @ currentInv @ xt_.transpose(1, 2) @ innerInv,
            dim1=-2, dim2=-1,
        ).sum(-1)
        delInd = torch.argmin(-1 * traceEst).item()

        xt_ = X[indsAll[delInd]].unsqueeze(0).cuda()
        innerInv = torch.inverse(
            -1 * torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)
        ).detach()
        currentInv = (
            currentInv
            - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv
        ).detach()[0]
        del indsAll[delInd]

    del xt_, innerInv, currentInv
    torch.cuda.empty_cache()
    gc.collect()
    return indsAll


def query_bait(strategy, n):
    """Acquisition rule plugging into the active-learning harness."""
    idxs_unlabeled = np.arange(strategy.n_pool)[~strategy.idxs_lb]
    xt = strategy.get_exp_grad_embedding(strategy.X, strategy.Y)

    batchSize = 1000
    fisher = torch.zeros(xt.shape[-1], xt.shape[-1])
    for i in range(int(np.ceil(len(strategy.X) / batchSize))):
        xt_ = xt[i * batchSize:(i + 1) * batchSize].cuda()
        fisher += torch.sum(
            torch.matmul(xt_.transpose(1, 2), xt_) / len(xt), 0
        ).detach().cpu()
        del xt_
        torch.cuda.empty_cache()
        gc.collect()

    init = torch.zeros(xt.shape[-1], xt.shape[-1])
    xt2 = xt[strategy.idxs_lb]
    for i in range(int(np.ceil(len(xt2) / batchSize))):
        xt_ = xt2[i * batchSize:(i + 1) * batchSize].cuda()
        init += torch.sum(
            torch.matmul(xt_.transpose(1, 2), xt_) / len(xt2), 0
        ).detach().cpu()
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
