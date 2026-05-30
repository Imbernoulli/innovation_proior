# Jigsaw — Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles

## Problem

Learn transferable convolutional features from unlabeled images by a pretext task that can only be solved through understanding object content and spatial structure — not through low-level shortcuts — and that works from single still images.

## Key idea

Cut an image into a 3×3 grid of nine tiles, shuffle them by a permutation drawn from a fixed, carefully chosen permutation set, and train a network to classify *which permutation* was applied. Reassembling all nine tiles jointly resolves the placement ambiguity that two-tile relative-position prediction suffers (placement is mutually exclusive, so constraints intersect). The network — the Context-Free Network (CFN) — processes each tile independently through nine weight-shared AlexNet columns up to fc6, joining only at fc7, so it must build genuine per-tile features before it is allowed to compare tiles. The transferable part is the shared per-tile conv stack.

## Method

- **Puzzle generation:** crop/resize to a 255×255 window, 3×3 grid; from each 85×85 cell sample a 64×64 tile at a random offset (leaving up to a ~21px gap between tiles); shuffle by a chosen permutation; the label is that permutation's index.
- **Network (CFN):** siamese-ennead. Nine identical AlexNet conv1–conv5 + fc6 columns with shared weights process tiles independently (receptive field = one tile). The nine fc6 vectors are concatenated and fed to fc7 → fc8 → softmax over the permutation set. Cross-tile "context" enters only at fc7. fc6 is small (4×4×256→512 ≈ 2M params), making the CFN lighter than AlexNet (27.5M vs 61M) while matching its ImageNet accuracy.
- **Permutation set:** N orderings (≈1000) of the 9!=362,880 chosen by **maximal average Hamming distance** via a greedy algorithm (seed randomly; repeatedly add the remaining permutation farthest in mean Hamming distance from the chosen set). Far-apart permutations make the task unambiguous; cardinality and average Hamming distance jointly control difficulty. Principle: a good self-supervised task is neither simple nor ambiguous.
- **Shortcut prevention:**
  - *Absolute position:* feed many distinct puzzles per image (~69 over training) with high-Hamming permutations, so no tile maps to a fixed cell.
  - *Low-level statistics:* normalize each tile's mean and std independently.
  - *Edge continuity:* the inter-tile gap (tile smaller than cell).
  - *Chromatic aberration:* central crop + resize, mix grayscale (30%) and color (70%) images, and spatially jitter color channels by ±{0,1,2} px per tile.
- **Training:** Caffe, SGD without batch norm, batch 256, base lr 0.01, 350K iterations (~2.5 days), 1.3M unlabeled ImageNet images.
- **Transfer:** copy the conv weights into a standard AlexNet (stride 4 first layer; stride 2 during puzzle training), randomly initialize the fully connected layers, fine-tune for classification / detection / segmentation.

## Code

```python
import numpy as np
import itertools
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist

def build_permutation_set(N=1000, selection='max'):
    """N orderings of 9 tiles with maximal average Hamming distance, greedy, fixed before training."""
    P_hat = np.array(list(itertools.permutations(range(9))))          # all 9! orderings
    P = None
    j = np.random.randint(len(P_hat))
    for _ in range(N):
        P = P_hat[j].reshape(1, -1) if P is None else np.concatenate([P, P_hat[j].reshape(1, -1)])
        P_hat = np.delete(P_hat, j, axis=0)
        D = cdist(P, P_hat, metric='hamming').mean(axis=0)
        j = D.argmax() if selection == 'max' else D.argsort()[len(D)//2]
    return P                                                          # [N, 9]

def make_puzzle(image, perms, cell=85, tile=64):
    crop = random_central_crop_resize(image, 255)                     # 3 x 85; weak chromatic aberration
    tiles = []
    for r in range(3):
        for c in range(3):
            cell_img = crop[r*cell:r*cell+cell, c*cell:c*cell+cell]
            oy, ox = np.random.randint(0, cell - tile + 1, size=2)
            t = cell_img[oy:oy+tile, ox:ox+tile]                      # random offset -> gap
            t = color_jitter_channels(t)
            t = (t - t.mean()) / (t.std() + 1e-6)                     # per-tile normalization
            tiles.append(t)
    k = np.random.randint(len(perms))
    order = perms[k]
    return torch.stack([tiles[order[i]] for i in range(9)]), k        # 9 tiles, label = perm index

class CFN(nn.Module):
    def __init__(self, num_perms=1000):
        super().__init__()
        self.conv = nn.Sequential(                                    # AlexNet conv1-conv5 (shared)
            nn.Conv2d(3, 96, 11, stride=2), nn.ReLU(True), nn.MaxPool2d(3, 2), LRN(),
            nn.Conv2d(96, 256, 5, padding=2, groups=2), nn.ReLU(True), nn.MaxPool2d(3, 2), LRN(),
            nn.Conv2d(256, 384, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(384, 384, 3, padding=1, groups=2), nn.ReLU(True),
            nn.Conv2d(384, 256, 3, padding=1, groups=2), nn.ReLU(True), nn.MaxPool2d(3, 2),
        )
        self.fc6 = nn.Sequential(nn.Linear(256*4*4, 512), nn.ReLU(True), nn.Dropout(0.5))
        self.fc7 = nn.Sequential(nn.Linear(9*512, 4096), nn.ReLU(True), nn.Dropout(0.5))
        self.fc8 = nn.Linear(4096, num_perms)

    def forward(self, tiles):                                         # [B, 9, 3, 64, 64]
        B = tiles.size(0)
        feats = [self.fc6(self.conv(tiles[:, i]).view(B, -1)) for i in range(9)]  # shared columns
        x = torch.cat(feats, dim=1)                                   # context joins at fc7
        return self.fc8(self.fc7(x))                                  # logits over permutation set

def train(net, loader, perms):
    opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    ce = nn.CrossEntropyLoss()
    for image_batch in loader:
        tiles, labels = zip(*[make_puzzle(im, perms) for im in image_batch])  # fresh puzzle each step
        loss = ce(net(torch.stack(tiles)), torch.tensor(labels))
        opt.zero_grad(); loss.backward(); opt.step()
```
