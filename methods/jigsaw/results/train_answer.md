I want convolutional features as good as the ones supervised ImageNet pretraining gives, but learned from unlabeled images alone. The route is a pretext task whose supervisory label comes free from the image itself and that can *only* be solved by understanding object content and spatial structure — so that the features picked up along the way are genuinely semantic and transfer to classification and detection. The hard part is choosing the task so that the network cannot cheat: a network will solve any pretext task by whatever is easiest, and the easiest route is usually a low-level shortcut — matching pixel mean and standard deviation across neighboring regions, following continuous edges and texture at shared borders, reading off a tile's location from chromatic aberration (the lens-induced relative shift between color channels that grows toward the image border), or, worst of all, mapping a fixed appearance to a fixed absolute position. Any of these solves the task while learning nothing about objects.

The closest prior idea is Doersch's context prediction: take a $3\times 3$ grid, fix the center tile, present a second tile from one of the eight surrounding cells, and classify its relative position. It is single-image, which I like, but it only ever shows the network *two* tiles at once, and that is its weakness — the relative position of two tiles can be genuinely ambiguous. If the center tile and the tile above are both uniform sky or fur, "directly above or above-left?" has no determinate answer from those two patches, so the network is trained on an underdetermined label and latches onto whatever spurious cue resolves it. The alternatives lean on multiple images: Wang & Gupta track a patch across video frames to learn a patch-similarity metric, and Agrawal predicts egomotion between two frames using odometry. Both use two views of the *same instance*, so they learn invariance to viewpoint and illumination of one object and fixate on shared low-level statistics like color and texture — exactly the structure-blind cues I want the features to ignore.

I propose learning representations by solving jigsaw puzzles. Instead of asking about the relative position of two tiles, cut the image into a full $3\times 3$ grid of nine tiles, scramble them, and ask the network to recover the arrangement of all nine at once. The reason this beats the two-tile version is the ambiguity argument made precise: any single tile may be ambiguous about where it belongs, but placement is *mutually exclusive* — each tile occupies exactly one cell — so when all nine are considered jointly, the per-tile constraints intersect and the ambiguity sets shrink, often to a unique solution. To know that a featureless sky tile goes top-right I do not need that tile to be self-identifying; I need the other eight placed, and the last cell is then forced. Solving the puzzle therefore requires reasoning about what the parts are and where they jointly go, which is exactly the part-based-model intuition. Rather than predict a raw permutation of nine tiles (awkward, and it invites a degeneracy I return to), I fix a *set* of permutations ahead of time, index them, scramble the tiles by one chosen permutation, and train the network to classify *which permutation index* was used. Puzzle-solving becomes a plain classification problem over the size of the permutation set, and it forces one joint decision about the whole arrangement instead of nine independent per-tile guesses.

The architecture is where the first shortcut hides, so I design it to forbid that shortcut. The naive option stacks the nine tiles along the channel dimension ($9\times 3 = 27$ input channels) into one fat AlexNet — but then all nine tiles meet at the very first convolution, and the cheapest way to find the arrangement is to match texture and edge continuity where tiles abut, which needs zero object understanding and leaves the features useless. So I process each tile *completely independently* through its own column — the same AlexNet conv1–conv5 plus the first fully connected layer fc6, with weights *shared* across all nine tiles — and only then, at fc7, concatenate the nine fc6 vectors and let the network reason about arrangement. Nine identical columns up to fc6, joined at fc7: a siamese-ennead network I call the Context-Free Network (CFN), context-free because the relationship between tiles is deliberately withheld until the last layers, after each column has already built a genuine high-level per-tile feature. Two consequences make this the right design. First, sharing weights up to fc6 means there is a single tile-feature extractor, and *that* is precisely what transfers; the cross-tile reasoning in fc7/fc8 is puzzle-specific and is discarded at transfer time. Second, the CFN is not crippled as a feature extractor — fed full tiles it matches AlexNet on ImageNet classification (about $57\%$ top-1 either way) while being far lighter, because fc6 now maps a $4\times 4\times 256$ tile feature to 512 ($\approx 2\mathrm{M}$ params) rather than AlexNet's $6\times 6\times 256\to 4096$ ($\approx 37.5\mathrm{M}$), so the whole CFN is $27.5\mathrm{M}$ params against AlexNet's $61\mathrm{M}$.

The central failure mode is subtle, and naming it gives the fix. Write the output as a part-based density,
$$p(S \mid A_1,\dots,A_9) \;=\; p(S \mid F_1,\dots,F_9)\,\prod_i p(F_i \mid A_i),$$
where $S$ is the tile configuration, $A_i$ the appearance of part $i$, and $F_i$ the intermediate feature of tile $i$. I want each $F_i$ to be semantic — to describe *what* the part is — so the arrangement is inferred from content. But suppose I only ever show one puzzle per image, so each tile always sits in the same cell. Then the network can satisfy the task by learning $F_i \to$ "absolute cell position," making $p(S \mid F_1,\dots,F_9)$ factorize as $\prod_i p(L_i \mid F_i)$ with each tile's location $L_i$ read straight off its feature; the features encode arbitrary 2D position and nothing semantic. The fix follows from naming the failure: never let any tile have a fixed cell. Feed *many distinct* puzzles of the same image (it works out to roughly 69 different puzzles per image over training), so a given tile appears across many positions; "map appearance to a fixed location" is then no longer valid, because the same appearance is the answer "position 1" in one puzzle and "position 7" in another, and the only way to predict the index is to reason about the joint content-based arrangement of all nine tiles.

This makes the permutation set the real hyperparameter, and its geometry matters in two opposing ways. There are $9! = 362{,}880$ orderings; I use a subset of size $N$ (around 1000). If the chosen permutations are too *similar* — differing in only two tile positions — the task is ambiguous in Doersch's way: if those two swapped tiles look alike, no network can tell the permutations apart, and I am training on noise. So I want them *far apart* in Hamming distance, the number of positions in which two permutations differ; large Hamming distance means distinguishing any two requires reading many tiles' content and little chance two near-identical-looking arrangements share a label. But too few permutations, or a set spread so far the task becomes trivial, lets the network memorize a handful of orderings without learning rich features. The sweet spot is a moderate number of mutually distant permutations — the crystallized principle being that a good self-supervised task is neither so easy the network sidesteps understanding nor so ambiguous the label is unpredictable. I construct the set greedily: start from all $9!$ permutations and an empty set, seed it with a random permutation, then repeatedly add the remaining permutation whose mean Hamming distance to the already-chosen set is largest (take the argmax over candidates), until $N$ are selected. Swapping argmax for a middle pick gives control sets for ablation; the set is fixed before training begins.

Closing the absolute-position shortcut still leaves the low-level cues the architecture alone does not kill, and each gets its own countermeasure. Adjacent tiles share similar pixel mean and standard deviation, so I normalize each tile's mean and standard deviation *independently*, $t \leftarrow (t - \mathrm{mean}(t))/(\mathrm{std}(t)+\varepsilon)$, erasing that cue. Edge continuity and matching texture at a shared border need no object understanding, so I leave a gap: from each $85\times 85$ cell I sample a smaller $64\times 64$ tile at a random offset, leaving up to a ${\sim}21$px margin between neighbors and no continuous edge to follow. Chromatic aberration is attacked three ways: crop the central square and resize so all tiles come from near the center where aberration is weak and uniform; mix grayscale ($30\%$) and color ($70\%$) images so color cues cannot be relied on; and spatially jitter each tile's color channels by $\pm\{0,1,2\}$ px to destroy the residual shift. By ablation on downstream detection each matters — removing the gap hurts most, removing per-tile normalization next, removing color jitter least — but all three together give the best transfer. Training is plain classification over permutation indices: SGD without batch normalization, batch 256, base learning rate $0.01$, 350K iterations (about 2.5 days) on 1.3M unlabeled ImageNet images; at transfer I copy the shared conv weights into a standard AlexNet (stride-4 first layer at transfer, stride-2 during puzzle training), randomly initialize the fully connected layers, and fine-tune for classification, detection, or segmentation.

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
