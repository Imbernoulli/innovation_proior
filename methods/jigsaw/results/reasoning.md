I want convolutional features as good as the supervised ImageNet ones, but without the labels. The way to get there is to invent a pretext task whose label comes free from the image, and that can *only* be solved by understanding object content and structure — so that the features I learn along the way are genuinely semantic and transfer to classification and detection. So the real design work is choosing the task, and choosing it so the network can't cheat.

Let me look at what's already out there and find the gap. Doersch's context-prediction task takes a 3×3 grid, fixes the center tile, and asks the network to classify which of the 8 surrounding positions a second tile came from. It's single-image, which I like. But it only ever shows the network *two* tiles at a time, and that's its weakness: the relative position of two tiles can be genuinely ambiguous. If the center tile and the tile above it are both, say, patches of sky or fur, then "is this tile directly above, or above-left?" has no determinate answer from those two patches alone — the network is being asked to predict something underdetermined, and it'll learn whatever spurious cue resolves the ambiguity. The other two approaches I know lean on multiple images: Wang & Gupta track a patch across video frames and learn a similarity metric; Agrawal predicts egomotion between two frames with odometry labels. Both use two views of the *same instance*, so the features they learn are about being invariant to viewpoint/illumination of that one object — they latch onto shared low-level statistics like color and texture, not the high-level structure that separates one object's parts from another's. Two cars of different colors, two dogs of different fur: those methods would fixate on the color/texture difference; I'd rather the features ignore that and key on shape and parts.

So the gap is: a single-image task that forces reasoning over *all* the parts at once, so that ambiguities can cancel. Here's the move. Instead of asking about the relative position of two tiles, cut the image into a full 3×3 grid of nine tiles, scramble them, and ask the network to *unscramble* — to recover the spatial arrangement of all nine simultaneously. A jigsaw puzzle. The reason this beats the two-tile version is exactly the ambiguity argument: any single tile might be ambiguous about where it belongs, but the placement is *mutually exclusive* — each tile occupies exactly one cell — so when all nine are considered jointly, the constraints intersect and the ambiguity sets shrink, often to a single solution. To know that a uniform sky tile goes top-right, I don't need that tile to be self-identifying; I need the other eight to be placed, and the last spot is forced. The task makes the network reason about parts and their joint configuration, which is precisely the part-based-model intuition: solving the puzzle means knowing what the parts are and where they go.

Now, how do I phrase "unscramble" as a learning target? The literal output would be a permutation of nine tiles — but predicting a permutation directly (nine position outputs) is awkward and, worse, invites a failure I'll come back to. Cleaner: fix a *set* of permutations ahead of time, index them, scramble the tiles by one chosen permutation, and ask the network to classify *which permutation index* was used. That turns puzzle-solving into a plain classification problem over the size of my permutation set, and crucially it forces a single joint decision about the whole arrangement rather than nine independent per-tile guesses.

Let me design the network, because the architecture is where the first shortcut hides. The naive thing is to stack the nine tiles along the channel dimension — 9×3 = 27 input channels — and run one fat AlexNet. But think about what that network will do: with all nine tiles mixed at the very first convolution, the cheapest way to figure out the arrangement is to look at *low-level correlations across tile boundaries* — matching texture and edge continuity where two tiles abut. That solves the puzzle (humans do it too) but requires zero understanding of the global object, so the features would be junk for transfer. I need to *forbid* the network from comparing tiles until it has first formed a real per-tile representation. So: process each tile completely independently through its own conv stack — same AlexNet conv1–conv5 and first fully connected layer fc6, with weights *shared* across all nine tiles — and only *then*, at fc7, concatenate the nine fc6 vectors and let the network reason about arrangement. Nine identical columns up to fc6, joined at fc7: a siamese-ennead network. Each column's receptive field is one tile; there is no cross-tile information flow until the features are already high-level. I'll call it context-free because context — the relationship between tiles — is deliberately withheld until the last layers.

Two things fall out of this design that I should check. First, sharing weights up to fc6 means there's a single tile-feature extractor, and *that* is the thing I'll transfer; the cross-tile reasoning in fc7/fc8 is puzzle-specific and gets thrown away at transfer time, which is exactly what I want. Second, I should make sure this CFN isn't crippled as a feature extractor compared to plain AlexNet. If I feed it full tiles and test it on ImageNet classification, it should match AlexNet — and it does, essentially (about 57% top-1 either way), while having far fewer parameters, because fc6 now operates on a small 4×4×256 tile feature (≈2M params) rather than AlexNet's 6×6×256→4096 (≈37.5M). Good: the architecture is a fair stand-in for AlexNet, so transferred weights are directly comparable.

Now the central failure mode, and it's subtle. Write the network's output as a part-based pdf: p(S | A₁,…,A₉) = p(S | F₁,…,F₉) · Πᵢ p(Fᵢ | Aᵢ), where S is the tile configuration, Aᵢ the appearance of part i, and Fᵢ the intermediate feature for tile i. I want the Fᵢ to be semantic — to describe *what* the part is, so the arrangement can be inferred from content. But there's a degenerate solution. Suppose I only ever show the network *one* puzzle per image, i.e. each tile always appears in the same cell. Then the network can satisfy the task by learning Fᵢ → "absolute cell position," so that p(S | F₁,…,F₉) factorizes into Πᵢ p(Lᵢ | Fᵢ) with each tile's location Lᵢ read straight off its feature. In that solution the features encode arbitrary 2D position and *nothing semantic*. The network solved the puzzle and learned nothing I want.

The fix follows directly from naming the failure: don't let any tile have a fixed cell. Feed *many different* puzzles of the same image, so a given tile must appear in many — ideally all nine — positions over training. Then "map this appearance to a fixed location" is no longer a valid strategy, because the same appearance is the answer "position 1" in one puzzle and "position 7" in another; the only way to predict the index is to reason about the *joint* arrangement of all nine tiles by content. Concretely I'll arrange for each image to be seen as dozens of distinct puzzles over the course of training (it works out to roughly 69 different puzzles per image given the training length), and — this is the lever — I'll choose the permutation set so the tiles get shuffled *as much as possible*.

That makes the permutation set the real hyperparameter of the method, so let me think carefully about how to choose it. There are 9! = 362,880 possible orderings; I'll use a subset, and the subset's geometry matters in two opposing ways. If the permutations in my set are too *similar* to each other — differing only in the positions of two tiles — then the task is ambiguous in the same way Doersch's was: if those two swapped tiles happen to look alike, no network can tell the two permutations apart, and I'm training on noise. So I want the permutations to be *far apart*, measured by Hamming distance — the number of tile positions in which two permutations differ. Large Hamming distance means any two permutations disagree in many positions, so distinguishing them requires reading many tiles' content, and there's little chance two near-identical-looking arrangements share a label. On the other side, if I use too *few* permutations, or make them so spread out that the task is trivial, the network can memorize the handful of orderings without learning rich features. So there's a tension: more permutations and more mutual dissimilarity make the task harder *and* the features better up to a point, but pushed to the extreme (very few, maximally distant permutations) the task gets easy again. The sweet spot is a moderate number of permutations chosen to be mutually far apart. The principle that crystallizes: a good self-supervised task is neither so easy the network sidesteps understanding, nor so ambiguous that the label is unpredictable.

So I need to *construct* a set of, say, a thousand permutations with large average Hamming distance. Greedy selection does it. Start from all 9! permutations and an empty set; pick one permutation at random to seed the set; then repeatedly add the permutation (from those remaining) whose Hamming distance to the current set is largest — compute, for each candidate, the mean Hamming distance to everything already chosen, and take the argmax. Repeat until I have N permutations. (Swap the argmax for argmin and I'd get a minimal-distance set; for uniform sampling, a middling set — useful as ablation controls. The set is fixed before training begins.)

Even with the absolute-position shortcut closed, there are the low-level shortcuts the architecture alone doesn't kill, and each needs its own countermeasure. Let me go through them. Adjacent tiles share similar pixel mean and standard deviation, so the network could match neighbors by raw intensity statistics — I normalize the mean and standard deviation of each tile *independently*, erasing that cue. Edge continuity and matching texture right at a shared border is a strong cue that needs no object understanding — I leave a gap between tiles by cutting each tile *smaller* than its cell: from a cell I sample a 64×64 tile with a random offset, so there's empty margin (up to about 21 pixels) between neighbors and no continuous edge to follow. Chromatic aberration — the lens-induced relative shift between color channels that grows toward the image borders — lets the network localize a tile from its color-fringing alone; I attack it three ways: crop the central square and resize (so all tiles come from near the center where aberration is weak and uniform), train on a mix of grayscale and color images so color cues can't be relied on, and spatially jitter the color channels of each tile by a pixel or two to destroy the residual shift. If I check these by ablation on downstream detection, each matters: removing the gap hurts most, removing the per-tile normalization next, removing color jitter least — but all three together give the best transfer.

Let me lay out the pieces in code, mirroring how the CFN and the permutation set actually go.

```python
import numpy as np
import itertools
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist

# ---------------------------------------------------------------
# Permutation set: choose N orderings of 9 tiles with MAXIMAL
# average Hamming distance, greedily. Generated once, before
# training. Far-apart permutations -> the task is unambiguous and
# distinguishing them forces reading many tiles' content.
# ---------------------------------------------------------------
def build_permutation_set(N=1000, selection='max'):
    P_hat = np.array(list(itertools.permutations(range(9))))   # all 9! orderings
    P = None
    j = np.random.randint(len(P_hat))                          # random seed permutation
    for i in range(N):
        P = P_hat[j].reshape(1, -1) if P is None else np.concatenate([P, P_hat[j].reshape(1, -1)])
        P_hat = np.delete(P_hat, j, axis=0)
        D = cdist(P, P_hat, metric='hamming').mean(axis=0)     # mean Hamming dist to chosen set
        j = D.argmax() if selection == 'max' else D.argsort()[len(D)//2]  # farthest (or middle)
    return P                                                    # [N, 9]

# ---------------------------------------------------------------
# Turn one image into a puzzle: 3x3 grid, sample a smaller tile from
# each cell (-> gap, kills edge continuity), per-tile mean/std
# normalize (kills low-level-statistic matching), color jitter +
# grayscale mix (kills chromatic aberration), shuffle by a chosen
# permutation, target = its index.
# ---------------------------------------------------------------
def make_puzzle(image, perms, cell=85, tile=64):
    crop = random_central_crop_resize(image, 255)             # central crop/resize -> weak aberration
    tiles = []
    for r in range(3):
        for c in range(3):
            cell_img = crop[r*cell:r*cell+cell, c*cell:c*cell+cell]  # 3x3 grid of 85x85 cells
            off_y, off_x = np.random.randint(0, cell - tile + 1, size=2)
            t = cell_img[off_y:off_y+tile, off_x:off_x+tile]   # random offset -> inter-tile gap
            t = color_jitter_channels(t)                       # +/- few px per channel
            t = (t - t.mean()) / (t.std() + 1e-6)              # per-tile normalization
            tiles.append(t)
    k = np.random.randint(len(perms))                          # choose a permutation
    order = perms[k]
    shuffled = [tiles[order[i]] for i in range(9)]             # scramble
    return torch.stack(shuffled), k                            # 9 tiles, label = permutation index

# ---------------------------------------------------------------
# Context-Free Network: 9 weight-shared AlexNet columns up to fc6;
# tiles are processed INDEPENDENTLY (no cross-tile context) until
# fc6, so the network must build real per-tile features before it is
# allowed to compare them. Context enters only at fc7.
# ---------------------------------------------------------------
class CFN(nn.Module):
    def __init__(self, num_perms=1000):
        super().__init__()
        self.conv = nn.Sequential(                             # AlexNet conv1-conv5 (shared)
            nn.Conv2d(3, 96, 11, stride=2), nn.ReLU(True), nn.MaxPool2d(3, 2), LRN(),
            nn.Conv2d(96, 256, 5, padding=2, groups=2), nn.ReLU(True), nn.MaxPool2d(3, 2), LRN(),
            nn.Conv2d(256, 384, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(384, 384, 3, padding=1, groups=2), nn.ReLU(True),
            nn.Conv2d(384, 256, 3, padding=1, groups=2), nn.ReLU(True), nn.MaxPool2d(3, 2),
        )
        self.fc6 = nn.Sequential(nn.Linear(256*4*4, 512), nn.ReLU(True), nn.Dropout(0.5))
        self.fc7 = nn.Sequential(nn.Linear(9*512, 4096), nn.ReLU(True), nn.Dropout(0.5))  # context joins here
        self.fc8 = nn.Linear(4096, num_perms)                  # classify the permutation index

    def forward(self, tiles):                                  # tiles: [B, 9, 3, 64, 64]
        B = tiles.size(0)
        feats = []
        for i in range(9):                                     # each tile through the SAME column
            z = self.conv(tiles[:, i])
            z = self.fc6(z.view(B, -1))
            feats.append(z)
        x = torch.cat(feats, dim=1)                            # concatenate 9 tile features
        return self.fc8(self.fc7(x))                           # joint reasoning -> permutation logits

# ---------------------------------------------------------------
# Training: plain classification over permutation indices.
# Many puzzles per image (each image seen as ~dozens of distinct
# puzzles) so no tile has a fixed cell -> kills the absolute-
# position shortcut. Transfer: copy conv weights into AlexNet.
# ---------------------------------------------------------------
def train(net, loader, perms):
    opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    ce = nn.CrossEntropyLoss()
    for image_batch in loader:
        tiles, labels = zip(*[make_puzzle(im, perms) for im in image_batch])  # fresh puzzle each time
        tiles, labels = torch.stack(tiles), torch.tensor(labels)
        loss = ce(net(tiles), labels)
        opt.zero_grad(); loss.backward(); opt.step()
```

So the chain: I wanted free single-image supervision that forces object understanding; two-tile relative-position prediction was the closest prior idea but suffered ambiguity because it never sees all the parts together, and the multi-image methods learned low-level instance invariances instead of structure. Reassembling a full 3×3 jigsaw fixes the ambiguity — joint, mutually-exclusive placement lets constraints intersect — and predicting the permutation index makes it a clean classification. The architecture has to *withhold* cross-tile context until the features are high-level, or the network cheats on boundary texture; hence the context-free siamese-ennead network with weights shared up to fc6 and joining only at fc7, which doubles as the transferable AlexNet feature extractor. The dangerous shortcut is the network mapping appearance to absolute position; feeding many high-Hamming-distance puzzles per image removes it, and choosing the permutation set by maximal average Hamming distance keeps the task hard-but-unambiguous. Per-tile normalization, inter-tile gaps, and chromatic-aberration countermeasures close the remaining low-level shortcuts, so the only way left to solve the puzzle is to understand the parts.
