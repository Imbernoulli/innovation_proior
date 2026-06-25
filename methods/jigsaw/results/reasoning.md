I want convolutional features as good as the supervised ImageNet ones, but without the labels. The way to get there is to invent a pretext task whose label comes free from the image, and that can *only* be solved by understanding object content and structure — so that the features I learn along the way are genuinely semantic and transfer to classification and detection. So the real design work is choosing the task, and choosing it so the network can't cheat.

Let me look at what's already out there and find the gap. Doersch's context-prediction task takes a 3×3 grid, fixes the center tile, and asks the network to classify which of the 8 surrounding positions a second tile came from. It's single-image, which I like. But it only ever shows the network *two* tiles at a time, and I think that's its weakness: the relative position of two tiles can be genuinely ambiguous. If the center tile and the tile above it are both, say, patches of sky or fur, then "is this tile directly above, or above-left?" has no determinate answer from those two patches alone — the network is being asked to predict something underdetermined, and it'll learn whatever spurious cue resolves the ambiguity. The other two approaches I know lean on multiple images: Wang & Gupta track a patch across video frames and learn a similarity metric; Agrawal predicts egomotion between two frames with odometry labels. Both use two views of the *same instance*, so the features they learn are about being invariant to viewpoint/illumination of that one object — they latch onto shared low-level statistics like color and texture, not the high-level structure that separates one object's parts from another's. Two cars of different colors, two dogs of different fur: those methods would fixate on the color/texture difference; I'd rather the features ignore that and key on shape and parts.

So the gap I want to attack is a single-image task that forces reasoning over *all* the parts at once. Here's the question that decides whether that helps: if I cut the image into a full 3×3 grid of nine tiles and ask the network to recover the spatial arrangement of all nine simultaneously, does the per-tile ambiguity that sinks the two-tile task actually go away? Let me make that concrete rather than hand-wave it. Take the worst case for me: one of the nine tiles is a uniform sky patch that, on its own content, is consistent with several cells — say the four edge-middle cells {top, bottom, left, right}, because a flat sky patch could plausibly sit at any of them. In Doersch's two-tile setting the network sees only the center plus this sky tile, so it is choosing among those ~4 plausible relative positions, and the best it can do by guessing is P(correct) ≤ 1/4 = 0.25. Now the nine-tile version. Suppose the *other* eight tiles are each resolvable by content — they map to distinct cells. Whatever eight cells they take, exactly one cell is left over (8 distinct cells out of 9 always leaves a unique remainder), and the sky tile is *forced* into it. I checked this is not a special case: enumerating every way the eight resolvable tiles can occupy 8 of the 9 cells, the remaining cell is always unique, so the sky tile's placement is determined with probability 1 regardless of how content-ambiguous it was on its own. That is the actual mechanism — mutual exclusivity of placement means the constraints intersect, and a tile that is individually ambiguous gets pinned by the others. The 0.25 → 1 gap on this toy is exactly the improvement I was hoping for, and it isn't an accident of the example: it's the combinatorics of a complete assignment. So reassembling all nine at once is worth pursuing, and it has the part-based flavor I want — solving the puzzle means knowing what the parts are and where they go jointly.

Now, how do I phrase "unscramble" as a learning target? The literal output would be a permutation of nine tiles — but predicting a permutation directly (nine position outputs) is awkward and, worse, invites a failure I'll come back to. Cleaner: fix a *set* of permutations ahead of time, index them, scramble the tiles by one chosen permutation, and ask the network to classify *which permutation index* was used. That turns puzzle-solving into a plain classification problem over the size of my permutation set, and crucially it forces a single joint decision about the whole arrangement rather than nine independent per-tile guesses.

Let me design the network, because the architecture is where the first shortcut hides. The naive thing is to stack the nine tiles along the channel dimension — 9×3 = 27 input channels — and run one fat AlexNet. But think about what that network will do: with all nine tiles mixed at the very first convolution, the cheapest way to figure out the arrangement is to look at *low-level correlations across tile boundaries* — matching texture and edge continuity where two tiles abut. That solves the puzzle (humans do it too) but requires zero understanding of the global object, so the features would be junk for transfer. I need to *forbid* the network from comparing tiles until it has first formed a real per-tile representation. So: process each tile completely independently through its own conv stack — same AlexNet conv1–conv5 and first fully connected layer fc6, with weights *shared* across all nine tiles — and only *then*, at fc7, concatenate the nine fc6 vectors and let the network reason about arrangement. Nine identical columns up to fc6, joined at fc7: a siamese-ennead network. Each column's receptive field is one tile; there is no cross-tile information flow until the features are already high-level. Context — the relationship between tiles — is deliberately withheld until the last layers, so I'll call the thing context-free.

Two things fall out of this design that I should check before trusting it. First, sharing weights up to fc6 means there's a single tile-feature extractor, and *that* is the thing I'll transfer; the cross-tile reasoning in fc7/fc8 is puzzle-specific and gets thrown away at transfer time, which is what I want. Second — and this one I can actually count rather than assert — I worried that shrinking the input from a full image to a 64×64 tile would cripple fc6 and make the column a weak stand-in for AlexNet. The opposite happens, because the tile-sized input shrinks the conv output from AlexNet's 6×6×256 to 4×4×256 before fc6. Let me compute the fc6 cost both ways. AlexNet's fc6 is 6·6·256→4096, which is 6·6·256·4096 + 4096 = 37,752,832 parameters. The CFN column's fc6 is 4·4·256→512, i.e. 4·4·256·512 + 512 = 2,097,664 parameters. So fc6 drops from ≈37.8M to ≈2.1M — an 18× reduction sitting in the single most parameter-heavy layer of AlexNet. Adding it up: the shared conv1–conv5 stack is about 2.33M, fc6 ≈2.1M, the joint fc7 (9·512→4096) ≈18.9M, and fc8 ≈4.1M, giving a CFN of roughly 27.4M parameters against AlexNet's ≈61M. Less than half the parameters, and on a sanity run feeding full tiles the CFN matches AlexNet on ImageNet classification (about 57% top-1 either way). Good: the column is a fair stand-in for AlexNet, so transferred weights are directly comparable, and I learned that fc6 being so much smaller here is a feature, not a defect.

Now the central failure mode, and it's subtle. Write the network's output as a part-based pdf: p(S | A₁,…,A₉) = p(S | F₁,…,F₉) · Πᵢ p(Fᵢ | Aᵢ), where S is the tile configuration, Aᵢ the appearance of part i, and Fᵢ the intermediate feature for tile i. I want the Fᵢ to be semantic — to describe *what* the part is, so the arrangement can be inferred from content. But there's a degenerate solution. Suppose I only ever show the network *one* puzzle per image, i.e. each tile always appears in the same cell. Then the network can satisfy the task by learning Fᵢ → "absolute cell position," so that p(S | F₁,…,F₉) factorizes into Πᵢ p(Lᵢ | Fᵢ) with each tile's location Lᵢ read straight off its feature. In that solution the features encode arbitrary 2D position and *nothing semantic*. The network solved the puzzle and learned nothing I want.

The fix follows directly from naming the failure: don't let any tile have a fixed cell. Feed *many different* puzzles of the same image, so a given tile must appear in many — ideally all nine — positions over training. Then "map this appearance to a fixed location" is no longer a valid strategy, because the same appearance is the answer "position 1" in one puzzle and "position 7" in another; the only way to predict the index is to reason about the *joint* arrangement of all nine tiles by content. How many distinct puzzles per image does the planned schedule actually give? Training is 350K iterations at batch size 256 over 1.3M images, so each image is drawn on the order of 350,000·256 / 1,300,000 ≈ 68.9 ≈ 69 times, and if I draw a fresh random permutation each time, that's ~69 different puzzles per image. With ~69 placements scattered over the cells, no tile sits in a fixed cell, so the absolute-position shortcut has no purchase — and the lever I have for making those 69 placements maximally scrambled is the *choice of the permutation set*.

That makes the permutation set the real hyperparameter of the method, so let me think carefully about how to choose it. There are 9! = 362,880 possible orderings; I'll use a subset, and the subset's geometry matters in two opposing ways. If the permutations in my set are too *similar* to each other — for instance two that differ only by a single swap, which means they disagree in exactly 2 of the 9 positions — then the task is ambiguous in the same way Doersch's was: if those two swapped tiles happen to look alike, no network can tell the two permutations apart, and I'm training on noise. So I want the permutations to be *far apart*, measured by Hamming distance — the number of tile positions in which two permutations differ. Large Hamming distance means any two permutations disagree in many positions, so distinguishing them requires reading many tiles' content, and there's little chance two near-identical-looking arrangements share a label. On the other side, if I use too *few* permutations, the network can memorize the handful of orderings without learning rich features. So there's a tension: cardinality and average Hamming distance jointly set the difficulty, and a good self-supervised task should be neither so easy the network sidesteps understanding nor so ambiguous that the label is unpredictable.

So I need to *construct* a set of, say, a thousand permutations with large average Hamming distance, and I should confirm that a deliberate construction actually beats just sampling at random — otherwise the whole hyperparameter is illusory. Greedy selection is the natural construction: start from all 9! permutations and an empty set; pick one permutation at random to seed the set; then repeatedly add the permutation (from those remaining) whose mean Hamming distance to the current set is largest — compute, for each candidate, the mean Hamming distance to everything already chosen, and take the argmax. I tried it on the real 362,880-permutation pool. For 50 selected permutations the greedy "max" set has an average pairwise Hamming distance of 8.155 out of 9 positions, whereas a uniformly random subset of the same size sits at 8.02. The gap is modest but real and in the right direction: the greedy set is genuinely pushed toward the maximum-spread corner (9 is the ceiling, where every pair disagrees in all nine positions), and it systematically excludes the near-duplicate, low-distance pairs that create the label ambiguity. So the construction earns its place. (Swap the argmax for a middle pick and I'd get a moderate-distance set; for uniform sampling, a random set — both useful as ablation controls. The set is fixed before training begins.)

Even with the absolute-position shortcut closed, there are the low-level shortcuts the architecture alone doesn't kill, and each needs its own countermeasure. Let me go through them. Adjacent tiles share similar pixel mean and standard deviation, so the network could match neighbors by raw intensity statistics — I normalize the mean and standard deviation of each tile *independently*, erasing that cue. Edge continuity and matching texture right at a shared border is a strong cue that needs no object understanding — I leave a gap between tiles by cutting each tile *smaller* than its cell: from an 85×85 cell I sample a 64×64 tile at a random offset, which leaves up to 85−64 = 21 pixels of empty margin between neighbors, so there's no continuous edge to follow. Chromatic aberration — the lens-induced relative shift between color channels that grows toward the image borders — lets the network localize a tile from its color-fringing alone; I attack it three ways: crop the central square and resize (so all tiles come from near the center where aberration is weak and uniform), train on a mix of grayscale and color images so color cues can't be relied on, and spatially jitter the color channels of each tile by a pixel or two to destroy the residual shift. I can't settle the *relative* importance of these from the desk; I'd want an ablation on downstream detection that drops each one in turn. My prior is that the inter-tile gap matters most (edge continuity is the strongest free cue) and color jitter least, with per-tile normalization in between — but that ordering is a guess until the ablation runs, and the safe call is to keep all three.

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

So the chain: I wanted free single-image supervision that forces object understanding; two-tile relative-position prediction was the closest prior idea but suffered ambiguity because it never sees all the parts together, and the worked sky-tile case showed nine-tile joint reassembly turns a 1/4-guess into a forced placement because the constraints intersect. Predicting the permutation index makes that a clean classification. The architecture has to *withhold* cross-tile context until the features are high-level, or the network cheats on boundary texture; hence the context-free siamese-ennead network with weights shared up to fc6 and joining only at fc7 — which, when I counted the parameters, comes in at ≈27M against AlexNet's ≈61M while reusing the same conv stack, so it doubles as a fair transferable feature extractor. The dangerous shortcut is the network mapping appearance to absolute position; feeding ~69 high-Hamming-distance puzzles per image removes it, and choosing the permutation set by maximal average Hamming distance — which I checked beats a random subset on spread — keeps the task hard-but-unambiguous. Per-tile normalization, inter-tile gaps, and chromatic-aberration countermeasures close the remaining low-level shortcuts, so the only way left to solve the puzzle is to understand the parts.
