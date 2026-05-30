OK, let me think about object detection from scratch. The thing I actually want is dead simple to state: hand me an image, and I'll tell you every object in it — a tight box and a label for each. The trouble is the gap between how easy that is to *say* and how baroque the best systems are to *build*.

Let me look at what the best detectors actually do, because the shape of their pain is going to tell me where to push. The whole field, basically without exception, treats detection as classification run many times. You have a classifier — "is this an object, and which class?" — and you evaluate it at lots of places. DPM (Felzenszwalb, Girshick, McAllester, Ramanan) does this with HOG features and a sliding window over an image pyramid: a root filter plus part filters tied together by a deformable star model, scored by a latent SVM, and then you do non-maximal suppression to clean up overlapping firings. R-CNN (Girshick, Donahue, Darrell, Malik) does it smarter — instead of sliding everywhere, Selective Search (Uijlings et al.) proposes about two thousand candidate regions, you warp each one, push it through a CNN, score the CNN features with per-class SVMs, then a linear regressor nudges the box, then NMS. Fast R-CNN (Girshick) cleans that up a lot: run the conv net once over the whole image, RoI-pool each proposal off the shared feature map, and train a softmax classifier and a box regressor jointly on top.

So where exactly does each of these hurt? Two things keep recurring, and I want to name them precisely because they're the design pressure.

The first is speed, and the speed problem is not where people first assume. In Fast R-CNN the convolution is shared and fast; what kills you is Selective Search — one to two seconds per image, on CPU, before the net even runs. R-CNN is worse, tens of seconds, because it also runs the CNN on two thousand separate crops. Even Faster R-CNN (Ren, He, Girshick, Sun), which learns the proposals with a region proposal network and is genuinely fast, is still a two-stage thing: propose, then classify-and-refine. At its accurate settings it's still under real time. And I want *real* time — thirty frames a second, low latency, for video and for things like a robot or a car.

The second is more subtle and I think more interesting. Every one of these systems is a *pipeline of separately trained parts*. R-CNN: Selective Search (tuned for recall), then a CNN, then SVMs (tuned for per-class accuracy), then a box regressor (tuned for localization), then NMS. Each stage optimizes its own proxy. Nothing in that chain is trained against the actual thing I care about, which is detection quality on the whole image. You can't backpropagate through Selective Search. So the system is never end-to-end optimized for its own goal. That smells wrong. If I could make detection *one function*, one network, I could train it directly against detection.

And there's a third thing, an empirical signature, that I keep coming back to. If I take Fast R-CNN — one of the very best detectors going — and run the Hoiem, Chodpathumwan & Dai error analysis on it, sorting its top detections into correct / localization / similar / other / background, the striking fact is how many of its top detections are *background*: around thirteen-plus percent are boxes with no object in them at all. It's almost three times as likely to fire on background as you'd hope. Why? Because it classifies an *isolated proposal*. The classifier sees a cropped patch and nothing around it. A patch of grass, a rock, a hay bale — out of context, locally, it can look like an object, and the classifier has no way to know it's surrounded by a field. That error mode is baked into the very idea of classifying regions one at a time. If the network instead looked at the *whole image* when scoring any box, it would have the context to reject those.

So now I can state what I want, and it's three things that happen to be mutually reinforcing. One network, one forward pass over the full image (fast, and end-to-end trainable). Trained directly against detection. And it sees the whole image when it predicts, so it has global context and stops hallucinating objects in the background.

Now — is there any precedent for a net that just *emits boxes* instead of classifying pre-cut windows? Yes, a thin thread of it. OverFeat (Sermanet et al.) has one ConvNet do localization, applying its fully-connected layers convolutionally so you get a grid of box regressions in one efficient sweep — but it optimizes localization, not detection, its regressor still only sees a local window, and it needs a lot of post-processing to glue the dense outputs into coherent detections. MultiBox (Erhan, Szegedy, Toshev, Anguelov) trains a CNN to directly regress a fixed set of boxes with confidences — proof a net can output geometry directly — but the boxes are class-agnostic and it's still just a proposal generator feeding a downstream classifier. And there's grasp detection (Redmon & Angelova), where a CNN regresses a single rectangle from an image over a coarse spatial grid. That last one is interesting precisely because it's the crudest: a grid, regress a box. Let me hold onto "grid + regress."

Let me try to build the thing directly and see what breaks. The cleanest fantasy: feed the image to a CNN, have it spit out a list of boxes with labels, train the whole thing on detection. The immediate wall is the output. A CNN with fully-connected layers on top produces a *fixed-size* tensor. But the number of objects in an image is variable — could be zero, could be twenty. How do I get a variable-length set of boxes out of a fixed-size output? And there's a worse, sneakier problem hiding under it: even if I fix the count, how does the network decide *which* output slot predicts *which* object? If I just have it emit, say, ten boxes and match each to the nearest ground-truth box by some assignment at training time, then two output units both drifting toward the same big central object will fight each other, and nothing breaks the symmetry — every output unit is looking at the same global features and has the same job, so they'll all collapse toward the same easy prediction. I need to *divide the labor spatially* so different output units own different parts of the image and stop competing.

That's exactly the move the grasp grid hints at. Divide the image into an S×S grid. Now make a rule: the cell that the object's *center* falls into is the one responsible for predicting that object. This is the load-bearing idea, so let me make sure I believe it. It does several things at once. It gives me a fixed-size output — S×S slots, period, regardless of how many objects there are. It breaks the symmetry — each cell owns a region, so two cells aren't competing to predict the same centrally-located object; the object belongs to whichever cell contains its center and to no other. And it gives me a free, cheap version of duplicate-suppression: since an object lives in exactly one cell, the network is structurally discouraged from emitting the same object from many places. R-CNN and DPM lean hard on NMS to kill duplicates; here the grid does most of that work for me by construction.

So each cell predicts a box: its center coordinates and its size. Let me think about how to parameterize that, because regression targets that aren't bounded will be miserable to train. The center — I'll predict (x, y) as an *offset within the cell*, normalized so they're in [0,1] relative to the cell's bounds. That ties the prediction to the responsible cell, which is what I want, and it's bounded. The width and height — I'll normalize those by the *whole image* width and height, so they're in [0,1] too. Bounded targets everywhere. Good.

Now, does one box per cell suffice? Picture a cell that, across the training set, sometimes has to predict a tall narrow thing (a standing person) and sometimes a wide flat thing (a car). A single regressor in that cell is forced to average those, and it'll be mediocre at both. If instead the cell predicts B boxes, the boxes can *specialize* — one drifts toward tall aspect ratios, the other toward wide — as long as I have a training rule that lets them differentiate. So let each cell predict B boxes. How big should B be? Each box costs me five numbers in the output and I want the tensor small for speed, so I want the *smallest* B that buys me specialization. That's two. B=2: cheap, and enough for two predictors to divide the aspect-ratio/size space between them.

But if a cell emits two boxes for one object, at test time I'll have duplicates again, and at training time which box do I push toward the truth? I want, per object, exactly *one* of the two predictors to be responsible. The natural rule: whichever of the cell's B boxes currently has the highest IOU with the ground-truth box is declared responsible for this object, and only that one gets the box-regression gradient. The other is left alone (for this object). Over training this is self-reinforcing — the box that happens to fit tall objects better keeps getting picked for tall objects, so it keeps specializing on them. That's the specialization I wanted, and it improves recall because the two predictors cover more of the shape space together than one could.

Now the label. Each box needs more than geometry; it needs a sense of "is there actually an object here, and is my box any good?" Call it a confidence score. What should that number *mean*? I want it high only when two things hold: there genuinely is an object, and the box I drew is accurate. So define it as the product Pr(Object) · IOU(predicted, truth). Stare at that. If there's no object, Pr(Object) is zero, so confidence should be zero. If there is an object, Pr(Object) is one and the confidence collapses to exactly the IOU between my box and the truth. So the target for a box's confidence isn't a flat "1" — it's *the actual overlap quality of that box*. That's lovely: the network learns to report its own localization quality. A box that's an object but poorly placed should honestly say so with a low confidence. That's a far more useful signal than a binary objectness, and it's exactly what I'll want for ranking and thresholding at test time.

What about the class label? My first instinct is to attach a class distribution to each of the B boxes. But wait — the B boxes in a cell are not B different objects. They're B competing *hypotheses about the same object*, the one whose center is in this cell. The class is a property of that object/location, not of which candidate rectangle I drew around it. So predicting a separate class distribution per box is redundant and wasteful. Predict *one* set of class probabilities per cell, conditional on there being an object: Pr(Class_i | Object), shared across all B boxes. This also keeps the output tensor small — the cell emits B·5 numbers for the boxes plus C class numbers, not B·(5+C). With S=7, B=2, and VOC's C=20 classes, the whole image output is a 7×7×30 tensor. That's it. One tensor.

Let me check the test-time semantics fall out cleanly. Each cell gives me Pr(Class_i | Object), and each box in the cell gives me its confidence Pr(Object)·IOU. Multiply them:

Pr(Class_i | Object) · Pr(Object) · IOU = Pr(Class_i) · IOU.

So per box, per class, I get a number that is the class probability times the localization quality — exactly the score I'd want to threshold and rank by. It encodes both "is this the right class" and "is this box tight." And across the whole image that's only 7·7·2 = 98 boxes to consider, against Selective Search's two thousand. Far fewer candidates, and each one came from looking at the whole image.

Now the network that produces this tensor. I need a strong feature extractor; detection data is scarce, so I'll pretrain a convolutional backbone on ImageNet classification and then adapt it. For the backbone design I'll borrow from GoogLeNet (Szegedy et al.) — but I don't need the full inception machinery. The cheap, load-bearing trick from Network-in-Network (Lin, Chen, Yan) and inception is the 1×1 convolution: it adds a nonlinearity and, crucially, *reduces the channel count* before an expensive 3×3, which keeps compute down. So I'll just alternate 1×1 reduction layers with 3×3 convolutions — twenty-four convolutional layers — and put two fully-connected layers on top to do the actual box/class regression into that 7×7×30 tensor. Leaky ReLU (slope 0.1) on the hidden layers so units don't die; a linear activation on the final layer because its outputs are bounded regression targets, not logits I want to squash.

A resolution wrinkle: I'll pretrain the conv layers on ImageNet at 224×224 (classification doesn't need fine detail and it's cheaper). But detection *does* need fine spatial detail to localize, so for detection I double the input to 448×448. And since I'm converting a classification net to detection, I'll add fresh layers on top — there's evidence (Ren, He, Girshick, Zhang, Sun) that adding both convolutional and connected layers to a pretrained net helps — so on go four new conv layers and the two FC layers, randomly initialized.

Now the part that actually decides whether this works: the loss. The dream of this whole approach is that I get to train against detection directly with one objective. The crudest choice is sum-squared error over the whole output tensor — boxes, confidences, classes, all of it. It's trivial to optimize and it's differentiable end-to-end, which was the whole point. So let me start there and pressure-test it, because I already suspect plain SSE is misaligned with what I care about (average precision), and I want to find exactly where it breaks.

First problem. SSE weights localization error and classification error equally. There's no reason those should carry the same weight; getting the box right is the hard, valuable part. So I'll want a knob to upweight the coordinate terms.

Second problem, and this one is fatal if I ignore it. Look at the output: it's 7×7 cells, but a typical image has a handful of objects. So the overwhelming majority of cells contain *no object*. For every one of those empty cells, the loss is pushing its boxes' confidence toward zero. Now count the gradient. There are dozens of empty cells each pushing "confidence → 0," versus a few cells with objects pushing "confidence → IOU." The empty-cell gradient swamps the object-cell gradient. The net discovers the easy minimum: predict confidence near zero *everywhere*. Early in training the confidences all collapse, the gradient from the rare positive cells can't overcome the flood, and training diverges. I watched this coming the moment I wrote down 7×7 cells with a few objects.

The fix has to rebalance these two failure modes, and a single mechanism handles both. Introduce two weights. λ_coord, larger than one, multiplies the coordinate-error terms — that fixes problem one (localization matters more) and also helps problem two (it amplifies the signal from the cells that *do* have objects). And λ_noobj, smaller than one, multiplies the confidence error for boxes in cells with *no* object — that directly damps the flood from empty cells so it stops overpowering everything. Concretely λ_coord = 5 and λ_noobj = 0.5. Five and a half: coordinates count five times, empty-cell confidence counts half. The asymmetry is the whole point.

Third problem. SSE treats an absolute error in a box the same regardless of the box's size. But it shouldn't: if I'm twenty pixels off on the width of a three-hundred-pixel bus, who cares — the IOU barely moves. Twenty pixels off on a thirty-pixel bird and I've blown the box entirely; the IOU craters. So the same absolute width error should hurt *more* on a small box than a large one, and plain squared error on w and h does the opposite of accounting for that — it's scale-blind. I want to compress the large values so a fixed absolute error there contributes less. The trick: regress the *square root* of width and height instead of width and height. Then the squared error is on sqrt(w), not w. The derivative of sqrt at large w is small, so a given absolute change in a big box maps to a small change in sqrt-space and a small loss; the same absolute change in a small box, where sqrt is steeper, maps to a larger change and larger loss. It doesn't perfectly equalize the IOU sensitivity, but it tilts the loss the right way for almost free. So: predict √w and √h.

Now I can assemble the whole objective. I need indicator bookkeeping: let 𝟙ᵢⱼ^obj be 1 when box j in cell i is the *responsible* predictor for the object centered in cell i (highest current IOU among the cell's boxes), 𝟙ᵢⱼ^noobj its complement, and 𝟙ᵢ^obj be 1 when an object's center is in cell i at all. With those, the loss is five sums:

- λ_coord · Σᵢ Σⱼ 𝟙ᵢⱼ^obj [ (x − x̂)² + (y − ŷ)² ] — center error, only on the responsible box.
- λ_coord · Σᵢ Σⱼ 𝟙ᵢⱼ^obj [ (√w − √ŵ)² + (√h − √ĥ)² ] — size error in sqrt-space, only on the responsible box.
- Σᵢ Σⱼ 𝟙ᵢⱼ^obj (C − Ĉ)² — confidence error for the responsible box, whose target C is its IOU with the truth.
- λ_noobj · Σᵢ Σⱼ 𝟙ᵢⱼ^noobj (C − Ĉ)² — confidence error for the boxes that aren't responsible / cells with no object, target zero, downweighted.
- Σᵢ 𝟙ᵢ^obj Σ_c (p(c) − p̂(c))² — class error, only when the cell actually contains an object.

Two of these conditionals deserve a second look because they encode real choices. The class term carries 𝟙ᵢ^obj: I only penalize class predictions for cells that contain an object. That's the right reading of "Pr(Class | Object)" — the class distribution is *conditional* on an object being there, so it's meaningless (and shouldn't be trained) in empty cells. And the coordinate terms carry 𝟙ᵢⱼ^obj, the per-box responsibility flag: I only penalize the geometry of the one predictor that owns the object, which is exactly the specialization rule from before — the other box isn't dragged toward this object, so it's free to specialize elsewhere.

Let me make sure the responsibility assignment is consistent with the confidence definition, because they're coupled. The responsible box is the one with the current highest IOU with the truth; its confidence target is *that IOU*; and only it gets coordinate gradient. So the box that's already fitting best is the one told to keep improving and to report its overlap honestly — a clean, self-reinforcing loop. (In the actual implementation there's a small wrinkle: when none of a cell's boxes overlaps the truth at all, IOU is zero for all of them and you can't rank by it, so you fall back to picking the box with the smallest coordinate RMSE to the truth. Same intent — pick the closest one and make it responsible.)

Training schedule, briefly, because the divergence risk is real and it shapes the recipe. I can't just slam in a high learning rate — with this loss and a flood of empty cells the gradients are unstable and the model blows up. So warm up: ramp from 1e-3 to 1e-2 over the first epochs, hold 1e-2 for a long stretch, then step down to 1e-3 and 1e-4. Momentum 0.9, weight decay 5e-4, batch 64. Detection data is scarce so overfitting is a threat: dropout 0.5 after the first fully-connected layer to prevent co-adaptation (Hinton et al.), plus aggressive augmentation — random scaling and translation up to 20% of image size, and random exposure/saturation jitter in HSV.

At test time it's almost anticlimactic, which is the entire selling point: one forward pass produces the 7×7×30 tensor, I read off the 98 boxes and their class-specific scores, threshold, and I'm essentially done. The grid already suppressed most duplicates by construction. The one residual case is a large object, or one sitting on a cell boundary, that gets confidently claimed by more than one cell. For those I run non-maximal suppression — but note the difference in stakes: for R-CNN and DPM, NMS is load-bearing because they generate thousands of overlapping firings; here it's a small cleanup that adds a couple of points of mAP, not the thing holding the system together.

I should be honest about what this design *costs*, because the same choices that make it fast and global also box me in. The grid imposes a hard spatial constraint: each cell predicts only B=2 boxes and only one class. So if many small objects crowd into one cell — a flock of birds — I structurally cannot emit them all. The features feeding the predictor are coarse, because the conv stack downsamples the image several times, so unusual aspect ratios and small objects are harder to pin down. And even with the sqrt fix, the loss still doesn't fully respect that a small absolute error matters more on a small box — so I'd expect localization to be my dominant error source, the mirror image of Fast R-CNN's background-false-positive problem. That trade is one I'm willing to make for a single end-to-end network that runs in real time and reasons about the whole image. And there's an upside I'd bet on but want to verify: because the model learns shape and layout from whole images rather than classifying local crops, it should carry over to domains where pixel statistics shift — artwork — better than a proposal-based system whose proposals are tuned to natural-image texture.

Let me write the network, the loss, and the training loop, tying each block back to the reasoning.

```python
import torch
import torch.nn as nn

# Backbone: GoogLeNet-inspired but no inception. The cheap, load-bearing trick
# is the 1x1 reduction (NIN / inception) before each expensive 3x3. 24 conv layers,
# then 2 FC layers that regress the S x S x (B*5 + C) detection tensor.
# Each tuple = (kernel, out_channels, stride, padding); "M" = 2x2 maxpool;
# [..., n] = the bracketed block repeated n times.
architecture_config = [
    (7, 64, 2, 3), "M",
    (3, 192, 1, 1), "M",
    (1, 128, 1, 0), (3, 256, 1, 1), (1, 256, 1, 0), (3, 512, 1, 1), "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4], (1, 512, 1, 0), (3, 1024, 1, 1), "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2], (3, 1024, 1, 1), (3, 1024, 2, 1),
    (3, 1024, 1, 1), (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)  # leaky so units don't die
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, split_size=7, num_boxes=2, num_classes=20):
        super().__init__()
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(architecture_config)
        self.fcs = self._create_fcs(split_size, num_boxes, num_classes)
    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))  # -> S*S*(C + B*5)
    def _create_conv_layers(self, arch):
        layers, in_c = [], self.in_channels
        for x in arch:
            if type(x) == tuple:
                layers += [CNNBlock(in_c, x[1], kernel_size=x[0], stride=x[2], padding=x[3])]
                in_c = x[1]
            elif x == "M":
                layers += [nn.MaxPool2d(2, 2)]
            elif type(x) == list:
                c1, c2, reps = x[0], x[1], x[2]
                for _ in range(reps):
                    layers += [CNNBlock(in_c, c1[1], kernel_size=c1[0], stride=c1[2], padding=c1[3])]
                    layers += [CNNBlock(c1[1], c2[1], kernel_size=c2[0], stride=c2[2], padding=c2[3])]
                    in_c = c2[1]
        return nn.Sequential(*layers)
    def _create_fcs(self, S, B, C):
        # 2 FC layers do the actual box/class regression; linear final activation.
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.Dropout(0.5),          # detection data is scarce -> regularize
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (C + B * 5)),  # the 7x7x30 tensor
        )


class YoloLoss(nn.Module):
    """Sum-squared error, reshaped to respect detection. Per-cell layout used here:
    [ C class scores | box1: conf,x,y,w,h | box2: conf,x,y,w,h ] -> length C + B*5 = 30."""
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S, self.B, self.C = S, B, C
        self.lambda_coord = 5      # upweight localization + amplify the few object cells
        self.lambda_noobj = 0.5    # damp the flood of empty-cell confidence gradients

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # Responsibility: of the B boxes in a cell, the one with higher IOU to the
        # truth owns the object and gets the coordinate/confidence gradient.
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        _, bestbox = torch.max(ious, dim=0)          # 0 or 1: which box is responsible
        exists_box = target[..., 20].unsqueeze(3)    # 1_i^obj: is there an object in this cell

        # --- coordinate loss (responsible box only); sqrt on w,h so a fixed absolute
        #     error costs less on big boxes than small ones. sign/abs guard keeps sqrt
        #     differentiable through possibly-negative raw predictions. ---
        box_predictions = exists_box * (
            bestbox * predictions[..., 26:30] + (1 - bestbox) * predictions[..., 21:25]
        )
        box_targets = exists_box * target[..., 21:25]
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # --- object confidence loss: target is the IOU of the responsible box. ---
        pred_box = bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )

        # --- no-object confidence loss: every box in an empty cell, target 0, downweighted. ---
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        # --- class loss: only for cells that contain an object (Pr(Class | Object)). ---
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2),
        )

        return (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )
```

So the whole causal chain, end to end: I wanted a detector that's fast, end-to-end trainable, and globally aware, because the pipelines of the day were slow, jointly-untrainable, and blind to context (which showed up as background false positives). To get one network and one forward pass I had to emit a fixed-size output and break the symmetry between predictors, which forced the S×S grid with center-based responsibility — that simultaneously gives a fixed output, divides the spatial labor, and suppresses duplicates for free. Each cell predicts B=2 boxes so the predictors can specialize by aspect ratio, a shared per-cell class distribution because the boxes are competing hypotheses about one object, and a confidence equal to Pr(Object)·IOU so the score reports localization quality and multiplies cleanly with the class prob at test time. Training it with plain sum-squared error exposed three misalignments — localization underweighted, empty cells flooding the gradient into divergence, and scale-blind size error — which I patched with λ_coord=5, λ_noobj=0.5, and the √w,√h reparameterization respectively. The backbone is a GoogLeNet-style 24-conv stack with 1×1 reductions, pretrained on ImageNet at 224 and fine-tuned for detection at 448. One tensor out, 98 boxes, a touch of NMS, real time.
