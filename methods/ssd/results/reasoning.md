I start with the bottleneck that is impossible to ignore. The accurate detectors all spend their budget in a loop over candidate boxes. First they hypothesize regions; then each region is pooled or cropped; then a classifier and regressor operate on that region. Computing the convolutional map once helps, and learning the proposals helps, but as long as every candidate gets its own feature-resampling step, the detector is not really a single computation over the image.

So I want the network to emit detections directly. Not proposals that must be looked at again, and not a small set of coarse global boxes, but final class scores and final localization corrections from one feed-forward pass.

The immediate danger is that this sounds like the single-pass detectors that already lose accuracy. I need to understand why they lose it. A detector that predicts from one deep feature map has one stride and one grid. If the stride is coarse, a small object has only a few cells, sometimes less than one useful cell, to describe it. If several small objects are near each other, the grid cell assignment becomes crowded. If the head is fully connected on top of the whole feature map, the prediction is no longer a small local rule repeated across the image; it is a large global regression problem.

The proposal networks have a better local representation. At a feature-map location, place several reference boxes and predict offsets to those references. The center offsets are measured in units of the reference width and height, and the width and height are log ratios. This turns a hard absolute-coordinate problem into a small residual correction. It also gives the convolutional predictor a consistent job at every spatial location.

But a proposal network still stops halfway. It predicts objectness and candidate boxes, then passes those boxes to a second detector. I do not want objectness plus a second detector. I want, for each reference box, the class scores themselves and the four offsets themselves.

That still leaves scale. If I attach this predictor to only one deep feature map, I have only repaired the head, not the small-object problem. The network already computes a sequence of feature maps. Earlier maps are spatially finer; later maps are coarser and more semantic. The image pyramid idea says different resolutions help with different object sizes, but an image pyramid costs multiple forward passes. The internal feature maps are the cheaper version: a feature pyramid already exists inside one pass.

So I attach predictors to several feature maps. The high-resolution map gets small reference boxes and many locations; the low-resolution maps get larger boxes and fewer locations. This lets each map specialize to a scale band instead of asking one map to cover everything.

I want to pin down what "specialize" is allowed to mean here, because the obvious worry is whether the default-box size on a map has to match that map's receptive field. If it does, the design becomes a fragile receptive-field-matching exercise. But the predictor on a given map is a learned 3x3 convolution producing residual offsets; it does not need the prior to coincide with its receptive field, only to start close enough that the offset is small. So I size the boxes by a smooth progression and let the matching decide which map owns which object, rather than computing receptive fields.

Now I need the actual default boxes. The centers should be tied to cell centers. For a square feature map, the clean formula is center at ((i + 0.5) / |f_k|, (j + 0.5) / |f_k|). I can also specify explicit pixel steps for each source map; for the 300-pixel model these steps are 8, 16, 32, 64, 100, and 300. This handles the 10 by 10 and 3 by 3 maps even though they do not divide 300 exactly.

For size, I want a regular progression from small to large. If there are m prediction maps, define

  s_k = s_min + (s_max - s_min)(k - 1)/(m - 1).

With s_min = 0.2 and s_max = 0.9 and m = 6, k = 1..6 gives 0.2, 0.34, 0.48, 0.62, 0.76, 0.9. In pixels on a 300 image that is 60, 102, 144, 186, 228, 270. So the deepest map carries boxes that are nearly the whole image and the progression is roughly linear in size, which is what I wanted from a coarse-to-fine pyramid. The one thing this table does not cover is genuinely small objects: even the first scale is 60 pixels. I will come back to that.

For aspect ratio a, I use

  w = s_k sqrt(a),  h = s_k / sqrt(a).

The area stays s_k^2 regardless of aspect ratio, since w*h = s_k^2*sqrt(a)/sqrt(a), and the ratio w/h is exactly a. So changing aspect ratio reshapes the box without changing how much image it covers.

A square object whose size falls between two adjacent scales might be poorly covered, so I add another square default box at the geometric mean scale:

  s'_k = sqrt(s_k s_{k+1}).

That gives, per location, two squares (the s_k square and the s'_k square) plus a box and its reciprocal for each aspect ratio. So with aspect ratios 2, 3 the count is 2 + 2*2 = 6, and with only aspect ratio 2 it is 2 + 2 = 4. Using ratios 1, 2, 3, 1/2, 1/3 plus the extra square is the six-box case; dropping the extreme pair 3 and 1/3 leaves four boxes per location.

When I instantiate the 300 by 300 detector, the small-object hole I flagged above forces one special case. The generic formula makes the first scale 0.2 (60 px), but the highest-resolution 38 by 38 map is the only real chance for small objects, so I give it a smaller default size: 30 pixels, or 0.1 of the image, with the extra square bridging to 60 pixels. The bridge lands at s' = sqrt(30*60) = 42.4 px, between 30 and 60, so conv4_3 carries a 30 px square, a 42.4 px square, and the two aspect-2 boxes — a sensible small-object set. With conv4_3 pulled down to 0.1, I re-spread the remaining five maps evenly across the rest of the range: from 0.2 up toward 0.9 in four equal gaps of 0.17, which is 0.20, 0.37, 0.54, 0.71, 0.88, and each map's extra square then bridges to the next map's scale, with the last bridge reaching just past the image at 1.05. The six source maps then use min sizes 30, 60, 111, 162, 213, and 264; max sizes 60, 111, 162, 213, 264, and 315; and box counts 4, 6, 6, 6, 4, and 4.

Now the count of default boxes matters, because it determines the whole imbalance problem later. Across 38^2, 19^2, 10^2, 5^2, 3^2, and 1^2 cells with those per-cell counts:

  38^2 * 4 = 5776
  19^2 * 6 = 2166
  10^2 * 6 = 600
  5^2 * 6  = 150
  3^2 * 4  = 36
  1^2 * 4  = 4

Summing: 5776 + 2166 + 600 + 150 + 36 + 4 = 8732 default boxes. So the high-resolution map alone contributes 5776 of them, two thirds of the total, which confirms that most of the spatial density — and therefore most of the small-object capacity — lives on conv4_3. This is far denser than a 7 by 7 grid with two boxes per cell, but still cheap because the predictions are just convolutions.

The predictor itself is simple. If a feature map has p channels and a cell has k default boxes, a 3 by 3 by p convolution emits 4k localization channels and another emits ck confidence channels. After permuting and flattening, the detector has one localization vector and one class-score vector per default box.

There is a subtlety with the early high-resolution map I have just made responsible for most of the boxes. conv4_3 is shallow, so its feature magnitudes are not on the same scale as the deeper maps that have passed through more layers. If I feed raw conv4_3 features into the same kind of predictor, the gradients across sources will be mismatched. I L2-normalize each spatial feature vector and then learn a per-channel scale. The normalization fixes the magnitude; the learnable scale lets the network undo it if it wants. Initializing that scale at 20 puts the normalized features back into a usable range rather than the unit norm the L2 step would otherwise force.

The matching problem is the next wall, and it is a real one because the output is fixed in advance while the image contains a variable number of objects. The simplest rule is to match each object to its single best default box. That guarantees every object has exactly one positive, but consider what it does to the 8732-box detector: for an object that genuinely overlaps a dozen nearby defaults well, eleven of them get labeled background even though they sit right on the object. Those eleven then push high non-background confidence as false negatives and fight the one positive. With thousands of defaults that is far too harsh a labeling. So I use a two-step assignment: first, each ground-truth box claims its best-overlap default so no object is orphaned; second, every default whose best Jaccard overlap with any ground truth is at least 0.5 also becomes positive for that ground truth. Now one object can train several overlapping boxes, and the duplicates this creates are exactly what non-maximum suppression removes at inference, so I am not creating a problem downstream.

The localization target follows the anchor-regression parameterization. For default box d and matched ground truth g,

  g_hat_cx = (g_cx - d_cx) / d_w,
  g_hat_cy = (g_cy - d_cy) / d_h,
  g_hat_w  = log(g_w / d_w),
  g_hat_h  = log(g_h / d_h).

To check the sign convention and that decoding is the exact inverse: take a default at d = (0.5, 0.5, 0.2, 0.2) and a ground truth shifted right and made wider, g = (0.6, 0.5, 0.3, 0.2). The center-x residual is (0.6 - 0.5)/0.2 = 0.5, positive, as it should be for an object to the right of the default. The width target is log(0.3/0.2) = log(1.5) = +0.405, positive, as it should be for an object wider than the default; a narrower object would give a negative log. The y and h targets are 0 because g matches d there. Decoding adds the scaled center offset back to the default center and exponentiates the scaled size offset: cx = 0.5 + 0.5*0.2 = 0.6, w = 0.2*exp(0.405) = 0.2*1.5 = 0.3. That returns g exactly, so encode and decode are inverse and the sign convention is consistent.

For numerical scaling I divide the x and y center targets by 0.1 and the width and height log targets by 0.2, and at inference I multiply by those same variances before decoding — the same substitution as above with an extra factor of 0.1 that decoding immediately undoes, so the variance only rescales the regression target's numeric range and does not touch the geometry.

The loss is the sum of a confidence term and a localization term, divided by the number N of matched defaults:

  L = (L_conf + alpha L_loc) / N.

I set alpha to 1. The localization term is Smooth L1 and only positives contribute, because there is no ground-truth box for background. The confidence term is softmax cross-entropy over all classes including background, with class 0 reserved for background. A positive box is trained toward its matched object class; a selected negative box is trained toward background. I need to handle N = 0 explicitly: if an image (or crop) has no matched positive, there is no localization target and, as I will set up next, hard-negative mining selects negatives in proportion to positives, so it selects none either. Both terms are empty, and dividing by N = 0 would be a NaN, so the loss must be forced to zero in that case.

The class imbalance is severe, and I now have the exact number to see how severe. With 8732 default boxes and usually only a handful of objects, after matching maybe a few dozen boxes are positive and the rest — well over 8000 — are background. If I keep all negatives, the gradient is overwhelmingly "this is background" and the positives are drowned. Random negative sampling spends the budget on easy background that the classifier already gets right. The useful negatives are the hard ones: background boxes currently receiving high non-background confidence, i.e. the would-be false positives. So after matching I compute confidence loss for the negative boxes, sort them by that loss in descending order, and keep at most three negatives per positive. The 3:1 ratio keeps the gradient dominated by the decision boundary cases rather than by trivial background, and it ties the negative count to the positive count so the balance holds whether an image has one object or ten.

I also need to keep boundary boxes. Some defaults cross the normalized image boundary, especially on coarse maps or with wide aspect ratios. If I clip or discard them too early, I change the intended tiling and lose large-object coverage. The detector can train with those boundary boxes and decode them later; final output can still be clipped to the image for evaluation. That is why the prior configuration sets clip to false.

For the backbone, I use VGG-16 without its classification head, keeping the fully convolutional form by converting the old fully connected layers into convolutions. That conversion has a catch. VGG's fc6 sees the 7 by 7 by 512 pool5 output, so as a dense layer it has 7*7*512 = 25088 inputs per output unit; written as a convolution it becomes a 7 by 7 kernel with thousands of output channels, over a hundred million parameters — more than the entire convolutional stack below it. The fix is to subsample the fc6 and fc7 parameters when converting them, keeping only a stride-sampled slice of each kernel. The second catch is pool5 itself. I want the feature map after conv5 to stay at 19 by 19 rather than drop again, so I change pool5 from 2 by 2 with stride 2 to 3 by 3 with stride 1: the padding keeps the map size and nothing downsamples. But every layer after pool5 now sees features sampled twice as densely as its weights were trained for, which would halve its receptive field. The remedy is to space out the kernel taps — dilate the following convolutions — so each tap skips the same factor that the removed stride would have subsampled by. Same receptive field, same parameter count, one less stride. Then I append extra convolutional layers that shrink the spatial size down to the 1 by 1 source, giving me the six prediction maps the box configuration assumes.

Training has to manufacture scale variation because there is no region crop at detection time. The matching is fixed to whatever sizes appear, so the only way each map sees objects in its scale band is to vary object size in the data. Random crops act like zoom-in operations: sample a patch with minimum Jaccard overlap chosen from values such as 0.1, 0.3, 0.5, 0.7, and 0.9, or use a random patch; keep boxes whose centers remain inside; resize, flip, and distort color. Zooming in helps large and medium objects but does nothing for the small-object maps, so I also need the opposite operation: place the image on a larger mean-filled canvas before cropping, so the original objects become smaller in the training crop and conv4_3's small boxes actually get matched.

At inference I decode all default boxes, discard very low class scores with a 0.01 threshold, and run per-class NMS at Jaccard 0.45, then keep the top 200 detections per image. The low score threshold lets weak true positives survive into NMS; NMS collapses the deliberate duplicate matches from training back to one box per object without reintroducing a proposal-classification loop.

What remains is a single network with no proposal stage: the parts of proposal networks that earned their keep — local convolutional predictors, reference boxes, offset regression — now attach directly to final class scores and box outputs, spread dense across both spatial location and feature-map scale, so the one-pass detector is not trapped by the single coarse grid that costs OverFeat and YOLO their accuracy.
