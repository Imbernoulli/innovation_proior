I start with the bottleneck that is impossible to ignore. The accurate detectors all spend their budget in a loop over candidate boxes. First they hypothesize regions; then each region is pooled or cropped; then a classifier and regressor operate on that region. Computing the convolutional map once helps, and learning the proposals helps, but as long as every candidate gets its own feature-resampling step, the detector is not really a single computation over the image.

So I want the network to emit detections directly. Not proposals that must be looked at again, and not a small set of coarse global boxes, but final class scores and final localization corrections from one feed-forward pass.

The immediate danger is that this sounds like the single-pass detectors that already lose accuracy. I need to understand why they lose it. A detector that predicts from one deep feature map has one stride and one grid. If the stride is coarse, a small object has only a few cells, sometimes less than one useful cell, to describe it. If several small objects are near each other, the grid cell assignment becomes crowded. If the head is fully connected on top of the whole feature map, the prediction is no longer a small local rule repeated across the image; it is a large global regression problem.

The proposal networks have a better local representation. At a feature-map location, place several reference boxes and predict offsets to those references. The center offsets are measured in units of the reference width and height, and the width and height are log ratios. This turns a hard absolute-coordinate problem into a small residual correction. It also gives the convolutional predictor a consistent job at every spatial location.

But a proposal network still stops halfway. It predicts objectness and candidate boxes, then passes those boxes to a second detector. I do not want objectness plus a second detector. I want, for each reference box, the class scores themselves and the four offsets themselves.

That still leaves scale. If I attach this predictor to only one deep feature map, I have only repaired the head, not the small-object problem. The network already computes a sequence of feature maps. Earlier maps are spatially finer; later maps are coarser and more semantic. The image pyramid idea says different resolutions help with different object sizes, but an image pyramid costs multiple forward passes. The internal feature maps are the cheaper version: a feature pyramid already exists inside one pass.

So I attach predictors to several feature maps. The high-resolution map gets small reference boxes and many locations; the low-resolution maps get larger boxes and fewer locations. This lets each map specialize to a scale band instead of asking one map to cover everything. I do not need the default boxes to equal the exact receptive field. I only need the assignment to be learnable: boxes of the right approximate size should be predicted from maps with appropriate spatial support.

Now I need the actual default boxes. The centers should be tied to cell centers. For a square feature map, the clean formula is center at ((i + 0.5) / |f_k|, (j + 0.5) / |f_k|). I can also specify explicit pixel steps for each source map; for the 300-pixel model these steps are 8, 16, 32, 64, 100, and 300. This handles the 10 by 10 and 3 by 3 maps even though they do not divide 300 exactly.

For size, I want a regular progression from small to large. If there are m prediction maps, define

  s_k = s_min + (s_max - s_min)(k - 1)/(m - 1).

With s_min = 0.2 and s_max = 0.9, this spreads square-box sizes across the maps. For aspect ratio a, I use

  w = s_k sqrt(a),  h = s_k / sqrt(a).

The area stays s_k^2 and the width-height ratio is a. A square object between two adjacent scales might be poorly covered, so I add another square default box at the geometric mean scale:

  s'_k = sqrt(s_k s_{k+1}).

That gives six boxes per location when I use aspect ratios 1, 2, 3, 1/2, and 1/3 plus the extra square. On some maps I can drop the extreme ratio pair 3 and 1/3, leaving four boxes per location.

When I instantiate the 300 by 300 detector, I need one special case. The generic formula makes the first scale 0.2, but the highest-resolution 38 by 38 map is the only real chance for small objects, so I give it a smaller default size: 30 pixels, or 0.1 of the image, with the extra square bridging to 60 pixels. The six source maps then use min sizes 30, 60, 111, 162, 213, and 264; max sizes 60, 111, 162, 213, 264, and 315; and box counts 4, 6, 6, 6, 4, and 4. Across 38^2, 19^2, 10^2, 5^2, 3^2, and 1^2 cells, that is

  38^2 * 4 + 19^2 * 6 + 10^2 * 6 + 5^2 * 6 + 3^2 * 4 + 1^2 * 4 = 8732

default boxes. This is far denser than a 7 by 7 grid with two boxes per cell, but still cheap because the predictions are just convolutions.

The predictor itself is simple. If a feature map has p channels and a cell has k default boxes, a 3 by 3 by p convolution emits 4k localization channels and another emits ck confidence channels. After permuting and flattening, the detector has one localization vector and one class-score vector per default box. I also need to normalize the early high-resolution map before predicting from it, because its feature magnitude is not on the same scale as the deeper maps. L2-normalize each spatial feature vector and learn a per-channel scale initialized to 20.

The matching problem is the next wall. Every output is fixed in advance, but the image contains a variable number of objects. If I only match each object to one default box, I guarantee every object has a positive, but I also punish nearby boxes that are genuinely good overlaps. With thousands of defaults, that is too harsh. I use a two-step assignment: first, each ground-truth box claims its best-overlap default so no object is orphaned; second, every default whose best Jaccard overlap with any ground truth is at least 0.5 becomes positive for that ground truth. Now one object can train several overlapping boxes, and duplicate detections can be removed later by non-maximum suppression.

The localization target follows the anchor-regression parameterization. For default box d and matched ground truth g,

  g_hat_cx = (g_cx - d_cx) / d_w,
  g_hat_cy = (g_cy - d_cy) / d_h,
  g_hat_w  = log(g_w / d_w),
  g_hat_h  = log(g_h / d_h).

The signs matter. If the object center is to the right of the default center, the x target is positive. If the object is wider than the default, the width target is positive because log(g_w/d_w) is positive. If it is smaller, the target is negative. Decoding reverses this by adding the scaled center offsets back to the default center and exponentiating the scaled size offsets.

For numerical scaling, I divide the x and y center targets by 0.1 and the width and height log targets by 0.2. At inference I multiply by those same variances before decoding. That changes target scale, not the underlying geometry.

The loss is the sum of a confidence term and a localization term, divided by the number N of matched defaults:

  L = (L_conf + alpha L_loc) / N.

I set alpha to 1. The localization term is Smooth L1 and only positives contribute, because there is no ground-truth box for background. The confidence term is softmax cross-entropy over all classes including background, with class 0 reserved for background. A positive box is trained toward its matched object class; a selected negative box is trained toward background. If N is zero, the whole loss must be zero, because there is no positive localization target and negative mining with a 3:1 ratio selects no negatives.

The class imbalance is severe. With 8732 default boxes and usually only a few objects, almost every box is background. If I keep all negatives, the gradient becomes mostly "background" and the positives are drowned. Random negative sampling wastes most of the budget on easy background. I need the difficult background boxes, the ones currently receiving high non-background confidence. So after matching I compute confidence loss for negative boxes, sort by loss, and keep at most three negatives per positive. This gives the classifier useful false positives without letting background dominate training.

I also need to keep boundary boxes. Some defaults cross the normalized image boundary, especially on coarse maps or with wide aspect ratios. If I clip or discard them too early, I change the intended tiling and lose large-object coverage. The detector can train with those boundary boxes and decode them later; final output can still be clipped to the image for evaluation.

For the backbone, I use VGG-16 without its classification head. I keep the fully convolutional form by converting the old fully connected layers into convolutions. The pool5 stride becomes 1, and the following convolution uses holes so the receptive field remains large without another downsampling step. Then I append extra convolutional layers that shrink the spatial size down to the 1 by 1 source.

Training has to manufacture scale variation because there is no region crop at detection time. Random crops act like zoom-in operations: sample a patch with minimum Jaccard overlap chosen from values such as 0.1, 0.3, 0.5, 0.7, and 0.9, or use a random patch, keep boxes whose centers remain inside, resize, flip, and distort color. To help small objects, I also need the opposite operation: place the image on a larger mean-filled canvas before cropping, so the original objects become smaller in the training crop.

At inference I decode all default boxes, discard very low class scores, and run per-class NMS. With a confidence threshold of 0.01, NMS threshold 0.45, and a final keep-top-200 limit, the detector keeps duplicates under control without reintroducing a proposal-classification loop.

The final shape is now clear. A single network takes the image, computes a VGG-based stack plus extra feature maps, L2-normalizes the early high-resolution source, runs small convolutional predictors on six maps, emits class scores and variance-scaled center/log-size offsets for 8732 default boxes, trains by best-plus-threshold matching with Smooth L1 and softmax, mines hard negatives at 3:1, and uses NMS only after the one forward pass. The proposal stage is gone, but the useful parts of proposal networks remain: local convolutional predictors, reference boxes, and offset regression. The missing piece was to make those references dense across both location and feature-map scale, so the one-pass detector is not trapped by one coarse grid.
