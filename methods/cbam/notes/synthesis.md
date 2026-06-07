# CBAM synthesis

## Verified arXiv id
1807.06521 "CBAM: Convolutional Block Attention Module" (Woo, Park, Lee, Kweon; KAIST/Adobe/Lunit, ECCV 2018).
Canonical code: github.com/Jongchan/attention-module → MODELS/cbam.py (fetched to code/).

## Pain point / research question
CNN improvements focused on depth, width, cardinality. A different axis: attention — emphasize informative features, suppress noise. Convolution blends cross-channel AND spatial info, so refine along both principal axes: channel ("what") and spatial ("where"). Want a lightweight, plug-and-play module insertable at every conv block, negligible param/compute overhead, end-to-end trainable.

## The method (exact equations)
Input intermediate feature F ∈ R^{C×H×W}. CBAM sequentially infers:
- 1D channel attention M_c ∈ R^{C×1×1}
- 2D spatial attention M_s ∈ R^{1×H×W}
Overall (eq first):
  F' = M_c(F) ⊗ F        (channel attention broadcast over spatial)
  F'' = M_s(F') ⊗ F'     (spatial attention broadcast over channels)
⊗ = element-wise multiply with broadcasting. F'' = final refined output.

### Channel attention module (eq third)
Squeeze spatial dim with BOTH avg-pool and max-pool → two C×1×1 descriptors F^c_avg, F^c_max. Feed each through a SHARED MLP (one hidden layer, hidden size C/r, r = reduction ratio). Sum the two outputs, sigmoid.
  M_c(F) = σ( MLP(AvgPool(F)) + MLP(MaxPool(F)) )
         = σ( W_1(W_0(F^c_avg)) + W_1(W_0(F^c_max)) )
W_0 ∈ R^{C/r × C}, W_1 ∈ R^{C × C/r}; ReLU after W_0; W_0,W_1 shared between both inputs.
Why both pools? avg-pool (SE uses only this) gathers smooth global stats; max-pool gathers distinctive-feature clues; together → finer attention. r=16 default.
Channel-attention-with-avg-only == SE module exactly.

### Spatial attention module (eq forth)
Pool along the CHANNEL axis with avg and max → two 1×H×W maps F^s_avg, F^s_max. Concatenate → 2×H×W. Apply one 7×7 conv → 1×H×W. Sigmoid.
  M_s(F) = σ( f^{7×7}( [AvgPool(F); MaxPool(F)] ) )
Channel-axis pooling highlights informative regions (Zagoruyko & Komodakis 2017, attention transfer). 7×7 conv = large receptive field for spatial context. Symmetric in spirit to channel module.

### Arrangement
Channel and spatial = complementary (what vs where). Options: sequential or parallel. Found: sequential > parallel. Within sequential: channel-first > spatial-first (slightly). So order = channel then spatial.

## Code grounding (cbam.py)
- ChannelGate: MLP = [Flatten, Linear(C, C//r), ReLU, Linear(C//r, C)]. For pool in ['avg','max']: avg_pool2d / max_pool2d over (H,W) → MLP → sum into channel_att_sum. scale = sigmoid(sum).unsqueeze(2,3).expand_as(x); return x*scale. reduction_ratio=16.
- ChannelPool: cat( max over channel dim (keepdim), mean over channel dim ) → 2×H×W.
- SpatialGate: compress=ChannelPool; spatial=BasicConv(2,1,kernel=7,padding=3,relu=False) (Conv+BN, no ReLU); scale=sigmoid(out); return x*scale.
- CBAM: forward → x=ChannelGate(x); if not no_spatial: x=SpatialGate(x); return x. (channel then spatial, sequential.)
- BasicConv: Conv2d(bias=False) + BN(eps1e-5, mom0.01) + optional ReLU.
Note: code also supports 'lp'/'lse' pool types but default pool_types=['avg','max'].

## Integration (ResNet)
CBAM placed inside each residual block, applied to the block's conv output before the residual addition (so the refinement is on the residual branch). Param overhead negligible.

## Ancestors (load-bearing)
- SE / Squeeze-and-Excitation (Hu 2017): channel attention via global AVG pool → FC reduce → ReLU → FC → sigmoid → rescale. The direct predecessor for channel attention; CBAM adds max-pool and spatial attention.
- Residual Attention Network (Wang 2017): encoder-decoder 3D attention mask; heavier. CBAM decomposes into separate channel+spatial → cheaper.
- Attention transfer (Zagoruyko & Komodakis 2017): channel-pooled spatial maps highlight informative regions.
- CAM (Zhou 2016): global avg pool localizes object extent.
- ResNet (He 2016), ResNeXt, WideResNet, DenseNet: the base nets.
- Channel = feature detector view (Zeiler & Fergus 2014).

## Evaluation settings (no outcomes)
ImageNet-1K classification (ResNet/WideResNet/ResNeXt/MobileNet backbones; SGD lr 0.1 drop every 30 ep, 90 epochs, single 224 crop, top-1/top-5 error). MS COCO + VOC 2007 detection. Grad-CAM visualization. r=16 fixed in ablations.

## Scaffold ↔ final code correspondence
Pre-method scaffold: a feature-refinement module slot that takes F∈R^{C×H×W} and returns a refined map of same shape, insertable in a conv block; an attention sub-module stub. Final code fills: ChannelGate (dual-pool shared-MLP sigmoid channel gate), SpatialGate (channel-pool concat → 7×7 conv → sigmoid spatial gate), CBAM wiring them sequentially channel→spatial, each as multiply-by-attention.
