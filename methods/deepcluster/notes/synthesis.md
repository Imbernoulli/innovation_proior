# DeepCluster synthesis (grounding notes)

arXiv 1807.05520 — "Deep Clustering for Unsupervised Learning of Visual Features", Caron, Bojanowski, Joulin, Douze (Facebook AI Research), ECCV 2018. Verified by title search.
Canonical impl: facebookresearch/deepcluster (clustering.py, main.py). Grounds preprocessing, k-means, uniform sampler, top-layer reinit.

## Pain point / research question
Want general-purpose visual features WITHOUT labels, trainable at internet scale. ImageNet supervision is saturating (Stock & Cisse 2017: little error left), and ImageNet is small (1M images) and curated for object classification. Bigger datasets (billions) would need infeasible manual annotation; raw metadata (hashtags) introduces biases. So need UNSUPERVISED end-to-end convnet training that scales.

## Background / tools and where each falls short
- Classical unsupervised: clustering, dim reduction, density estimation (used in vision: bag-of-features clusters handcrafted descriptors). Work on any domain, but designed for LINEAR models on FIXED features; barely work when features must be learned simultaneously.
- THE central failure: training a convnet WITH k-means directly -> trivial solution: features zeroed, all clusters collapse to one entity. (Joint learning of classifier + labels is degenerate.)
- Self-supervised (Doersch 2015 patch position, Noroozi 2016 jigsaw, Pathak 2016 inpainting, colorization, video coherence): replace labels with pseudo-labels from a hand-designed pretext task. Gap: DOMAIN-DEPENDENT, need expert knowledge to design a pretext that yields transferable features.
- Generative (VAE Kingma 2013, GAN Goodfellow 2014, BiGAN/ALI Donahue 2016/Dumoulin 2016): GAN discriminator features disappointing; adding an encoder helps. Gap: indirect, features not the primary objective.
- Prior clustering+convnet: Coates&Ng 2012 (k-means pretrain but layer-by-layer bottom-up, not end-to-end); Yang 2016 (recurrent joint feature+cluster, small datasets, hard to scale); Bojanowski&Joulin 2017 (info-preserving loss, discriminates between images like exemplar SVM). None tested at scale on modern convnets.

## KEY OBSERVATION (motivating, pre-method, knowable)
A RANDOM convnet (theta ~ Gaussian, no training) already gives features far above chance: an MLP on the last conv layer of a random AlexNet hits 12% on ImageNet (chance 0.1%). This weak signal comes from the convolutional structure itself (strong prior on the input). IDEA: bootstrap the discriminative power from this weak signal.

## THE METHOD
Supervised baseline objective: theta = convnet f_theta, g_W classifier on top.
  min_{theta,W} (1/N) sum_n ell( g_W(f_theta(x_n)), y_n )    (eq sup), ell = multinomial logistic (neg log-softmax).
DeepCluster: ALTERNATE
  (A) Cluster the features f_theta(x_n) with k-means -> pseudo-labels y_n.
  (B) Update theta (and classifier) by predicting these pseudo-labels via eq (sup), minibatch SGD + backprop.
k-means jointly learns d x k centroid matrix C and assignments y_n:
  min_{C in R^{dxk}} (1/N) sum_n min_{y_n in {0,1}^k} || f_theta(x_n) - C y_n ||_2^2   s.t.  y_n^T 1_k = 1.   (eq kmeans)
Use only the optimal assignments y_n* as pseudo-labels; DISCARD the centroid matrix C.
Focus on k-means (PIC also works; choice not crucial).

## Avoiding trivial solutions (two distinct fixes)
General problem: any method jointly learning a discriminative classifier AND the labels has trivial solutions; discriminative clustering suffers even for linear models (Xu 2005). Classical fixes constrain/penalize min points per cluster, but those terms are over the WHOLE dataset -> not scalable.
1. EMPTY CLUSTERS. A discriminative model's optimal boundary can put ALL inputs in one cluster (Xu 2005); no mechanism prevents empty clusters. FIX (from feature quantization, Johnson 2017 faiss): during k-means, when a cluster becomes empty, randomly pick a non-empty cluster, use its centroid + small random perturbation as the new centroid for the empty one, and split the non-empty cluster's points between the two.
2. TRIVIAL PARAMETRIZATION. If most images go to a few clusters, theta only discriminates those; worst case all-but-one are singletons -> convnet predicts same output regardless of input. (Same as supervised with highly unbalanced classes; metadata is Zipf.) FIX: sample images by a UNIFORM distribution over the (pseudo-)classes = weight each input's loss contribution by the INVERSE of its assigned cluster's size.

## Implementation details
- Architecture: AlexNet (5 conv: 96,256,384,384,256 filters; 3 FC). Remove LRN, add batch norm. Also VGG-16 + BN.
- SOBEL: apply fixed linear transform (Sobel filters) to remove color & increase local contrast (unsupervised methods overfit to color; Sobel removes it). From Bojanowski 2017, Paulin 2015.
- Training data: ImageNet 1.3M / 1000 classes (default); also YFCC100M Flickr (uncured).
- Optimization: cluster CENTRAL-CROPPED features; data augmentation (random flips + crops of random sizes/aspect ratios) during network training -> invariance to augmentation. Dropout, constant step size, L2 weight decay, momentum 0.9, minibatch 256.
- Clustering features: PCA-reduce to 256 dims, whiten, L2-normalize. faiss k-means (Johnson 2017).
- Re-cluster EVERY EPOCH (nearly optimal on ImageNet). k-means takes ~1/3 of time (need a forward pass over full dataset).
- Number of clusters k = 10,000 (over-segmentation beneficial even though ImageNet has 1000 classes).
- Train 500 epochs (~12 days on P100, AlexNet).
- Hyperparam selection: downstream Pascal VOC classification (no finetuning), Krahenbuhl's code.
- PER-EPOCH TOP-LAYER REINIT (from code): cluster identities are arbitrary/permuted each epoch (a cluster index has no consistent meaning across re-clusterings), so the final classification (top) layer is REINITIALIZED (random Normal(0,0.01), bias 0) each epoch; the backbone features persist. This is essential and follows from cluster labels being meaningless between rounds.

## Evaluation metrics (diagnostic tools, pre-method)
NMI (normalized mutual information): NMI(A;B) = I(A;B)/sqrt(H(A)H(B)); 0 if independent, 1 if deterministically predictable. Used to measure (a) cluster<->ImageNet-label dependence over training and (b) stability between epoch t-1 and t clusters.

## Design-decision -> why
- Alternate cluster (pseudo-labels) then classify: turns unsupervised learning into the well-understood supervised pipeline, reusing all its tricks (batchnorm, SGD, augmentation). Bootstraps from the weak random-convnet signal.
- k-means specifically: standard, scalable (faiss), choice not crucial.
- Discard centroids, keep only assignments: the goal is features, not the clustering; the classifier relearns the boundary from pseudo-labels.
- Empty-cluster reassignment: prevents the trivial all-in-one-cluster k-means solution at scale (cheap, local, not a global penalty).
- Uniform sampling over clusters (inverse-size weighting): prevents the convnet from collapsing onto a few dominant clusters (trivial parametrization).
- Sobel (drop color): color is a too-easy shortcut for unsupervised features; removing it forces shape/texture learning.
- PCA+whiten+L2 before k-means: decorrelate and equalize feature scales so Euclidean k-means is meaningful; reduce dim for speed.
- Re-cluster every epoch + reinit top layer: features drift, so re-derive pseudo-labels; cluster IDs permute, so the classifier head must be reborn.
- k=10000 > 1000: over-segmentation gives finer pseudo-classes -> richer features.

NO unsourced facts. k=10000, 12% random, 500 epochs, PCA256, momentum 0.9, batch 256 all from text. Top-layer reinit and uniform sampler from official code.
