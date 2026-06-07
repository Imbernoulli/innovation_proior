# f_q, f_k: encoder networks for anchor
# (query) and target (keys) respectively.
# loader: minibatch sampler from ReplayBuffer
# B-batch_size, C-channels, H,W-spatial_dims
# x : shape : [B, C, H, W]
# C = c * num_frames; c=3 (R/G/B) or 1 (gray)
# m: momentum, e.g. 0.95
# z_dim: latent dimension
f_k.params = f_q.params
W = rand(z_dim, z_dim) # bilinear product.
for x in loader: # load minibatch from buffer
 x_q = aug(x) # random augmentation
 x_k = aug(x) # different random augmentation
 z_q = f_q.forward(x_q)
 z_k = f_k.forward(x_k)
 z_k = z_k.detach() # stop gradient
 proj_k = matmul(W, z_k.T) # bilinear product
 logits = matmul(z_q, proj_k) # B x B
  # subtract max from logits for stability
 logits = logits - max(logits, axis=1) 
 labels = arange(logits.shape[0])
 loss = CrossEntropyLoss(logits, labels) 
 loss.backward()
 update(f_q.params) # Adam
 update(W) # Adam
 f_k.params = m*f_k.params+(1-m)*f_q.params