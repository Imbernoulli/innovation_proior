# o = original unaugmented observation
# aug = augmentation
# contrastive = InfoNCE loss
o_anchor, o_target = aug(o), aug(o)
curl_loss = contrastive(o_anchor, o_target)
sac_loss = critic_loss(o) + actor_loss(o)
loss = curl_loss + sac_loss 
params = update(params, grad(loss, params))
