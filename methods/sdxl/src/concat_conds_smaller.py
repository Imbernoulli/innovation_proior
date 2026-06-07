from einops import rearrange
import torch

batch_size=16
# channel dimension of pooled output of text encoder(s)
pooled_dim = 512

def fourier_embedding(inputs, outdim=256, max_period=10000):
    """
    Classical sinusoidal timestep embedding
    as commonly used in diffusion models
    :param inputs: batch of integer scalars shape [b,]
    :param outdim: embedding dimension
    :param max_period: max freq added
    :return: batch of embeddings of shape [b, outdim]
    """
    ...

def cat_along_channel_dim(
        x:torch.Tensor,) -> torch.Tensor:
    if x.ndim == 1:
        x = x[...,None]
    assert x.ndim == 2
    b, d_in = x.shape
    x = rearrange(x, "b din -> (b din)")
    # fourier fn adds additional dimension
    emb = fourier_embedding(x)
    d_f = emb.shape[-1]
    emb = rearrange(emb, "(b din) df -> b (din df)",
                        b=b, din=d_in, df=d_f)
    return emb

def concat_embeddings(
        # batch of size and crop conditioning cf. Sec. 3.2
        c_size:torch.Tensor,
        c_crop:torch.Tensor,
        # batch of aspect ratio conditioning cf. Sec. 3.3
        c_ar:torch.Tensor,
        # final output of text encoders after pooling cf. Sec. 3.1
        c_pooled_txt:torch.Tensor, ) -> torch.Tensor:
    # fourier feature for size conditioning
    c_size_emb = cat_along_channel_dim(c_size)
    # fourier feature for size conditioning
    c_crop_emb = cat_along_channel_dim(c_crop)
    # fourier feature for size conditioning
    c_ar_emb = cat_along_channel_dim(c_ar)
    # the concatenated output is mapped to the same
    # channel dimension than the noise level conditioning
    # and added to that conditioning before being fed to the unet
    return torch.cat([c_pooled_txt,
                      c_size_emb,
                      c_crop_emb,
                      c_ar_emb], dim=1)

# simulating c_size and c_crop as in Sec. 3.2
c_size=torch.zeros((batch_size, 2)).long()
c_crop=torch.zeros((batch_size, 2)).long()
# simulating c_ar and pooled text encoder output as in Sec. 3.3
c_ar=torch.zeros((batch_size, 2)).long()
c_pooled=torch.zeros((batch_size, pooled_dim)).long()

# get concatenated embedding
c_concat = concat_embeddings(c_size, c_crop, c_ar, c_pooled)

