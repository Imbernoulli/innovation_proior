def encode(x,z_dim):
    """
    ConvNet encoder
    args:
        B-batch_size, C-channels
        H,W-spatial_dims
        x : shape : [B, C, H, W]
        C = 3 * num_frames; 3 - R/G/B
        z_dim: latent dimension
    """

    x = x / 255.
    
    # c: channels, f: filters
    # k: kernel, s: stride
    
    z = Conv2d(c=x.shape[1], f=32, k=3, s=2)])(x)
    z = ReLU(z)
    
    for _ in range(num_layers - 1):
        z = Conv2d((c=32, f=32, k=3, s=1))(z)
        z = ReLU(z)
    
    z = flatten(z)
    
    # in: input dim, out: output_dim, h: hiddens
    
    z = mlp(in=z.size(),out=z_dim,h=1024)
    z = LayerNorm(z)
    z = tanh(z)