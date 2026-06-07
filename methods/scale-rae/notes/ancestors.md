# Load-bearing ancestor notes (verified against primary sources)

- **RAE parent — Zheng, Ma, Tong, Xie 2025, "Diffusion Transformers with Representation
  Autoencoders" (arXiv 2510.11690, saved in refs/).** Direct parent. Freeze a pretrained
  representation encoder, train a ViT decoder (L1+LPIPS+adversarial), diffuse with a DiT on the frozen
  high-dim tokens. Three enabling choices: width >= token dim (Thm 1: width d<n gives loss floor
  sum_{i=d+1}^n lambda_i, tail eigenvalues of cov(eps-x); noise inflates manifold to full rank);
  dimension-dependent noise shift (alpha=sqrt(m/n), m=N*d, base n=4096; t_m=alpha t_n/(1+(alpha-1)t_n);
  removing it: ImageNet gFID ~5 -> ~23); wide DDT head (backbone M produces conditioning z_t, wide
  head H produces velocity, to buy width cheaply when backbone < latent dim); noise-augmented decoding
  (decoder trained on z+n, n~N(0,sigma^2 I), sigma~|N(0,tau^2)|). Gap: only validated on ImageNet.

- **SD3 — Esser et al. 2024 (arXiv 2403.03206, saved in refs/).** Rectified flow x_t=(1-t)x+t eps,
  velocity v=eps-x, MSE loss; logit-normal timestep sampling; resolution-dependent timestep shift from
  the SNR argument sigma(t,n)=(t/(1-t)) sqrt(1/n). Gap: shift derived for spatial resolution / low
  channels, not high channel dim.

- **DiT — Peebles & Xie 2023.** Transformer denoiser over latent patch tokens, adaLN-zero conditioning
  (predict per-block shift/scale/gate, gates zero-init). Gap: shown on low-dim VAE latents only.

- **LightningDiT — Yao et al. 2025 (VA-VAE).** Modernized DiT block: RMSNorm, SwiGLU, RoPE, optional
  QK-norm, Gaussian Fourier t-embed; large convergence speedups. Adopted as the DiT backbone.

- **MetaQuery — Pan et al. 2025 (arXiv 2504.06256).** Learnable query tokens as interface between a
  (frozen) MLLM and a diffusion model; query hidden states + MLP connector = diffusion conditioning.
  Reported limited LLM-scaling benefit (frozen LLM). Gap: used with VAE-latent diffusion.

- **SigLIP-2 — Tschannen et al. 2025 (arXiv 2502.14786).** Sigmoid VL loss + self-supervised + decoder
  losses -> strong dense patch features. So400M p14 @224 -> 256 tokens, d=1152. The frozen encoder.

- **Latent diffusion / LDM — Rombach et al. 2022; VAE — Kingma & Welling 2014.** Diffuse in a
  compressed VAE latent. Gap: low-dim, non-semantic -> two-tower unified designs + reconstruction
  ceiling.

Official code snapshot: code/Scale-RAE (ZitengWangNYU/Scale-RAE). Key files grounding the deliverables:
diffusion/rf.py (RectifiedFlow + shift), models/lightningDiT.py (DiT block/forward),
scale_rae_arch.py (latent_queries, projectors, diffusion loss), multimodal_decoder/decoder.py (ViT
decoder).
