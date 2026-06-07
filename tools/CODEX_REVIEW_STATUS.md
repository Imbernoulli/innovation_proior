# Codex Independent-Review Status (forensic reconstruction)

Generated: 2026-06-06. Forensic reconstruction from local session logs only (Claude Code transcripts under `~/.claude/projects/**`, the Codex CLI rollouts under `~/.codex/sessions/**`, and repo sentinels). No reliance on memory.

## Headline counts

- **Total method directories:** 365
- **Reviewed (codex_reviewed=true):** 106
- **Not reviewed:** 250
- **Uncertain:** 9 (Codex applied edits but the rollout was cut off before a confirmed task_complete/changelog)

Outcome breakdown: `completed`=106, `errored`=53, `limited`=16, `none`=181, `uncertain`=9

## Methodology & confidence

Two independent evidence streams were parsed and merged (most-favorable real evidence wins):

1. **Claude-side direct invocations** — every `node codex-companion.mjs task "Review AND FIX …"` Bash tool_use across all sessions/subagents (280 invocations, 119 distinct methods). Each was paired with its tool_result; background runs were followed through their `=== <bgid> ===` streamed notifications. Completion = `[codex] File changes completed` / `Turn completed` / a `file:line` changelog in that method's own segment.
2. **Codex-side rollouts** — 308 `~/.codex/sessions/**` rollouts touch paper2reasoning methods (the `codex:rescue` task-queue and CLI path, which the Claude-side parser cannot see). Completion = a `task_complete` event carrying a substantive `last_agent_message` changelog, corroborated by `patch_apply_end` edit events. For the few multi-method orchestration/summary rollouts, completion was attributed per-method only when that method's path appears in the changelog.

Confidence is **high** for `completed` (corroborated by applied patches + changelog) and for the `limited` timeline (verbatim 'You've hit your usage limit' strings with reset times). Confidence is **medium** for the `uncertain` bucket: Codex genuinely applied file edits there, but the session ended before a confirmed changelog, so an independent review cannot be asserted. Defaulted to NOT reviewed wherever there was no positive completion.

## Codex usage-limit timeline (chronological)

When Codex was unavailable and therefore could NOT be used as the independent reviewer. Timestamps are UTC; reset labels are the verbatim 'try again at' strings seen in the logs.

| UTC window | events | example methods being gated | what happened |
|---|---|---|---|
| 2026-05-29T11:xx | 17 | byol, cyclegan, dpo, faster-rcnn, flashattention, gat | first wall — 'You've hit your usage limit … try again at 3:28 PM' (Codex 5-hour quota) |
| 2026-05-29T15:xx | 1 | rope | quota still exhausted on early retries |
| 2026-05-29T16:xx | 28 | detr, efficientnet, flow-matching, fno, gae, glow | second wall — bulk subagent gate fan-out all bounced on 'usage limit … try again at 10:13 PM / 8:33 PM' |
| 2026-05-30T02:xx | 21 | alphazero, biggan, cpc, dino, edm, fno | third wall — overnight parallel gate wave hit 'usage limit … try again at 4:27 AM' |
| 2026-05-30T07:xx | 1 | barlow-twins | barlow-twins bounced with a raw 429 / stream error during kill-and-retry sweep |
| 2026-05-30T08:xx | 6 | bart, fpn | fourth wall — bart/fpn retry loop repeatedly hit 'usage limit … try again at Jun 1st 1:52 PM' |
| 2026-06-05T17:xx | 1 | deeplabv3 | late codex-rollout retry (deeplabv3) hit usage limit |

Total distinct limit-hit invocation events: **75**. The Codex gate stalled repeatedly across these windows; the bulk of the not-reviewed backlog is a direct consequence.

## Suspicious sentinels

`.codex_done` files present in the repo that are NOT backed by any confirmed Codex completion — i.e. stamped by self-verification fallback, not an independent Codex pass. These need re-gating:

- **adam** — outcome `uncertain`; UNCERTAIN: Codex rollout applied file edits (patch_apply_end) but was interrupted before a confirmed task_complete/changelog; review not confirmed
- **elmo** — outcome `uncertain`; UNCERTAIN: Codex rollout applied file edits (patch_apply_end) but was interrupted before a confirmed task_complete/changelog; review not confirmed
- **linear-attention** — outcome `uncertain`; UNCERTAIN: Codex rollout applied file edits (patch_apply_end) but was interrupted before a confirmed task_complete/changelog; review not confirmed
- **srgan** — outcome `errored`; Codex task started but never produced a changelog/edits (indeterminate, no completion captured)
- **ssd** — outcome `errored`; Codex task started but never produced a changelog/edits (indeterminate, no completion captured)
- **stochastic-depth** — outcome `errored`; Codex task started but never produced a changelog/edits (indeterminate, no completion captured)
- **swa** — outcome `uncertain`; UNCERTAIN: Codex rollout applied file edits (patch_apply_end) but was interrupted before a confirmed task_complete/changelog; review not confirmed

The three explicitly called out in SESSION_REVIEW_NOTES — **srgan**, **ssd**, **stochastic-depth** — are all confirmed here: each has a Codex rollout that reached an empty `task_complete` with ZERO `patch_apply_end` edits (Codex did no work), so their sentinels are false. Note that several other previously-doubted sentinels (e.g. pixelcnn, qr-dqn, speculative-decoding, lookahead, scaling-laws, ffjord, iaf, ddpm, gan, gcn, weight-norm, chinchilla, llama) WERE in fact genuinely reviewed via the Codex rescue/CLI path and are correctly marked reviewed.

## Methods still needing re-gating (NOT reviewed)

250 methods with no confirmed independent Codex review:

```text
adapter
alexnet
alphafold2
ape-x
autocorrelation-inequalities
awac
awq
bart
batchnorm
beta-vae
blip2
bpe
cap-set
cbam
centernet
chain-of-thought
christofides-tsp
chromatic-number-plane
circle-packing-in-square
coca
conformer
constitutional-ai
constructive-ramsey
contriever
controlnet
convnext
cooley-tukey-fft
count-min-sketch
curl
cutmix
cutout
dalle
darts
data2vec
dcgan
deberta
deep-compression
deep-ensembles
deepcluster
deeplabv3
deeponet
deepseek-r1
deepwalk
deformable-detr
deit
depth-anything
diffpool
diffuser
dino
dinov2
distilbert
dit
double-dqn
dpm-solver
dpr
dreambooth
dreamer
dreamerv3
dreamfusion
drq-v2
efficientdet
egnn
enas
encodec
erdos-minimum-overlap
esm2
fast-matrix-multiplication
fast-rcnn
fcos
finite-field-kakeya
fitnets
flamingo
flan
flash-attention-2
fnet
gae
gaussian-splatting
gpipe
gpt2
gptq
gradient-checkpointing
graphsaint
grokking
grpo
heilbronn-triangle
held-karp-bound
hifi-gan
hrnet
hubert
hyperloglog
ibot
icm
imagebind
imagen
impala
inception-v3
infogan
instance-norm
instant-ngp
iql
iterative-rounding-sndp
iwae
jigsaw
johnson-lindenstrauss
karatsuba-multiplication
karger-min-cut
kfac
kissing-number
knowledge-distillation
label-smoothing
lamb
lcm
least-to-most
lion
llava
llm-int8
lottery-ticket
low-autocorrelation-sequences
lsgan
lsq
maf
mappo
mask-rcnn
mask2former
maskgit
maxcut-sdp
mc-dropout
medusa
megatron-lm
miller-rabin-primality
minhash-lsh
mip-nerf
mistral
mixtral
mlp-mixer
mnasnet
mobilenet
mobilenetv2
mobilenetv3
moe
movement-pruning
mpnn
multiplicative-weights
muon
muzero
nasnet
nerf
neural-ode
nice
node2vec
noisy-nets
non-local-nn
ntk
pbt
performer
pix2pix
pixelrnn
planet
pointnet
pointnet-plusplus
ppg
ppo
prefix-tuning
prompt-tuning
qlora
qmix
r-cnn
radam
rag
rainbow
randaugment
react
realm
rectified-flow
reformer
regnet
reservoir-sampling
resnet
resnext
retnet
retro
ring-attention
rnd
rwkv
s4
sam
schnet
schonhage-strassen
sdxl
segformer
segment-anything
self-consistency
self-instruct
senet
sentence-bert
set-transformer
shampoo
shufflenet
sidon-sets
siglip
simclr
simsiam
smoothquant
snip
soft-q-learning
sophia
spatial-transformer
spectral-norm
sphere-packing-lattices
squeezenet
srgan
ssd
stochastic-depth
streaming-llm
stylegan2
submodular-greedy
sum-free-sets
sums-and-differences-sets
swav
swin
switch-transformer
tacotron2
tammes-problem
td3-bc
temperature-scaling
textual-inversion
thomson-problem
toolformer
transformer
transformer-xl
tree-of-thoughts
ulmfit
vae
vall-e
var
vdm
vicreg
vllm
vqgan
vqvae
vqvae2
wav2vec2
wavenet
whisper
wide-resnet
winograd-convolution
world-models
xception
yolo
zero
```

### Uncertain (Codex edited but completion unconfirmed — re-gate to confirm)

```text
adam
edm
efficientnet
electra
elmo
fpn
linear-attention
progressive-gan
swa
```

## Gaps / caveats

- Background Codex task `.output` files for the original session (`~/.codex/.tmp` and `/tmp/claude-*/.../tasks/*.output`) have been garbage-collected, so foreground/background completions were recovered from streamed transcript notifications and the Codex rollouts rather than those files.
- A handful of `codex:rescue` queue tasks (e.g. for adam, gcn, srgan, stochastic-depth) were observed in `queued`/`running`/`editing` state with no terminal `completed` in the captured logs; their Codex rollouts confirm no changelog was produced, so they are treated as not reviewed.
- 'When' for a reviewed method is the timestamp of its completing event (UTC).
- Per-method durable markers `.codex_review.json` were written for every `completed` method.