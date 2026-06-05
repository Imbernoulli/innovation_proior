# Session Review Handoff

Date recorded: 2026-06-06

This file records the local Claude Code and Codex session state for the
paper-to-reasoning batch. It is meant as a handoff note for continuing work on
another server.

## Sources inspected

- Claude Code history: `/Users/moonshot/.claude/history.jsonl`
- Main Claude Code session: `/Users/moonshot/.claude/projects/-Users-moonshot/f600e444-e46c-4e2c-a6e9-e555f4f530fc.jsonl`
- Claude Code subagent/task notifications under the same session
- Codex session index/history under `/Users/moonshot/.codex/`
- Repo-local review tools and artifacts under `tools/`
- Current repo state under `methods/`

The relevant Claude Code session is `f600e444-e46c-4e2c-a6e9-e555f4f530fc`.
It moved into `/Users/moonshot/paper2reasoning` and generated/reviewed the
paper-to-reasoning deliverables in batches.

## Important session findings

The Claude Code session records that Codex review was requested for generated
deliverables, including the `context.md`, `reasoning.md`, and `answer.md`
files. It also records that the Codex gate was interrupted by usage/rate limits.

Critical point: do not treat every `results/.codex_done` sentinel as proof of a
real Codex review. Near the end of the Claude Code session, the agent noted that
some gate agents fell back to self-verification under quota pressure and may
have stamped `.codex_done` without an independent Codex pass.

Explicitly suspicious sentinels:

- `methods/srgan/results/.codex_done`
- `methods/ssd/results/.codex_done`
- `methods/stochastic-depth/results/.codex_done`

These three should be re-run through the real Codex gate before being treated as
reviewed.

The same Claude Code session also ended with background generation agents still
pending. The final visible state included an API/session limit and a pending
background-agent count of 18 after one late generation wave completed. The repo
state below is therefore the practical source of truth for what exists locally.

## Current repo snapshot

Commands used to count the state:

```bash
find methods -mindepth 1 -maxdepth 1 -type d | wc -l
find methods -mindepth 1 -maxdepth 1 -type d -exec test -f '{}/results/context.md' ';' -print | wc -l
find methods -name .codex_done | wc -l
```

Current counts:

- Method directories total: 328
- Method directories with `results/context.md`: 290
- `results/.codex_done` sentinels: 48
- Result sets without a `.codex_done` sentinel: 242
- Method directories without `results/context.md`: 38

`tools/codex_gate_report.txt` shows the sequential sweep completing only
`adafactor` and `alibi` before stalling on `alphazero`. The other sentinels came
from parallel Claude Code gate agents or manual/self-verification paths. Preserve
that distinction when continuing.

## Continue Codex review

Existing helper scripts:

- `tools/codex_gate_sweep.sh`: older sequential sweep over 92 methods.
- `tools/codex_gate_chunk.sh`: one chunk worker, resumable by `.codex_done`.
- `tools/chunks/c1.txt` through `tools/chunks/c6.txt`: older chunk files.
- `tools/codex_todo.json`: broad historical todo snapshot.

Recommended next steps on the next server:

1. Recreate or verify the Codex companion path inside `tools/codex_gate_chunk.sh`
   and `tools/codex_gate_sweep.sh`.
2. Re-run real Codex gate for `srgan`, `ssd`, and `stochastic-depth`.
3. Generate a fresh no-sentinel list from the current repo before launching a
   full sweep. The existing chunk files are useful, but they should not be
   assumed to cover all 242 current no-sentinel result sets.

Useful commands:

```bash
# Result sets with no .codex_done sentinel.
comm -23 \
  <(find methods -mindepth 1 -maxdepth 1 -type d -exec test -f '{}/results/context.md' ';' -print | sed 's#methods/##' | sort) \
  <(find methods -mindepth 1 -maxdepth 1 -type d -exec test -f '{}/results/.codex_done' ';' -print | sed 's#methods/##' | sort)

# Method dirs not yet ready for Codex gate because results/context.md is missing.
find methods -mindepth 1 -maxdepth 1 -type d '!' -exec test -f '{}/results/context.md' ';' -print | sed 's#methods/##' | sort

# Optional strict reset of suspicious sentinels before re-gating.
for m in srgan ssd stochastic-depth; do
  rm -f "methods/$m/results/.codex_done"
done
```

## No-sentinel result sets

These 242 method result sets exist locally but currently have no
`results/.codex_done` sentinel:

```text
adapter
alexnet
ape-x
awac
awq
bahdanau-attention
bart
batchnorm
bert
beta-vae
bigbird
biggan
blip2
bpe
bsms-gnn
byol
c51
cbam
centernet
chain-of-thought
chebnet
classifier-free-guidance
clip
conformer
consistency-models
constitutional-ai
contriever
controlnet
convnext
cpc
cql
cutmix
cutout
cyclegan
dalle
darts
data2vec
dcgan
deberta
deep-compression
deep-sets
deepcluster
deeplabv3
deeponet
deepseek-r1
deepwalk
deformable-detr
deit
densenet
detr
diffpool
dino
distilbert
double-dqn
dpm-solver
dpo
dpr
dqn
dreamer
dreamerv3
dropout
dueling-dqn
edm
efficientnet
electra
enas
encodec
fast-rcnn
faster-rcnn
fcos
fitnets
flan
flash-attention-2
flashattention
fnet
fpn
gae
gail
glide
gpipe
gpt2
gpt3
gptq
gradient-checkpointing
graphormer
graphsage
graphsaint
grokking
group-norm
gumbel-softmax
her
hifi-gan
hippo
hubert
hyena
ibot
icm
impala
inception-v3
infogan
instance-norm
instructgpt
iql
iwae
jigsaw
kfac
knowledge-distillation
label-smoothing
lamb
lars
latent-diffusion
layernorm
least-to-most
lion
llm-int8
lora
lottery-ticket
lsgan
lsq
mae
maf
mamba
mask-rcnn
medusa
megatron-lm
mip-nerf
mistral
mixtral
mixup
mnasnet
mobilenet
moco
moe
movement-pruning
mpnn
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
outlier-rescaling
performer
pix2pix
pixelrnn
planet
pointnet
pointnet-plusplus
ppg
ppo
prefix-tuning
progressive-gan
prompt-tuning
qlora
r-cnn
radam
rag
rainbow
randaugment
react
realm
realnvp
rectified-flow
reformer
regnet
resnet
resnext
retnet
retro
ring-attention
rmsnorm
rnd
rope
rwkv
s4
sac
sam
scale-rae
score-sde
segformer
self-consistency
self-instruct
senet
sentence-bert
seq2seq
set-transformer
shampoo
shufflenet
simclr
simsiam
smoothquant
snip
soft-q-learning
sonicmoe
sophia
spacy
spatial-transformer
spectral-norm
squeezenet
streaming-llm
stylegan
stylegan2
swav
swin
switch-transformer
t5
tacotron2
td3
temperature-scaling
textual-inversion
toolformer
transformer
transformer-xl
tree-of-thoughts
trpo
ulmfit
unet
vae
vall-e
vdm
vgg
vicreg
vit
vllm
vqgan
vqvae
vqvae2
wav2vec2
wavenet
wgan
wgan-gp
whisper
wide-resnet
word2vec
world-models
xception
yolo
zero
```

## Methods missing result sets

These 38 method directories do not currently have `results/context.md`, so they
need generation or repair before Codex review:

```text
alphafold2
coca
curl
deep-ensembles
depth-anything
diffuser
dinov2
dit
dreambooth
dreamfusion
drq-v2
efficientdet
egnn
esm2
flamingo
gaussian-splatting
grpo
hrnet
imagebind
imagen
instant-ngp
lcm
llava
mappo
mask2former
maskgit
mc-dropout
mlp-mixer
mobilenetv2
mobilenetv3
pbt
qmix
schnet
sdxl
segment-anything
siglip
td3-bc
var
```
