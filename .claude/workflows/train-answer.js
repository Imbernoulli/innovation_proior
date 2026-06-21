export const meta = {
  name: 'train-answer',
  description: 'Backfill results/train_answer.md (scientist discovery write-up) for methods that have context/reasoning/answer but no fourth file',
  phases: [{ title: 'Write', detail: 'one agent per method: read 3 source files, write train_answer.md' }],
}

// Method slugs to process (relative to methods/<slug>/results/).
// args overrides SLUGS when provided as an array.
const SLUGS = [
  "link-cut-tree",   "lin-kernighan",   "lin-kernighan-tsp",   "lion",   "llama",   "lll-reduction",
  "llm-int8",   "local-minimax-optimization",   "lof",   "loglinear",   "longformer",   "lookahead",
  "looped-transformer",   "lorenz-chaos",   "low-autocorrelation-sequences",   "lowenheim-skolem",   "lqr",   "lsgan",
  "lshade",   "luby-rackoff",   "luffy",   "mappo-critic",   "markov-game-nash-selfplay",   "markowitz",
  "markowitz-portfolio",   "mart",   "mask-rcnn",   "matchingnet",   "mat-critic",   "matrix-concentration-tropp",
  "maxcut-sdp",   "mha",   "mice",   "mifgsm",   "miller-rabin-primality",   "minhash-lsh",
  "minimax-lower-bounds-lecam",   "mip-neq-nexp",   "mip-nerf",   "mirror-descent",   "mirror-symmetry",   "mish",
  "mistral",   "mixtral",   "mixup",   "mla",   "mla-attention",   "moe",
  "moea-d",   "moead",   "monsky-theorem",   "montgomery-multiplication",   "morse-theory",   "mos-algorithm",
  "movement-pruning",   "mpc",   "mpnn",   "mppi",   "mqa",   "multiplicative-weights",
  "muon",   "muon-tomography",   "muzero",   "myerson-auction",   "nash-embedding",   "nash-equilibrium-existence",
  "nasnet",   "natural-policy-gradient",   "nbeats",   "needleman-wunsch",   "nerf",   "nesterov-acceleration",
  "neural-ode",   "neural-tangent-kernel",   "neyman-pearson-lemma",   "ngu",   "nice",   "nll-entropy",
  "no-cloning-theorem",   "node2vec",   "noether-normalization",   "noether-theorem",   "no-free-lunch",   "noi-mincut-hu",
  "noi-palindromic-tree-weng",   "noi-sbt-chen",   "noise-decay",   "noisy-nets",   "none",   "non-local-nn",
  "normalized-loss-functions",   "notears",   "notears-mlp",   "nsga2",   "nsga3",   "optuna-cma",
  "orpo",   "orthogonal-reg",   "outlier-rescaling",   "pac-bayes",   "pac-learning-valiant",   "pac-mdp-rmax",
  "pagerank-spectral",   "parsimony-gp",   "physicscup-2022-photon-rocket",   "physicscup-2022-projective-optics",   "physicscup-2023-mhd-flux",   "physicscup-2023-spaceship",
  "physicscup-2024-satellites",   "pid-lag",   "pifold",   "pix2pix",   "pixelcnn",   "pixelrnn",
  "pixle",   "planck-quantization",   "planet",   "platt-scaling",   "pna-readout",   "poincare-duality",
  "pointnet",   "pointnet-plusplus",   "policy-gradient-theorem",   "poly-loss",   "polynomial-method-combinatorics",   "pontryagin-duality",
  "potential-based-reward-shaping",   "powered-descent-convexification",   "powersgd",   "ppg",   "ppo",   "ppo-lag",
  "ppo-penalty",   "pre-activation",   "prefix-tuning",   "primal-dual-steiner-forest",   "prime-number-theorem",   "prioritized-replay",
  "probabilistic-method",   "probabilistic-method-erdos",   "progressive-gan",   "prompt-tuning",   "prores",   "proteinmpnn",
  "protonet",   "proton-therapy-impt",   "proximal-gradient-ista",   "pseudorandomness-ggm",   "putnam-1992-a6-sphere",   "putnam-2016-a4",
  "putnam-2018-b4",   "qaoa",   "qk-norm",   "q-learning-ucb-jin",   "q-learning-watkins",   "qlora",
  "qr-dqn",   "qsgd",   "quadratic",   "quadruped-gait",   "quantum-no-cloning",   "quicksort",
  "react",   "realm",   "realnvp",   "rebrac",   "rectified-flow",   "recursion-theorem",
  "reddiff",   "reformer",   "regnet",   "reinforce-plus-plus-baseline",   "reinforce-score-function",   "reingold-undirected-connectivity",
  "relaxloss",   "relu-sq-torch",   "relu-squared",   "renormalization-group",   "reservoir-sampling",   "resnext",
  "restricted-isometry-property",   "retnet",   "retro",   "reward-free-exploration",   "reynolds-boids",   "rhg",
  "rice-theorem",   "ridge",   "riemann-mapping-theorem",   "riesz-representation",   "robot-arm-time-optimal",   "rope",
  "roth-theorem",   "round-to-nearest",   "rpo",   "rsa-spectrum-assignment",   "rsa-trapdoor",   "rs-kd",
  "rubin-causal-model",   "rvea",   "rwkv",   "s4",   "sabre-routing",   "sac",
  "sac-rnd",   "sagpool",   "salun",   "sam",   "sard-theorem",   "savitch-theorem",
  "scaffold",   "scale-opa",   "scale-rae",   "scaling-laws",   "scarf-ss-policy",   "schnet",
  "schonhage-strassen",   "score",   "score-sde",   "scrub",   "seag",   "seal",
  "se-block",   "second-moment-method",   "seg",   "segformer",   "segment-tree-beats",   "selinger-join-order",
  "semantic-security-gm",   "senet",   "sentence-bert",   "seq2seq",   "sequence-level",   "set-transformer",
  "set-transformer-aggregation",   "sgd",   "sgd-robbins-monro",   "sha256-merkle-damgard",   "shallow-mf",   "shampoo",
  "shannon-channel-coding",   "shannon-information-theory",   "shapley-value",   "shifting-bottleneck",   "shor-algorithm",   "shufflenet",
  "si",   "signsgd",   "sigreg",   "silu",   "simba",   "simclr",
  "simnorm",   "simp",   "simplex-method",   "simpo",   "simsiam",   "simulated-annealing",
  "sinkhorn-ot",   "skew-reverse-kl",   "ski-rental",   "sldagent-style",   "s-learner",   "smoothed-analysis-simplex",
  "smoothquant",   "sms-emoa",   "snip",   "soap",   "softcap-ce",   "soft-q-learning",
  "softs",   "solomonoff-induction",   "sonicmoe",   "spacy",   "sparsefool",   "sparse-pgd",
  "sparse-rs",   "spatial-transformer",   "spea2",   "special-relativity",   "spectral-clustering-cheeger",   "spectral-norm",
  "spectral-signature",   "spectral-theorem-hilbert",   "spectre",   "spme-battery",   "spontaneous-symmetry-breaking",   "spot",
  "spsa",   "square",   "squat-subspace-4bit",   "squeezenet",   "ssd",   "ssm",
  "stack-rnn",   "standard",   "standard-dpsgd",   "standard-gp",   "stemgnn",   "stid",
  "stochastic-depth",   "stone-weierstrass",   "storm",   "storm-plus",   "streaming-llm",   "streamingllm",
  "stylegan",   "stylegan2",   "subgradient-method",   "submodular-greedy",   "t5",   "tabu-search",
  "tacotron2",   "taid",   "taming",   "td-lambda-eligibility-traces",   "temperature-scaling",   "temporal-difference-learning",
  "ternary-158bit",   "textual-inversion",   "tpe-hyperopt",   "tra",   "trades",   "transformer-alibi",
  "transformer-nope",   "trimul",   "triton-gelu",   "trivialaugment",   "trpo",   "tsne",
  "turing-halting",   "tutte-matrix-matching",   "two-sat-scc",   "two-stage",   "ucb1",   "ulmfit",
  "umap",   "uncertainty",   "unet",   "unimol",   "unipc",   "unit-commitment",
  "vae",   "value-embed",   "vanilla",   "varibad",   "vasarhelyi-flocking",   "vc-dimension",
  "vdm",   "viterbi-algorithm",   "vqvae",   "vqvae2",   "warmup-cosine",   "wav2vec2",
  "wavelet-tree",   "wavenet",   "weather-routing",   "weight-norm",   "weil-conjectures",   "wgan",
  "wgan-gp",   "whisper",   "wide-resnet",   "wiles-fermat",   "winograd-convolution",   "word2vec",
  "world-models",   "wsd",   "wsd-sqrt",   "xception",   "xgboost",   "xgboost-style",
  "xlnet",   "yao-minimax-principle",   "yolo",   "yoneda-lemma",   "yufeizhao-dimension-method",   "yufei-zhao-lte",
  "zagier-two-squares",   "z-algorithm",   "zero",   "zeroinit",   "zero-knowledge-gmr",   "ziegler-nichols-pid",
  "zigzag",   "zig-zag-product",   "z-loss", 
]
const slugs = Array.isArray(args) && args.length ? args : SLUGS
if (!slugs.length) throw new Error('no slugs to process')

log(`Generating train_answer.md for ${slugs.length} method(s)`)

const SKILL = '.claude/skills/discovery-writeup/SKILL.md'

const prompt = (slug) => `You are writing the fourth deliverable for a single method in a research-methods corpus.

Follow the skill at ${SKILL} EXACTLY. Read it first.

Method: methods/${slug}/results/

Steps:
1. Read ALL THREE existing files in full: methods/${slug}/results/context.md, methods/${slug}/results/reasoning.md, methods/${slug}/results/answer.md. Do not skim.
2. Write methods/${slug}/results/train_answer.md as the scientist's own final write-up of their discovery, in continuous prose with three movements that flow without headers:
   (a) summarize the analysis — the problem and why the existing options fall short;
   (b) propose the method, NAME it, and explain its concrete mechanism in real detail — the defining equations/update rule, each component's purpose, the key design choices and why each beats the obvious alternative, and the load-bearing derivation steps. The reader must understand HOW it works and WHY it is built that way, not just its name;
   (c) end with the code.
3. HARD RULE on code: copy the method's final implementation VERBATIM from answer.md. Do not rewrite, rename, re-implement, or "clean up" anything. The code block(s) in train_answer.md must be byte-for-byte identical to the primary implementation in answer.md, so there is zero divergence from the other deliverables and zero chance of a new bug. If answer.md has a main method plus a genuine sibling variant, carry the primary one (and the variant too, verbatim, only if it is truly part of the method). After writing, diff your code against answer.md and confirm it matches.
4. Ground everything ONLY in the three existing files — invent no facts, numbers, or method pieces; do no web research. The files are already verified; you are reformatting/distilling, not researching. The existing files set the format; write a genuinely better, more faithful and more complete account than a thin example would.
5. Style: continuous prose, NO Markdown headers/sections/bullet-lists-describing-the-method. Write math in LaTeX: inline $...$ (e.g. $\\beta_2 = 0.999$) and display $$...$$ for the load-bearing equations (e.g. $$\\theta_t = \\theta_{t-1} - \\alpha\\,\\hat m_t / (\\sqrt{\\hat v_t} + \\epsilon)$$), using proper symbols (\\alpha, \\beta, \\hat, \\sqrt, subscripts). The prose stays header-free; LaTeX math is the one place Markdown is expected. English; in-frame (name the method, but no citation line / no "this paper" / no authors-venue-arXiv); no meta-commentary about the document or its purpose.

Verify before finishing: code identical to answer.md; no headers; no "this paper"/citation; method details actually explained; file written to methods/${slug}/results/train_answer.md.

Return a one-line confirmation: the slug, the word count of the prose, and "code matches answer.md" or the discrepancy.`

// Throttle: process CHUNK methods at a time (well below the default ~16-wide
// fan-out) to avoid tripping the server-side request limiter.
const CHUNK = 4
// Per-method automatic retry: a null result means the agent was rate-limited or
// died after its own internal backoff; re-run it up to MAX_ATTEMPTS times.
const MAX_ATTEMPTS = 6

async function writeWithRetry(slug) {
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    const label = `train_answer:${slug}` + (attempt > 1 ? ` (retry ${attempt - 1})` : '')
    const r = await agent(prompt(slug), { label, phase: 'Write' })
    if (r) return { slug, r, attempt }
    log(`${slug}: attempt ${attempt}/${MAX_ATTEMPTS} failed` + (attempt < MAX_ATTEMPTS ? ', retrying' : ' — giving up'))
  }
  return { slug, r: null, attempt: MAX_ATTEMPTS }
}

const done = []
for (let i = 0; i < slugs.length; i += CHUNK) {
  const chunk = slugs.slice(i, i + CHUNK)
  const res = (await parallel(chunk.map(s => () => writeWithRetry(s)))).filter(Boolean)
  done.push(...res)
  const ok = done.filter(d => d.r).length
  log(`chunk ${Math.floor(i / CHUNK) + 1}/${Math.ceil(slugs.length / CHUNK)}: ${ok}/${slugs.length} written so far`)
}

const okList = done.filter(d => d.r)
const failed = done.filter(d => !d.r).map(d => d.slug)
log(`Completed ${okList.length}/${slugs.length}` + (failed.length ? ` — still failed: ${failed.join(', ')}` : ''))
return okList.map(d => `${d.slug}: ${d.r}`)
