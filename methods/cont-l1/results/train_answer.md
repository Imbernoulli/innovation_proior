The pain that drives everything here is deployment. I have a 7B-parameter vision-language-action policy that takes an image plus a language instruction and emits a robot action, and the language-following and semantic generalization are already there from pretraining — that part works. What does not work is the action generation. The policy predicts a 7-dimensional delta end-effector pose the way a language model predicts text: each dimension is normalized to $[-1,+1]$, chopped into 256 bins, each bin assigned a token, and the tokens are decoded one at a time, left to right, under a causal mask. Producing a single timestep's action therefore costs $D$ sequential decoder passes — about a third of a second on an A100 — which caps throughput at 3-5 Hz, an order of magnitude below the 25-50+ Hz a high-frequency (or bimanual) controller needs. And the slowness is not just an inconvenience: it blocks the one improvement I know would help. Predicting and executing a chunk of $K$ future actions per query is well documented to raise manipulation success, because it shrinks the effective horizon of the task and so curbs the compounding-error problem of behavioral cloning — where each small prediction error nudges the robot further off the demonstrated distribution until it reaches a state it cannot recover from — and because a chunk can express temporally-correlated structure (a pause, say) that a single-step Markovian policy cannot. But under autoregressive decoding a chunk of $K$ timesteps costs $K \cdot D$ sequential passes; with $D=7$ and even $K=8$ that is fifty-six decoder calls per query. The decoding scheme and the thing I most want are in direct conflict, and on top of all this the 256-bin grid is a lossy bottleneck that caps action precision no matter how fast I run. The prior continuous-action imitation methods point at the right ingredients — one from-scratch chunking policy predicts all $K$ actions in a single non-causal pass and regresses them with an L1 loss it found more precise than L2, another models the chunk with diffusion to capture multimodality — but neither carries large-scale vision-language pretraining, and diffusion's iterative sampling reintroduces exactly the latency I need to kill. What is missing is a single fine-tuning recipe that folds the good ingredients onto the large pretrained trunk.

I propose OpenVLA-OFT in its continuous-L1 instantiation, Cont-L1, which rests on three choices made in sequence from the pain backward. The first is to ask why I am decoding sequentially at all. In language, left-to-right is forced — token $t+1$ genuinely depends on what token $t$ turned out to be. But the seven dimensions of one timestep's delta pose are just seven coordinates of one vector; there is no sense in which the gripper dimension must be generated after the z-position dimension. I am imposing an autoregressive ordering on a quantity that has none, purely because I inherited the language-model machinery. So I remove it. In the autoregressive setup the input to an action position is the previously-generated token; if I am not generating sequentially there is no previous token to feed, so I fill all $K \cdot D$ action-token positions with empty (zeroed) action embeddings that carry no action content and differ from one another only by positional encoding — they merely mark "an action goes here, at this slot in the chunk." And the causal mask, which exists precisely to enforce left-to-right dependence, is replaced by bidirectional attention over those slots, so every action position can read the full observation and every other action position. Now one forward pass produces a hidden state at all $K \cdot D$ action positions at once, collapsing the sequential-pass factor from $K \cdot D$ to 1. This is the same trick a from-scratch chunking policy uses with fixed per-timestep query embeddings; the only honest worry is that parallel prediction is less expressive than autoregressive, since I have thrown away the ability to condition position $j$ on the realized value of position $i$ — but with a 7B trunk and full bidirectional attention each slot still sees everything except the sampled values of the others, which is an expressivity question to validate, not a reason to keep an already-too-slow decode. Crucially, because decoding is now a single pass, adding more slots to go from $D$ to $K \cdot D$ lengthens the attention sequence but no longer multiplies decoder calls, so chunking — and with it the reduced effective horizon and the drop in compounding error — becomes affordable. The single-pass decode is what makes chunking deployable; they are not two ideas but one.

The second choice concerns what comes out of each action position: a hidden state the width of the LLM, which I still must turn into an action. The native recipe would project it to 256 bin-logits, but binning is the lossy bottleneck I keep paying for — 256 bins is a coarse grid and fine manipulation lives in the gaps, while adding bins for resolution makes each bin-token rarer across a few hundred demonstrations and so generalize worse, a trade where neither end is good. The cleaner move is to not discretize at all: replace the bin-logit layer with a small head that maps the action-position hidden states straight to real-valued normalized actions, and the precision ceiling simply disappears. The head consumes the hidden states grouped by timestep — the $D$ action positions belonging to timestep $k$ jointly determine that timestep's $D$-dimensional action — so the $(B, K \cdot D, \text{hidden})$ tensor is regrouped to $(B, K, D \cdot \text{hidden})$, each row the concatenation of one timestep's $D$ hidden states, and mapped to $(B, K, \text{action\_dim})$. For the head itself I use a pre-norm MLP-ResNet: an input projection (LayerNorm → Linear → ReLU) lifting the $D \cdot \text{hidden}$ input to the working width, two residual feedforward blocks, then a final LayerNorm and a linear projection down to action\_dim. Each block is LayerNorm → Linear → ReLU added back to its input; the residual connections give gradients a clean path so the head trains stably on top of the LoRA-adapted trunk, and putting normalization before the transformation (pre-norm) keeps the residual stream's gradients well-behaved, the same reasoning that favors pre-norm feedforward blocks in transformers. The head is about four layers deep and tiny relative to the 7B trunk, so it adds negligible inference cost — which is the whole point of choosing a head over a heavier generative decoder.

The third choice is the loss, and it interacts with the messy reality of human demonstration data. The obvious default is squared error, L2, minimizing the mean of $(\text{predicted} - \text{ground\_truth})^2$. But demonstrations are noisy and a human might act slightly differently across episodes at the same observation, so the target is effectively a spread of plausible actions. L2 penalizes residuals quadratically, is pulled hard toward outliers, and its optimum is the conditional mean — and the mean of several distinct valid motions can be a blended action valid under none of them. L1, the mean absolute error, penalizes linearly, is far less swayed by large deviations, and its optimum is the median, which is the better summary for action prediction: robust to noisy outliers, and exactly the choice the from-scratch chunking policy reports moving to for more precise action modeling. So my objective is

$$L = \frac{1}{K D} \sum_{k=1}^{K} \sum_{d=1}^{D} \left| \hat{a}_{k,d} - a_{k,d} \right|,$$

the mean absolute error between the predicted and ground-truth action chunks on the normalized $[-1,+1]$ actions — one `L1Loss` call. I considered diffusion instead, and its advantage is real: modeling the action chunk with a conditional denoising process represents genuinely multimodal action distributions, capturing several distinct valid ways to act rather than collapsing to a median. I decline it for two decisive reasons in this setting. First, cost: diffusion needs tens of sequential denoising steps at inference, so even riding on top of my single-pass decode each chunk costs tens of passes and I am back in the latency hole, with slower training too. Second, the bet: simple regression usually loses to diffusion because of under-capacity — a small network hedges by averaging modes — but a 7B trunk conditioned on a rich observation can often pin down which mode the situation calls for and regress it cleanly, making the multimodality diffusion buys largely redundant here, so L1 should roughly match diffusion's success rate while being dramatically cheaper. The honest limitation is that the median-seeking L1 head cannot represent truly multimodal demonstrations, where several genuinely different action sequences are all correct for one input; it suits focused, consistent-strategy demonstrations. Around the method, I adapt the 7B trunk with LoRA (rank 32) rather than full fine-tuning, which would be wasteful and overfit on hundreds of demonstrations, and train the head and any small proprioceptive-state projector alongside the adapters; $K=8$ and $D=7$ are fixed by protocol, the chunk predicted in one pass and executed open-loop, and because actions are normalized to $[-1,+1]$ the L1 scale is meaningful enough that "train until mean normalized L1 < 0.01" is a sensible convergence criterion.

Tracing the whole chain: the native autoregressive-discrete decode is too slow and its quantization caps precision and its slowness blocks chunking; asking why decode sequentially leads to positionally-marked empty action embeddings plus bidirectional attention so all $K \cdot D$ positions emerge in one pass, which makes chunking essentially free; removing the discretization bottleneck means regressing continuous actions from the action-position hidden states with a small pre-norm MLP-ResNet head; and L1 trains it, robust and precise and single-pass-cheap where diffusion would not be. Throughput jumps because $K \cdot D$ sequential passes became one, success rises because chunking cuts compounding error and continuous regression keeps the action precise, and the recipe stays simple enough to fold extra camera views and proprioceptive state into the same single forward pass. The MLP-ResNet head and the L1 training step:

```python
import torch
import torch.nn as nn

# Robot constants (LIBERO single-arm)
ACTION_DIM = 7
NUM_ACTIONS_CHUNK = 8


class MLPResNetBlock(nn.Module):
    """Pre-norm residual feedforward block: LayerNorm -> Linear -> ReLU, + input."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        identity = x
        x = self.ffn(x)
        return x + identity


class MLPResNet(nn.Module):
    """MLP with residual blocks: input projection -> blocks -> output projection."""
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList(
            [MLPResNetBlock(dim=hidden_dim) for _ in range(num_blocks)]
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer_norm1(x)
        x = self.fc1(x)
        x = self.relu(x)
        for block in self.mlp_resnet_blocks:
            x = block(x)
        x = self.layer_norm2(x)
        x = self.fc2(x)
        return x


class L1RegressionActionHead(nn.Module):
    """Continuous-action head trained via L1 regression over an action chunk."""
    def __init__(self, input_dim=4096, hidden_dim=4096, action_dim=7):
        super().__init__()
        self.action_dim = action_dim
        # Input width = hidden * ACTION_DIM: the D action-position hidden states
        # of a timestep are consumed jointly to predict that timestep's D-dim action.
        self.model = MLPResNet(
            num_blocks=2,
            input_dim=input_dim * ACTION_DIM,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def predict_action(self, actions_hidden_states):
        # actions_hidden_states: (B, K * ACTION_DIM, hidden_dim)
        batch_size = actions_hidden_states.shape[0]
        rearranged = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        return self.model(rearranged)  # (B, K, action_dim)


def zero_action_token_embeddings(input_embeddings, all_actions_mask):
    """Inner OpenVLA-OFT forward step for L1 regression.
    The action token IDs keep the positions; their embeddings are zeroed so
    they carry no teacher-forced action content."""
    return input_embeddings * ~all_actions_mask.unsqueeze(-1)


def l1_training_step(vla, action_head, batch, num_patches,
                     get_current_action_mask, get_next_actions_mask):
    # Ground-truth action chunk, normalized to [-1, +1]: (B, K, action_dim)
    ground_truth_actions = batch["actions"].to(torch.bfloat16)

    # Parallel decode: vla.forward zeroes action-token input embeddings in the
    # L1 path; the design uses bidirectional action-token attention,
    # so all K*D action positions are predicted in a single forward pass.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = vla(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"].to(torch.bfloat16),
            labels=batch["labels"],
            output_hidden_states=True,
        )

    # Select the K*D action-token hidden states from the last layer.
    last_hidden_states = output.hidden_states[-1]                 # (B, seq_len, hidden)
    text_hidden_states = last_hidden_states[:, num_patches:-1]    # drop vision prefix
    ground_truth_token_ids = batch["labels"][:, 1:]
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
    batch_size = batch["input_ids"].shape[0]
    actions_hidden_states = (
        text_hidden_states[current_action_mask | next_actions_mask]
        .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
        .to(torch.bfloat16)
    )                                                            # (B, K*D, hidden)

    # Predict the continuous action chunk and take the mean L1 loss.
    predicted_actions = action_head.predict_action(actions_hidden_states)  # (B, K, D)
    loss = torch.nn.L1Loss()(ground_truth_actions, predicted_actions)
    return loss, predicted_actions
```
