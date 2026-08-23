# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import logging
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.adaptive_sampling import (
    AdaptiveSamplingConfig,
    adaptive_advantage_values,
    flat_group_mask,
    group_needs_deepening,
    group_indices,
    has_positive,
    next_round_target,
    plan_retention,
)
from verl.utils.reward_score.overlong_penalty import OverlongPenaltyConfig, overlong_penalty
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

# NOTE: `fit()` binds a LOCAL name `logger` to a Tracking instance, so this
# module-level logger is only visible to the other methods -- which is what the
# adaptive-resampling helpers below want.
logger = logging.getLogger(__name__)


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)

        self.use_rm = need_reward_model(self.config)

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # Per-prompt adaptive resampling (group deepening). Disabled by default:
        # with ADAPTIVE_N_ENABLE unset this object is inert and the fit loop takes
        # exactly the code path it takes today. See adaptive_sampling.py.
        self.adaptive_cfg = AdaptiveSamplingConfig.from_env()
        # v2.1(B) requeue state: FIFO of first-time budget-exhausted prompts
        # (full pre-pop rows), the uids of re-injected retries, and the
        # second-drop tally. In-memory only: a job restart empties the buffer,
        # which merely converts pending retries into ordinary drops.
        self._requeue_buffer: list[dict] = []
        self._requeue_retry_uids: set[str] = set()
        self._requeue_second_drops: int = 0
        self._requeue_retried_total: int = 0
        self._requeue_snapshot: Optional[DataProto] = None
        # v2.4 pipelined deepening (ADAPTIVE_N_OVERLAP=1): uid -> group size this
        # prompt must be sampled at the next time it appears in a batch. Written
        # when a group comes back without a positive; consumed by
        # _expand_gen_batch when the requeue FIFO re-injects the prompt.
        self._pending_deepen: dict[str, int] = {}
        # v2.1(A) gate: the clamp also covers the penalty-without-adaptive case.
        # Fail LOUDLY on malformed FS_OVERLONG_* -- an except-swallow here would
        # silently disable the zero-raw advantage clamp in the penalty-only
        # configuration, resurrecting the wrong-answer-uptraining pathology the
        # clamp exists to kill (review m4).
        self._overlong_driver_enabled = OverlongPenaltyConfig.from_env(
            max_resp_len=self.config.data.max_response_length
        ).enable
        if self.adaptive_cfg.enable:
            print(f"[adaptive-n] per-prompt adaptive resampling ENABLED: {self.adaptive_cfg.describe()}")
            if self.config.algorithm.get("use_kl_in_reward", False):
                # The deepened baseline (mean/std over all K samples) is measured from
                # rm_scores, but only the RETAINED rows ever get a KL term added to
                # token_level_rewards -- so the baseline and the scores it centres
                # would be on two different scales. Fail loudly instead of silently
                # mis-centring every deepened group.
                raise ValueError(
                    "ADAPTIVE_N_ENABLE=1 is incompatible with algorithm.use_kl_in_reward=True: the "
                    "K-sample baseline is measured pre-KL while token_level_rewards is post-KL. Use "
                    "actor.use_kl_loss instead (which is what this repo's runner already does)."
                )
            # GRPO_VECTORIZED computes the same (score - group_mean)/group_std that
            # _apply_adaptive_advantage reproduces, so both are safe. GRPO_PASSK and
            # the non-group estimators are not.
            if self.config.algorithm.adv_estimator not in (
                AdvantageEstimator.GRPO,
                AdvantageEstimator.GRPO_VECTORIZED,
            ):
                raise ValueError(
                    f"ADAPTIVE_N_ENABLE=1 supports adv_estimator=grpo / grpo_vectorized only, got "
                    f"{self.config.algorithm.adv_estimator}. The fold-back re-derives the GRPO "
                    f"group baseline, which other estimators do not use."
                )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self.checkpoint_manager = None

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _dump_long_responses_on_spike(
        self,
        batch: DataProto,
        reward_extra_infos_dict: dict,
        metrics: dict,
        dump_dir: str,
        threshold_tokens: int = 20000,
        top_k: int = 5,
    ):
        """When response_length exceeds threshold (spike), dump the longest responses for inspection.
        Useful for debugging abnormal response length peaks during GRPO training.
        """
        max_len = metrics.get("response_length/max", 0)
        mean_len = metrics.get("response_length/mean", 0)
        if max_len < threshold_tokens and mean_len < threshold_tokens:
            return

        max_response_len = batch.batch["responses"].shape[-1]
        prompt_mask = batch.batch["attention_mask"][:, :-max_response_len].bool()
        response_mask = batch.batch["attention_mask"][:, -max_response_len:].bool()
        lengths = response_mask.sum(-1).cpu().tolist()

        # Indices of top-k longest responses
        indexed = [(i, lengths[i]) for i in range(len(lengths))]
        indexed.sort(key=lambda x: -x[1])
        top_indices = [idx for idx, _ in indexed[:top_k] if lengths[idx] >= threshold_tokens]
        if not top_indices:
            return

        os.makedirs(dump_dir, exist_ok=True)
        filename = os.path.join(dump_dir, f"spike_step{self.global_steps}.jsonl")

        prompts = batch.batch["prompts"]
        responses = batch.batch["responses"]
        scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()

        def _detect_repetition(text: str) -> dict:
            """Quick repetition check: unique n-gram ratio and dup lines."""
            if not text or len(text) < 100:
                return {"has_repetition": False, "unique_ngram_ratio": 1.0, "dup_line_ratio": 0}
            words = text.split()
            n, step = 12, 6
            ngrams = [tuple(words[i : i + n]) for i in range(0, len(words) - n + 1, step)]
            unique_ratio = len(set(ngrams)) / len(ngrams) if len(ngrams) >= 10 else 1.0
            lines = [L.strip() for L in text.split("\n") if L.strip()]
            dup = sum(1 for j in range(1, len(lines)) if lines[j] == lines[j - 1])
            dup_ratio = dup / len(lines) if lines else 0
            return {
                "has_repetition": unique_ratio < 0.4 or dup_ratio > 0.15,
                "unique_ngram_ratio": round(unique_ratio, 3),
                "dup_line_ratio": round(dup_ratio, 3),
            }

        lines = []
        for i in top_indices:
            valid_resp_len = int(lengths[i])
            prompt_ids = prompts[i]
            response_ids = responses[i][:valid_resp_len]
            input_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
            output_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            try:
                item = batch[i]
                gt = item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
            except (IndexError, KeyError, TypeError):
                gt = None
            rep = _detect_repetition(output_text)
            entry = {
                "step": self.global_steps,
                "index": int(i),
                "response_length": valid_resp_len,
                "score": scores[i],
                "ground_truth": gt,
                "repetition": rep,
                "input_preview": input_text[:500] + "..." if len(input_text) > 500 else input_text,
                "output_preview": output_text[:2000] + "..." if len(output_text) > 2000 else output_text,
                "output_full": output_text,
            }
            for k, v in reward_extra_infos_dict.items():
                if isinstance(v, (list, np.ndarray)) and len(v) == len(lengths):
                    entry[k] = v[i]
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")
        n_rep = sum(1 for line in lines if json.loads(line).get("repetition", {}).get("has_repetition"))
        rep_note = f", {n_rep}/{len(top_indices)} show repetition" if n_rep else ""
        print(f"[Diagnostic] Response length spike detected (max={max_len}, mean={mean_len:.0f}){rep_note}. "
              f"Dumped {len(top_indices)} longest samples to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    # ------------------------------------------------------------------
    # Per-prompt adaptive resampling ("group deepening").
    # See verl/trainer/ppo/adaptive_sampling.py for the full rationale, and in
    # particular for the group-size / loss-denominator decision.
    # ------------------------------------------------------------------

    @staticmethod
    def _group_scores(scores: np.ndarray, prompt_index: np.ndarray, num_prompts: int) -> list[np.ndarray]:
        """Slice a flat per-row score vector into per-prompt groups."""
        buckets: list[list[float]] = [[] for _ in range(num_prompts)]
        for row, p in enumerate(prompt_index):
            buckets[int(p)].append(float(scores[row]))
        return [np.asarray(b, dtype=np.float64) for b in buckets]

    def _raw_scores_np(self, output: DataProto, penalized: np.ndarray) -> np.ndarray:
        """Per-row RAW (pre-penalty) scores -- the currency of every v2 decision.

        Three sources, in order of preference:
          1. the ``reward_pre_overlong`` column the worker-side penalty logs
             (exact float, present whenever FS_OVERLONG_PENALTY=1 with logging);
          2. deterministic reconstruction: the penalty is a pure function of the
             valid response length, so raw = penalized - penalty(length);
          3. the penalty is off entirely -> rm_scores ARE raw.
        """
        col = output.non_tensor_batch.get("reward_pre_overlong")
        if col is not None:
            vals = np.array([np.nan if v is None else float(v) for v in col], dtype=np.float64)
            if not np.isnan(vals).any():
                return vals
            # partially missing (schema-aligned fill rows): fall through to recompute
        ol_cfg = OverlongPenaltyConfig.from_env(max_resp_len=self.config.data.max_response_length)
        if not ol_cfg.enable:
            return penalized.copy()
        prompt_length = output.batch["prompts"].size(1)
        valid_resp = output.batch["attention_mask"][:, prompt_length:].sum(dim=1).cpu().numpy()
        pens = np.array(
            [
                overlong_penalty(int(length), ol_cfg.max_resp_len, ol_cfg.buffer_len, ol_cfg.penalty_factor)
                for length in valid_resp
            ],
            dtype=np.float64,
        )
        return penalized - pens

    def _adaptive_deepen(
        self,
        gen_batch: DataProto,
        gen_batch_output: DataProto,
        timing_raw: dict,
        prompt_index: Optional[np.ndarray] = None,
    ) -> tuple[DataProto, np.ndarray, dict]:
        """Resample flat groups on the SAME prompt until they acquire spread.

        Args:
            gen_batch: the un-repeated prompt batch (B rows, carries ``uid``).
            gen_batch_output: rollout output for ``gen_batch.repeat(n)`` (B*n rows).

        Returns:
            (output, prompt_index, metrics) where ``prompt_index[i]`` is the row of
            ``gen_batch`` that produced output row ``i``. In the default
            ``keep="subsample"`` mode ``len(output) == B*n`` and ``prompt_index`` is
            exactly ``repeat_interleave(arange(B), n)`` -- i.e. the shape the
            trainer already expects, so nothing downstream changes.
        """
        cfg = self.adaptive_cfg
        n = self.config.actor_rollout_ref.rollout.n
        num_prompts = len(gen_batch)
        # In overlap mode the wave is NOT a uniform repeat(n): carried-over
        # prompts were expanded deeper in-wave, so the caller passes the real
        # row->prompt map. Everywhere else it is exactly repeat_interleave.
        identity_index = (
            np.asarray(prompt_index, dtype=np.int64)
            if prompt_index is not None
            else np.repeat(np.arange(num_prompts, dtype=np.int64), n)
        )
        metrics: dict[str, float] = {}

        if "rm_scores" not in gen_batch_output.batch.keys():
            # Rewards are not available yet (colocate reward-model path). We cannot
            # tell which groups are flat, so deepening is impossible here. Warn once
            # and behave exactly as before.
            # Codex review finding 5: skipping is only safe for a UNIFORM wave. In
            # overlap mode the wave can be ragged (a carried prompt was expanded
            # in-wave); returning it unfolded would train that prompt at 2n rows
            # with no HT weights -- the exact defect 21546af fixed. Refuse loudly
            # instead of silently mis-weighting the update.
            if prompt_index is not None and (np.bincount(identity_index, minlength=num_prompts) != n).any():
                raise RuntimeError(
                    "[adaptive-n] overlap wave is ragged but rm_scores are absent: the deepened "
                    "groups cannot be folded back. Use a streaming/in-loop reward (rm_scores in the "
                    "rollout output) or disable ADAPTIVE_N_OVERLAP."
                )
            logger.warning(
                "[adaptive-n] rm_scores absent from rollout output -- adaptive resampling needs "
                "in-rollout rewards (streaming RewardLoopWorker / in-loop agent scores). Skipping."
            )
            return gen_batch_output, identity_index, {"adaptive/skipped_no_rm_scores": 1.0}

        # STEP 1 of the v2 pipeline: judge the trigger on RAW (pre-penalty)
        # scores. `scores` (penalized) is the training signal the baseline will
        # eventually be computed over; `raw` decides everything else. Under the
        # zero_variance (v1) trigger the raw/penalized split is carried but the
        # trigger is the old flatness test on the penalized scores.
        scores = gen_batch_output.batch["rm_scores"].sum(dim=-1).detach().cpu().numpy().astype(np.float64)
        raw = self._raw_scores_np(gen_batch_output, scores)
        groups = self._group_scores(scores, identity_index, num_prompts)
        raw_groups = self._group_scores(raw, identity_index, num_prompts)

        dead_before = [p for p in range(num_prompts) if flat_group_mask(groups[p], cfg.eps)]
        metrics["adaptive/dead_groups_before"] = len(dead_before) / max(num_prompts, 1)
        metrics["adaptive/dead_groups_before_count"] = float(len(dead_before))

        needs = [p for p in range(num_prompts) if group_needs_deepening(raw_groups[p], groups[p], cfg)]
        if cfg.trigger == "no_positive":
            metrics["adaptive/groups_no_positive_before"] = len(needs) / max(num_prompts, 1)
            metrics["adaptive/groups_no_positive_before_count"] = float(len(needs))

        # v2.1(D)(iii): the OTHER degenerate end. A group where every sample is
        # a raw positive and the penalized scores are tied contributes zero
        # gradient but a full share of the token-mean denominator -- DAPO filters
        # both ends, we previously only handled the zero end. These groups are
        # solved, not starving: no deepening, no requeue -- just masked out of
        # the loss like dropped groups. (Judged on PENALIZED scores: if the
        # penalty differentiates equally-correct answers by length, that is a
        # real "prefer shorter" gradient and the group stays.)
        saturated: set[int] = set()
        if cfg.trigger == "no_positive":
            saturated = {
                p
                for p in range(num_prompts)
                if bool(np.all(raw_groups[p] > cfg.pos_eps)) and flat_group_mask(groups[p], cfg.eps)
            }
            metrics["adaptive/groups_saturated"] = float(len(saturated))

        # Agent allow-list: one extra mlsbench_agent sample is a whole multi-minute
        # episode, so a run may want to deepen only the cheap single_turn rows.
        candidates = list(needs)
        agent_names = gen_batch.non_tensor_batch.get("agent_name")
        if cfg.agents and agent_names is not None:
            allowed = set(cfg.agents)
            candidates = [p for p in candidates if str(agent_names[p]) in allowed]
        if cfg.max_prompts > 0:
            candidates = candidates[: cfg.max_prompts]

        extra_outputs: list[DataProto] = []
        extra_prompt_index: list[np.ndarray] = []
        # Codex review finding 1 (2026-08-12): sizing every group at n is only true
        # for a uniform wave. In overlap mode a carried prompt arrives already
        # expanded (2n/4n) and the synchronous loop that maintained group_sizes
        # never runs, so the stale 16 made the ladder restart at 32 forever: the
        # ceiling drop test (group_sizes >= max_n) could never fire and the prompt
        # cycled until eviction. Take the sizes from the wave that actually came back.
        _counts = np.bincount(identity_index, minlength=num_prompts)
        group_sizes = {p: int(_counts[p]) for p in range(num_prompts)}
        active = list(candidates)
        rounds = 0
        extra_rollouts = 0

        # v2.4 OVERLAP MODE: issue no extra wave at all. A starving group is
        # masked out of this step (it has no positive, so per the standing rule
        # it must not be scored) and its prompt is queued to come back in the
        # NEXT step's main wave at the deeper size, where its tokens are
        # generated alongside everything else instead of in a serial tail.
        # `active`/`rounds` stay empty so all the accounting below is unchanged.
        if cfg.overlap or cfg.inwave:
            active = []

        with marked_timer("adaptive_resample", timing_raw, color="magenta"):
            while active and rounds < 64:
                current = group_sizes[active[0]]
                target = next_round_target(current, cfg)
                if target <= current:
                    break  # ceiling reached
                add = target - current

                if cfg.max_extra > 0 and extra_rollouts + add * len(active) > cfg.max_extra:
                    room = max(0, (cfg.max_extra - extra_rollouts) // max(add, 1))
                    if room <= 0:
                        logger.info("[adaptive-n] extra-rollout budget exhausted at round %d", rounds)
                        break
                    active = active[:room]

                retry_prompts = gen_batch.select_idxs(np.asarray(active, dtype=np.int64))
                retry_batch = retry_prompts.repeat(repeat_times=add, interleave=True)
                retry_prompt_index = np.repeat(np.asarray(active, dtype=np.int64), add)

                # AgentLoopManager.generate_sequences chunks across workers and
                # requires an exactly divisible batch; pad, then drop the pad rows.
                divisor = max(1, len(self.async_rollout_manager.agent_loop_workers))
                retry_batch, pad_size = pad_dataproto_to_divisor(retry_batch, divisor)
                retry_out = self.async_rollout_manager.generate_sequences(retry_batch)
                retry_out = unpad_dataproto(retry_out, pad_size)
                retry_out.meta_info.pop("timing", None)
                retry_out.meta_info.pop("metrics", None)

                if "rm_scores" not in retry_out.batch.keys():
                    # _postprocess only writes rm_scores when EVERY row in the chunk
                    # scored (agent_loop.py:775). One failed reward in a deepening
                    # round would otherwise KeyError and take the whole step down.
                    # Drop this round's samples and keep the base rollout.
                    logger.warning(
                        "[adaptive-n] deepening round %d returned no rm_scores (a reward failed); "
                        "discarding the round and continuing with the base rollout.",
                        rounds,
                    )
                    break

                extra_outputs.append(retry_out)
                extra_prompt_index.append(retry_prompt_index)
                extra_rollouts += len(retry_prompt_index)
                rounds += 1

                # STEP 2: re-judge each active group on RAW scores; stop a group's
                # deepening the moment a positive appears (no_positive) or the
                # moment spread appears (zero_variance).
                retry_scores = retry_out.batch["rm_scores"].sum(dim=-1).detach().cpu().numpy().astype(np.float64)
                retry_raw = self._raw_scores_np(retry_out, retry_scores)
                still_active = []
                for p in active:
                    sel = retry_prompt_index == p
                    groups[p] = np.concatenate([groups[p], retry_scores[sel]])
                    raw_groups[p] = np.concatenate([raw_groups[p], retry_raw[sel]])
                    group_sizes[p] = groups[p].size
                    if group_needs_deepening(raw_groups[p], groups[p], cfg) and group_sizes[p] < cfg.max_n:
                        still_active.append(p)
                active = still_active

        rescued = [p for p in candidates if not group_needs_deepening(raw_groups[p], groups[p], cfg)]
        # STEP 4: drop ONLY the groups that were actually selected for deepening
        # (candidates) and still lack a correct exemplar after their budget.
        # Groups the agent allow-list / prompt cap excluded from deepening were
        # never given their chance -- deleting them from the loss would silently
        # discard every no-positive group of an excluded agent (e.g. ALL MLS
        # groups under ADAPTIVE_N_AGENTS=single_turn_agent). They stay in the
        # batch with normal advantages; the v2.1(A) clamp caps every raw<=0 row
        # at advantage <= 0, so the penalty-only-uptraining pathology cannot
        # reach them either. (Reviewer decision M3, 2026-08-10.)
        # "Deepened to budget" is literal: the group reached its full individual
        # ceiling (max_n). A candidate the shared per-step rollout budget cut off
        # early was not given its full chance -- it stays in the loss (clamped),
        # is not requeued, and simply competes again if the sampler ever
        # revisits it.
        dropped: set[int] = set()
        if cfg.trigger == "no_positive":
            dropped = {
                p
                for p in candidates
                if group_sizes[p] >= cfg.max_n and not has_positive(raw_groups[p], cfg.pos_eps)
            }
        # v2.4 overlap: a starving group that has NOT yet reached its ceiling is
        # also masked out of this step (no positive => must not be scored), but
        # unlike a true drop it carries a deeper target into the requeue FIFO,
        # so the next wave samples it at 2x. `carried` rides `dropped` through
        # the masking/requeue code below; the two are told apart by deepen_to.
        carried: dict[int, int] = {}
        if cfg.overlap and not cfg.inwave and cfg.trigger == "no_positive":
            for p in candidates:
                if p in dropped or has_positive(raw_groups[p], cfg.pos_eps):
                    continue
                target = min(next_round_target(group_sizes[p], cfg), cfg.max_n)
                if target > group_sizes[p]:
                    carried[p] = target
            dropped |= set(carried)
            metrics["adaptive/overlap_carried"] = float(len(carried))
        metrics["adaptive/groups_deepened"] = float(len(candidates))
        metrics["adaptive/groups_rescued"] = float(len(rescued))
        if cfg.trigger == "no_positive":
            metrics["adaptive/groups_rescued_positive"] = float(len(rescued))
            metrics["adaptive/groups_dropped"] = float(len(dropped))
            # Kept no-positive groups are now expected exactly when deepening was
            # policy-limited (allow-list / max_prompts); they are safe under the
            # clamp. The count equals needs - candidates by construction -- the
            # smoke asserts that identity rather than zero.
            kept_no_positive = sum(
                1 for p in range(num_prompts) if p not in dropped and not has_positive(raw_groups[p], cfg.pos_eps)
            )
            metrics["adaptive/kept_groups_without_positive"] = float(kept_no_positive)
            metrics["adaptive/groups_excluded_from_deepening"] = float(len(needs) - len(candidates))

            # v2.1(D)(i): stage-hit histogram -- at what group size was each
            # rescue found? -- plus the pass-rate distribution of the groups
            # that will actually train.
            for p in rescued:
                metrics[f"adaptive/rescued_at_{int(group_sizes[p])}"] = (
                    metrics.get(f"adaptive/rescued_at_{int(group_sizes[p])}", 0.0) + 1.0
                )
            pass_rates = [
                float((raw_groups[p] > cfg.pos_eps).mean())
                for p in range(num_prompts)
                if p not in dropped and p not in saturated
            ]
            if pass_rates:
                pr = np.asarray(pass_rates)
                metrics["adaptive/group_pass_rate_mean"] = float(pr.mean())
                metrics["adaptive/group_pass_rate_min"] = float(pr.min())
                metrics["adaptive/group_pass_rate_max"] = float(pr.max())

            # v2.1(B): first-time budget-exhausted prompts go to the requeue FIFO
            # (retried once in a later batch); second-time failures are dropped
            # for good. Requeue happens by uid: rows re-injected by fit() carry
            # uids registered in self._requeue_retry_uids.
            if cfg.requeue_enable:
                uids = gen_batch.non_tensor_batch.get("uid")
                second_drops = 0
                queued_now = 0
                for p in sorted(dropped):
                    uid = str(uids[p]) if uids is not None else None
                    # v2.4 overlap: a carried group has NOT exhausted its budget --
                    # it is mid-ladder by construction, so the once-only retry rule
                    # (which exists to stop budget-exhausted prompts from cycling
                    # forever) must not apply to it. Its ceiling is cfg.max_n,
                    # enforced when `carried` is built.
                    if uid is not None and uid in self._requeue_retry_uids and p not in carried:
                        second_drops += 1  # already retried once: gone for good
                        continue
                    if self._requeue_snapshot is not None:
                        row = self._requeue_snapshot.select_idxs(np.asarray([p], dtype=np.int64))
                        # The snapshot was taken AFTER uid assignment, but the
                        # batch this row will be concat'ed into has NO uid yet
                        # (uids are assigned after injection). DataProto.concat
                        # keys off the FIRST piece (protocol.py:201-210), so a
                        # stale uid column would trip its assert -- and any
                        # missing key would silently mis-align column lengths.
                        # Strip uid here so the schemas match exactly.
                        row.non_tensor_batch.pop("uid", None)
                        # Hard cap (review M2): the buffer holds deepcopied full
                        # prompt rows; without a cap a long run with a hard tail
                        # grows it without bound. Oldest-first eviction to a
                        # second-drop keeps the accounting honest.
                        if len(self._requeue_buffer) >= self.adaptive_cfg.requeue_buffer_cap:
                            self._requeue_buffer.pop(0)
                            second_drops += 1
                            logger.warning(
                                "[adaptive-n] requeue buffer full (%d); evicting oldest entry as a second drop",
                                self.adaptive_cfg.requeue_buffer_cap,
                            )
                        entry = {"row": row, "step": self.global_steps}
                        if p in carried:
                            entry["deepen_to"] = int(carried[p])
                        self._requeue_buffer.append(entry)
                        queued_now += 1
                self._requeue_second_drops += second_drops
                metrics["adaptive/requeue_size"] = float(len(self._requeue_buffer))
                metrics["adaptive/requeue_queued_now"] = float(queued_now)
                metrics["adaptive/requeue_second_drops"] = float(self._requeue_second_drops)
        metrics["adaptive/extra_rollouts"] = float(extra_rollouts)
        metrics["adaptive/rounds"] = float(rounds)
        metrics["adaptive/max_group_size"] = float(max(group_sizes.values()) if group_sizes else n)

        dead_after = [p for p in range(num_prompts) if flat_group_mask(groups[p], cfg.eps)]
        metrics["adaptive/dead_groups_after"] = len(dead_after) / max(num_prompts, 1)
        metrics["adaptive/dead_groups_after_count"] = float(len(dead_after))

        # Overlap mode has no extra rounds to merge, but the incoming wave itself
        # can be ragged: a carried prompt was expanded to 2n/4n IN the main wave.
        # Those groups must still fold back to exactly n rows with HT weights --
        # otherwise a deepened group enters the loss at double width and eats a
        # double share of the token-mean denominator (the very hazard keep="all"
        # is warned about). Skip the early return whenever any group != n.
        oversized = bool((cfg.overlap or cfg.inwave) and (np.bincount(identity_index, minlength=num_prompts) != n).any())
        if not extra_outputs and not oversized:
            self._annotate_adaptive(gen_batch_output, None, num_prompts, identity_index, n)
            if dropped or saturated:
                self._mask_dropped_groups(gen_batch_output, identity_index, dropped | saturated, metrics)
            return gen_batch_output, identity_index, metrics

        # ---- merge the base rollout with every deepening round -----------------
        pieces = [gen_batch_output] + extra_outputs
        base_timing = gen_batch_output.meta_info.pop("timing", None)
        align = getattr(type(self.async_rollout_manager), "_align_outputs_for_concat", None)
        if align is not None:
            align(pieces)
        else:
            # a custom agent_loop_manager_class without the helper: unify the
            # non-tensor schema ourselves so DataProto.concat cannot fail
            keys = sorted({k for p in pieces for k in p.non_tensor_batch})
            for p in pieces:
                for k in keys:
                    if k not in p.non_tensor_batch:
                        missing = np.empty(len(p), dtype=object)
                        missing.fill(None)
                        p.non_tensor_batch[k] = missing
        # reward_extra_keys can differ between rounds; concat asserts equality on
        # overlapping non-metric meta_info keys, so unify it first.
        merged_extra_keys = sorted({k for p in pieces for k in p.meta_info.get("reward_extra_keys", [])})
        for p in pieces:
            if merged_extra_keys:
                p.meta_info["reward_extra_keys"] = merged_extra_keys
            p.meta_info.pop("timing", None)
        merged = DataProto.concat(pieces)
        if base_timing is not None:
            merged.meta_info["timing"] = base_timing
        merged_index = np.concatenate([identity_index] + extra_prompt_index)

        merged_scores = merged.batch["rm_scores"].sum(dim=-1).detach().cpu().numpy().astype(np.float64)
        rng = np.random.default_rng(cfg.seed + self.global_steps)

        merged_raw = self._raw_scores_np(merged, merged_scores)

        if cfg.keep == "all":
            logger.warning(
                "[adaptive-n] keep='all': the batch is now ragged (%d rows instead of %d). "
                "ppo_mini_batch_size slicing, the token-mean denominator and seq-balancing all "
                "shift, and a %d-sample group will dominate the update. This is an experimental "
                "mode -- see adaptive_sampling.py.",
                len(merged),
                num_prompts * n,
                int(max(group_sizes.values())),
            )
            self._annotate_adaptive(merged, groups, num_prompts, merged_index, n, rng=rng, keep_all=True)
            if dropped or saturated:
                self._mask_dropped_groups(merged, merged_index, dropped | saturated, metrics)
            return merged, merged_index, metrics

        # ---- STEP 3/4: fold each group back to exactly n rows ------------------
        rows_by_prompt: list[list[int]] = [[] for _ in range(num_prompts)]
        for row, p in enumerate(merged_index):
            rows_by_prompt[int(p)].append(row)

        keep_rows: list[int] = []
        weights: list[float] = []
        g_mean: list[float] = []
        g_std: list[float] = []
        enlarged_flag: list[float] = []
        for p in range(num_prompts):
            rows = np.asarray(rows_by_prompt[p], dtype=np.int64)
            if p in dropped:
                # STEP 4: no correct exemplar at budget -> the group is dropped.
                # Keep its BASE n rows purely to preserve the batch shape; they
                # are annotated inert (enlarged=0, weight=1) and their tokens are
                # masked out below, so no penalty, no advantage, and no presence
                # in the token-mean denominator.
                chosen = rows[:n]
                keep_rows.extend(chosen.tolist())
                weights.extend([1.0] * len(chosen))
                g_mean.extend([0.0] * len(chosen))
                g_std.extend([1.0] * len(chosen))
                enlarged_flag.extend([0.0] * len(chosen))
                continue
            # STEP 3: a kept group. Positives (raw > pos_eps) are the protected
            # stratum -- a 1-in-128 success must survive the fold. The GRPO
            # baseline (mean/std) is over the PENALIZED scores of all K samples;
            # the penalty shapes length only inside groups anchored by a correct
            # answer.
            protect = merged_raw[rows] > cfg.pos_eps if cfg.trigger == "no_positive" else None
            plan = plan_retention(merged_scores[rows], n=n, cfg=cfg, rng=rng, protect=protect)
            chosen = rows[plan.keep_idx]
            keep_rows.extend(chosen.tolist())
            weights.extend(plan.weights.tolist())
            g_mean.extend([plan.mean] * len(chosen))
            g_std.extend([plan.std] * len(chosen))
            enlarged_flag.extend([1.0 if plan.enlarged else 0.0] * len(chosen))

        keep_rows_np = np.asarray(keep_rows, dtype=np.int64)
        folded = merged.select_idxs(keep_rows_np)
        folded.non_tensor_batch["adaptive_weight"] = np.asarray(weights, dtype=np.float64)
        folded.non_tensor_batch["adaptive_group_mean"] = np.asarray(g_mean, dtype=np.float64)
        folded.non_tensor_batch["adaptive_group_std"] = np.asarray(g_std, dtype=np.float64)
        folded.non_tensor_batch["adaptive_enlarged"] = np.asarray(enlarged_flag, dtype=np.float64)

        assert len(folded) == num_prompts * n, (
            f"[adaptive-n] fold-back must restore the batch shape: got {len(folded)}, want {num_prompts * n}"
        )
        # The folded batch is uniform (n rows per prompt, prompt-major), which is
        # NOT what identity_index describes in overlap mode -- there the incoming
        # wave was ragged. Hand the caller the index that actually maps the rows
        # it is getting. (Identical to identity_index on every non-overlap path.)
        folded_index = np.repeat(np.arange(num_prompts, dtype=np.int64), n)
        if dropped or saturated:
            self._mask_dropped_groups(folded, folded_index, dropped | saturated, metrics)
        else:
            folded.non_tensor_batch["adaptive_dropped"] = np.zeros(len(folded), dtype=np.float64)
        metrics["adaptive/weight_max"] = float(np.max(weights)) if weights else 1.0
        metrics["adaptive/weight_min"] = float(np.min(weights)) if weights else 1.0
        return folded, folded_index, metrics

    def _filter_truncated(self, output: DataProto) -> dict:
        """DAPO overlong filtering (FS_OVERLONG_FILTER=1): zero the response_mask
        of every rollout that hit the generation cap.

        Rationale measured on rlv10: a truncated rollout carries a full-magnitude
        negative reward over max_response_length tokens, so under token-mean those
        rows held 56-60% of the gradient in the two arms that destabilised. That
        update is pure unlikelihood and cannot teach stopping -- the trajectory
        contains no stop action to up-weight -- so entropy rose, generations grew,
        and truncation fed itself. Masking removes them from BOTH the numerator and
        the token-mean denominator (same mechanism as _mask_dropped_groups).

        Length metrics read attention_mask, not response_mask, so clip_ratio and
        response_length/* keep reporting the truth after filtering -- we must be
        able to SEE the truncation we stopped training on.
        """
        if os.environ.get("FS_OVERLONG_FILTER", "0").strip().lower() not in ("1", "true", "yes"):
            return {}
        flags = output.non_tensor_batch.get("overlong_truncated")
        if flags is None:
            return {"overlong_filter/rows": 0.0}
        mask = np.asarray([1.0 if (f is not None and float(f) > 0) else 0.0 for f in flags])
        n = int(mask.sum())
        if n == 0:
            return {"overlong_filter/rows": 0.0, "overlong_filter/token_frac": 0.0}
        rm = output.batch["response_mask"]
        before = float(rm.sum().item())
        idx = torch.tensor(np.nonzero(mask)[0], dtype=torch.long, device=rm.device)
        if n >= rm.size(0):
            # Every row truncated: masking all of them would leave an empty loss.
            # Keep the batch, report loudly -- this is a red alert, not a no-op.
            logger.error(
                "[overlong-filter] ALL %d rows truncated; refusing to mask the whole batch", n
            )
            return {"overlong_filter/rows": float(n), "overlong_filter/all_truncated": 1.0}
        rm[idx] = 0
        after = float(rm.sum().item())
        return {
            "overlong_filter/rows": float(n),
            "overlong_filter/row_frac": n / rm.size(0),
            "overlong_filter/token_frac": (before - after) / max(before, 1.0),
        }

    def _mask_dropped_groups(
        self,
        output: DataProto,
        prompt_index: np.ndarray,
        dropped: set,
        metrics: dict,
    ) -> None:
        """Zero the response_mask of every dropped group's rows, in place.

        response_mask is the loss mask everywhere downstream: GRPO multiplies the
        advantage by it (compute_grpo_outcome_advantage), and token-mean agg_loss
        divides by loss_mask.sum() (core_algos.agg_loss). Zeroing it therefore
        removes the group from BOTH the numerator and the denominator -- a
        dropped group does not dilute the effective learning rate the way a
        zero-advantage-but-counted group does (~25% dilution in the audit).

        Guard: if EVERY group is dropped, token-mean would divide by zero. One
        sentinel group keeps its mask but has its advantage forced to exactly 0
        via the adaptive-weight machinery (weight=0, enlarged=1), so the step is
        a clean no-op instead of NaN.
        """
        drop_rows = np.asarray([int(p) in dropped for p in prompt_index], dtype=bool)
        if not drop_rows.any():
            output.non_tensor_batch["adaptive_dropped"] = np.zeros(len(output), dtype=np.float64)
            return

        if drop_rows.all():
            # No sentinel: agg_loss floors the token-mean denominator at 1, so a
            # fully-masked batch yields loss exactly 0 for pg, KL and entropy
            # alike (all three aggregate through agg_loss with response_mask).
            # The previous zero-advantage sentinel kept its tokens in the KL
            # term and turned the "no-op" step into a concentrated KL pull on
            # one arbitrary group (review m1).
            logger.warning(
                "[adaptive-n] every group in the batch was dropped (no positives anywhere); "
                "masking the entire batch -- this step contributes exactly zero gradient."
            )

        mask = output.batch["response_mask"]
        masked_tokens = int(mask[torch.from_numpy(drop_rows)].sum().item())
        total_tokens = int(mask.sum().item())
        mask[torch.from_numpy(drop_rows)] = torch.zeros_like(mask[torch.from_numpy(drop_rows)])
        output.batch["response_mask"] = mask
        output.non_tensor_batch["adaptive_dropped"] = drop_rows.astype(np.float64)
        metrics["adaptive/dropped_rows"] = float(drop_rows.sum())
        metrics["adaptive/dropped_token_frac"] = masked_tokens / max(total_tokens, 1)

    @staticmethod
    def _annotate_adaptive(output, groups, num_prompts, prompt_index, n, rng=None, keep_all=False):
        """Attach neutral adaptive_* columns so downstream code has a uniform schema."""
        size = len(output)
        if keep_all and groups is not None:
            means, stds, flags = [], [], []
            for p in prompt_index:
                g = groups[int(p)]
                means.append(float(g.mean()))
                stds.append(float(np.std(g, ddof=1)) if g.size > 1 else 1.0)
                flags.append(1.0 if g.size > n else 0.0)
            output.non_tensor_batch["adaptive_group_mean"] = np.asarray(means, dtype=np.float64)
            output.non_tensor_batch["adaptive_group_std"] = np.asarray(stds, dtype=np.float64)
            output.non_tensor_batch["adaptive_enlarged"] = np.asarray(flags, dtype=np.float64)
        else:
            output.non_tensor_batch["adaptive_group_mean"] = np.zeros(size, dtype=np.float64)
            output.non_tensor_batch["adaptive_group_std"] = np.ones(size, dtype=np.float64)
            output.non_tensor_batch["adaptive_enlarged"] = np.zeros(size, dtype=np.float64)
        output.non_tensor_batch["adaptive_weight"] = np.ones(size, dtype=np.float64)
        output.non_tensor_batch.setdefault("adaptive_dropped", np.zeros(size, dtype=np.float64))

    def _apply_adaptive_advantage(self, batch: DataProto, norm_adv_by_std_in_grpo: bool) -> DataProto:
        """Recompute advantages for deepened groups against their K-sample baseline.

        Rows that were never deepened keep the advantage ``compute_advantage``
        produced, bit-for-bit. For deepened rows we replace it with

            a_i = (score_i - mean_K) / (std_K + eps) * w_i

        where ``(mean_K, std_K)`` come from ALL K samples (a lower-variance
        baseline, and the only correct one once retention is stratified) and
        ``w_i`` is the Horvitz-Thompson weight that unbiases the retention draw.
        """
        if "adaptive_enlarged" not in batch.non_tensor_batch:
            return batch
        enlarged = np.asarray(batch.non_tensor_batch["adaptive_enlarged"], dtype=np.float64)
        if not np.any(enlarged > 0):
            return batch

        scores = batch.batch["token_level_rewards"].sum(dim=-1).detach().cpu().numpy().astype(np.float64)
        means = np.asarray(batch.non_tensor_batch["adaptive_group_mean"], dtype=np.float64)
        stds = np.asarray(batch.non_tensor_batch["adaptive_group_std"], dtype=np.float64)
        weights = np.asarray(batch.non_tensor_batch["adaptive_weight"], dtype=np.float64)

        sel = enlarged > 0
        new_vals = adaptive_advantage_values(
            scores=scores[sel],
            mean=means[sel],
            std=stds[sel],
            weights=weights[sel],
            norm_adv_by_std=norm_adv_by_std_in_grpo,
        )
        advantages = batch.batch["advantages"]
        response_mask = batch.batch["response_mask"]
        rows = torch.from_numpy(np.flatnonzero(sel)).to(advantages.device)
        vals = torch.tensor(new_vals, dtype=advantages.dtype, device=advantages.device).unsqueeze(-1)
        advantages[rows] = vals * response_mask[rows].to(advantages.dtype)
        batch.batch["advantages"] = advantages
        # GRPO sets returns == advantages (core_algos.compute_grpo_outcome_advantage)
        if "returns" in batch.batch.keys():
            returns = batch.batch["returns"]
            returns[rows] = vals * response_mask[rows].to(returns.dtype)
            batch.batch["returns"] = returns
        return batch

    def _clamp_nonpositive_raw_advantage(self, batch: DataProto) -> tuple[DataProto, dict]:
        """v2.1(A): a row whose RAW reward is not positive must never receive
        positive advantage -- ``adv <= 0``, in every group, deepened or natural.

        Penalty-gating alone has a residual hole: in a deepened group with one
        positive and many penalty-bearing truncations, the group mean sits below
        zero, so a wrong-but-complete row (raw exactly 0) clears the mean and
        gets uptrained again. This clamp closes the pathology BY CONSTRUCTION:
        never-uptrain-a-wrong-answer becomes an invariant of the update rule
        rather than a property of a particular group composition. (Related in
        spirit to NSR/PKPO-style sign-aware shaping.)

        Rows with positive raw rewards are untouched; rows with raw <= pos_eps
        keep any NEGATIVE advantage they already have (that gradient is real:
        "do less of this") and only forfeit the positive part.
        """
        if "advantages" not in batch.batch.keys():
            return batch, {}
        scores_key = "token_level_scores" if "token_level_scores" in batch.batch.keys() else "rm_scores"
        penalized = batch.batch[scores_key].sum(dim=-1).detach().cpu().numpy().astype(np.float64)
        raw = self._raw_scores_np(batch, penalized)
        sel = raw <= self.adaptive_cfg.pos_eps
        if not np.any(sel):
            return batch, {"adaptive/clamped_rows": 0.0}
        rows = torch.from_numpy(np.flatnonzero(sel)).to(batch.batch["advantages"].device)
        adv = batch.batch["advantages"]
        before = adv[rows]
        clamped_any = (before > 0).any().item()
        adv[rows] = torch.clamp(before, max=0.0)
        batch.batch["advantages"] = adv
        if "returns" in batch.batch.keys():
            ret = batch.batch["returns"]
            ret[rows] = torch.clamp(ret[rows], max=0.0)
            batch.batch["returns"] = ret
        n_clamped = int((before > 0).any(dim=-1).sum().item()) if clamped_any else 0
        return batch, {"adaptive/clamped_rows": float(n_clamped)}

    def _inject_requeued_prompts(self, batch: DataProto) -> tuple[DataProto, int]:
        """v2.1(B): re-inject first-failure prompts from the FIFO into this gen
        batch, ONCE each. Called BEFORE uid assignment, so retries get fresh
        uids like any other row; the caller registers those uids so a second
        budget exhaustion drops the prompt for good.

        The injected count is rounded down to a multiple of ppo_mini_batch_size
        (prompt units): the trainer-level batch size B is validated against the
        mini-batch size at startup, and keeping B + r a multiple of it means the
        mini-batch split, the grad-accumulation count and seq-balancing never
        see a ragged batch.
        """
        cfg = self.adaptive_cfg
        if not (cfg.enable and cfg.trigger == "no_positive" and cfg.requeue_enable):
            return batch, 0
        eligible = [
            item for item in self._requeue_buffer if self.global_steps - item["step"] >= cfg.requeue_after_steps
        ]
        if not eligible:
            return batch, 0
        mini = int(self.config.actor_rollout_ref.actor.ppo_mini_batch_size)
        # M2 fix: with cap = int(frac * B) alone, any config where cap < mini
        # rounds take to 0 FOREVER (e.g. B=32, mini=16, frac=0.25 -> cap=8) --
        # a silent no-op while the buffer grows without bound. Floor the cap at
        # one mini-batch so a full mini's worth of retries can always flow.
        cap = max(int(cfg.requeue_max_frac * len(batch)), mini)
        take = min(len(eligible), cap)
        take = (take // mini) * mini if mini > 0 else take
        if take <= 0:
            # only reachable when eligible < mini: benign deferral until a full
            # mini-batch of retries accumulates. Loud anyway, per review M2.
            logger.info(
                "[adaptive-n] requeue deferred: %d eligible < mini-batch %d (buffer %d)",
                len(eligible),
                mini,
                len(self._requeue_buffer),
            )
            return batch, 0
        chosen = eligible[:take]
        for item in chosen:
            self._requeue_buffer.remove(item)
        self._requeue_retried_total += len(chosen)
        rows = [item["row"] for item in chosen]
        # Schema guard: concat keys off the FIRST piece and silently mis-aligns
        # column lengths if a later piece is missing a key. The stash strips the
        # uid column; anything else differing means the dataset schema changed
        # mid-run -- fail loudly instead of corrupting the batch.
        batch_keys = set(batch.non_tensor_batch.keys())
        for item, row in zip(chosen, rows, strict=True):
            row_keys = set(row.non_tensor_batch.keys())
            if row_keys != batch_keys:
                raise RuntimeError(
                    f"[adaptive-n] requeued row schema mismatch: row has {sorted(row_keys - batch_keys)} "
                    f"extra and lacks {sorted(batch_keys - row_keys)} vs the incoming batch. "
                    f"Refusing to inject (would corrupt DataProto.concat)."
                )
        logger.info(
            "[adaptive-n] re-injecting %d requeued prompt(s) (buffer %d left) into a batch of %d",
            len(chosen),
            len(self._requeue_buffer),
            len(batch),
        )
        # v2.4 overlap: retries get FRESH uids, so the deepen target cannot be
        # keyed by uid. Injected rows land at the tail in `chosen` order, so
        # record target-by-position instead; _expand_gen_batch consumes it.
        base = len(batch)
        self._pending_deepen = {
            base + i: int(item.get("deepen_to", 0)) for i, item in enumerate(chosen) if item.get("deepen_to", 0) > 0
        }
        return DataProto.concat([batch] + rows), len(chosen)

    def _inwave_prompt_index(self, gen_batch: DataProto, gen_batch_output: DataProto, metrics: dict) -> np.ndarray:
        """Rebuild row->prompt for an in-wave-deepened rollout.

        The AgentLoopWorker tags every returned row with the uid of the prompt it
        was drawn from (extra_fields["adaptive_group"]); a deepened group simply
        contributes more rows with the same tag. Fails loudly rather than guessing
        if the tag is missing or does not cover the batch -- a wrong row->prompt
        map would silently mix groups in the GRPO baseline.
        """
        tags = gen_batch_output.non_tensor_batch.get("adaptive_group")
        uids = gen_batch.non_tensor_batch.get("uid")
        if tags is None or uids is None:
            raise RuntimeError(
                "[adaptive-n/inwave] rollout rows carry no 'adaptive_group' tag (or the prompt batch "
                "has no uid). The workers must run with ADAPTIVE_N_INWAVE=1 too -- refusing to guess "
                "the row->prompt map."
            )
        pos = {str(u): i for i, u in enumerate(uids)}
        try:
            index = np.asarray([pos[str(t)] for t in tags], dtype=np.int64)
        except KeyError as e:
            raise RuntimeError(f"[adaptive-n/inwave] unknown group tag {e} not present in this batch's uids") from e
        counts = np.bincount(index, minlength=len(gen_batch))
        if (counts == 0).any():
            missing = int((counts == 0).sum())
            raise RuntimeError(f"[adaptive-n/inwave] {missing} prompt(s) came back with no rows at all")
        n = self.config.actor_rollout_ref.rollout.n
        extra = int(counts.sum() - n * len(gen_batch))
        metrics["adaptive/inwave_extra_rows"] = float(extra)
        metrics["adaptive/inwave_groups_deepened"] = float(int((counts > n).sum()))
        metrics["adaptive/inwave_max_group_size"] = float(int(counts.max()))
        return index

    def _expand_gen_batch(self, gen_batch: DataProto, n: int) -> tuple[DataProto, np.ndarray, dict[str, float]]:
        """Expand prompts to rollout rows. Normally this is a plain
        ``repeat(n, interleave=True)``; in overlap mode a prompt carried over
        from a previous step is expanded to its deeper target instead, so the
        deepening samples ride inside THIS main wave (no separate low-occupancy
        round). Returns (expanded, prompt_index, metrics).
        """
        cfg = self.adaptive_cfg
        pending = self._pending_deepen if (cfg.enable and cfg.overlap) else {}
        self._pending_deepen = {}
        if not pending:
            expanded = gen_batch.repeat(repeat_times=n, interleave=True)
            return expanded, np.repeat(np.arange(len(gen_batch), dtype=np.int64), n), {}

        # Per-prompt counts, honouring the shared per-step extra-rollout budget.
        counts = np.full(len(gen_batch), n, dtype=np.int64)
        extra = 0
        for idx in sorted(pending):
            if idx >= len(gen_batch):
                continue  # defensive: batch shrank between injection and expansion
            target = min(int(pending[idx]), cfg.max_n)
            add = max(0, target - n)
            if cfg.max_extra > 0 and extra + add > cfg.max_extra:
                # Budget clipping must stay on the n-grid: AgentLoopManager chunks
                # the wave across workers and requires exact divisibility, which a
                # ragged remainder would break.
                add = ((max(0, cfg.max_extra - extra)) // n) * n
            if add <= 0:
                continue
            counts[idx] = n + add
            extra += add

        idxs = np.repeat(np.arange(len(gen_batch), dtype=np.int64), counts)
        expanded = gen_batch.select_idxs(idxs)
        metrics = {
            "adaptive/overlap_prompts_deepened": float(int((counts > n).sum())),
            "adaptive/overlap_extra_rows": float(extra),
            "adaptive/overlap_max_group_size": float(int(counts.max())),
        }
        logger.info(
            "[adaptive-n] overlap: %d carried prompt(s) expanded in-wave (+%d rows, max n=%d)",
            int((counts > n).sum()),
            extra,
            int(counts.max()),
        )
        return expanded, idxs, metrics

    def _compute_reward_colocate(self, batch: DataProto) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        """
        compute reward use colocate reward model
        """
        assert self.reward_loop_manager is not None, "RewardLoopManager is None"
        batch_reward = self.reward_loop_manager.compute_rm_score(batch)
        return batch_reward

    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            if self.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():
                # for colocate reward models, we need to sleep rollout model
                # to spare GPU memory for reward model
                self.checkpoint_manager.sleep_replicas()
                batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                # wake up rollout model
                # replace with wake_up method once supported
                self.checkpoint_manager.update_weights()

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            reward_tensor, reward_extra_info = extract_reward(test_batch)

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}
        if result_b is None:
            result_b = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            if self.use_legacy_worker_impl == "disable":
                # convert critic_cfg into TrainingWorkerConfig
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create reward loop manager
        from verl.experimental.reward_loop import RewardLoopManager

        # initalize reward loop manager
        # reward model (colocate or standalone): get resource_pool
        # no reward model: resource_pool = None
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
        self.async_rollout_manager = AgentLoopManager(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rollout_resource_pool=actor_rollout_resource_pool,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=self.async_rollout_manager.rollout_replicas,
        )

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, compute_loss=False)
            output = self.critic_wg.infer_batch(batch_td)
            output = output.get()
            values = tu.get(output, "values")
            values = no_padding_2_padding(values, batch_td)
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            values = self.critic_wg.compute_values(batch)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            metadata = {"calculate_entropy": False, "compute_loss": False}
            if self.ref_in_actor:
                metadata["no_lora_adapter"] = True
            tu.assign_non_tensor(batch_td, **metadata)
            if self.ref_in_actor:
                output = self.actor_rollout_wg.compute_log_prob(batch_td)
            else:
                output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            # gather output
            log_probs = tu.get(output, "log_probs")
            # step 4. No padding to padding
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        if self.use_legacy_worker_impl == "disable":
            # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            # gather output
            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            # step 4. No padding to padding
            entropy = no_padding_2_padding(entropy, batch_td)
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            old_log_prob_mfu = 0
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_output = tu.get(actor_output, "metrics")
            actor_output = rename_dict(actor_output, "actor/")
            # modify key name
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            actor_output = self.actor_rollout_wg.update_actor(batch)

        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            output = self.critic_wg.train_mini_batch(batch_td)
            output = output.get()
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            # modify key name
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(batch)
        return critic_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # v2.1(B): re-inject requeued (first-failure) prompts BEFORE uid
                # assignment so retries look like ordinary rows. The injected
                # count is a multiple of ppo_mini_batch_size, so B+r never
                # produces a ragged mini-batch split.
                batch, num_requeued = self._inject_requeued_prompts(batch)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )
                if num_requeued > 0:
                    # the LAST num_requeued rows are the retries; a second budget
                    # exhaustion on these uids drops the prompt for good.
                    retry_uids = batch.non_tensor_batch["uid"][-num_requeued:]
                    self._requeue_retry_uids.update(str(u) for u in retry_uids)
                    metrics["adaptive/requeue_retried"] = float(num_requeued)
                    metrics["adaptive/requeue_retried_total"] = float(self._requeue_retried_total)

                # v2.1(B): snapshot the full pre-pop rows so budget-exhausted
                # prompts can be requeued with everything generation needs.
                if (
                    self.adaptive_cfg.enable
                    and self.adaptive_cfg.trigger == "no_positive"
                    and self.adaptive_cfg.requeue_enable
                ):
                    self._requeue_snapshot = deepcopy(batch)
                else:
                    self._requeue_snapshot = None

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output, gen_prompt_index, expand_metrics = self._expand_gen_batch(
                    gen_batch, self.config.actor_rollout_ref.rollout.n
                )
                metrics.update(expand_metrics)
                if self.adaptive_cfg.enable and self.adaptive_cfg.inwave:
                    # The wave itself will come back ragged (the workers deepen
                    # starving groups in place); the row->prompt map is rebuilt
                    # from the group tag they attach, not from the input shape.
                    gen_prompt_index = None

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if curr_step_profile:
                            self.async_rollout_manager.start_profile()
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        # Per-prompt adaptive resampling: deepen the groups that came
                        # back with no score spread (and would therefore emit exactly
                        # zero GRPO gradient) by drawing MORE samples of the SAME
                        # prompt. Must run before sleep_replicas() -- it needs the
                        # rollout engines awake. No-op unless ADAPTIVE_N_ENABLE=1.
                        adaptive_prompt_index = None
                        if self.adaptive_cfg.enable:
                            if self.adaptive_cfg.inwave:
                                gen_prompt_index = self._inwave_prompt_index(gen_batch, gen_batch_output, metrics)
                            gen_batch_output, adaptive_prompt_index, adaptive_metrics = self._adaptive_deepen(
                                gen_batch, gen_batch_output, timing_raw, prompt_index=gen_prompt_index
                            )
                            metrics.update(adaptive_metrics)
                        # DAPO overlong filtering: drop truncated rollouts from the
                        # loss entirely (see agent_loop.py where they are flagged).
                        metrics.update(self._filter_truncated(gen_batch_output))

                        self.checkpoint_manager.sleep_replicas()
                        if curr_step_profile:
                            self.async_rollout_manager.stop_profile()

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if curr_step_profile:
                                self.async_rollout_manager.start_profile()
                            gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            self.checkpoint_manager.sleep_replicas()
                            if curr_step_profile:
                                self.async_rollout_manager.stop_profile()
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                batch_reward = self._compute_reward_colocate(batch)
                                batch = batch.union(batch_reward)

                            # Compute or extract reward for REMAX baseline
                            reward_baseline_tensor = batch.batch["rm_scores"].sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout.
                    # With adaptive resampling, row i of gen_batch_output may come from
                    # a different prompt than plain repeat-interleave implies, so use
                    # the index the deepening pass reports. In the default
                    # keep="subsample" mode that index IS repeat-interleave, making
                    # this branch identical to the line it replaces.
                    if adaptive_prompt_index is not None:
                        batch = batch.select_idxs(adaptive_prompt_index)
                    else:
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            batch_reward = self._compute_reward_colocate(batch)
                            batch = batch.union(batch_reward)

                        # extract reward_tensor and reward_extra_infos_dict for training
                        reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                router_mode = getattr(
                                    self.config.actor_rollout_ref.actor.router_replay, "mode", "disabled"
                                )
                                if router_mode == "R2":
                                    batch.batch.pop("routed_experts")
                                else:
                                    old_log_prob.batch.pop("routed_experts")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                        # Deepened groups must be scored against their K-sample
                        # baseline and carry their Horvitz-Thompson weight; every
                        # other row keeps the advantage computed just above,
                        # bit-for-bit. No-op when adaptive resampling is off.
                        if self.adaptive_cfg.enable:
                            batch = self._apply_adaptive_advantage(batch, norm_adv_by_std_in_grpo)

                        # v2.1(A): the always-on safety net. No row with a
                        # non-positive RAW reward may carry positive advantage --
                        # closes the penalty-gating residual hole (deepened group
                        # with one positive + many truncations has mean < 0, so
                        # raw-0 rows clear the mean) by construction, everywhere.
                        # Active whenever either feature is on; with both off it
                        # is a mathematical no-op for non-negative scorers and is
                        # skipped entirely.
                        if self.adaptive_cfg.clamp_zero_raw and (
                            self.adaptive_cfg.enable or self._overlong_driver_enabled
                        ):
                            batch, clamp_metrics = self._clamp_nonpositive_raw_advantage(batch)
                            metrics.update(clamp_metrics)

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)

                        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        # Check if the conditions for saving a checkpoint are met.
                        # The conditions include a mandatory condition (1) and
                        # one of the following optional conditions (2/3/4):
                        # 1. The save frequency is set to a positive value.
                        # 2. It's the last training step.
                        # 3. The current step number is a multiple of the save frequency.
                        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                        # update weights from trainer to rollout
                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights()

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # Dump longest responses when spike detected (for debugging response length peaks)
                dump_dir = self.config.trainer.get("dump_long_responses_dir", None)
                if dump_dir:
                    threshold = self.config.trainer.get("dump_long_responses_threshold", 20000)
                    top_k = self.config.trainer.get("dump_long_responses_top_k", 5)
                    self._dump_long_responses_on_spike(
                        batch=batch,
                        reward_extra_infos_dict=reward_extra_infos_dict,
                        metrics=metrics,
                        dump_dir=dump_dir,
                        threshold_tokens=threshold,
                        top_k=top_k,
                    )
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                # Durable per-step metrics sink (2026-08-11): wandb-offline buffers
                # history in the service process and only writes it on graceful
                # finish, so TIMEOUT/OOM-killed runs lose every step's metrics
                # (observed: 3 completed step-1 runs, 0 history rows recoverable).
                # Append one JSON line per step next to the rollout dumps instead.
                try:
                    _mdir = os.environ.get("FS_METRICS_DIR") or os.path.join(
                        self.config.trainer.get("rollout_data_dir", "") or ".", ""
                    )
                    if _mdir and _mdir != ".":
                        os.makedirs(_mdir, exist_ok=True)
                        with open(os.path.join(_mdir, "metrics.jsonl"), "a") as _mf:
                            _mf.write(
                                json.dumps(
                                    {"step": self.global_steps, **{k: v for k, v in metrics.items()}},
                                    default=float,
                                )
                                + "\n"
                            )
                except Exception as _me:  # never let the sink kill training
                    print(f"[metrics-sink] write failed (non-fatal): {_me}")

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
