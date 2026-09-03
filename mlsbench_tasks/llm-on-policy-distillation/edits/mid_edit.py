"""Mid-edit for the llm-on-policy-distillation task.

Wires TRL's GKDTrainer to dispatch its distillation loss into an editable
file (``custom_distill_loss.py``).

Three changes to ``trl/experimental/gkd/gkd_trainer.py``:

  1. Add an import of ``compute_distill_loss`` near the top.
  2. Force ``use_liger_gkd_loss = False`` at the start of ``compute_loss``
     so the standard (editable) path runs.
  3. Replace the ``self.generalized_jsd_loss(...)`` call with a
     ``compute_distill_loss(...)`` call and emit a TRAIN_METRICS line for
     parser feedback.

Line numbers refer to the original TRL v1.4.0 source. Ops are listed
bottom-to-top so prior ops do not shift line numbers downstream.
"""

from pathlib import Path

_TEMPLATE = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE.read_text()

_FILE = "trl/trl/experimental/gkd/gkd_trainer.py"

_IMPORT_LINE = "from .custom_distill_loss import compute_distill_loss  # MLS-Bench editable loss"

_DISABLE_LIGER = (
    "        # MLS-Bench: force the standard (editable) path; ignore the fused Liger kernel.\n"
    "        self.use_liger_gkd_loss = False"
)

_CALL_PATCH = (
    "            # MLS-Bench: align student/teacher vocab. Qwen2.5-Math-7B vocab\n"
    "            # is 152064 (multiple of 128 for tensor-parallel padding) while\n"
    "            # Qwen2.5-0.5B base is 151936. The trailing rows of the teacher\n"
    "            # head are unused embeddings, so dropping them is safe.\n"
    "            _vs = shifted_student_logits.size(-1)\n"
    "            if shifted_teacher_logits.size(-1) != _vs:\n"
    "                shifted_teacher_logits = shifted_teacher_logits[..., :_vs]\n"
    "            # MLS-Bench: dispatch to the editable compute_distill_loss.\n"
    "            loss = compute_distill_loss(\n"
    "                student_logits=shifted_student_logits,\n"
    "                teacher_logits=shifted_teacher_logits,\n"
    "                labels=shifted_labels,\n"
    "                beta=self.beta,\n"
    "                temperature=self.temperature,\n"
    "                reduction=\"batchmean\",\n"
    "                step=int(getattr(self.state, \"global_step\", 0)),\n"
    "                total_steps=int(\n"
    "                    getattr(self.state, \"max_steps\", 0)\n"
    "                    or getattr(self.args, \"max_steps\", 0)\n"
    "                    or 1\n"
    "                ),\n"
    "                lmbda=float(getattr(self, \"lmbda\", 0.5)),\n"
    "            )\n"
    "            # MLS-Bench feedback line (parsed by tasks/llm-on-policy-distillation/parser.py)\n"
    "            try:\n"
    "                _step = int(getattr(self.state, \"global_step\", 0))\n"
    "                _log_every = max(1, int(getattr(self.args, \"logging_steps\", 10) or 10))\n"
    "                if _step % _log_every == 0:\n"
    "                    _lr = float(self._get_learning_rate()) if hasattr(self, \"_get_learning_rate\") else 0.0\n"
    "                    print(\n"
    "                        f\"TRAIN_METRICS step={_step} loss={float(loss.detach()):.6f} \"\n"
    "                        f\"lr={_lr:.3e} lmbda={float(getattr(self, 'lmbda', 0.0)):.3f} \"\n"
    "                        f\"beta={float(getattr(self, 'beta', 0.0)):.3f}\",\n"
    "                        flush=True,\n"
    "                    )\n"
    "            except Exception:\n"
    "                pass"
)


OPS = [
    # 1. Create the editable file.
    {
        "op": "create",
        "file": "trl/trl/experimental/gkd/custom_distill_loss.py",
        "content": _CUSTOM_PY,
    },
    # 2. (bottom of gkd_trainer.py) Replace the call site: lines 388–393 are the
    #    `loss = self.generalized_jsd_loss(...)` block in TRL v1.4.0.
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 388,
        "end_line": 393,
        "content": _CALL_PATCH,
    },
    # 3. Inject the Liger-bypass right after `def compute_loss(...):` (line 295).
    {
        "op": "insert",
        "file": _FILE,
        "after_line": 295,
        "content": _DISABLE_LIGER,
    },
    # 4. Inject the import after the existing imports (line 43 is the last
    #    `from .gkd_config import GKDConfig` line).
    {
        "op": "insert",
        "file": _FILE,
        "after_line": 43,
        "content": _IMPORT_LINE,
    },
]
