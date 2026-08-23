#!/usr/bin/env python3
"""Launch vLLM after applying local compatibility patches."""

from __future__ import annotations


def patch_transformers_tokenizer_for_vllm() -> None:
    from transformers import PreTrainedTokenizerBase

    if hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        return

    @property
    def all_special_tokens_extended(self):
        seen = set()
        tokens = []

        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            token = self._special_tokens_map.get(attr)
            if token is not None and str(token) not in seen:
                tokens.append(token)
                seen.add(str(token))

        for token in getattr(self, "_extra_special_tokens", []):
            if str(token) not in seen:
                tokens.append(token)
                seen.add(str(token))

        return tokens

    PreTrainedTokenizerBase.all_special_tokens_extended = all_special_tokens_extended


def main() -> int:
    patch_transformers_tokenizer_for_vllm()
    from vllm.entrypoints.cli.main import main as vllm_main

    return vllm_main()


if __name__ == "__main__":
    raise SystemExit(main())
