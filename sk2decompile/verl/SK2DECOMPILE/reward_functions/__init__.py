"""
SK2Decompile — Reference Reward Functions for GRPO Training.

This module provides reference implementations of reward functions used in the
SK2Decompile RL training pipeline. These are example implementations that
demonstrate the reward design described in Section 3.5 of the paper:

  SK2Decompile: LLM-based Two-Phase Binary Decompilation from Skeleton to Skin
  (arXiv:2509.22114)

Reference implementations:
- exe_type: Compilability + placeholder identifier Jaccard similarity
- sim_exe: Compilability + word-level Jaccard similarity
- embedding_gte: Tree-sitter identifier extraction + GTE embedding cosine similarity
- embedding_qwen3: Tree-sitter identifier extraction + Qwen3 embedding cosine similarity

To integrate into VERL, copy these files into verl/utils/reward_score/ and
register routing branches in __init__.py. See README.md for details.
"""

from . import exe_type, sim_exe, embedding_gte, embedding_qwen3

__all__ = ["exe_type", "sim_exe", "embedding_gte", "embedding_qwen3"]
