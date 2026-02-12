"""
Reference reward function: Qwen3 Embedding-based Identifier Similarity.

This is a reference implementation of the Identifier Naming reward (Eq. 4)
described in the SK2Decompile paper (arXiv:2509.22114, Section 3.5).

Evaluates decompiled C code by:
1. Using tree-sitter to parse C code and extract identifiers (func/var/type/field)
2. Building a naming summary string per code sample
3. Computing cosine similarity between Qwen3 embeddings of the two summaries
4. Squaring the similarity score to sharpen the reward signal

Final score = cosine_similarity^2

Requires:
- A running OpenAI-compatible embedding server (e.g., vLLM serving Qwen3-Embedding-0.6B)
- tree-sitter and tree-sitter-c packages

Environment variables:
- QWEN3_EMBEDDING_MODEL_PATH: Model name/path (default: "Qwen3-Embedding-0.6B")
- QWEN3_EMBEDDING_API_KEY or OPENAI_API_KEY: API key (default: "none")
- QWEN3_EMBEDDING_API_BASE: API base URL (default: "http://127.0.0.1:8000/v1")
"""

import math
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

from openai import OpenAI
from tree_sitter import Language, Parser
import tree_sitter_c as tsc

# ---- OpenAI Embedding Client ----

_MODEL_NAME = os.getenv("QWEN3_EMBEDDING_MODEL_PATH", "Qwen3-Embedding-0.6B")
_API_KEY = os.getenv("QWEN3_EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY") or "none"
_API_BASE = os.getenv("QWEN3_EMBEDDING_API_BASE", "http://127.0.0.1:8000/v1")
_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if _API_BASE:
            _client = OpenAI(api_key=_API_KEY, base_url=_API_BASE)
        elif _API_KEY:
            _client = OpenAI(api_key=_API_KEY)
        else:
            _client = OpenAI()
    return _client


def _embed_two(text_a: str, text_b: str) -> Tuple[List[float], List[float]]:
    """Embed two texts in a single API call, return their embedding vectors."""
    client = _get_client()
    resp = client.embeddings.create(model=_MODEL_NAME, input=[text_a, text_b])
    emb_a = [float(x) for x in resp.data[0].embedding]
    emb_b = [float(x) for x in resp.data[1].embedding]
    return emb_a, emb_b


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---- Tree-sitter C: Identifier Extraction ----

C_LANG = Language(tsc.language())
_TS_PARSER = Parser(C_LANG)


def _classify_node(node):
    """
    Classify a tree-sitter node into identifier categories:
    - func: function names (definitions + calls)
    - var: variable names (parameters / local / global)
    - type: type names
    - field: struct field names
    """
    node_type = node.type
    name = node.text.decode("utf8")

    if node_type == "type_identifier":
        return "type", name
    if node_type == "field_identifier":
        return "field", name
    if node_type != "identifier":
        return None, None

    parent = node.parent
    if parent:
        parent_type = parent.type
        if parent_type == "function_declarator" and parent.child_by_field_name("declarator") == node:
            return "func", name
        if parent_type == "call_expression" and parent.child_by_field_name("function") == node:
            return "func", name
        if parent_type in ("init_declarator", "parameter_declaration", "declaration", "pointer_declarator"):
            return "var", name

    return "var", name


def _extract_identifiers_ts(code: str, max_per_type: int = 64) -> Dict[str, List[str]]:
    """Extract identifiers from C code using tree-sitter, classified by type."""
    tree = _TS_PARSER.parse(code.encode("utf8"))
    result: Dict[str, List[str]] = {"func": [], "var": [], "type": [], "field": []}

    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        id_type, name = _classify_node(node)
        if id_type in result and len(result[id_type]) < max_per_type:
            result[id_type].append(name)
        stack.extend(node.children)

    return result


# ---- Summary Construction & Similarity ----


def _build_summary_text(identifiers: Dict[str, List[str]], max_per_type: int = 64) -> str:
    """
    Build a naming summary string from classified identifiers.
    Example: "func: foo bar || type: my_type || field: field1 field2 || var: i j k"
    """
    parts: List[str] = []
    for kind in ("func", "type", "field", "var"):
        names = identifiers.get(kind, [])
        if not names:
            continue
        segment = f"{kind}: " + " ".join(names[:max_per_type])
        parts.append(segment)
    return " || ".join(parts)


def _identifier_similarity_ts(candidate_text: str, reference_text: str):
    """
    Compute identifier-level similarity using embedding cosine similarity.

    Steps:
    1. Extract identifiers from both texts using tree-sitter
    2. Build naming summary strings
    3. Embed both summaries in a single API call
    4. Return cosine similarity as name_score

    Returns:
        name_score: float in [0, 1]
    """
    cand_ids = _extract_identifiers_ts(candidate_text)
    ref_ids = _extract_identifiers_ts(reference_text)

    cand_summary = _build_summary_text(cand_ids)
    ref_summary = _build_summary_text(ref_ids)

    if not cand_summary or not ref_summary:
        return 0.0

    emb_cand, emb_ref = _embed_two(cand_summary, ref_summary)
    return _cosine_similarity(emb_cand, emb_ref)


# ---- Main Reward Function ----


def compute_score(solution_str, ground_truth, extra_info=None):
    """
    Compute reward based on identifier naming similarity using Qwen3 embeddings.
    Returns score^2 to sharpen the reward signal.
    """
    if not isinstance(solution_str, str):
        solution_str = "" if solution_str is None else str(solution_str)
    if not isinstance(ground_truth, str):
        ground_truth = "" if ground_truth is None else str(ground_truth)

    candidate_text = solution_str.strip()
    reference_text = ground_truth.strip()

    if not candidate_text or not reference_text:
        return 0.0

    name_score = _identifier_similarity_ts(candidate_text, reference_text)
    return name_score * name_score
