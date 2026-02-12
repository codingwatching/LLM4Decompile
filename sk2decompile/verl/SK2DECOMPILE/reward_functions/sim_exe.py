"""
Reference reward function: Compilability + Word-level Jaccard Similarity.

This is a reference implementation of an alternative Structure Recovery reward
for the SK2Decompile RL training pipeline (arXiv:2509.22114, Section 3.5).

Evaluates decompiled C code by:
1. Computing word-level Jaccard similarity between candidate and ground truth
2. Checking if the code compiles with gcc (compilability score: 0 or 1)

Final score = jaccard_similarity + compilability_score if jaccard > 0.5, else 0.
"""

import os
import subprocess
import tempfile


def compute_score(solution_str, ground_truth, extra_info=None):
    sim_score = jaccard_similarity(solution_str, ground_truth)
    compile_score = compileable_score(solution_str, ground_truth, extra_info)

    if sim_score > 0.5:
        return sim_score + compile_score
    return 0


def jaccard_similarity(str1, str2):
    """Compute word-level Jaccard similarity between two strings."""
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    if union == 0:
        return 0.0
    return intersection / union


def compileable_score(solution_str, ground_truth, extra_info=None):
    """
    Check if the candidate C code compiles with gcc.

    Args:
        extra_info: Optional dict with 'header' key containing C header declarations.

    Returns:
        1.0 if compilable, 0.0 otherwise.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            source_file = os.path.join(tmpdir, "temp.c")
            object_file = os.path.join(tmpdir, "temp.o")
            header = extra_info.get('header', '') if extra_info else ''

            with open(source_file, 'w') as f:
                f.write(f'{header}\n\n{solution_str}')

            proc = subprocess.run(
                ['gcc', '-c', source_file, '-o', object_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
                check=True
            )
            return 1.0 if proc.returncode == 0 else 0.0
        except Exception:
            return 0.0
