"""Model-family derivation for judge-diversity claim steering.

Two annotations only count as independent evidence when they come from
different model families — the annotator fleet decodes deterministically
(temperature 0), so a same-family repeat is a re-run, not a second opinion.
The claim path uses this mapping to route each judge toward pairs its
family hasn't labeled yet.
"""

from __future__ import annotations

# Coarse family keywords, matched against the lowercased basename of a
# HuggingFace model id. Basename (not org) so quantized re-uploads land in
# the same family: "Qwen/Qwen2.5-7B-Instruct-AWQ" and
# "mlx-community/Qwen2.5-7B-Instruct-4bit" are both "qwen".
KNOWN_FAMILIES = ("qwen", "mistral", "phi", "llama", "gemma", "deepseek")


def model_family(model_id: str | None) -> str | None:
    """Derive a coarse model family from a HuggingFace model id.

    Returns None when no known family matches — callers skip steering
    rather than guess.
    """
    if not model_id:
        return None
    basename = model_id.rsplit("/", 1)[-1].lower()
    for family in KNOWN_FAMILIES:
        if family in basename:
            return family
    return None
