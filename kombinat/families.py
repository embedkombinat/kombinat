"""Model-family derivation for judge-diversity claim steering.

Two annotations only count as independent evidence when they come from
different model families — the annotator fleet decodes deterministically
(temperature 0), so a same-family repeat is a re-run, not a second opinion.
The claim path uses this mapping to route each judge toward pairs its
family hasn't labeled yet.
"""

from __future__ import annotations

import re

# Coarse family keywords, matched against the lowercased basename of a
# HuggingFace model id. Basename (not org) so quantized re-uploads land in
# the same family: "Qwen/Qwen2.5-7B-Instruct-AWQ" and
# "mlx-community/Qwen2.5-7B-Instruct-4bit" are both "qwen".
KNOWN_FAMILIES = ("qwen", "mistral", "phi", "llama", "gemma", "deepseek")


def family_sql_pattern(family: str) -> str:
    """Postgres `~*` pattern matching a family the same way model_family does.

    `\\m` (start-of-word) mirrors the prefix word boundary used in Python;
    SQL callers must apply it to the basename of the stored model_id so the
    two sides of steering never disagree about what counts as a family.
    """
    return rf"\m{family}"


def model_family(model_id: str | None) -> str | None:
    """Derive a coarse model family from a HuggingFace model id.

    Families match only at the start of a word within the basename: a bare
    substring test would classify "dolphin-2.9-llama3-8b" as "phi"
    (dol-PHI-n) instead of "llama". A trailing boundary is deliberately not
    required so version suffixes still match ("qwen2.5", "llama3",
    "phi-3.5"). Returns None when no known family matches — callers skip
    steering rather than guess.
    """
    if not model_id:
        return None
    basename = model_id.rsplit("/", 1)[-1].lower()
    for family in KNOWN_FAMILIES:
        if re.search(rf"\b{family}", basename):
            return family
    return None
