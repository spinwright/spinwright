"""Model IDs by tier (SPEC §4.2 + Modification 5).

Reasoning: extraction, bottleneck identification, patch proposal.
Classification: candidate ranking, eligibility second-pass.
Summarization: compressing large profiler output before reasoning sees it.
"""

REASONING = "claude-opus-4-7"
CLASSIFICATION = "claude-sonnet-4-6"
SUMMARIZATION = "claude-haiku-4-5-20251001"

DEFAULT_MAX_TOKENS = 8192
