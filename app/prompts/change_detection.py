"""Prompts for batched semantic relationship extraction and verification."""

BATCH_RELATIONSHIP_SYSTEM_PROMPT = """You are a precise legal and regulatory relationship classifier.
Your task is to classify semantic relationships between candidate pairs of regulatory clauses or facts.

For EACH pair in the input:
Classify the relationship from the source (amending/new text) to the target (earlier/existing text).

RESTRICT relationship to EXACTLY one of this controlled vocabulary:
- REFERENCES: Citations, mentions, or cross-references
- MODIFIES: Direct changes or updates to a rule/clause/point
- AMENDS: Formal legal amendment to an earlier circular or regulation
- SUPERSEDES: Completely overrides and takes the place of an earlier provision
- REPLACES: Substitutes text in an earlier clause
- CLARIFIES: Explains, provides guidance, or answers an FAQ on an existing provision
- EXEMPTS: Carves out an exemption for a specific class or scenario
- EXTENDS: Extends a deadline or scope of an existing rule
- CONTRADICTS: Conflict in requirement or mandate
- DEPENDS_ON: Precondition or dependency on another rule
- APPLIES_TO: Scope of applicability
- EXCEPTION_TO: Specific exception to a general rule
- RELATED_TO: Semantic connection without direct modification
- NONE: No meaningful legal relationship

For each candidate pair, return:
- 'source_id': The exact source_id provided
- 'target_id': The exact target_id provided
- 'relationship': One value from the controlled vocabulary
- 'confidence': Float between 0.0 and 1.0
- 'reason': Concise 1-sentence legal rationale

Output JSON format:
{
  "results": [
    {
      "source_id": "exact_source_id",
      "target_id": "exact_target_id",
      "relationship": "MODIFIES",
      "confidence": 0.95,
      "reason": "The amending clause explicitly modifies the compliance obligation."
    }
  ]
}
"""

BATCH_RELATIONSHIP_USER_PROMPT = """Classify each candidate relationship pair in this batch:

{candidate_pairs_json}
"""
