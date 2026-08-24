"""Candidate relationship generator filtering candidate pairs using deterministic signals."""
import re
from typing import Any, Dict, List, Optional
from app.core.config import get_settings
from app.core.logging import logger
from app.models.chunk import Chunk
from app.models.fact import Fact
from app.models.reference import DocumentReference

REGULATORY_KEYWORDS = [
    r"\bsupersed\w*",
    r"\bamend\w*",
    r"\bmodifi\w*",
    r"\brevis\w*",
    r"\bclarifi\w*",
    r"\bclarification\b",
    r"\breplac\w*",
    r"\bsubstitut\w*",
    r"\bin partial modification of\b",
    r"\bread with\b",
    r"\bsubject to\b",
    r"\bexcept\b",
    r"\bnotwithstanding\b",
    r"\bshall apply\b",
    r"\bshall cease\b",
    r"\bwithdrawn\b",
    r"\brescinded\b",
    r"\bstands? deleted\b",
    r"\bdelet\w*",
]

KEYWORD_REGEX = re.compile("|".join(REGULATORY_KEYWORDS), re.IGNORECASE)


class CandidateGenerator:
    """Generates and scores candidate relationship pairs between source and target facts/clauses."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.top_k = self.settings.RELATIONSHIP_TOP_K

    def generate_candidates(
        self,
        source_facts: List[Fact],
        target_facts: List[Fact],
        references: Optional[List[DocumentReference]] = None,
        source_chunks: Optional[List[Chunk]] = None,
        target_chunks: Optional[List[Chunk]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate top-K scored candidate pairs between source facts and target facts."""
        if not source_facts or not target_facts:
            return []

        scored_candidates: List[Dict[str, Any]] = []
        has_explicit_ref = bool(references and len(references) > 0)

        # Build lookup maps for chunk content if available
        src_chunk_map = {c.chunk_id: c for c in (source_chunks or [])}
        tgt_chunk_map = {c.chunk_id: c for c in (target_chunks or [])}

        for sf in source_facts:
            sf_chunk = src_chunk_map.get(sf.chunk_id)
            sf_full_text = f"{sf.source_text} {sf.object} {sf_chunk.content if sf_chunk else ''}"

            for tf in target_facts:
                tf_chunk = tgt_chunk_map.get(tf.chunk_id)
                tf_full_text = f"{tf.source_text} {tf.object} {tf_chunk.content if tf_chunk else ''}"

                score = 0.0
                reasons: List[str] = []

                # Signal 1: Explicit cross-document citation match
                if has_explicit_ref:
                    score += 0.35
                    reasons.append("explicit_citation")

                # Signal 2: Regulatory change / evolution keywords in source text
                kw_matches = KEYWORD_REGEX.findall(sf_full_text)
                if kw_matches:
                    score += 0.30 + min(0.15, len(kw_matches) * 0.05)
                    reasons.append(f"keywords({','.join(set(kw_matches[:3]))})")

                # Signal 3: Subject / Entity overlap
                if sf.subject and tf.subject and sf.subject.lower() == tf.subject.lower():
                    score += 0.20
                    reasons.append("subject_match")

                # Signal 4: Specific Point / Clause number match in text
                src_clause_match = re.search(r"\b(?:point|clause|item|section|annexure)\s*([0-9a-zA-Z]+)", sf_full_text, re.IGNORECASE)
                tgt_clause_match = re.search(r"\b(?:point|clause|item|section|annexure)\s*([0-9a-zA-Z]+)", tf_full_text, re.IGNORECASE)
                if src_clause_match and tgt_clause_match and src_clause_match.group(1).lower() == tgt_clause_match.group(1).lower():
                    score += 0.40
                    reasons.append(f"clause_match({src_clause_match.group(1)})")

                # Signal 5: Lexical term overlap
                src_words = set(re.findall(r"\b\w{4,}\b", sf.object.lower()))
                tgt_words = set(re.findall(r"\b\w{4,}\b", tf.object.lower()))
                if src_words and tgt_words:
                    jaccard = len(src_words & tgt_words) / len(src_words | tgt_words)
                    score += jaccard * 0.25
                    if jaccard > 0.2:
                        reasons.append(f"jaccard({jaccard:.2f})")

                # Filter: Only keep candidates that have meaningful signals (score >= 0.3)
                if score >= 0.3:
                    scored_candidates.append({
                        "source_id": sf.fact_id,
                        "target_id": tf.fact_id,
                        "source_text": sf.source_text or sf.object,
                        "target_text": tf.source_text or tf.object,
                        "source_fact": sf,
                        "target_fact": tf,
                        "score": round(score, 3),
                        "reasons": reasons,
                    })

        # Sort by candidate relevance score descending
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)

        # Select Top-K candidates
        top_candidates = scored_candidates[: self.top_k]
        logger.info(
            "Generated %d total candidate pairs, filtered to Top-%d (score threshold >= 0.3)",
            len(scored_candidates),
            len(top_candidates),
        )
        return top_candidates
