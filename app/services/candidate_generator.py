"""Candidate relationship generator filtering candidate pairs using strict regulatory signals."""
import re
from typing import Any, Dict, List, Optional, Tuple
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


def is_metadata_text(text: str) -> bool:
    """Detect if text is pure document metadata or header."""
    t = text.strip()
    if len(t) < 20:
        return True
    return bool(
        re.search(
            r"^(?:date:|department:|download ref|circular ref|national stock|page \d+|cin\s*:|tel no|to all members|yours faithfully|for national stock|regd\.? office)",
            t,
            re.IGNORECASE,
        )
    )


class CandidateGenerator:
    """Generates and scores candidate relationship pairs between source and target facts using strict regulatory anchors."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.top_k = self.settings.RELATIONSHIP_TOP_K
        self.max_candidates_per_source = 2  # Strict fan-out protection

    def generate_candidates(
        self,
        source_facts: List[Fact],
        target_facts: List[Fact],
        references: Optional[List[DocumentReference]] = None,
        source_chunks: Optional[List[Chunk]] = None,
        target_chunks: Optional[List[Chunk]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate high-confidence candidate pairs between source facts and target facts."""
        if not source_facts or not target_facts:
            return []

        scored_candidates: List[Dict[str, Any]] = []
        has_explicit_ref = bool(references and len(references) > 0)

        # Build lookup maps for chunk content if available
        src_chunk_map = {c.chunk_id: c for c in (source_chunks or [])}
        tgt_chunk_map = {c.chunk_id: c for c in (target_chunks or [])}

        # Track candidate count per source fact to prevent 1-to-N fan-out
        source_fact_matches: Dict[str, List[Dict[str, Any]]] = {}

        for sf in source_facts:
            if is_metadata_text(sf.source_text) or is_metadata_text(sf.object):
                continue

            sf_chunk = src_chunk_map.get(sf.chunk_id)
            sf_full_text = f"{sf.source_text} {sf.object} {sf_chunk.content if sf_chunk else ''}"

            for tf in target_facts:
                if is_metadata_text(tf.source_text) or is_metadata_text(tf.object):
                    continue

                # Same fact cannot amend itself
                if sf.fact_id == tf.fact_id or sf.document_id == tf.document_id:
                    continue

                tf_chunk = tgt_chunk_map.get(tf.chunk_id)
                tf_full_text = f"{tf.source_text} {tf.object} {tf_chunk.content if tf_chunk else ''}"

                score = 0.0
                reasons: List[str] = []
                has_hard_anchor = False

                # Signal 1: Exact Rule Reference Match (e.g. "Rule 8(1)(f)" == "Rule 8(1)(f)")
                if sf.rule_reference and tf.rule_reference and sf.rule_reference.lower() == tf.rule_reference.lower():
                    score += 0.60
                    has_hard_anchor = True
                    reasons.append(f"rule_match({sf.rule_reference})")
                elif sf.rule_reference and tf.rule_reference and sf.rule_reference.lower() != tf.rule_reference.lower():
                    # Heavy penalty if rules explicitly conflict
                    score -= 0.50
                    reasons.append("rule_conflict")

                # Signal 2: Exact Point / Clause Reference Match (e.g. "Point 1" == "Point 1")
                src_clause = sf.clause_reference or self._extract_clause_id(sf_full_text)
                tgt_clause = tf.clause_reference or self._extract_clause_id(tf_full_text)
                if src_clause and tgt_clause:
                    if src_clause.lower() == tgt_clause.lower():
                        score += 0.50
                        has_hard_anchor = True
                        reasons.append(f"clause_match({src_clause})")
                    else:
                        # Heavy penalty if points explicitly conflict (e.g. Point 1 vs Point 9)
                        score -= 0.60
                        reasons.append(f"clause_conflict({src_clause}_vs_{tgt_clause})")

                # Signal 3: Explicit amendment keywords citing target
                kw_matches = KEYWORD_REGEX.findall(sf_full_text)
                if kw_matches:
                    score += 0.20
                    reasons.append(f"keywords({','.join(set(kw_matches[:2]))})")

                # Signal 4: Document-level citation
                if has_explicit_ref:
                    score += 0.20
                    reasons.append("document_ref")

                # Signal 5: Subject / Entity overlap
                if sf.subject and tf.subject:
                    sf_sub = sf.subject.lower().strip()
                    tf_sub = tf.subject.lower().strip()
                    is_generic = sf_sub in {
                        "company", "securities", "market", "date", "trading member", 
                        "member", "all members", "intermediary", "intermediaries", 
                        "random entity", "listed companies"
                    }
                    if sf_sub == tf_sub:
                        if has_explicit_ref and not is_generic:
                            score += 0.35
                            # Subject match acts as anchor when there is also text overlap
                            reasons.append(f"subject_match_ref({sf.subject[:25]})")
                        else:
                            score += 0.15
                            reasons.append(f"subject_match({sf.subject[:25]})")
                    elif not is_generic and (sf_sub in tf_sub or tf_sub in sf_sub):
                        if has_explicit_ref:
                            score += 0.30
                            reasons.append(f"subject_containment_ref({sf.subject[:25]})")
                        else:
                            score += 0.10
                            reasons.append("subject_partial")

                # Signal 6: Lexical Jaccard similarity
                src_words = set(re.findall(r"\b\w{3,}\b", (sf.object or sf.source_text).lower()))
                tgt_words = set(re.findall(r"\b\w{3,}\b", (tf.object or tf.source_text).lower()))
                jaccard = 0.0
                if src_words and tgt_words:
                    jaccard = len(src_words & tgt_words) / len(src_words | tgt_words)
                    if jaccard > 0.25:
                        score += min(0.25, jaccard * 0.30)
                        reasons.append(f"jaccard({jaccard:.2f})")

                # If we have a non-generic subject match/containment with direct ref AND text similarity, enable hard anchor
                if any("subject_match_ref" in r or "subject_containment_ref" in r for r in reasons) and jaccard >= 0.20:
                    has_hard_anchor = True

                # Strict Gate: Candidate MUST have a hard anchor (rule/clause match or score >= 0.55)
                decision = "ACCEPT" if (has_hard_anchor and score >= 0.55) or score >= 0.70 else "REJECT"
                
                logger.debug(
                    "[AMENDMENT_CANDIDATE] new_fact=%s (%s) | old_fact=%s (%s) | score=%.2f | decision=%s | reasons=%s",
                    sf.fact_id[:8],
                    sf.clause_reference or sf.rule_reference or "N/A",
                    tf.fact_id[:8],
                    tf.clause_reference or tf.rule_reference or "N/A",
                    score,
                    decision,
                    ",".join(reasons),
                )

                if decision == "ACCEPT":
                    candidate_entry = {
                        "source_id": sf.fact_id,
                        "target_id": tf.fact_id,
                        "source_text": sf.source_text or sf.object,
                        "target_text": tf.source_text or tf.object,
                        "source_fact": sf,
                        "target_fact": tf,
                        "score": round(score, 3),
                        "reasons": reasons,
                    }
                    source_fact_matches.setdefault(sf.fact_id, []).append(candidate_entry)

        # Enforce strict per-source cardinality cap (max 2 candidates per new fact)
        for sf_id, matches in source_fact_matches.items():
            matches.sort(key=lambda x: x["score"], reverse=True)
            for m in matches[: self.max_candidates_per_source]:
                scored_candidates.append(m)

        # Sort overall list by score descending
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        top_candidates = scored_candidates[: self.top_k]

        logger.info(
            "Generated %d high-confidence candidate pairs (strict threshold >= 0.60, max %d/fact, hard anchor required)",
            len(top_candidates),
            self.max_candidates_per_source,
        )
        return top_candidates

    def _extract_clause_id(self, text: str) -> Optional[str]:
        """Extract standardized point or clause identifier from text snippet."""
        m = re.search(r"\b(?:Point(?:\s*No\.?)?|Clause)\s*(\d+[a-zA-Z]?)", text, re.IGNORECASE)
        if m:
            return f"Point {m.group(1)}"
        m_num = re.search(r"^\s*(\d+)\.\s+", text)
        if m_num:
            return f"Point {m_num.group(1)}"
        return None

