"""Deterministic 1-to-1 fact matcher with component-level comparison and explainability."""
import difflib
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.models.fact import Fact, FactPredicate, normalize_text
from app.services.fact_extractor import validate_fact_evidence
from evaluation.models import (
    ErrorCategory,
    GoldFact,
    MatchComponentScores,
    MatchDecision,
    RegulatoryImpact,
)

# Standard abbreviation and entity normalization map
SYNONYM_MAP = {
    "miis": "market infrastructure institutions",
    "mii": "market infrastructure institutions",
    "exchanges": "stock exchanges",
    "exchange": "stock exchanges",
    "depositories": "depositories",
    "depository": "depositories",
    "amc": "asset management companies",
    "amcs": "asset management companies",
    "sebi-ipef": "investor protection and education fund",
    "sebi ipef": "investor protection and education fund",
    "ipef": "investor protection and education fund",
    "tft": "trade for trade",
    "fdsrl": "financial disincentives surveillance related lapses",
    "nclt": "national company law tribunal",
    "cdr": "corporate debt restructuring",
}


def clean_normalized_text(text: Optional[str]) -> str:
    """Normalize text by lowering, removing punctuation, and expanding common acronyms."""
    if not text:
        return ""
    # Collapse whitespace and lowercase
    s = normalize_text(text)
    # Remove surrounding quotes, brackets, and terminal punctuation
    s = re.sub(r"[^\w\s\d]", " ", s)
    tokens = s.split()
    # Expand synonyms
    expanded = [SYNONYM_MAP.get(t, t) for t in tokens]
    return " ".join(expanded)


def token_similarity(text1: Optional[str], text2: Optional[str]) -> float:
    """Compute token-level Jaccard similarity combined with character sequence ratio."""
    norm1 = clean_normalized_text(text1)
    norm2 = clean_normalized_text(text2)

    if not norm1 and not norm2:
        return 1.0
    if not norm1 or not norm2:
        return 0.0

    if norm1 == norm2:
        return 1.0

    # Token Jaccard
    tokens1 = set(norm1.split())
    tokens2 = set(norm2.split())
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    jaccard = len(intersection) / len(union) if union else 0.0

    # Substring / Containment bonus
    containment = 0.0
    if norm1 in norm2 or norm2 in norm1:
        shorter = min(len(norm1), len(norm2))
        longer = max(len(norm1), len(norm2))
        containment = shorter / longer if longer > 0 else 0.0

    # Sequence matcher ratio
    seq_ratio = difflib.SequenceMatcher(None, norm1, norm2).ratio()

    return max(jaccard, containment * 0.9, seq_ratio)


def normalize_clause_ref(clause: Optional[str]) -> str:
    """Normalize clause strings (e.g. 'Clause 1.1.1' -> '1.1.1', 'Point 2' -> '2')."""
    if not clause:
        return ""
    s = str(clause).strip().lower()
    s = re.sub(r"^(clause|point|section|para|paragraph|rule)\s*", "", s)
    return s.strip()


def clauses_align(c1: Optional[str], c2: Optional[str]) -> bool:
    """Check if two clause references are compatible."""
    if not c1 or not c2:
        return True  # If either is absent, do not penalize clause match
    norm1 = normalize_clause_ref(c1)
    norm2 = normalize_clause_ref(c2)
    if norm1 == norm2:
        return True
    # Sub-clause alignment (e.g. 1.1 matches 1.1.1)
    if norm1.startswith(norm2) or norm2.startswith(norm1):
        return True
    return False


def compare_qualifier_values(gold_val: Optional[str], pred_val: Optional[str]) -> float:
    """Compare single qualifier strings (threshold, time, condition)."""
    g_clean = clean_normalized_text(gold_val)
    p_clean = clean_normalized_text(pred_val)

    # Both empty/none -> perfect preservation
    if not g_clean and not p_clean:
        return 1.0
    # Gold required qualifier but prediction dropped it
    if g_clean and not p_clean:
        return 0.0
    # Prediction added qualifier not in gold (slight penalty or allowed)
    if not g_clean and p_clean:
        return 0.7  # minor discrepancy

    # Extract digits/amounts for threshold/time strictness
    g_nums = re.findall(r"\b\d+\b", g_clean)
    p_nums = re.findall(r"\b\d+\b", p_clean)
    if g_nums and p_nums:
        if set(g_nums) == set(p_nums):
            return 0.9 + 0.1 * token_similarity(g_clean, p_clean)
        return 0.2  # numbers do not match

    return token_similarity(g_clean, p_clean)


def compare_qualifier_lists(gold_list: List[str], pred_list: List[str]) -> float:
    """Compare conditions or exceptions lists."""
    if not gold_list and not pred_list:
        return 1.0
    if gold_list and not pred_list:
        return 0.0
    if not gold_list and pred_list:
        return 0.7

    g_joined = "; ".join(gold_list)
    p_joined = "; ".join(pred_list)
    return token_similarity(g_joined, p_joined)


class FactMatcher:
    """Deterministic, explainable 1-to-1 matching engine."""

    def __init__(
        self,
        match_threshold: float = 0.65,
        ambiguous_threshold: float = 0.50,
        subject_threshold: float = 0.60,
        object_threshold: float = 0.50,
    ) -> None:
        self.match_threshold = match_threshold
        self.ambiguous_threshold = ambiguous_threshold
        self.subject_threshold = subject_threshold
        self.object_threshold = object_threshold

    def compare_pair(self, gold: GoldFact, pred: Fact) -> MatchComponentScores:
        """Compute detailed component-level scores between a gold fact and predicted fact."""
        # 1. Subject score
        sub_score = token_similarity(gold.subject, pred.subject)

        # 2. Predicate match
        gold_canon = gold.get_canonical_predicate()
        pred_canon = pred.predicate if isinstance(pred.predicate, FactPredicate) else None
        if pred_canon is None:
            try:
                pred_canon = FactPredicate(str(pred.predicate).upper())
            except ValueError:
                pred_canon = None

        pred_match = False
        if gold_canon and pred_canon:
            pred_match = (gold_canon == pred_canon)
        elif gold.predicate.upper() == str(pred.predicate).upper():
            pred_match = True

        # 3. Object score
        obj_score = token_similarity(gold.object, pred.object)

        # 4. Qualifier scores
        cond_score = compare_qualifier_lists(gold.conditions, pred.conditions)
        exc_score = compare_qualifier_lists(gold.exceptions, pred.exceptions)
        thresh_score = compare_qualifier_values(gold.threshold, pred.threshold)
        time_score = compare_qualifier_values(gold.time, pred.time)

        # 5. Evidence grounding / overlap score
        evidence_score = 1.0
        if gold.evidence_text and pred.source_text:
            evidence_score = token_similarity(gold.evidence_text, pred.source_text)
        elif gold.evidence_text and not pred.source_text:
            evidence_score = 0.0

        # 6. Clause alignment
        clause_match = clauses_align(gold.clause_reference or gold.section_ref, pred.clause_reference or pred.rule_reference)

        # Overall composite score calculation
        # Core triple weight: 70%, Qualifiers: 20%, Evidence & Clause: 10%
        pred_numeric = 1.0 if pred_match else 0.0
        triple_score = (0.35 * sub_score) + (0.30 * pred_numeric) + (0.35 * obj_score)
        qual_score = (cond_score + exc_score + thresh_score + time_score) / 4.0
        context_score = (0.6 * evidence_score) + (0.4 * (1.0 if clause_match else 0.0))

        overall = (0.70 * triple_score) + (0.20 * qual_score) + (0.10 * context_score)

        return MatchComponentScores(
            subject_score=round(sub_score, 4),
            predicate_match=pred_match,
            object_score=round(obj_score, 4),
            condition_score=round(cond_score, 4),
            exception_score=round(exc_score, 4),
            threshold_score=round(thresh_score, 4),
            time_score=round(time_score, 4),
            evidence_score=round(evidence_score, 4),
            clause_match=clause_match,
            overall_score=round(overall, 4),
        )

    def match(
        self,
        gold_facts: List[GoldFact],
        predictions: List[Fact],
        include_unverified: bool = False,
    ) -> List[MatchDecision]:
        """Perform 1-to-1 deterministic matching between gold facts and predictions."""
        # 1. Filter gold facts by verification status if needed
        evaluated_gold = [g for g in gold_facts if (include_unverified or g.is_verified)]

        if not evaluated_gold and not predictions:
            return []

        # 2. Build pairwise score matrix
        pairs: List[Tuple[float, int, int, MatchComponentScores]] = []
        for g_idx, gold in enumerate(evaluated_gold):
            for p_idx, pred in enumerate(predictions):
                scores = self.compare_pair(gold, pred)
                pairs.append((scores.overall_score, g_idx, p_idx, scores))

        # Sort candidate pairs descending by overall score
        pairs.sort(key=lambda x: x[0], reverse=True)

        matched_gold_indices: Set[int] = set()
        matched_pred_indices: Set[int] = set()
        decisions: List[MatchDecision] = []

        # 3. Greedy best-first 1-to-1 assignment
        for score, g_idx, p_idx, scores in pairs:
            if g_idx in matched_gold_indices or p_idx in matched_pred_indices:
                continue

            gold = evaluated_gold[g_idx]
            pred = predictions[p_idx]

            # Check matching criteria
            is_match = (
                scores.predicate_match
                and scores.subject_score >= self.subject_threshold
                and scores.object_score >= self.object_threshold
                and scores.overall_score >= self.match_threshold
            )

            is_ambiguous = (
                not is_match
                and scores.overall_score >= self.ambiguous_threshold
                and (scores.subject_score >= 0.50 or scores.object_score >= 0.40)
            )

            if is_match:
                matched_gold_indices.add(g_idx)
                matched_pred_indices.add(p_idx)
                decisions.append(
                    MatchDecision(
                        gold_fact_id=gold.gold_fact_id,
                        predicted_fact_id=pred.fact_id,
                        status="MATCH",
                        scores=scores,
                        match_reason=f"High alignment across subject ({scores.subject_score:.2f}), predicate, and object ({scores.object_score:.2f})",
                        gold_fact=gold.model_dump(),
                        predicted_fact=pred.model_dump(),
                    )
                )
            elif is_ambiguous:
                # Mark as ambiguous match for manual adjudication
                matched_gold_indices.add(g_idx)
                matched_pred_indices.add(p_idx)
                decisions.append(
                    MatchDecision(
                        gold_fact_id=gold.gold_fact_id,
                        predicted_fact_id=pred.fact_id,
                        status="AMBIGUOUS",
                        scores=scores,
                        error_category=ErrorCategory.AMBIGUOUS_MATCH,
                        regulatory_impact=RegulatoryImpact.LOW,
                        match_reason=f"Borderline alignment (score={scores.overall_score:.2f}). Predicate match: {scores.predicate_match}",
                        gold_fact=gold.model_dump(),
                        predicted_fact=pred.model_dump(),
                    )
                )

        # 4. Handle remaining predictions (False Positives or Duplicates)
        for p_idx, pred in enumerate(predictions):
            if p_idx in matched_pred_indices:
                continue

            # Check if this prediction is a duplicate of an already-matched prediction
            is_dup = False
            for d in decisions:
                if d.predicted_fact:
                    dup_sub = token_similarity(pred.subject, d.predicted_fact.get("subject"))
                    dup_obj = token_similarity(pred.object, d.predicted_fact.get("object"))
                    dup_pred = (str(pred.predicate).upper() == str(d.predicted_fact.get("predicate")).upper())
                    if dup_sub >= 0.8 and dup_obj >= 0.8 and dup_pred:
                        is_dup = True
                        break

            if is_dup:
                decisions.append(
                    MatchDecision(
                        predicted_fact_id=pred.fact_id,
                        status="UNMATCHED_PRED",
                        scores=MatchComponentScores(overall_score=0.0),
                        error_category=ErrorCategory.DUPLICATE_PREDICTION,
                        regulatory_impact=RegulatoryImpact.LOW,
                        match_reason="Duplicate prediction repeating already extracted fact",
                        predicted_fact=pred.model_dump(),
                    )
                )
            else:
                decisions.append(
                    MatchDecision(
                        predicted_fact_id=pred.fact_id,
                        status="UNMATCHED_PRED",
                        scores=MatchComponentScores(overall_score=0.0),
                        error_category=ErrorCategory.UNSUPPORTED_OR_FABRICATED_FACT,
                        regulatory_impact=RegulatoryImpact.HIGH,
                        match_reason="Predicted fact does not match any gold standard fact",
                        predicted_fact=pred.model_dump(),
                    )
                )

        # 5. Handle remaining gold facts (False Negatives / Missed obligations)
        for g_idx, gold in enumerate(evaluated_gold):
            if g_idx in matched_gold_indices:
                continue

            is_obligation = (gold.predicate in ["REQUIRES", "PROHIBITS", "PAYMENT_DEADLINE"])
            impact = RegulatoryImpact.HIGH if is_obligation else RegulatoryImpact.MEDIUM

            decisions.append(
                MatchDecision(
                    gold_fact_id=gold.gold_fact_id,
                    status="UNMATCHED_GOLD",
                    scores=MatchComponentScores(overall_score=0.0),
                    error_category=ErrorCategory.MISSING_OBLIGATION if is_obligation else ErrorCategory.WRONG_OBJECT,
                    regulatory_impact=impact,
                    match_reason=f"Gold fact '{gold.gold_fact_id}' was not extracted by the model",
                    gold_fact=gold.model_dump(),
                )
            )

        return decisions
