"""Error classification and regulatory impact analyzer for RegGraph evaluation."""
from typing import Any, Dict, List, Tuple
from collections import defaultdict

from evaluation.models import (
    ErrorCategory,
    MatchDecision,
    RegulatoryImpact,
)


class ErrorAnalyzer:
    """Classifies extraction errors and assesses potential regulatory compliance risk."""

    @classmethod
    def analyze_decisions(
        cls,
        decisions: List[MatchDecision],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int], Dict[str, int]]:
        """Extract false positives, false negatives, component errors, and summary distributions."""
        false_positives: List[Dict[str, Any]] = []
        false_negatives: List[Dict[str, Any]] = []
        component_errors: List[Dict[str, Any]] = []

        error_counts: Dict[str, int] = defaultdict(int)
        impact_counts: Dict[str, int] = defaultdict(int)

        for d in decisions:
            # 1. Unmatched or Ambiguous Predictions -> False Positives
            if d.status == "UNMATCHED_PRED":
                cat = d.error_category or ErrorCategory.UNSUPPORTED_OR_FABRICATED_FACT
                imp = d.regulatory_impact or RegulatoryImpact.HIGH
                error_counts[cat.value] += 1
                impact_counts[imp.value] += 1

                pred = d.predicted_fact or {}
                false_positives.append({
                    "predicted_fact_id": d.predicted_fact_id,
                    "subject": pred.get("subject"),
                    "predicate": pred.get("predicate"),
                    "object": pred.get("object"),
                    "clause_reference": pred.get("clause_reference"),
                    "evidence": pred.get("source_text"),
                    "error_category": cat.value,
                    "regulatory_impact": imp.value,
                    "reason": d.match_reason,
                })

            elif d.status == "AMBIGUOUS":
                cat = ErrorCategory.AMBIGUOUS_MATCH
                imp = RegulatoryImpact.LOW
                error_counts[cat.value] += 1
                impact_counts[imp.value] += 1

                pred = d.predicted_fact or {}
                false_positives.append({
                    "predicted_fact_id": d.predicted_fact_id,
                    "subject": pred.get("subject"),
                    "predicate": pred.get("predicate"),
                    "object": pred.get("object"),
                    "clause_reference": pred.get("clause_reference"),
                    "evidence": pred.get("source_text"),
                    "error_category": cat.value,
                    "regulatory_impact": imp.value,
                    "reason": d.match_reason,
                })

            # 2. Unmatched or Ambiguous Gold Facts -> False Negatives
            if d.status in ["UNMATCHED_GOLD", "AMBIGUOUS"]:
                cat = d.error_category or ErrorCategory.MISSING_OBLIGATION
                imp = d.regulatory_impact or RegulatoryImpact.HIGH
                error_counts[cat.value] += 1
                impact_counts[imp.value] += 1

                gold = d.gold_fact or {}
                false_negatives.append({
                    "gold_fact_id": d.gold_fact_id,
                    "document_title": gold.get("document_title"),
                    "clause_reference": gold.get("clause_reference") or gold.get("section_ref"),
                    "subject": gold.get("subject"),
                    "predicate": gold.get("predicate"),
                    "object": gold.get("object"),
                    "threshold": gold.get("threshold"),
                    "time": gold.get("time"),
                    "evidence_text": gold.get("evidence_text"),
                    "error_category": cat.value,
                    "regulatory_impact": imp.value,
                    "reason": d.match_reason,
                })

            # 3. Component-level defects in MATCH decisions
            if d.status == "MATCH" and d.gold_fact and d.predicted_fact:
                gold = d.gold_fact
                pred = d.predicted_fact

                # Check Condition loss
                if gold.get("conditions") and d.scores.condition_score < 0.7:
                    cat = ErrorCategory.LOST_CONDITION
                    imp = RegulatoryImpact.MEDIUM
                    error_counts[cat.value] += 1
                    impact_counts[imp.value] += 1
                    component_errors.append({
                        "gold_fact_id": d.gold_fact_id,
                        "predicted_fact_id": d.predicted_fact_id,
                        "component_type": "CONDITION",
                        "gold_value": "; ".join(gold.get("conditions", [])),
                        "predicted_value": "; ".join(pred.get("conditions", [])),
                        "score": d.scores.condition_score,
                        "error_category": cat.value,
                        "regulatory_impact": imp.value,
                    })

                # Check Exception loss
                if gold.get("exceptions") and d.scores.exception_score < 0.7:
                    cat = ErrorCategory.LOST_EXCEPTION
                    imp = RegulatoryImpact.CRITICAL
                    error_counts[cat.value] += 1
                    impact_counts[imp.value] += 1
                    component_errors.append({
                        "gold_fact_id": d.gold_fact_id,
                        "predicted_fact_id": d.predicted_fact_id,
                        "component_type": "EXCEPTION",
                        "gold_value": "; ".join(gold.get("exceptions", [])),
                        "predicted_value": "; ".join(pred.get("exceptions", [])),
                        "score": d.scores.exception_score,
                        "error_category": cat.value,
                        "regulatory_impact": imp.value,
                    })

                # Check Threshold mismatch
                if gold.get("threshold") and d.scores.threshold_score < 0.8:
                    cat = ErrorCategory.LOST_OR_INCORRECT_THRESHOLD
                    imp = RegulatoryImpact.HIGH
                    error_counts[cat.value] += 1
                    impact_counts[imp.value] += 1
                    component_errors.append({
                        "gold_fact_id": d.gold_fact_id,
                        "predicted_fact_id": d.predicted_fact_id,
                        "component_type": "THRESHOLD",
                        "gold_value": gold.get("threshold"),
                        "predicted_value": pred.get("threshold"),
                        "score": d.scores.threshold_score,
                        "error_category": cat.value,
                        "regulatory_impact": imp.value,
                    })

                # Check Time / Deadline mismatch
                if gold.get("time") and d.scores.time_score < 0.8:
                    cat = ErrorCategory.LOST_OR_INCORRECT_DEADLINE
                    imp = RegulatoryImpact.HIGH
                    error_counts[cat.value] += 1
                    impact_counts[imp.value] += 1
                    component_errors.append({
                        "gold_fact_id": d.gold_fact_id,
                        "predicted_fact_id": d.predicted_fact_id,
                        "component_type": "TIME",
                        "gold_value": gold.get("time"),
                        "predicted_value": pred.get("time"),
                        "score": d.scores.time_score,
                        "error_category": cat.value,
                        "regulatory_impact": imp.value,
                    })

        return (
            false_positives,
            false_negatives,
            component_errors,
            dict(error_counts),
            dict(impact_counts),
        )
