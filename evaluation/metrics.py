"""Metric computation engine with zero-denominator safety and component-level tracking."""
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict

from evaluation.models import (
    ComponentMetrics,
    CoreMetrics,
    GoldFact,
    MatchDecision,
    PredicateBreakdown,
)
from app.models.fact import Fact, FactPredicate


class MetricsCalculator:
    """Calculates factual accuracy metrics across match decisions."""

    @staticmethod
    def safe_div(num: float, den: float) -> Optional[float]:
        """Safely divide two numbers, returning None if denominator is zero."""
        if den == 0:
            return None
        return round(num / den, 4)

    @classmethod
    def calculate_core_metrics(cls, decisions: List[MatchDecision]) -> CoreMetrics:
        """Compute True Positives, False Positives, False Negatives, Precision, Recall, F1."""
        tp = sum(1 for d in decisions if d.status == "MATCH")
        # Ambiguous matches are treated conservatively as non-matches (both FP and FN)
        ambiguous = sum(1 for d in decisions if d.status == "AMBIGUOUS")
        unmatched_pred = sum(1 for d in decisions if d.status == "UNMATCHED_PRED")
        unmatched_gold = sum(1 for d in decisions if d.status == "UNMATCHED_GOLD")

        fp = unmatched_pred + ambiguous
        fn = unmatched_gold + ambiguous

        precision = cls.safe_div(tp, tp + fp)
        recall = cls.safe_div(tp, tp + fn)

        f1 = None
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = round(2.0 * precision * recall / (precision + recall), 4)

        return CoreMetrics(
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1_score=f1,
        )

    @classmethod
    def calculate_component_metrics(
        cls,
        decisions: List[MatchDecision],
        gold_facts: List[GoldFact],
        predictions: List[Fact],
    ) -> ComponentMetrics:
        """Compute fine-grained component accuracy, qualifier preservation, and duplicate rate."""
        matched_pairs = [d for d in decisions if d.status == "MATCH"]

        # 1. Subject, Predicate, Object Accuracy on Matched Pairs
        if matched_pairs:
            subj_acc = round(sum(d.scores.subject_score for d in matched_pairs) / len(matched_pairs), 4)
            pred_acc = round(sum(1.0 if d.scores.predicate_match else 0.0 for d in matched_pairs) / len(matched_pairs), 4)
            obj_acc = round(sum(d.scores.object_score for d in matched_pairs) / len(matched_pairs), 4)
        else:
            subj_acc, pred_acc, obj_acc = None, None, None

        # 2. Qualifier Preservation Rates (Denominator = Gold facts having that qualifier)
        # Conditions
        gold_with_conds = [d for d in matched_pairs if d.gold_fact and d.gold_fact.get("conditions")]
        cond_preserved = sum(1 for d in gold_with_conds if d.scores.condition_score >= 0.7)
        cond_rate = cls.safe_div(cond_preserved, len(gold_with_conds))

        # Exceptions
        gold_with_excs = [d for d in matched_pairs if d.gold_fact and d.gold_fact.get("exceptions")]
        exc_preserved = sum(1 for d in gold_with_excs if d.scores.exception_score >= 0.7)
        exc_rate = cls.safe_div(exc_preserved, len(gold_with_excs))

        # Thresholds
        gold_with_thresh = [d for d in matched_pairs if d.gold_fact and d.gold_fact.get("threshold")]
        thresh_preserved = sum(1 for d in gold_with_thresh if d.scores.threshold_score >= 0.8)
        thresh_rate = cls.safe_div(thresh_preserved, len(gold_with_thresh))

        # Time / Deadlines
        gold_with_time = [d for d in matched_pairs if d.gold_fact and d.gold_fact.get("time")]
        time_preserved = sum(1 for d in gold_with_time if d.scores.time_score >= 0.8)
        time_rate = cls.safe_div(time_preserved, len(gold_with_time))

        # 3. Evidence Support Rate across Predictions
        if predictions:
            # Check source text presence and length
            supported_count = sum(1 for p in predictions if p.source_text and len(p.source_text.strip()) > 15)
            evidence_support_rate = cls.safe_div(supported_count, len(predictions))
        else:
            evidence_support_rate = None

        # 4. Duplicate Prediction Rate
        dup_count = sum(1 for d in decisions if d.error_category and d.error_category.value == "DUPLICATE_PREDICTION")
        dup_rate = cls.safe_div(dup_count, len(predictions)) if predictions else None

        # 5. Clause Coverage
        gold_clauses: Set[str] = {
            g.clause_reference or g.section_ref
            for g in gold_facts
            if (g.clause_reference or g.section_ref)
        }
        covered_clauses: Set[str] = set()
        for d in matched_pairs:
            c = d.gold_fact.get("clause_reference") or d.gold_fact.get("section_ref") if d.gold_fact else None
            if c:
                covered_clauses.add(c)
        clause_coverage = cls.safe_div(len(covered_clauses), len(gold_clauses))

        return ComponentMetrics(
            subject_accuracy=subj_acc,
            predicate_accuracy=pred_acc,
            object_accuracy=obj_acc,
            condition_preservation_rate=cond_rate,
            exception_preservation_rate=exc_rate,
            threshold_preservation_rate=thresh_rate,
            time_preservation_rate=time_rate,
            evidence_support_rate=evidence_support_rate,
            duplicate_prediction_rate=dup_rate,
            clause_coverage=clause_coverage,
        )

    @classmethod
    def calculate_predicate_breakdown(
        cls,
        decisions: List[MatchDecision],
        gold_facts: List[GoldFact],
        predictions: List[Fact],
    ) -> List[PredicateBreakdown]:
        """Compute performance breakdown segmented by FactPredicate."""
        def _get_pred_name(val: Any) -> str:
            if isinstance(val, FactPredicate):
                return val.value
            s = str(val).split(".")[-1].strip().upper()
            return s

        gold_counts: Dict[str, int] = defaultdict(int)
        for g in gold_facts:
            canon = g.get_canonical_predicate()
            name = canon.value if canon else _get_pred_name(g.predicate)
            gold_counts[name] += 1

        pred_counts: Dict[str, int] = defaultdict(int)
        for p in predictions:
            pred_counts[_get_pred_name(p.predicate)] += 1

        matched_counts: Dict[str, int] = defaultdict(int)
        for d in decisions:
            if d.status == "MATCH" and d.gold_fact:
                gold_pred = d.gold_fact.get("predicate", "")
                name = _get_pred_name(gold_pred)
                matched_counts[name] += 1

        all_preds = sorted(list(set(gold_counts.keys()) | set(pred_counts.keys())))
        breakdowns: List[PredicateBreakdown] = []

        for p in all_preds:
            g_cnt = gold_counts.get(p, 0)
            p_cnt = pred_counts.get(p, 0)
            m_cnt = matched_counts.get(p, 0)

            prec = cls.safe_div(m_cnt, p_cnt)
            rec = cls.safe_div(m_cnt, g_cnt)
            f1 = None
            if prec is not None and rec is not None and (prec + rec) > 0:
                f1 = round(2.0 * prec * rec / (prec + rec), 4)

            breakdowns.append(
                PredicateBreakdown(
                    predicate=p,
                    gold_count=g_cnt,
                    predicted_count=p_cnt,
                    matched_count=m_cnt,
                    precision=prec,
                    recall=rec,
                    f1_score=f1,
                )
            )

        return breakdowns
