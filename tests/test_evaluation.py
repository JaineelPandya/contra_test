"""Comprehensive unit tests for RegGraph factual accuracy evaluation pipeline.

Covers:
1. Perfect predictions (TP=N, FP=0, FN=0, P=1, R=1, F1=1)
2. Completely missed facts (FN=N, TP=0, FP=0)
3. Completely incorrect predictions (FP=M, FN=N, TP=0)
4. Partial subject/predicate/object matches & ambiguity flagging
5. Conditions dropped (LOST_CONDITION detection)
6. Exceptions dropped (LOST_EXCEPTION detection & CRITICAL impact)
7. Threshold and deadline mismatches (LOST_OR_INCORRECT_THRESHOLD / DEADLINE)
8. Duplicate predictions (cannot inflate recall; counted as FP)
9. One-to-one matching invariant
10. Empty datasets and zero denominator safety
11. Unverified gold rows excluded by default
12. Unverified gold rows included when flag set
13. Missing evidence handling
14. Multiple documents and clauses isolation
15. Deterministic reproducibility
16. Malformed and noisy input rows resilience
17. Seed workbook import and parsing
"""
import pytest
from pathlib import Path
from evaluation.models import (
    ErrorCategory,
    GoldFact,
    RegulatoryImpact,
    VerificationStatus,
)
from evaluation.importer import DatasetImporter
from evaluation.matcher import FactMatcher
from evaluation.metrics import MetricsCalculator
from evaluation.error_analyzer import ErrorAnalyzer
from app.models.fact import Fact, FactPredicate


def make_gold_fact(
    gold_id: str = "G-001",
    subject: str = "Stock exchanges",
    predicate: str = "REQUIRES",
    obj: str = "Place affected securities in TFT segment",
    conditions=None,
    exceptions=None,
    threshold=None,
    time=None,
    clause_ref="1.1",
    evidence="Trading shall take place in TFT segment",
    is_verified: bool = True,
) -> GoldFact:
    return GoldFact(
        gold_fact_id=gold_id,
        document_title="Master Circular on Surveillance",
        clause_reference=clause_ref,
        subject=subject,
        predicate=predicate,
        object=obj,
        conditions=conditions or [],
        exceptions=exceptions or [],
        threshold=threshold,
        time=time,
        evidence_text=evidence,
        annotation_status="VERIFIED" if is_verified else "Seed annotation — human review required",
        is_verified=is_verified,
    )


def make_pred_fact(
    fact_id: str = "P-001",
    subject: str = "Stock exchanges",
    predicate: FactPredicate = FactPredicate.REQUIRES,
    obj: str = "Place affected securities in TFT segment",
    conditions=None,
    exceptions=None,
    threshold=None,
    time=None,
    clause_ref="1.1",
    evidence="Trading shall take place in TFT segment",
) -> Fact:
    return Fact(
        fact_id=fact_id,
        document_id="doc_eval",
        chunk_id="chunk_1",
        subject=subject,
        predicate=predicate,
        object=obj,
        conditions=conditions or [],
        exceptions=exceptions or [],
        threshold=threshold,
        time=time,
        clause_reference=clause_ref,
        source_text=evidence,
    )


# ==============================================================================
# 1. CORE MATCHING & METRICS TESTS
# ==============================================================================

def test_perfect_predictions() -> None:
    gold = [
        make_gold_fact("G1", "Stock exchanges", "REQUIRES", "Enforce TFT trading"),
        make_gold_fact("G2", "MIIs", "REQUIRES", "Credit financial disincentive"),
    ]
    preds = [
        make_pred_fact("P1", "Stock exchanges", FactPredicate.REQUIRES, "Enforce TFT trading"),
        make_pred_fact("P2", "MIIs", FactPredicate.REQUIRES, "Credit financial disincentive"),
    ]

    matcher = FactMatcher()
    decisions = matcher.match(gold, preds)
    assert len(decisions) == 2
    assert all(d.status == "MATCH" for d in decisions)

    core = MetricsCalculator.calculate_core_metrics(decisions)
    assert core.true_positives == 2
    assert core.false_positives == 0
    assert core.false_negatives == 0
    assert core.precision == 1.0
    assert core.recall == 1.0
    assert core.f1_score == 1.0


def test_completely_missed_facts() -> None:
    gold = [
        make_gold_fact("G1", "Stock exchanges", "REQUIRES", "Enforce TFT trading"),
        make_gold_fact("G2", "MIIs", "REQUIRES", "Credit financial disincentive"),
    ]
    preds = []

    matcher = FactMatcher()
    decisions = matcher.match(gold, preds)
    assert len(decisions) == 2
    assert all(d.status == "UNMATCHED_GOLD" for d in decisions)

    core = MetricsCalculator.calculate_core_metrics(decisions)
    assert core.true_positives == 0
    assert core.false_positives == 0
    assert core.false_negatives == 2
    assert core.precision is None
    assert core.recall == 0.0
    assert core.f1_score is None


def test_completely_incorrect_predictions() -> None:
    gold = [make_gold_fact("G1", "Stock exchanges", "REQUIRES", "Enforce TFT trading")]
    preds = [make_pred_fact("P1", "Unrelated Actor", FactPredicate.PROHIBITS, "Unrelated action entirely")]

    matcher = FactMatcher()
    decisions = matcher.match(gold, preds)

    core = MetricsCalculator.calculate_core_metrics(decisions)
    assert core.true_positives == 0
    assert core.false_positives == 1
    assert core.false_negatives == 1
    assert core.precision == 0.0
    assert core.recall == 0.0


def test_partial_match_and_ambiguity_flagging() -> None:
    # Gold fact and prediction have borderline similarity
    gold = [make_gold_fact("G1", "Stock exchanges", "REQUIRES", "Establish surveillance mechanism")]
    pred = [make_pred_fact("P1", "Exchanges", FactPredicate.REQUIRES, "Maintain records of trading")]

    matcher = FactMatcher(match_threshold=0.65, ambiguous_threshold=0.45)
    decisions = matcher.match(gold, pred)

    # Must either be flagged ambiguous or unmatched, NEVER silently counted as MATCH TP
    assert not any(d.status == "MATCH" for d in decisions)


# ==============================================================================
# 2. QUALIFIER PRESERVATION & ERROR TAXONOMY TESTS
# ==============================================================================

def test_conditions_dropped() -> None:
    gold = [make_gold_fact("G1", conditions=["If total turnover exceeds 100 Crores"])]
    pred = [make_pred_fact("P1", conditions=[])]

    matcher = FactMatcher()
    decisions = matcher.match(gold, pred)
    assert decisions[0].status == "MATCH"

    comp = MetricsCalculator.calculate_component_metrics(decisions, gold, pred)
    assert comp.condition_preservation_rate == 0.0

    _, _, comp_errors, error_counts, _ = ErrorAnalyzer.analyze_decisions(decisions)
    assert any(ce["component_type"] == "CONDITION" for ce in comp_errors)
    assert error_counts.get("LOST_CONDITION") == 1


def test_exceptions_dropped_triggers_critical_impact() -> None:
    gold = [make_gold_fact("G1", exceptions=["Original scrips on which derivatives are available"])]
    pred = [make_pred_fact("P1", exceptions=[])]

    matcher = FactMatcher()
    decisions = matcher.match(gold, pred)
    assert decisions[0].status == "MATCH"

    comp = MetricsCalculator.calculate_component_metrics(decisions, gold, pred)
    assert comp.exception_preservation_rate == 0.0

    _, _, comp_errors, error_counts, impact_counts = ErrorAnalyzer.analyze_decisions(decisions)
    assert any(ce["component_type"] == "EXCEPTION" for ce in comp_errors)
    assert error_counts.get("LOST_EXCEPTION") == 1
    assert impact_counts.get("CRITICAL", 0) >= 1


def test_threshold_and_deadline_mismatches() -> None:
    gold = [make_gold_fact("G1", threshold="10 Lakhs", time="15 working days")]
    pred = [make_pred_fact("P1", threshold="50 Lakhs", time="30 calendar days")]

    matcher = FactMatcher()
    decisions = matcher.match(gold, pred)
    assert decisions[0].status == "MATCH"

    comp = MetricsCalculator.calculate_component_metrics(decisions, gold, pred)
    assert comp.threshold_preservation_rate == 0.0
    assert comp.time_preservation_rate == 0.0

    _, _, comp_errors, error_counts, _ = ErrorAnalyzer.analyze_decisions(decisions)
    assert error_counts.get("LOST_OR_INCORRECT_THRESHOLD") == 1
    assert error_counts.get("LOST_OR_INCORRECT_DEADLINE") == 1


# ==============================================================================
# 3. DUPLICATE & 1-TO-1 MATCHING INVARIANTS
# ==============================================================================

def test_duplicate_predictions_cannot_inflate_recall() -> None:
    gold = [make_gold_fact("G1", "Stock exchanges", "REQUIRES", "Enforce TFT trading")]
    # Model hallucinated 3 identical predictions for 1 gold fact
    preds = [
        make_pred_fact("P1", "Stock exchanges", FactPredicate.REQUIRES, "Enforce TFT trading"),
        make_pred_fact("P2", "Stock exchanges", FactPredicate.REQUIRES, "Enforce TFT trading"),
        make_pred_fact("P3", "Stock exchanges", FactPredicate.REQUIRES, "Enforce TFT trading"),
    ]

    matcher = FactMatcher()
    decisions = matcher.match(gold, preds)

    # Must match at most ONE gold fact
    matches = [d for d in decisions if d.status == "MATCH"]
    assert len(matches) == 1

    # Remaining 2 predictions must be flagged as duplicates
    unmatched_preds = [d for d in decisions if d.status == "UNMATCHED_PRED"]
    assert len(unmatched_preds) == 2
    assert all(d.error_category == ErrorCategory.DUPLICATE_PREDICTION for d in unmatched_preds)

    core = MetricsCalculator.calculate_core_metrics(decisions)
    assert core.true_positives == 1
    assert core.false_positives == 2
    assert core.false_negatives == 0
    assert core.recall == 1.0
    assert round(core.precision, 4) == 0.3333


def test_one_to_one_matching_invariant() -> None:
    gold = [
        make_gold_fact("G1", "Actor A", "REQUIRES", "Action 1"),
        make_gold_fact("G2", "Actor A", "REQUIRES", "Action 2"),
    ]
    preds = [
        make_pred_fact("P1", "Actor A", FactPredicate.REQUIRES, "Action 1"),
        make_pred_fact("P2", "Actor A", FactPredicate.REQUIRES, "Action 1"),
    ]

    matcher = FactMatcher()
    decisions = matcher.match(gold, preds)

    # Check invariant: no gold fact ID appears in multiple MATCH decisions
    matched_gold_ids = [d.gold_fact_id for d in decisions if d.status == "MATCH"]
    assert len(matched_gold_ids) == len(set(matched_gold_ids))

    # Check invariant: no pred fact ID appears in multiple MATCH decisions
    matched_pred_ids = [d.predicted_fact_id for d in decisions if d.status == "MATCH"]
    assert len(matched_pred_ids) == len(set(matched_pred_ids))


# ==============================================================================
# 4. EDGE CASES & VERIFICATION GATING
# ==============================================================================

def test_empty_datasets_and_zero_denominators() -> None:
    matcher = FactMatcher()
    decisions = matcher.match([], [])
    assert decisions == []

    core = MetricsCalculator.calculate_core_metrics([])
    assert core.true_positives == 0
    assert core.false_positives == 0
    assert core.false_negatives == 0
    assert core.precision is None
    assert core.recall is None
    assert core.f1_score is None

    comp = MetricsCalculator.calculate_component_metrics([], [], [])
    assert comp.subject_accuracy is None
    assert comp.predicate_accuracy is None
    assert comp.condition_preservation_rate is None
    assert comp.clause_coverage is None


def test_unverified_gold_rows_excluded_by_default() -> None:
    gold = [
        make_gold_fact("G1", is_verified=True),
        make_gold_fact("G2", is_verified=False),
        make_gold_fact("G3", is_verified=False),
    ]
    preds = [make_pred_fact("P1")]

    matcher = FactMatcher()
    # Default: include_unverified=False
    decisions = matcher.match(gold, preds, include_unverified=False)

    evaluated_gold_ids = {d.gold_fact_id for d in decisions if d.gold_fact_id}
    assert "G1" in evaluated_gold_ids
    assert "G2" not in evaluated_gold_ids
    assert "G3" not in evaluated_gold_ids


def test_unverified_gold_rows_included_when_flag_set() -> None:
    gold = [
        make_gold_fact("G1", is_verified=True),
        make_gold_fact("G2", is_verified=False),
    ]
    preds = [make_pred_fact("P1")]

    matcher = FactMatcher()
    decisions = matcher.match(gold, preds, include_unverified=True)

    evaluated_gold_ids = {d.gold_fact_id for d in decisions if d.gold_fact_id}
    assert "G1" in evaluated_gold_ids
    assert "G2" in evaluated_gold_ids


def test_missing_evidence_drops_support_rate() -> None:
    gold = [make_gold_fact("G1")]
    # Prediction with missing source text
    pred = [make_pred_fact("P1", evidence="")]

    comp = MetricsCalculator.calculate_component_metrics([], gold, pred)
    assert comp.evidence_support_rate == 0.0


def test_multiple_documents_and_clauses_isolation() -> None:
    gold = [
        make_gold_fact("G1", "Stock exchanges", "REQUIRES", "Duty A", clause_ref="1.1"),
        make_gold_fact("G2", "Depositories", "REQUIRES", "Duty B", clause_ref="2.3"),
    ]
    preds = [
        make_pred_fact("P1", "Stock exchanges", FactPredicate.REQUIRES, "Duty A", clause_ref="1.1"),
        make_pred_fact("P2", "Depositories", FactPredicate.REQUIRES, "Duty B", clause_ref="2.3"),
    ]

    matcher = FactMatcher()
    decisions = matcher.match(gold, preds)
    assert len(decisions) == 2
    match_map = {d.gold_fact_id: d.predicted_fact_id for d in decisions if d.status == "MATCH"}
    assert match_map["G1"] == "P1"
    assert match_map["G2"] == "P2"


def test_deterministic_reproducibility() -> None:
    gold = [
        make_gold_fact("G1", "Stock exchanges", "REQUIRES", "Duty A"),
        make_gold_fact("G2", "Depositories", "REQUIRES", "Duty B"),
    ]
    preds = [
        make_pred_fact("P1", "Stock exchanges", FactPredicate.REQUIRES, "Duty A"),
        make_pred_fact("P2", "Depositories", FactPredicate.REQUIRES, "Duty B"),
    ]

    matcher = FactMatcher()
    d1 = matcher.match(gold, preds)
    d2 = matcher.match(gold, preds)

    assert [d.model_dump(exclude={"decision_id"}) for d in d1] == [
        d.model_dump(exclude={"decision_id"}) for d in d2
    ]


def test_malformed_and_noisy_inputs_resilience() -> None:
    noisy_dict = {
        "gold_fact_id": "NOISY-001",
        "subject": "  Depositories  ",
        "predicate": "permitted",  # non-standard casing/variant
        "object": "Share account transfers",
        "conditions_scope": "None stated",
        "exceptions_exclusions": "None",
        "thresholds_amounts": None,
        "deadline_timing": "Immediately",
        "annotation_status": "Seed annotation — human review required",
    }
    fact = DatasetImporter.dict_to_gold_fact(noisy_dict)
    assert fact.is_verified is False
    assert fact.conditions == []
    assert fact.exceptions == []
    assert fact.threshold is None
    assert fact.time == "Immediately"
    assert fact.get_canonical_predicate() == FactPredicate.PERMITS


def test_seed_workbook_import_exists_and_parses() -> None:
    xlsx_path = Path("evaluation/data/sebi_surveillance_goldset_seed.xlsx")
    assert xlsx_path.exists(), "Seed workbook should exist in evaluation/data/"

    facts = DatasetImporter.load_dataset(xlsx_path)
    assert len(facts) == 128
    assert all(not f.is_verified for f in facts), "Seed annotations must all be unverified"
    assert any(f.predicate == "REQUIRES" for f in facts)
