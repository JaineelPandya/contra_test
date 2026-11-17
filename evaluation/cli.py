"""Command-line interface for the RegGraph Gold-Standard Evaluation Pipeline.

Usage:
  # Evaluate using cached predictions (zero LLM calls):
  python -m evaluation.cli --dataset evaluation/data/gold_facts_seed.jsonl --cached exports/facts_2026_0b5dd645.json --include-unverified

  # Run extraction on a circular PDF and evaluate:
  python -m evaluation.cli --pdf tests/sample_data/circular_2026_amendment.pdf --dataset evaluation/data/gold_facts_seed.jsonl

  # Convert and validate seed workbook:
  python -m evaluation.cli --import-seed evaluation/data/sebi_surveillance_goldset_seed.xlsx
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional
from uuid import uuid4
from datetime import datetime, timezone

from app.core.config import get_settings
from app.core.logging import logger
from app.models.fact import Fact
from evaluation.error_analyzer import ErrorAnalyzer
from evaluation.importer import DatasetImporter
from evaluation.matcher import FactMatcher
from evaluation.metrics import MetricsCalculator
from evaluation.models import (
    EvaluationSummary,
    GoldFact,
)
from evaluation.reporter import EvaluationReporter
from evaluation.runner import EvaluationRunner, compute_prompt_hash


def run_pipeline(
    dataset_path: str,
    cached_predictions_path: Optional[str] = None,
    pdf_path: Optional[str] = None,
    chunk_limit: Optional[int] = None,
    include_unverified: bool = False,
    results_dir: str = "evaluation/results",
    reports_dir: str = "evaluation/reports",
    match_threshold: float = 0.65,
) -> EvaluationSummary:
    """Execute end-to-end evaluation pipeline and generate all reports."""
    run_id = f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
    logger.info("Initializing RegGraph Evaluation Pipeline [run_id=%s]", run_id)

    # 1. Load Gold Dataset
    gold_facts: List[GoldFact] = DatasetImporter.load_dataset(dataset_path)
    total_gold = len(gold_facts)
    verified_gold = sum(1 for g in gold_facts if g.is_verified)
    unverified_gold = total_gold - verified_gold

    evaluated_gold = [g for g in gold_facts if (include_unverified or g.is_verified)]
    is_verified_baseline = (len(evaluated_gold) > 0 and all(g.is_verified for g in evaluated_gold))

    logger.info(
        "Gold facts loaded: total=%d, verified=%d, unverified=%d, active_in_eval=%d (include_unverified=%s)",
        total_gold,
        verified_gold,
        unverified_gold,
        len(evaluated_gold),
        include_unverified,
    )

    if not evaluated_gold:
        logger.warning(
            "⚠️ No verified gold facts available for official evaluation! "
            "All %d rows in '%s' are unverified seed annotations. "
            "To inspect seed alignment exploratory metrics, re-run with '--include-unverified'.",
            total_gold,
            dataset_path,
        )

    # 2. Obtain Model Predictions
    runner = EvaluationRunner(results_dir=results_dir)
    predictions: List[Fact] = []

    if cached_predictions_path:
        logger.info("Loading pre-extracted cached predictions from %s", cached_predictions_path)
        predictions = runner.load_cached_predictions(cached_predictions_path)
    elif pdf_path:
        logger.info("Running extraction pipeline on PDF: %s", pdf_path)
        predictions, _ = runner.extract_from_pdf(
            pdf_path=pdf_path,
            chunk_limit=chunk_limit,
            run_id=run_id,
        )
    else:
        # Default fallback: check if exports/facts_2026_0b5dd645.json exists
        default_export = Path("exports/facts_2026_0b5dd645.json")
        if default_export.exists():
            logger.info("No prediction source specified. Defaulting to cached export at %s", default_export)
            predictions = runner.load_cached_predictions(default_export)
        else:
            logger.error("No predictions provided! Specify either '--cached <path>' or '--pdf <path>'.")
            sys.exit(1)

    # 3. Deterministic 1-to-1 Matching
    matcher = FactMatcher(match_threshold=match_threshold)
    decisions = matcher.match(
        gold_facts=gold_facts,
        predictions=predictions,
        include_unverified=include_unverified,
    )

    # 4. Error Classification & Regulatory Impact
    (
        false_positives,
        false_negatives,
        component_errors,
        error_counts,
        impact_counts,
    ) = ErrorAnalyzer.analyze_decisions(decisions)

    # 5. Metric Calculation with Zero-Denominator Safety
    core_metrics = MetricsCalculator.calculate_core_metrics(decisions)
    component_metrics = MetricsCalculator.calculate_component_metrics(
        decisions=decisions,
        gold_facts=evaluated_gold,
        predictions=predictions,
    )
    predicate_breakdown = MetricsCalculator.calculate_predicate_breakdown(
        decisions=decisions,
        gold_facts=evaluated_gold,
        predictions=predictions,
    )

    # 6. Assemble Summary Model
    settings = get_settings()
    summary = EvaluationSummary(
        run_id=run_id,
        model_name=getattr(settings, f"{settings.LLM_PROVIDER.upper()}_MODEL", "unknown"),
        provider_name=settings.LLM_PROVIDER,
        prompt_version_or_hash=compute_prompt_hash(),
        dataset_path=str(dataset_path),
        total_gold_facts=total_gold,
        verified_gold_facts=verified_gold,
        unverified_gold_facts=unverified_gold,
        evaluated_gold_facts=len(evaluated_gold),
        total_predictions=len(predictions),
        core_metrics=core_metrics,
        component_metrics=component_metrics,
        predicate_breakdown=predicate_breakdown,
        error_counts=error_counts,
        regulatory_impact_counts=impact_counts,
        is_verified_baseline=is_verified_baseline,
    )

    # 7. Write Output Artifacts
    reporter = EvaluationReporter(results_dir=results_dir, reports_dir=reports_dir)
    json_path = reporter.write_json_summary(summary)
    decisions_path = reporter.write_match_decisions(decisions, run_id)
    fp_path = reporter.write_csv_table(false_positives, f"false_positives_{run_id}.csv")
    fn_path = reporter.write_csv_table(false_negatives, f"false_negatives_{run_id}.csv")
    comp_err_path = reporter.write_csv_table(component_errors, f"component_errors_{run_id}.csv")
    report_path = reporter.write_markdown_report(
        summary=summary,
        false_positives=false_positives,
        false_negatives=false_negatives,
        component_errors=component_errors,
        decisions=decisions,
    )

    # 8. Print Clean Console Summary
    print("\n" + "=" * 70)
    print("           REGGRAPH FACTUAL ACCURACY EVALUATION RESULTS")
    print("=" * 70)
    print(f"Run ID:                  {run_id}")
    print(f"Dataset Path:            {dataset_path}")
    print(f"Evaluated Gold Facts:    {len(evaluated_gold)} (Total: {total_gold}, Verified: {verified_gold})")
    print(f"Total Predictions:       {len(predictions)}")
    print(f"Verified Baseline:       {'YES' if is_verified_baseline else 'NO (Exploratory / Seed Run)'}")
    print("-" * 70)
    print("CORE FACT METRICS:")
    p_str = f"{core_metrics.precision * 100:.1f}%" if core_metrics.precision is not None else "N/A"
    r_str = f"{core_metrics.recall * 100:.1f}%" if core_metrics.recall is not None else "N/A"
    f1_str = f"{core_metrics.f1_score * 100:.1f}%" if core_metrics.f1_score is not None else "N/A"
    print(f"  Precision:             {p_str}  (TP={core_metrics.true_positives}, FP={core_metrics.false_positives})")
    print(f"  Recall:                {r_str}  (TP={core_metrics.true_positives}, FN={core_metrics.false_negatives})")
    print(f"  F1-Score:              {f1_str}")
    print("-" * 70)
    print("COMPONENT & QUALIFIER PRESERVATION:")
    sub_str = f"{component_metrics.subject_accuracy * 100:.1f}%" if component_metrics.subject_accuracy is not None else "N/A"
    pred_str = f"{component_metrics.predicate_accuracy * 100:.1f}%" if component_metrics.predicate_accuracy is not None else "N/A"
    obj_str = f"{component_metrics.object_accuracy * 100:.1f}%" if component_metrics.object_accuracy is not None else "N/A"
    exc_str = f"{component_metrics.exception_preservation_rate * 100:.1f}%" if component_metrics.exception_preservation_rate is not None else "N/A"
    thresh_str = f"{component_metrics.threshold_preservation_rate * 100:.1f}%" if component_metrics.threshold_preservation_rate is not None else "N/A"
    time_str = f"{component_metrics.time_preservation_rate * 100:.1f}%" if component_metrics.time_preservation_rate is not None else "N/A"
    print(f"  Subject Accuracy:      {sub_str}")
    print(f"  Predicate Accuracy:    {pred_str}")
    print(f"  Object Accuracy:       {obj_str}")
    print(f"  Exception Preserved:   {exc_str}")
    print(f"  Threshold Preserved:   {thresh_str}")
    print(f"  Deadline Preserved:    {time_str}")
    print("-" * 70)
    print("OUTPUT ARTIFACTS GENERATED:")
    print(f"  Summary JSON:          {json_path}")
    print(f"  Match Decisions:       {decisions_path}")
    print(f"  False Positives CSV:   {fp_path}")
    print(f"  False Negatives CSV:   {fn_path}")
    print(f"  Component Errors CSV:  {comp_err_path}")
    print(f"  Audit Report:          {report_path}")
    print("=" * 70 + "\n")

    return summary


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="RegGraph End-to-End Gold-Standard Factual Accuracy Evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="evaluation/data/gold_facts_seed.jsonl",
        help="Path to gold dataset (.jsonl, .csv, or .xlsx). Default: evaluation/data/gold_facts_seed.jsonl",
    )
    parser.add_argument(
        "--cached", "-c",
        type=str,
        default=None,
        help="Path to pre-extracted predictions JSONL or JSON (runs offline evaluation without LLM calls).",
    )
    parser.add_argument(
        "--pdf", "-p",
        type=str,
        default=None,
        help="Path to circular PDF to extract and evaluate.",
    )
    parser.add_argument(
        "--chunk-limit",
        type=int,
        default=None,
        help="Limit number of chunks to process when extracting live from PDF.",
    )
    parser.add_argument(
        "--include-unverified",
        action="store_true",
        default=False,
        help="Include unverified seed annotations in accuracy computation (exploratory mode).",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.65,
        help="Matching threshold for gold fact alignment (default: 0.65).",
    )
    parser.add_argument(
        "--import-seed",
        type=str,
        default=None,
        help="Import and convert an Excel seed workbook (.xlsx) to canonical JSONL/CSV datasets.",
    )

    args = parser.parse_args()

    if args.import_seed:
        facts = DatasetImporter.load_dataset(args.import_seed)
        jsonl_out = DatasetImporter.save_dataset_jsonl(facts, "evaluation/data/gold_facts_seed.jsonl")
        csv_out = DatasetImporter.save_dataset_csv(facts, "evaluation/data/gold_facts_seed.csv")
        DatasetImporter.generate_template_csv("evaluation/data/gold_template.csv")
        print(f"Successfully converted {len(facts)} facts from {args.import_seed}:")
        print(f"  -> {jsonl_out}")
        print(f"  -> {csv_out}")
        return

    # Check dataset existence
    dpath = Path(args.dataset)
    if not dpath.exists():
        # Check xlsx fallback
        xlsx_alt = Path("evaluation/data/sebi_surveillance_goldset_seed.xlsx")
        if xlsx_alt.exists():
            args.dataset = str(xlsx_alt)
        else:
            print(f"Error: Dataset file not found at {args.dataset}")
            sys.exit(1)

    run_pipeline(
        dataset_path=args.dataset,
        cached_predictions_path=args.cached,
        pdf_path=args.pdf,
        chunk_limit=args.chunk_limit,
        include_unverified=args.include_unverified,
        match_threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
