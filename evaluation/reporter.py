"""Report generator producing JSON summary, CSV diagnostics, and Markdown audit report."""
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from evaluation.models import (
    EvaluationSummary,
    MatchDecision,
)


class EvaluationReporter:
    """Generates machine-readable results and human-readable audit reports."""

    def __init__(
        self,
        results_dir: Union[str, Path] = "evaluation/results",
        reports_dir: Union[str, Path] = "evaluation/reports",
    ) -> None:
        self.results_dir = Path(results_dir)
        self.reports_dir = Path(reports_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def write_json_summary(self, summary: EvaluationSummary) -> Path:
        """Write summary metrics to JSON."""
        out_path = self.results_dir / f"summary_metrics_{summary.run_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(summary.model_dump_json(indent=2))
        return out_path

    def write_match_decisions(self, decisions: List[MatchDecision], run_id: str) -> Path:
        """Write detailed match decisions to JSONL."""
        out_path = self.results_dir / f"match_decisions_{run_id}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for d in decisions:
                f.write(d.model_dump_json() + "\n")
        return out_path

    def write_csv_table(self, data: List[Dict[str, Any]], filename: str) -> Path:
        """Write list of dictionaries to a CSV file."""
        out_path = self.results_dir / filename
        if not data:
            # Write empty file with notice
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                f.write("status\nno_records_found\n")
            return out_path

        fieldnames = list(data[0].keys())
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)
        return out_path

    def write_markdown_report(
        self,
        summary: EvaluationSummary,
        false_positives: List[Dict[str, Any]],
        false_negatives: List[Dict[str, Any]],
        component_errors: List[Dict[str, Any]],
        decisions: List[MatchDecision],
    ) -> Path:
        """Generate human-readable Markdown evaluation audit report."""
        out_path = self.reports_dir / f"evaluation_report_{summary.run_id}.md"

        def fmt_pct(val: Optional[float]) -> str:
            if val is None:
                return "N/A (0 denominator)"
            return f"{val * 100:.1f}%"

        def fmt_float(val: Optional[float]) -> str:
            if val is None:
                return "N/A"
            return f"{val:.4f}"

        c = summary.core_metrics
        comp = summary.component_metrics

        md: List[str] = []
        md.append(f"# RegGraph Factual Accuracy Evaluation Report")
        md.append("")
        md.append(f"> **Run ID:** `{summary.run_id}`  ")
        md.append(f"> **Timestamp:** `{summary.timestamp}`  ")
        md.append(f"> **LLM Provider:** `{summary.provider_name or 'Default'}` | **Model:** `{summary.model_name or 'Default'}`  ")
        md.append(f"> **Prompt SHA-256:** `{summary.prompt_version_or_hash or 'N/A'}`  ")
        md.append(f"> **Dataset:** `{summary.dataset_path}`  ")
        md.append("")

        # 1. Dataset Verification Status
        md.append("## 1. Dataset Verification & Baseline Status")
        md.append("")
        if summary.is_verified_baseline:
            md.append("✅ **Official Verified Baseline:** Evaluation computed strictly on verified ground-truth annotations.")
        else:
            md.append("⚠️ **Exploratory Run Warning:** This run evaluated unverified seed annotations. Official accuracy baselines require human verification.")
        md.append("")
        md.append("| Metric | Count |")
        md.append("|---|---|")
        md.append(f"| Total Gold Facts in Dataset | {summary.total_gold_facts} |")
        md.append(f"| Human-Verified Gold Facts | {summary.verified_gold_facts} |")
        md.append(f"| Unverified / Seed Facts | {summary.unverified_gold_facts} |")
        md.append(f"| Gold Facts Included in This Run | {summary.evaluated_gold_facts} |")
        md.append(f"| Model Predictions Extracted | {summary.total_predictions} |")
        md.append("")

        # 2. Overall Core Metrics
        md.append("## 2. Core Fact-Level Accuracy")
        md.append("")
        md.append("| Metric | Score | Raw Counts |")
        md.append("|---|---|---|")
        md.append(f"| **Precision** | **{fmt_pct(c.precision)}** | TP: {c.true_positives} / (TP: {c.true_positives} + FP: {c.false_positives}) |")
        md.append(f"| **Recall** | **{fmt_pct(c.recall)}** | TP: {c.true_positives} / (TP: {c.true_positives} + FN: {c.false_negatives}) |")
        md.append(f"| **F1-Score** | **{fmt_pct(c.f1_score)}** | Harmonic mean of precision & recall |")
        md.append("")

        # 3. Component & Qualifier Metrics
        md.append("## 3. Component & Qualifier Preservation")
        md.append("")
        md.append("| Component / Metric | Score | Description / Denominator |")
        md.append("|---|---|---|")
        md.append(f"| **Subject Accuracy** | {fmt_pct(comp.subject_accuracy)} | Mean subject alignment score on matched pairs |")
        md.append(f"| **Predicate Accuracy** | {fmt_pct(comp.predicate_accuracy)} | Proportion of matched pairs with exact canonical predicate match |")
        md.append(f"| **Object Accuracy** | {fmt_pct(comp.object_accuracy)} | Mean object alignment score on matched pairs |")
        md.append(f"| **Condition Preservation** | {fmt_pct(comp.condition_preservation_rate)} | Fraction of gold conditions retained in matched facts |")
        md.append(f"| **Exception Preservation** | {fmt_pct(comp.exception_preservation_rate)} | Fraction of gold exemptions/carve-outs retained (Critical) |")
        md.append(f"| **Threshold Preservation** | {fmt_pct(comp.threshold_preservation_rate)} | Fraction of numeric/financial thresholds retained |")
        md.append(f"| **Time/Deadline Preservation** | {fmt_pct(comp.time_preservation_rate)} | Fraction of explicit deadlines/timelines retained |")
        md.append(f"| **Evidence Support Rate** | {fmt_pct(comp.evidence_support_rate)} | Predictions with verifiable verbatim source text |")
        md.append(f"| **Duplicate Prediction Rate** | {fmt_pct(comp.duplicate_prediction_rate)} | Redundant extra predictions over total predictions |")
        md.append(f"| **Clause Coverage** | {fmt_pct(comp.clause_coverage)} | Percentage of gold clauses with at least 1 extracted fact |")
        md.append("")

        # 4. Per-Predicate Breakdown
        md.append("## 4. Per-Predicate Performance Breakdown")
        md.append("")
        md.append("| Predicate | Gold Count | Extracted Count | Matched (TP) | Precision | Recall | F1 |")
        md.append("|---|---|---|---|---|---|---|")
        for pb in summary.predicate_breakdown:
            md.append(f"| `{pb.predicate}` | {pb.gold_count} | {pb.predicted_count} | {pb.matched_count} | {fmt_pct(pb.precision)} | {fmt_pct(pb.recall)} | {fmt_pct(pb.f1_score)} |")
        md.append("")

        # 5. Error Breakdown by Regulatory Impact
        md.append("## 5. Error Taxonomy & Regulatory Impact Distribution")
        md.append("")
        md.append("### Distribution by Regulatory Impact")
        md.append("")
        md.append("| Severity | Error Count | Impact Rationale |")
        md.append("|---|---|---|")
        md.append(f"| 🚨 **CRITICAL** | {summary.regulatory_impact_counts.get('CRITICAL', 0)} | Inverted obligations/prohibitions, omitted carve-outs, hallucinated rules |")
        md.append(f"| ⚠️ **HIGH** | {summary.regulatory_impact_counts.get('HIGH', 0)} | Missed obligations, wrong actor/target, corrupted thresholds/deadlines |")
        md.append(f"| ℹ️ **MEDIUM** | {summary.regulatory_impact_counts.get('MEDIUM', 0)} | Dropped conditions, clause misattribution, weak evidence grounding |")
        md.append(f"| 🔍 **LOW** | {summary.regulatory_impact_counts.get('LOW', 0)} | Duplicate extra facts, ambiguous borderline wording |")
        md.append("")

        md.append("### Distribution by Error Category")
        md.append("")
        md.append("| Error Category | Count |")
        md.append("|---|---|")
        for cat_name, cnt in sorted(summary.error_counts.items(), key=lambda x: x[1], reverse=True):
            md.append(f"| `{cat_name}` | {cnt} |")
        md.append("")

        # 6. Highest Impact Missed Facts (False Negatives)
        md.append("## 6. Critical & High-Impact Missed Facts (False Negatives)")
        md.append("")
        critical_fns = [fn for fn in false_negatives if fn.get("regulatory_impact") in ["CRITICAL", "HIGH"]][:5]
        if critical_fns:
            for idx, fn in enumerate(critical_fns, start=1):
                md.append(f"### {idx}. {fn.get('gold_fact_id')} — Clause `{fn.get('clause_reference')}`")
                md.append(f"- **Subject:** `{fn.get('subject')}`")
                md.append(f"- **Predicate:** `{fn.get('predicate')}`")
                md.append(f"- **Object:** `{fn.get('object')}`")
                if fn.get("threshold"):
                    md.append(f"- **Threshold:** `{fn.get('threshold')}`")
                if fn.get("time"):
                    md.append(f"- **Deadline:** `{fn.get('time')}`")
                md.append(f"- **Source Evidence:** *\"{fn.get('evidence_text')}\"*")
                md.append(f"- **Failure Reason:** {fn.get('reason')}")
                md.append("")
        else:
            md.append("*No critical or high-impact false negatives identified.*")
            md.append("")

        # 7. False Positives (Fabricated or Unsupported Facts)
        md.append("## 7. Unsupported Predictions (False Positives Sample)")
        md.append("")
        critical_fps = [fp for fp in false_positives if fp.get("regulatory_impact") in ["CRITICAL", "HIGH"]][:5]
        if critical_fps:
            for idx, fp in enumerate(critical_fps, start=1):
                md.append(f"### {idx}. Extracted Prediction `{fp.get('predicted_fact_id', '')[:8]}`")
                md.append(f"- **Extracted Triple:** `{fp.get('subject')}` -> `{fp.get('predicate')}` -> `{fp.get('object')}`")
                md.append(f"- **Cited Evidence:** *\"{fp.get('evidence')}\"*")
                md.append(f"- **Diagnosis:** {fp.get('reason')}")
                md.append("")
        else:
            md.append("*No critical false positives identified.*")
            md.append("")

        # 8. Ambiguous / Unresolved Matches
        ambiguous_matches = [d for d in decisions if d.status == "AMBIGUOUS"]
        md.append("## 8. Ambiguous Matches Requiring Human Adjudication")
        md.append("")
        if ambiguous_matches:
            md.append(f"Found **{len(ambiguous_matches)}** borderline alignment pairs needing human review:")
            md.append("")
            for idx, d in enumerate(ambiguous_matches[:5], start=1):
                g = d.gold_fact or {}
                p = d.predicted_fact or {}
                md.append(f"### {idx}. Gold `{d.gold_fact_id}` vs Prediction `{d.predicted_fact_id[:8] if d.predicted_fact_id else ''}`")
                md.append(f"- **Gold:** `{g.get('subject')}` [{g.get('predicate')}] `{g.get('object')}`")
                md.append(f"- **Pred:** `{p.get('subject')}` [{p.get('predicate')}] `{p.get('object')}`")
                md.append(f"- **Alignment Score:** `{d.scores.overall_score:.2f}` (Subject: `{d.scores.subject_score:.2f}`, Object: `{d.scores.object_score:.2f}`)")
                md.append(f"- **Adjudication Note:** {d.match_reason}")
                md.append("")
        else:
            md.append("*No ambiguous matches detected.*")
            md.append("")

        # 9. Next Steps and Limitations
        md.append("## 9. Limitations & Recommended Next Actions")
        md.append("")
        md.append("1. **Verify Seed Annotations:** Complete human verification for the 128 starter rows in `evaluation/data/sebi_surveillance_goldset_seed.xlsx` to produce official baseline numbers.")
        md.append("2. **Address Qualifier Dropping:** If exception or condition preservation is low, reinforce few-shot extraction prompts with explicit carve-out extraction rules.")
        md.append("3. **Inspect False Negatives:** Review the generated `evaluation/results/false_negatives_<run_id>.csv` to identify specific circular sections with low coverage.")
        md.append("")

        report_content = "\n".join(md)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        # Also write/update latest report symlink or copy
        latest_path = self.reports_dir / "evaluation_report_latest.md"
        with open(latest_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        return out_path
