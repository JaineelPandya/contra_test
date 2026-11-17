# RegGraph Factual Accuracy Evaluation Report

> **Run ID:** `eval_20261009_055052_6c7011`  
> **Timestamp:** `2026-10-09T05:50:53.323599+00:00`  
> **LLM Provider:** `groq` | **Model:** `openai/gpt-oss-20b`  
> **Prompt SHA-256:** `9c40b8c7f0bb0a06`  
> **Dataset:** `evaluation/data/gold_facts_seed.jsonl`  

## 1. Dataset Verification & Baseline Status

⚠️ **Exploratory Run Warning:** This run evaluated unverified seed annotations. Official accuracy baselines require human verification.

| Metric | Count |
|---|---|
| Total Gold Facts in Dataset | 128 |
| Human-Verified Gold Facts | 0 |
| Unverified / Seed Facts | 128 |
| Gold Facts Included in This Run | 0 |
| Model Predictions Extracted | 106 |

## 2. Core Fact-Level Accuracy

| Metric | Score | Raw Counts |
|---|---|---|
| **Precision** | **0.0%** | TP: 0 / (TP: 0 + FP: 106) |
| **Recall** | **N/A (0 denominator)** | TP: 0 / (TP: 0 + FN: 0) |
| **F1-Score** | **N/A (0 denominator)** | Harmonic mean of precision & recall |

## 3. Component & Qualifier Preservation

| Component / Metric | Score | Description / Denominator |
|---|---|---|
| **Subject Accuracy** | N/A (0 denominator) | Mean subject alignment score on matched pairs |
| **Predicate Accuracy** | N/A (0 denominator) | Proportion of matched pairs with exact canonical predicate match |
| **Object Accuracy** | N/A (0 denominator) | Mean object alignment score on matched pairs |
| **Condition Preservation** | N/A (0 denominator) | Fraction of gold conditions retained in matched facts |
| **Exception Preservation** | N/A (0 denominator) | Fraction of gold exemptions/carve-outs retained (Critical) |
| **Threshold Preservation** | N/A (0 denominator) | Fraction of numeric/financial thresholds retained |
| **Time/Deadline Preservation** | N/A (0 denominator) | Fraction of explicit deadlines/timelines retained |
| **Evidence Support Rate** | 100.0% | Predictions with verifiable verbatim source text |
| **Duplicate Prediction Rate** | 8.5% | Redundant extra predictions over total predictions |
| **Clause Coverage** | N/A (0 denominator) | Percentage of gold clauses with at least 1 extracted fact |

## 4. Per-Predicate Performance Breakdown

| Predicate | Gold Count | Extracted Count | Matched (TP) | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| `FACTPREDICATE.APPLIES_TO` | 0 | 3 | 0 | 0.0% | N/A (0 denominator) | N/A (0 denominator) |
| `FACTPREDICATE.CREDITED_TO` | 0 | 2 | 0 | 0.0% | N/A (0 denominator) | N/A (0 denominator) |
| `FACTPREDICATE.DEFINES` | 0 | 5 | 0 | 0.0% | N/A (0 denominator) | N/A (0 denominator) |
| `FACTPREDICATE.DELETES` | 0 | 1 | 0 | 0.0% | N/A (0 denominator) | N/A (0 denominator) |
| `FACTPREDICATE.EXCEPTION_TO` | 0 | 8 | 0 | 0.0% | N/A (0 denominator) | N/A (0 denominator) |
| `FACTPREDICATE.EXTENDS` | 0 | 1 | 0 | 0.0% | N/A (0 denominator) | N/A (0 denominator) |
| `FACTPREDICATE.PAYMENT_DEADLINE` | 0 | 2 | 0 | 0.0% | N/A (0 denominator) | N/A (0 denominator) |
| `FACTPREDICATE.PERMITS` | 0 | 8 | 0 | 0.0% | N/A (0 denominator) | N/A (0 denominator) |
| `FACTPREDICATE.PROHIBITS` | 0 | 6 | 0 | 0.0% | N/A (0 denominator) | N/A (0 denominator) |
| `FACTPREDICATE.REFERENCES` | 0 | 1 | 0 | 0.0% | N/A (0 denominator) | N/A (0 denominator) |
| `FACTPREDICATE.REQUIRES` | 0 | 61 | 0 | 0.0% | N/A (0 denominator) | N/A (0 denominator) |
| `FACTPREDICATE.SUBJECT_TO` | 0 | 6 | 0 | 0.0% | N/A (0 denominator) | N/A (0 denominator) |
| `FACTPREDICATE.SUBMITS_TO` | 0 | 2 | 0 | 0.0% | N/A (0 denominator) | N/A (0 denominator) |

## 5. Error Taxonomy & Regulatory Impact Distribution

### Distribution by Regulatory Impact

| Severity | Error Count | Impact Rationale |
|---|---|---|
| 🚨 **CRITICAL** | 0 | Inverted obligations/prohibitions, omitted carve-outs, hallucinated rules |
| ⚠️ **HIGH** | 97 | Missed obligations, wrong actor/target, corrupted thresholds/deadlines |
| ℹ️ **MEDIUM** | 0 | Dropped conditions, clause misattribution, weak evidence grounding |
| 🔍 **LOW** | 9 | Duplicate extra facts, ambiguous borderline wording |

### Distribution by Error Category

| Error Category | Count |
|---|---|
| `UNSUPPORTED_OR_FABRICATED_FACT` | 97 |
| `DUPLICATE_PREDICTION` | 9 |

## 6. Critical & High-Impact Missed Facts (False Negatives)

*No critical or high-impact false negatives identified.*

## 7. Unsupported Predictions (False Positives Sample)

### 1. Extracted Prediction `bd461488`
- **Extracted Triple:** `subscription to the issue of Non-Convertible Securities` -> `FactPredicate.PERMITS` -> `during trading window closure period`
- **Cited Evidence:** *"“Allowing subscription to the issue of Non-Convertible Securities during trading
window closure period” dated December 30, 2024 with reference number
SEBI/HO/ISD/ISD-PoD-2/P/CIR/2024/180; and"*
- **Diagnosis:** Predicted fact does not match any gold standard fact

### 2. Extracted Prediction `088b1ffb`
- **Extracted Triple:** `Immediate Relatives of Designated Persons` -> `FactPredicate.REQUIRES` -> `trading window closure period on account of declaration of financial results`
- **Cited Evidence:** *"“Trading Window closure period under Clause 4 of Schedule B read with Regulation
9 of Securities and Exchange Board of India (Prohibition of Insider Trading)
Regulations, 2015 – Extension of automated implementation of trading window
closure to Immediate Relatives of Designated Persons, on account of declaration of
financial results” dated April 21, 2025 with reference number SEBI/HO/ISD/ISD-PoD-
2/P/CIR/2025/55."*
- **Diagnosis:** Predicted fact does not match any gold standard fact

### 3. Extracted Prediction `ab742553`
- **Extracted Triple:** `Automated implementation of trading window closure` -> `FactPredicate.EXTENDS` -> `Immediate Relatives of Designated Persons`
- **Cited Evidence:** *"“Trading Window closure period under Clause 4 of Schedule B read with Regulation
9 of Securities and Exchange Board of India (Prohibition of Insider Trading)
Regulations, 2015 – Extension of automated implementation of trading window
closure to Immediate Relatives of Designated Persons, on account of declaration of
financial results” dated April 21, 2025 with reference number SEBI/HO/ISD/ISD-PoD-
2/P/CIR/2025/55."*
- **Diagnosis:** Predicted fact does not match any gold standard fact

### 4. Extracted Prediction `aada873b`
- **Extracted Triple:** `all directions/ instructions contained in the Circulars listed out in the Appendix to this Master Circular` -> `FactPredicate.DELETES` -> `to the extent they relate to Surveillance of Securities Market`
- **Cited Evidence:** *"With the issuance of this Master Circular, all directions/ instructions contained in the Circulars
listed out in the Appendix to this Master Circular shall stand rescinded to the extent they relate
to Surveillance of Securities Market."*
- **Diagnosis:** Predicted fact does not match any gold standard fact

### 5. Extracted Prediction `95902289`
- **Extracted Triple:** `actions taken under rescinded circulars prior to rescission` -> `FactPredicate.DEFINES` -> `corresponding provisions of this Master Circular`
- **Cited Evidence:** *"Notwithstanding such rescission:
a) anything done or any action taken or purported to have been done or taken under the
rescinded circulars, prior to such rescission, shall be deemed to have been done or
taken under the corresponding provisions of this Master Circular; and
b) any application made to the Board under the rescinded circulars prior to such
rescission, and pending before it, shall be deemed to have been made under the
corresponding provisions of this Master Circular; and
c) the previous operation of the rescinded circulars or anything done or suffered
thereunder, any right, privilege, obligation or liability acquired, accrued or incurred
under the rescinded circulars, any penalty, incurred in respect of any violation
committed against the rescinded circulars, or any investigation, legal proceeding
or remedy in respect of any such right, privilege, obligation, liability, penalty
as aforesaid, shall not be affected by such rescission and shall be enforceable as if
the rescinded circulars have continued to be in force."*
- **Diagnosis:** Predicted fact does not match any gold standard fact

## 8. Ambiguous Matches Requiring Human Adjudication

*No ambiguous matches detected.*

## 9. Limitations & Recommended Next Actions

1. **Verify Seed Annotations:** Complete human verification for the 128 starter rows in `evaluation/data/sebi_surveillance_goldset_seed.xlsx` to produce official baseline numbers.
2. **Address Qualifier Dropping:** If exception or condition preservation is low, reinforce few-shot extraction prompts with explicit carve-out extraction rules.
3. **Inspect False Negatives:** Review the generated `evaluation/results/false_negatives_<run_id>.csv` to identify specific circular sections with low coverage.
