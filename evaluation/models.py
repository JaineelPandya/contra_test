"""Data models for RegGraph factual accuracy evaluation."""
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4
from pydantic import BaseModel, Field

from app.models.fact import Fact, FactPredicate


class VerificationStatus(str, Enum):
    """Human verification status of an annotation."""
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    SEED = "SEED"
    REJECTED = "REJECTED"


class RegulatoryImpact(str, Enum):
    """Regulatory severity of an extraction error."""
    CRITICAL = "CRITICAL"    # Reversing obligation/prohibition, omitting carve-out, hallucination
    HIGH = "HIGH"            # Missing obligation, wrong actor/action, incorrect threshold/deadline
    MEDIUM = "MEDIUM"        # Missing condition, wrong clause reference, weak evidence
    LOW = "LOW"              # Duplicate prediction, stylistic wording variation


class ErrorCategory(str, Enum):
    """Granular error taxonomy for factual mismatches."""
    # Core triple errors
    MISSING_OBLIGATION = "MISSING_OBLIGATION"
    UNSUPPORTED_OR_FABRICATED_FACT = "UNSUPPORTED_OR_FABRICATED_FACT"
    WRONG_SUBJECT = "WRONG_SUBJECT"
    WRONG_PREDICATE = "WRONG_PREDICATE"
    WRONG_OBJECT = "WRONG_OBJECT"

    # Qualifier errors
    LOST_CONDITION = "LOST_CONDITION"
    LOST_EXCEPTION = "LOST_EXCEPTION"
    LOST_OR_INCORRECT_THRESHOLD = "LOST_OR_INCORRECT_THRESHOLD"
    LOST_OR_INCORRECT_DEADLINE = "LOST_OR_INCORRECT_DEADLINE"

    # Evidence & provenance errors
    UNSUPPORTED_EVIDENCE = "UNSUPPORTED_EVIDENCE"
    WRONG_CLAUSE_ATTRIBUTION = "WRONG_CLAUSE_ATTRIBUTION"

    # Alignment & redundancy errors
    DUPLICATE_PREDICTION = "DUPLICATE_PREDICTION"
    AMBIGUOUS_MATCH = "AMBIGUOUS_MATCH"


class GoldFact(BaseModel):
    """Gold-standard reference fact for accuracy evaluation."""
    gold_fact_id: str = Field(..., description="Stable gold fact ID, e.g. SEBI-SURV-001")
    document_title: str = Field(..., description="Authoritative title of the circular")
    document_ref: Optional[str] = Field(None, description="Official reference number of circular")
    document_id: Optional[str] = Field(None, description="Internal document UUID if linked")
    document_last_updated: Optional[str] = Field(None, description="Date of issuance or last revision")
    section_ref: Optional[str] = Field(None, description="Section or paragraph reference, e.g. 1.1")
    clause_reference: Optional[str] = Field(None, description="Granular clause reference, e.g. 1.1.1")
    source_chunk_id: Optional[str] = Field(None, description="Chunk UUID containing this fact")
    pdf_page: Optional[int] = Field(None, description="PDF page number of source clause")
    source_clause_text: Optional[str] = Field(None, description="Full text or excerpt of source clause")

    subject: str = Field(..., description="Regulated actor or entity")
    predicate: str = Field(..., description="Controlled predicate label, e.g. REQUIRES, PROHIBITS")
    object: str = Field(..., description="Action, requirement, or object entity")

    conditions: List[str] = Field(default_factory=list, description="Triggering conditions or scope")
    exceptions: List[str] = Field(default_factory=list, description="Explicit exclusions or carve-outs")
    threshold: Optional[str] = Field(None, description="Quantified financial/numerical threshold")
    time: Optional[str] = Field(None, description="Deadline or timing requirement")
    rule_reference: Optional[str] = Field(None, description="Cross-referenced statute or rule")
    evidence_text: Optional[str] = Field(None, description="Verbatim text span supporting the fact")
    fact_category: Optional[str] = Field(None, description="High-level category: OBLIGATION, SCOPE, etc.")

    annotation_status: str = Field(
        default="UNVERIFIED",
        description="Annotation status string from source dataset",
    )
    is_verified: bool = Field(
        default=False,
        description="True ONLY if verified by human against authoritative circular",
    )
    review_notes: Optional[str] = Field(None, description="Adjudication notes or caveats")

    def get_canonical_predicate(self) -> Optional[FactPredicate]:
        """Attempt to map predicate string to canonical FactPredicate enum."""
        try:
            return FactPredicate(self.predicate)
        except ValueError:
            # Common aliases
            mapping = {
                "PERMITTED": FactPredicate.PERMITS,
                "EXCLUDES": FactPredicate.EXCEPTION_TO,
                "EXEMPTS": FactPredicate.EXCEPTION_TO,
                "RESTRICTS": FactPredicate.PROHIBITS,
                "TRIGGERS": FactPredicate.REQUIRES,
                "EXPECTS": FactPredicate.REQUIRES,
            }
            return mapping.get(self.predicate.upper())


class MatchComponentScores(BaseModel):
    """Scores for individual components of a fact match."""
    subject_score: float = 0.0
    predicate_match: bool = False
    object_score: float = 0.0
    condition_score: float = 1.0     # 1.0 if neither has condition or both match
    exception_score: float = 1.0     # 1.0 if neither has exception or both match
    threshold_score: float = 1.0     # 1.0 if neither has threshold or both match
    time_score: float = 1.0          # 1.0 if neither has time or both match
    evidence_score: float = 1.0      # Evidence grounding / overlap score
    clause_match: bool = True        # Clause reference alignment
    overall_score: float = 0.0


class MatchDecision(BaseModel):
    """Alignment decision between a gold fact and an extracted prediction."""
    decision_id: str = Field(default_factory=lambda: str(uuid4()))
    gold_fact_id: Optional[str] = None
    predicted_fact_id: Optional[str] = None
    status: str = Field(..., description="MATCH, MISMATCH, AMBIGUOUS, UNMATCHED_GOLD, UNMATCHED_PRED")

    scores: MatchComponentScores = Field(default_factory=MatchComponentScores)

    error_category: Optional[ErrorCategory] = None
    regulatory_impact: Optional[RegulatoryImpact] = None
    match_reason: str = Field(..., description="Human-readable justification for the decision")

    gold_fact: Optional[Dict[str, Any]] = None
    predicted_fact: Optional[Dict[str, Any]] = None


class CoreMetrics(BaseModel):
    """Standard evaluation metrics with zero-denominator safety."""
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None


class ComponentMetrics(BaseModel):
    """Component-level precision/accuracy scores."""
    subject_accuracy: Optional[float] = None
    predicate_accuracy: Optional[float] = None
    object_accuracy: Optional[float] = None
    condition_preservation_rate: Optional[float] = None
    exception_preservation_rate: Optional[float] = None
    threshold_preservation_rate: Optional[float] = None
    time_preservation_rate: Optional[float] = None
    evidence_support_rate: Optional[float] = None
    duplicate_prediction_rate: Optional[float] = None
    clause_coverage: Optional[float] = None


class PredicateBreakdown(BaseModel):
    """Per-predicate performance statistics."""
    predicate: str
    gold_count: int = 0
    predicted_count: int = 0
    matched_count: int = 0
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None


class EvaluationSummary(BaseModel):
    """Complete summary of an evaluation run."""
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_name: Optional[str] = None
    provider_name: Optional[str] = None
    prompt_version_or_hash: Optional[str] = None

    dataset_path: str
    total_gold_facts: int = 0
    verified_gold_facts: int = 0
    unverified_gold_facts: int = 0
    evaluated_gold_facts: int = 0
    total_predictions: int = 0

    core_metrics: CoreMetrics
    component_metrics: ComponentMetrics
    predicate_breakdown: List[PredicateBreakdown] = Field(default_factory=list)
    error_counts: Dict[str, int] = Field(default_factory=dict)
    regulatory_impact_counts: Dict[str, int] = Field(default_factory=dict)

    is_verified_baseline: bool = Field(
        default=False,
        description="True if metrics are computed exclusively on verified ground truth",
    )
