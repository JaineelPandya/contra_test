"""Unit tests for Fact Extraction Hardening.

Verifies:
1. Evidence Grounding:
   - Valid evidence -> accepted
   - Missing evidence -> rejected
   - Hallucinated evidence -> rejected
   - Whitespace/case differences -> accepted
   - No fallback to chunk[:250]
2. Predicates:
   - REQUIRES, REPLACES, MODIFIES, REPEALS -> accepted
   - Invalid predicate -> rejected
   - Invalid predicate -> NEVER becomes REQUIRES
3. Conditions:
   - "provided that" -> condition populated
   - "subject to" -> condition populated
   - "if" -> condition populated when legally meaningful
   - "on the condition that", "only if", "where", "when" -> condition populated
4. Exceptions:
   - "except" -> exception populated
   - "unless" -> exception populated
   - "notwithstanding" -> exception/qualifier preserved
   - "shall not apply to", "does not apply to", "excluding" -> exception populated
5. Amendment Detection:
   - "shall be replaced" -> REPLACES
   - "shall be substituted" -> REPLACES
   - "shall be modified" -> MODIFIES
   - "supersedes" -> SUPERSEDES
   - "repealed" -> REPEALS
   - "reference to Circular X" -> NO amendment relationship
6. Entity Resolution:
   - "Asset Management Company" / "AMC" -> Canonical entity ORGANIZATION
"""
import pytest
from app.models.chunk import Chunk
from app.models.fact import Fact, FactPredicate, normalize_text
from app.llm.groq_provider import GroqProvider
from app.services.fact_extractor import extract_qualifier_spans, validate_fact_evidence
from app.services.entity_resolver import EntityResolver


# ==============================================================================
# 1. EVIDENCE GROUNDING TESTS
# ==============================================================================

def test_normalize_text_whitespace_and_casing() -> None:
    text1 = "Asset   Management \n\n Companies   shall   maintain  net worth."
    text2 = "asset management companies shall maintain net worth."
    assert normalize_text(text1) == normalize_text(text2)


def test_evidence_validation_accepted() -> None:
    chunk = Chunk(
        chunk_id="c1",
        document_id="d1",
        content="Asset Management Companies shall maintain a minimum net worth of INR 50 Crores.",
        token_count=15,
    )
    fact = Fact(
        document_id="d1",
        chunk_id="c1",
        subject="Asset Management Companies",
        predicate=FactPredicate.REQUIRES,
        object="Maintain minimum net worth",
        source_text="Asset Management Companies shall maintain a minimum net worth of INR 50 Crores.",
    )
    valid, reason = validate_fact_evidence(fact, chunk)
    assert valid is True
    assert reason is None


def test_evidence_validation_whitespace_difference_accepted() -> None:
    chunk = Chunk(
        chunk_id="c1",
        document_id="d1",
        content="Asset Management Companies shall maintain\n\na minimum net worth of INR 50 Crores.",
        token_count=15,
    )
    fact = Fact(
        document_id="d1",
        chunk_id="c1",
        subject="Asset Management Companies",
        predicate=FactPredicate.REQUIRES,
        object="Maintain minimum net worth",
        source_text="asset  management   companies shall maintain a minimum net worth of inr 50 crores.",
    )
    valid, reason = validate_fact_evidence(fact, chunk)
    assert valid is True


def test_evidence_validation_missing_rejected() -> None:
    chunk = Chunk(
        chunk_id="c1",
        document_id="d1",
        content="Asset Management Companies shall maintain a minimum net worth of INR 50 Crores.",
        token_count=15,
    )
    fact = Fact(
        document_id="d1",
        chunk_id="c1",
        subject="Asset Management Companies",
        predicate=FactPredicate.REQUIRES,
        object="Maintain minimum net worth",
        source_text="",
    )
    valid, reason = validate_fact_evidence(fact, chunk)
    assert valid is False
    assert reason == "empty_evidence"


def test_evidence_validation_hallucinated_rejected() -> None:
    chunk = Chunk(
        chunk_id="c1",
        document_id="d1",
        content="Asset Management Companies shall maintain a minimum net worth of INR 50 Crores.",
        token_count=15,
    )
    fact = Fact(
        document_id="d1",
        chunk_id="c1",
        subject="Brokers",
        predicate=FactPredicate.REQUIRES,
        object="Submit margin reports daily",
        source_text="Brokers shall deposit margin within 24 hours of trading.",
    )
    valid, reason = validate_fact_evidence(fact, chunk)
    assert valid is False
    assert reason == "evidence_not_found"


def test_groq_parse_raw_facts_rejects_missing_and_hallucinated_evidence() -> None:
    provider = GroqProvider()
    chunk = Chunk(
        chunk_id="chk-evidence-1",
        document_id="doc-1",
        content="Asset Management Companies shall maintain a minimum net worth of INR 50 Crores.",
        token_count=15,
    )

    raw_data = {
        "facts": [
            # 1. Valid grounded evidence -> accepted
            {
                "subject": "Asset Management Companies",
                "predicate": "REQUIRES",
                "object": "Maintain net worth of INR 50 Crores",
                "evidence": "Asset Management Companies shall maintain a minimum net worth of INR 50 Crores.",
                "chunk_id": "chk-evidence-1",
            },
            # 2. Empty evidence -> rejected
            {
                "subject": "Asset Management Companies",
                "predicate": "REQUIRES",
                "object": "Do something",
                "evidence": "",
                "chunk_id": "chk-evidence-1",
            },
            # 3. Hallucinated evidence not in chunk -> rejected
            {
                "subject": "Trading Member",
                "predicate": "REQUIRES",
                "object": "Hallucinated requirement",
                "evidence": "Trading Members shall submit quarterly audit reports to the committee.",
                "chunk_id": "chk-evidence-1",
            },
        ]
    }

    parsed = provider._parse_raw_facts(raw_data, "doc-1", [chunk], None)
    assert len(parsed) == 1
    assert parsed[0].subject == "Asset Management Companies"
    assert parsed[0].source_text == "Asset Management Companies shall maintain a minimum net worth of INR 50 Crores."


def test_groq_never_injects_chunk_slice_as_evidence() -> None:
    """Verifies that missing evidence is NEVER replaced with chunk.content[:250]."""
    provider = GroqProvider()
    chunk = Chunk(
        chunk_id="chk-slice-test",
        document_id="doc-1",
        content="This is the full chunk content text that must never be fabricated into an evidence field.",
        token_count=18,
    )

    raw_data = {
        "facts": [
            {
                "subject": "Trading Member",
                "predicate": "REQUIRES",
                "object": "Action without evidence field",
                "chunk_id": "chk-slice-test",
            }
        ]
    }

    parsed = provider._parse_raw_facts(raw_data, "doc-1", [chunk], None)
    # The fact must be rejected, not created with chunk[:250]
    assert len(parsed) == 0


# ==============================================================================
# 2. PREDICATE VALIDATION TESTS
# ==============================================================================

@pytest.mark.parametrize("pred_val", ["REQUIRES", "REPLACES", "MODIFIES", "REPEALS", "SUPERSEDES", "DELETES"])
def test_valid_predicates_accepted(pred_val: str) -> None:
    provider = GroqProvider()
    chunk = Chunk(
        chunk_id="chk-p1",
        document_id="doc-p1",
        content="Regulatory provision text for testing valid predicates.",
        token_count=10,
    )
    raw_data = {
        "facts": [
            {
                "subject": "Entity",
                "predicate": pred_val,
                "object": "Action",
                "evidence": "Regulatory provision text for testing valid predicates.",
                "chunk_id": "chk-p1",
            }
        ]
    }
    parsed = provider._parse_raw_facts(raw_data, "doc-p1", [chunk], None)
    assert len(parsed) == 1
    assert parsed[0].predicate.value == pred_val


def test_invalid_predicate_rejected_and_never_becomes_requires() -> None:
    """Invalid predicates must NEVER silently default to REQUIRES."""
    provider = GroqProvider()
    chunk = Chunk(
        chunk_id="chk-inv-1",
        document_id="doc-inv-1",
        content="Provision text that mentions an invalid predicate.",
        token_count=10,
    )
    raw_data = {
        "facts": [
            # Invalid fabricated predicates
            {
                "subject": "Entity",
                "predicate": "MANDATES",  # Not in FactPredicate
                "object": "Action",
                "evidence": "Provision text that mentions an invalid predicate.",
                "chunk_id": "chk-inv-1",
            },
            {
                "subject": "Entity",
                "predicate": "SUBSTITUTES_TOTALLY",  # Not in FactPredicate
                "object": "Action",
                "evidence": "Provision text that mentions an invalid predicate.",
                "chunk_id": "chk-inv-1",
            },
            {
                "subject": "Entity",
                "predicate": "ALTERS",  # Not in FactPredicate
                "object": "Action",
                "evidence": "Provision text that mentions an invalid predicate.",
                "chunk_id": "chk-inv-1",
            },
        ]
    }
    parsed = provider._parse_raw_facts(raw_data, "doc-inv-1", [chunk], None)
    # None of these should be accepted, and none should become REQUIRES
    assert len(parsed) == 0


def test_predicate_normalization_whitespace_and_case() -> None:
    """Formatting variations (lowercase, whitespace) should safely normalize."""
    provider = GroqProvider()
    chunk = Chunk(
        chunk_id="chk-norm-1",
        document_id="doc-norm-1",
        content="Provision text for case normalization test.",
        token_count=10,
    )
    raw_data = {
        "facts": [
            {
                "subject": "Entity",
                "predicate": "  requires  ",
                "object": "Action",
                "evidence": "Provision text for case normalization test.",
                "chunk_id": "chk-norm-1",
            },
            {
                "subject": "Entity",
                "predicate": "replaces",
                "object": "Action",
                "evidence": "Provision text for case normalization test.",
                "chunk_id": "chk-norm-1",
            },
        ]
    }
    parsed = provider._parse_raw_facts(raw_data, "doc-norm-1", [chunk], None)
    assert len(parsed) == 2
    assert parsed[0].predicate == FactPredicate.REQUIRES
    assert parsed[1].predicate == FactPredicate.REPLACES


# ==============================================================================
# 3. CONDITION EXTRACTION TESTS
# ==============================================================================

def test_condition_provided_that() -> None:
    text = "Trading members shall report transactions, provided that off-market trades are certified."
    qualifiers = extract_qualifier_spans(text)
    assert len(qualifiers["conditions"]) >= 1
    assert "provided that off-market trades are certified" in qualifiers["conditions"][0].lower()


def test_condition_subject_to() -> None:
    text = "AMCs shall maintain minimum net worth of INR 50 Crores, subject to the conditions specified in Clause 5."
    qualifiers = extract_qualifier_spans(text)
    assert len(qualifiers["conditions"]) >= 1
    assert "subject to the conditions specified in clause 5" in qualifiers["conditions"][0].lower()


def test_condition_if() -> None:
    text = "Trading members must submit annual audit reports if net turnover exceeds INR 100 Crores."
    qualifiers = extract_qualifier_spans(text)
    assert len(qualifiers["conditions"]) >= 1
    assert "if net turnover exceeds inr 100 crores" in qualifiers["conditions"][0].lower()


def test_additional_conditions_on_condition_that_only_if_where_when() -> None:
    c1 = extract_qualifier_spans("Members shall report daily, on the condition that trades were executed.")
    assert any("on the condition that" in c.lower() for c in c1["conditions"])

    c2 = extract_qualifier_spans("Only if approved by the Board, the limit shall be extended.")
    assert any("only if approved by the board" in c.lower() for c in c2["conditions"])

    c3 = extract_qualifier_spans("Where the broker fails to maintain records, a penalty shall be levied.")
    assert any("where the broker fails to maintain records" in c.lower() for c in c3["conditions"])

    c4 = extract_qualifier_spans("When required by SEBI, inspection shall be permitted.")
    assert any("when required by sebi" in c.lower() for c in c4["conditions"])


# ==============================================================================
# 4. EXCEPTION EXTRACTION TESTS
# ==============================================================================

def test_exception_except() -> None:
    text = "All funds must comply with investment ratios, except for infrastructure debt funds."
    qualifiers = extract_qualifier_spans(text)
    assert len(qualifiers["exceptions"]) >= 1
    assert "except for infrastructure debt funds" in qualifiers["exceptions"][0].lower()


def test_exception_unless() -> None:
    text = "Margin must be collected in advance, unless exempted under Rule 12."
    qualifiers = extract_qualifier_spans(text)
    assert len(qualifiers["exceptions"]) >= 1
    assert "unless exempted under rule 12" in qualifiers["exceptions"][0].lower()


def test_exception_notwithstanding() -> None:
    text = "This circular shall apply, notwithstanding anything contained in prior circulars."
    qualifiers = extract_qualifier_spans(text)
    assert len(qualifiers["exceptions"]) >= 1
    assert "notwithstanding anything contained in prior circulars" in qualifiers["exceptions"][0].lower()


def test_exception_shall_not_apply_to() -> None:
    text = "This requirement shall not apply to infrastructure debt funds."
    qualifiers = extract_qualifier_spans(text)
    assert len(qualifiers["exceptions"]) >= 1
    assert "shall not apply to infrastructure debt funds" in qualifiers["exceptions"][0].lower()


def test_spec_combined_condition_and_exception_example() -> None:
    """Exact example from Section 2 of the implementation prompt."""
    text = (
        "AMCs shall maintain minimum net worth of INR 50 Crores, "
        "subject to the conditions specified in Clause 5, "
        "except for infrastructure debt funds."
    )
    qualifiers = extract_qualifier_spans(text)
    assert qualifiers["conditions"] == ["subject to the conditions specified in Clause 5"]
    assert qualifiers["exceptions"] == ["except for infrastructure debt funds"]


def test_qualifiers_preserved_in_fallback_extraction() -> None:
    provider = GroqProvider()
    chunk = Chunk(
        chunk_id="chk-fb-qual",
        document_id="doc-fb",
        content=(
            "1. AMCs shall maintain minimum net worth of INR 50 Crores, "
            "subject to the conditions specified in Clause 5, "
            "except for infrastructure debt funds."
        ),
        token_count=35,
    )

    facts = provider._fallback_fact_extraction("doc-fb", [chunk], None)
    assert len(facts) >= 1
    assert facts[0].conditions == ["subject to the conditions specified in Clause 5"]
    assert facts[0].exceptions == ["except for infrastructure debt funds"]


# ==============================================================================
# 5. AMENDMENT DETECTION TESTS
# ==============================================================================

def test_amendment_detection_replaces() -> None:
    provider = GroqProvider()
    c1 = Chunk(chunk_id="c1", document_id="d1", content="Paragraph 5 shall be replaced with the following provision...", token_count=15)
    c2 = Chunk(chunk_id="c2", document_id="d1", content="Clause 2 shall be substituted by new compliance limits.", token_count=15)

    f1 = provider._fallback_fact_extraction("d1", [c1], None)
    f2 = provider._fallback_fact_extraction("d1", [c2], None)

    assert len(f1) == 1 and f1[0].predicate == FactPredicate.REPLACES
    assert len(f2) == 1 and f2[0].predicate == FactPredicate.REPLACES


def test_amendment_detection_modifies() -> None:
    provider = GroqProvider()
    c = Chunk(chunk_id="c1", document_id="d1", content="The existing requirement shall be modified from INR 50 Crores to INR 75 Crores.", token_count=20)
    facts = provider._fallback_fact_extraction("d1", [c], None)
    assert len(facts) == 1
    assert facts[0].predicate == FactPredicate.MODIFIES


def test_amendment_detection_supersedes() -> None:
    provider = GroqProvider()
    c = Chunk(chunk_id="c1", document_id="d1", content="This framework supersedes earlier circular dated 15 March 2022 on the subject.", token_count=20)
    facts = provider._fallback_fact_extraction("d1", [c], None)
    assert len(facts) == 1
    assert facts[0].predicate == FactPredicate.SUPERSEDES


def test_amendment_detection_repeals() -> None:
    provider = GroqProvider()
    c = Chunk(chunk_id="c1", document_id="d1", content="Rule 8(3)(f) is hereby repealed with immediate effect.", token_count=15)
    facts = provider._fallback_fact_extraction("d1", [c], None)
    assert len(facts) == 1
    assert facts[0].predicate == FactPredicate.REPEALS


def test_reference_is_not_amendment() -> None:
    """Circular citations must NEVER be treated as an amendment (REPLACES, MODIFIES, SUPERSEDES, REPEALS)."""
    provider = GroqProvider()
    c = Chunk(
        chunk_id="c1",
        document_id="d1",
        content="Reference is invited to Circular XYZ dated 10 January 2025 regarding margin collection framework.",
        token_count=20,
    )
    facts = provider._fallback_fact_extraction("d1", [c], None)
    for f in facts:
        assert f.predicate not in {
            FactPredicate.MODIFIES,
            FactPredicate.REPLACES,
            FactPredicate.SUPERSEDES,
            FactPredicate.REPEALS,
            FactPredicate.AMENDS,
            FactPredicate.DELETES,
        }
        # It must be REFERENCES
        assert f.predicate == FactPredicate.REFERENCES


# ==============================================================================
# 6. ENTITY RESOLUTION TESTS
# ==============================================================================

def test_entity_resolver_amc_and_mutual_fund() -> None:
    resolver = EntityResolver()
    chunk = Chunk(
        chunk_id="c1",
        document_id="d1",
        content="AMCs and Mutual Funds must comply with SEBI instructions.",
        token_count=15,
    )
    entities = resolver.resolve_from_facts([], [chunk])
    norm_names = {e.normalized_name for e in entities}
    assert "ASSET MANAGEMENT COMPANY" in norm_names
    assert "MUTUAL FUND" in norm_names
    assert "SECURITIES AND EXCHANGE BOARD OF INDIA" in norm_names
