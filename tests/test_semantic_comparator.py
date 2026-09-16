"""Tests for SemanticComparator, attribute extraction, and change classification."""
from unittest.mock import MagicMock
import pytest
from app.models.fact import Fact, FactPredicate
from app.services.qualifier_extractor import extract_qualifier_spans
from app.services.semantic_comparator import SemanticComparator


def test_extract_qualifier_spans_rich_attributes():
    """Verify condition, exception, threshold, time, and circular extraction."""
    text = (
        "The Financial Disincentive(s) under the framework of FDSRL, if imposed, "
        "shall be credited by the MII concerned, within 15 working days, to the Investor Protection and Education Fund "
        "('SEBI–IPEF') established under the SEBI Act. Added in line with Circular no. CIR/ISD/01/2015 dated May 11, 2015. "
        "Total annual revenue exceeding Rs. 50 crore shall be subject to audit, except where specifically exempted."
    )
    qualifiers = extract_qualifier_spans(text)
    assert "if imposed" in qualifiers["conditions"]
    assert any("exempt" in e.lower() for e in qualifiers["exceptions"])
    assert qualifiers["time"] is not None
    assert "15 working days" in qualifiers["time"].lower()
    assert qualifiers["threshold"] is not None
    assert "CIR/ISD/01/2015" in qualifiers["source_circular"]


def test_fact_object_never_empty():
    """Verify that Fact strictly rejects empty or whitespace objects."""
    with pytest.raises(Exception):
        Fact(
            document_id="doc_test",
            chunk_id="chk_test",
            subject="MII",
            predicate=FactPredicate.REQUIRES,
            object="",
            source_text="MII shall comply.",
        )


def test_semantic_comparator_classifications():
    """Verify 9-dimensional comparison produces accurate change classifications."""
    mock_repo = MagicMock()
    comparator = SemanticComparator(mock_repo)

    facts_2024 = [
        {
            "fact_id": "f24_tw",
            "subject": "Designated persons",
            "predicate": "PROHIBITS",
            "object": "in securities when the trading window is closed",
            "source_text": "Designated persons and their immediate relatives shall not trade in securities when the trading window is closed.",
            "clause_reference": "Point 4",
        },
        {
            "fact_id": "f24_pb",
            "subject": "Exchanges",
            "predicate": "PERMITS",
            "object": "keeping the price band open on the first day of trading",
            "source_text": "while keeping the price band open on the first day of trading",
            "clause_reference": "Point 1",
        },
        {
            "fact_id": "f24_news",
            "subject": "SEBI registered market intermediaries",
            "predicate": "REQUIRES",
            "object": "unauthenticated news circulated through various modes of communication",
            "source_text": "unauthenticated news circulated through various modes of communication",
            "clause_reference": "Point 2",
        },
    ]

    facts_2026 = [
        # 1. FDSRL
        {
            "fact_id": "f26_fdsrl",
            "subject": "MII",
            "predicate": "REQUIRES",
            "object": "FINANCIAL_DISINCENTIVE",
            "source_text": "Financial disincentives under FDSRL shall be credited by MII within 15 working days to SEBI-IPEF.",
            "time": "15 working days",
            "source_circular": "SEBI/HO/ISD/ISD-PoD-1/P/CIR/2024/73",
            "clause_reference": "Clause 3.5.3",
        },
        # 2. NCS Exception
        {
            "fact_id": "f26_ncs",
            "subject": "DESIGNATED_PERSONS",
            "predicate": "PERMITS",
            "object": "SUBSCRIPTION_TO_NCS",
            "source_text": "Trading window restriction shall not apply in respect of subscription to non-convertible securities by Designated Persons.",
            "source_circular": "SEBI/HO/ISD/ISD-PoD-2/P/CIR/2024/180",
            "clause_reference": "Clause 5.1.1",
        },
        # 3. Scope expansion
        {
            "fact_id": "f26_rel",
            "subject": "DEPOSITORIES",
            "predicate": "REQUIRES",
            "object": "FREEZE_PAN_ISIN",
            "source_text": "freeze PAN-ISIN for trading window closure for Designated Persons and their Immediate Relatives.",
            "source_circular": "SEBI/HO/ISD/ISD-PoD-2/P/CIR/2025/55",
            "clause_reference": "Clause 5.2.4",
        },
        # 4. Code of Conduct confirmation (cites 2015 circular)
        {
            "fact_id": "f26_coc",
            "subject": "Employees",
            "predicate": "REQUIRES",
            "object": "IMMEDIATE_CONFIRMATION",
            "source_text": "immediate confirmation of code of conduct compliance.",
            "source_circular": "CIR/ISD/01/2015",
            "clause_reference": "Clause 4.1.2.2",
        },
        # 5. Clarification
        {
            "fact_id": "f26_news",
            "subject": "Intermediaries",
            "predicate": "REQUIRES",
            "object": "maintain logs of blogs and unauthenticated news",
            "source_text": "maintain logs of blogs and unauthenticated news.",
            "clause_reference": "Point 2",
        },
    ]

    results = comparator.compare_facts(facts_2024, facts_2026)
    actions = {r["action"]: r for r in results}

    # Verify key classifications
    assert "NEW_OBLIGATION" in actions
    assert actions["NEW_OBLIGATION"]["change_type"] == "ACTUAL_CHANGE"
    assert "SEBI/HO/ISD/ISD-PoD-1/P/CIR/2024/73" in actions["NEW_OBLIGATION"]["source_circular"]

    assert "EXCEPTION_ADDED" in actions
    assert actions["EXCEPTION_ADDED"]["change_type"] == "ACTUAL_CHANGE"
    assert "SEBI/HO/ISD/ISD-PoD-2/P/CIR/2024/180" in actions["EXCEPTION_ADDED"]["source_circular"]

    assert "SCOPE_EXPANDED" in actions
    assert actions["SCOPE_EXPANDED"]["change_type"] == "ACTUAL_CHANGE"
    assert "SEBI/HO/ISD/ISD-PoD-2/P/CIR/2025/55" in actions["SCOPE_EXPANDED"]["source_circular"]

    assert "RESTATED" in actions
    assert actions["RESTATED"]["change_type"] == "EXISTING_RULE_RESTATED"
    assert "CIR/ISD/01/2015" in actions["RESTATED"]["source_circular"]

    assert "REQUIREMENT_REMOVED" in actions
    assert actions["REQUIREMENT_REMOVED"]["change_type"] == "ACTUAL_CHANGE"

    assert "CLARIFIED" in actions
    assert actions["CLARIFIED"]["change_type"] == "CLARIFICATION"
