"""Unit tests for Candidate Relationship Generator."""
import pytest
from app.models.fact import Fact, FactPredicate, FactStatus
from app.models.reference import DocumentReference
from app.services.candidate_generator import CandidateGenerator


def test_candidate_generator_keyword_scoring_and_top_k():
    generator = CandidateGenerator()
    generator.top_k = 3

    # Source facts from amending circular
    f1 = Fact(
        fact_id="src-1",
        document_id="doc-2026",
        chunk_id="chk-1",
        subject="Trading Member",
        predicate=FactPredicate.MODIFIES,
        object="Point 1 modified regarding client funds",
        source_text="Point 1 stands amended and substituted in partial modification of earlier rule.",
    )
    f2 = Fact(
        fact_id="src-2",
        document_id="doc-2026",
        chunk_id="chk-2",
        subject="Trading Member",
        predicate=FactPredicate.DELETES,
        object="Points 3 to 12 stand deleted",
        source_text="Point 3 to 12 stand deleted as clarified.",
    )
    f3 = Fact(
        fact_id="src-3",
        document_id="doc-2026",
        chunk_id="chk-3",
        subject="Random Entity",
        predicate=FactPredicate.REQUIRES,
        object="Unrelated operational notice",
        source_text="Unrelated operational notice for office administration.",
    )

    # Target facts from original circular
    target_facts = [
        Fact(
            fact_id=f"tgt-{i}",
            document_id="doc-2022",
            chunk_id=f"chk-tgt-{i}",
            subject="Trading Member",
            predicate=FactPredicate.REQUIRES,
            object=f"Point {i} requirements for maintenance of client records",
            source_text=f"Illustrative Point {i}: Member must maintain bank balances.",
        )
        for i in range(1, 13)
    ]

    ref = DocumentReference(
        source_document_id="doc-2026",
        target_document_id="doc-2022",
        target_reference_number="NSE/COMP/50957",
        reference_text="Ref No. NSE/COMP/50957",
    )

    candidates = generator.generate_candidates(
        source_facts=[f1, f2, f3],
        target_facts=target_facts,
        references=[ref],
    )

    # Must be filtered to top_k = 3
    assert len(candidates) <= 3
    assert len(candidates) > 0

    # Best candidates should have high scores
    for cand in candidates:
        assert cand["score"] >= 0.5
        assert "source_id" in cand
        assert "target_id" in cand
        assert "reasons" in cand


def test_digital_gold_unrelated_candidate_rejection():
    """Verify that unrelated FAQ facts do NOT pair with metadata or non-matching clauses."""
    generator = CandidateGenerator()

    # 1. Unrelated FAQ fact
    digital_gold_fact = Fact(
        fact_id="src-digital-gold",
        document_id="doc-2026",
        chunk_id="chk-faq-6",
        subject="Trading Member",
        predicate=FactPredicate.PROHIBITS,
        object="providing a platform to the clients for buying and selling of digital gold or any product not covered",
        clause_reference="Question 6",
        source_text="Question 6: Whether members are permitted to enter into any agreement for providing a platform for digital gold? Answer: No.",
    )

    # 2. Candidate old facts including metadata, dates, and different clauses
    old_metadata_date = Fact(
        fact_id="old-meta-date",
        document_id="doc-2022",
        chunk_id="chk-meta-1",
        subject="Trading Member",
        predicate=FactPredicate.REQUIRES,
        object="Date: January 07, 2022",
        source_text="Date: January 07, 2022",
    )
    old_metadata_ref = Fact(
        fact_id="old-meta-ref",
        document_id="doc-2022",
        chunk_id="chk-meta-2",
        subject="Trading Member",
        predicate=FactPredicate.REQUIRES,
        object="Download Ref No: NSE/COMP/50957",
        source_text="Download Ref No: NSE/COMP/50957",
    )
    old_point_1 = Fact(
        fact_id="old-point-1",
        document_id="doc-2022",
        chunk_id="chk-pt-1",
        subject="Trading Member",
        predicate=FactPredicate.PROHIBITS,
        object="Issuing Corporate Guarantees towards credit facilities availed by any entity",
        rule_reference="Rule 8(1)(f)",
        clause_reference="Point 1",
        source_text="Point No. 1. Issuing Corporate Guarantees towards credit facilities availed by any entity.",
    )
    old_broker_agent = Fact(
        fact_id="old-broker-agent",
        document_id="doc-2022",
        chunk_id="chk-pt-x",
        subject="Trading Member",
        predicate=FactPredicate.REQUIRES,
        object="broker or agent, not involving any personal financial liability.",
        source_text="broker or agent, not involving any personal financial liability.",
    )

    ref = DocumentReference(
        source_document_id="doc-2026",
        target_document_id="doc-2022",
        target_reference_number="NSE/COMP/50957",
        reference_text="in partial modification of Circular Ref No. NSE/COMP/50957",
    )

    candidates = generator.generate_candidates(
        source_facts=[digital_gold_fact],
        target_facts=[old_metadata_date, old_metadata_ref, old_point_1, old_broker_agent],
        references=[ref],
    )

    # Must NOT produce any false candidates against metadata or unrelated points
    matched_target_ids = {c["target_id"] for c in candidates}
    assert "old-meta-date" not in matched_target_ids
    assert "old-meta-ref" not in matched_target_ids
    assert "old-point-1" not in matched_target_ids
    assert "old-broker-agent" not in matched_target_ids
    assert len(candidates) == 0  # No valid anchor exists for digital gold against 2022 Point 1 or metadata

