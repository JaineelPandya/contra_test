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
        assert cand["score"] >= 0.3
        assert "source_id" in cand
        assert "target_id" in cand
        assert "reasons" in cand
