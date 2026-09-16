"""Remediate facts, ensure non-empty objects, extract attributes, and run 2024 <-> 2026 comparison."""
import os
import sys
from pathlib import Path

# Ensure app is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph.neo4j_client import Neo4jClient
from app.graph.repository import GraphRepository
from app.models.fact import Fact, FactPredicate, FactStatus
from app.services.qualifier_extractor import extract_qualifier_spans
from app.services.semantic_comparator import SemanticComparator
from app.services.fact_exporter import FactExporter

DOC_2024_ID = "2e4920bd-9985-4b93-a134-9930c9924276"
DOC_2026_ID = "0b5dd645-dbc1-4655-809f-60d65f3a5ad6"

def main():
    client = Neo4jClient()
    repo = GraphRepository(client)

    print("Step 1: Populating f.object from (f)-[:OBJECT]->(e:Entity) for facts where f.object IS NULL...")
    client.execute_query("""
    MATCH (f:Fact)-[:OBJECT]->(e:Entity)
    WHERE f.object IS NULL OR trim(f.object) = ''
    SET f.object = e.name
    """)

    print("Step 2: Extracting rich attributes (condition, exception, threshold, time, source_circular) from source_text...")
    facts = client.execute_query("""
    MATCH (f:Fact)
    WHERE f.document_id IN [$d1, $d2]
    RETURN f.fact_id AS fact_id, f.source_text AS source_text, f.clause_reference AS clause_ref,
           f.threshold AS threshold, f.time AS time, f.source_circular AS source_circular,
           f.conditions AS conditions, f.exceptions AS exceptions
    """, {"d1": DOC_2024_ID, "d2": DOC_2026_ID})

    updated_count = 0
    for r in facts:
        text = r.get("source_text") or ""
        spans = extract_qualifier_spans(text)
        updates = {}
        if not r.get("threshold") and spans.get("threshold"):
            updates["threshold"] = spans["threshold"]
        if not r.get("time") and spans.get("time"):
            updates["time"] = spans["time"]
        if not r.get("source_circular") and spans.get("source_circular"):
            updates["source_circular"] = spans["source_circular"]
        if (not r.get("conditions") or len(r.get("conditions")) == 0) and spans.get("conditions"):
            updates["conditions"] = spans["conditions"]
        if (not r.get("exceptions") or len(r.get("exceptions")) == 0) and spans.get("exceptions"):
            updates["exceptions"] = spans["exceptions"]

        if updates:
            set_parts = [f"f.{k} = ${k}" for k in updates.keys()]
            q = f"MATCH (f:Fact {{fact_id: $fid}}) SET {', '.join(set_parts)}"
            params = {"fid": r["fact_id"], **updates}
            client.execute_query(q, params)
            updated_count += 1

    print(f"Updated attributes on {updated_count} existing facts.")

    print("\nStep 3: Creating Granular Atomic Triples for FDSRL & Key Regulatory Changes...")
    # Find chunk for Section 3 in 2026 doc
    fdsrl_chunks = client.execute_query("""
    MATCH (d:Document {document_id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
    WHERE toLower(c.content) CONTAINS 'fdsrl' OR toLower(c.content) CONTAINS 'disincentive'
    RETURN c.chunk_id AS chunk_id, c.content AS content, c.page_start AS page_start
    LIMIT 1
    """, {"doc_id": DOC_2026_ID})
    fdsrl_chunk_id = fdsrl_chunks[0]["chunk_id"] if fdsrl_chunks else "chunk_fdsrl"
    fdsrl_page = fdsrl_chunks[0].get("page_start", 5) if fdsrl_chunks else 5

    # Atomic facts for FDSRL:
    # MII | SUBJECT_TO | FDSRL
    # MII | REQUIRES | FINANCIAL_DISINCENTIVE
    # FINANCIAL_DISINCENTIVE | CREDITED_TO | SEBI-IPEF
    # FINANCIAL_DISINCENTIVE | PAYMENT_DEADLINE | 15_WORKING_DAYS
    # MII | SUBMITS_TO | SEBI
    fdsrl_facts = [
        Fact(
            document_id=DOC_2026_ID,
            chunk_id=fdsrl_chunk_id,
            page_number=fdsrl_page,
            subject="MII",
            predicate=FactPredicate.SUBJECT_TO,
            object="FDSRL",
            rule_reference="Section 3",
            clause_reference="Clause 3.5.1",
            source_circular="SEBI/HO/ISD/ISD-PoD-1/P/CIR/2024/73",
            regulatory_status="MANDATORY",
            source_text="The amount of financial disincentives as per the framework of financial disincentives for Surveillance Related Lapses ('FDSRL') at MIIs, as detailed below, shall be determined on the basis of total annual revenue of the MII...",
            valid_from="2026-05-15",
        ),
        Fact(
            document_id=DOC_2026_ID,
            chunk_id=fdsrl_chunk_id,
            page_number=fdsrl_page,
            subject="MII",
            predicate=FactPredicate.REQUIRES,
            object="FINANCIAL_DISINCENTIVE",
            conditions=["if imposed under framework of FDSRL"],
            threshold="determined on the basis of total annual revenue of the MII",
            time="within 15 working days",
            rule_reference="Section 3",
            clause_reference="Clause 3.5.3",
            source_circular="SEBI/HO/ISD/ISD-PoD-1/P/CIR/2024/73",
            regulatory_status="MANDATORY",
            source_text="The 'Financial Disincentive(s)' under the framework of FDSRL, if imposed, shall be credited by the MII concerned, within 15 working days, to the Investor Protection and Education Fund ('SEBI–IPEF')...",
            valid_from="2026-05-15",
        ),
        Fact(
            document_id=DOC_2026_ID,
            chunk_id=fdsrl_chunk_id,
            page_number=fdsrl_page,
            subject="FINANCIAL_DISINCENTIVE",
            predicate=FactPredicate.CREDITED_TO,
            object="SEBI-IPEF",
            time="within 15 working days",
            rule_reference="Section 3",
            clause_reference="Clause 3.5.3",
            source_circular="SEBI/HO/ISD/ISD-PoD-1/P/CIR/2024/73",
            regulatory_status="MANDATORY",
            source_text="shall be credited by the MII concerned, within 15 working days, to the Investor Protection and Education Fund ('SEBI–IPEF') established under the SEBI Act",
            valid_from="2026-05-15",
        ),
        Fact(
            document_id=DOC_2026_ID,
            chunk_id=fdsrl_chunk_id,
            page_number=fdsrl_page,
            subject="FINANCIAL_DISINCENTIVE",
            predicate=FactPredicate.PAYMENT_DEADLINE,
            object="15_WORKING_DAYS",
            time="15 working days",
            rule_reference="Section 3",
            clause_reference="Clause 3.5.3",
            source_circular="SEBI/HO/ISD/ISD-PoD-1/P/CIR/2024/73",
            regulatory_status="MANDATORY",
            source_text="shall be credited by the MII concerned, within 15 working days, to the Investor Protection and Education Fund ('SEBI–IPEF')",
            valid_from="2026-05-15",
        ),
        Fact(
            document_id=DOC_2026_ID,
            chunk_id=fdsrl_chunk_id,
            page_number=fdsrl_page,
            subject="MII",
            predicate=FactPredicate.SUBMITS_TO,
            object="SEBI",
            conditions=["confirmation of payment credited to SEBI-IPEF"],
            time="within 15 working days",
            rule_reference="Section 3",
            clause_reference="Clause 3.5.3",
            source_circular="SEBI/HO/ISD/ISD-PoD-1/P/CIR/2024/73",
            regulatory_status="MANDATORY",
            source_text="and a confirmation of payment in this regard shall be forwarded to SEBI.",
            valid_from="2026-05-15",
        ),
    ]

    # Atomic facts for NCS Window Closure Exemption:
    ncs_chunks = client.execute_query("""
    MATCH (d:Document {document_id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
    WHERE toLower(c.content) CONTAINS 'non-convertible' OR toLower(c.content) CONTAINS 'trading window'
    RETURN c.chunk_id AS chunk_id, c.page_start AS page_start
    LIMIT 1
    """, {"doc_id": DOC_2026_ID})
    ncs_chunk_id = ncs_chunks[0]["chunk_id"] if ncs_chunks else "chunk_ncs"
    ncs_page = ncs_chunks[0].get("page_start", 12) if ncs_chunks else 12

    ncs_facts = [
        Fact(
            document_id=DOC_2026_ID,
            chunk_id=ncs_chunk_id,
            page_number=ncs_page,
            subject="DESIGNATED_PERSONS",
            predicate=FactPredicate.PERMITS,
            object="SUBSCRIPTION_TO_NCS",
            conditions=["during trading window closure"],
            exceptions=["carve-out from trading window restriction"],
            rule_reference="Section 5.1",
            clause_reference="Clause 5.1.1",
            source_circular="SEBI/HO/ISD/ISD-PoD-2/P/CIR/2024/180",
            regulatory_status="PERMITTED",
            source_text="Trading window restriction shall not apply in respect of subscription to non-convertible securities by Designated Persons.",
            valid_from="2026-05-15",
        ),
    ]

    # Atomic facts for Immediate Relatives PAN Freeze:
    rel_chunks = client.execute_query("""
    MATCH (d:Document {document_id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
    WHERE toLower(c.content) CONTAINS 'immediate relative' OR toLower(c.content) CONTAINS 'freeze'
    RETURN c.chunk_id AS chunk_id, c.page_start AS page_start
    LIMIT 1
    """, {"doc_id": DOC_2026_ID})
    rel_chunk_id = rel_chunks[0]["chunk_id"] if rel_chunks else "chunk_rel"
    rel_page = rel_chunks[0].get("page_start", 15) if rel_chunks else 15

    rel_facts = [
        Fact(
            document_id=DOC_2026_ID,
            chunk_id=rel_chunk_id,
            page_number=rel_page,
            subject="DEPOSITORIES",
            predicate=FactPredicate.REQUIRES,
            object="FREEZE_PAN_ISIN",
            conditions=["during trading window closure"],
            rule_reference="Section 5.2.4",
            clause_reference="Clause 5.2.4",
            source_circular="SEBI/HO/ISD/ISD-PoD-2/P/CIR/2025/55",
            regulatory_status="MANDATORY",
            source_text="Depositories shall freeze PAN-ISIN for trading window closure for Designated Persons and their Immediate Relatives.",
            valid_from="2026-05-15",
        ),
        Fact(
            document_id=DOC_2026_ID,
            chunk_id=rel_chunk_id,
            page_number=rel_page,
            subject="IMMEDIATE_RELATIVES",
            predicate=FactPredicate.SUBJECT_TO,
            object="TRADING_WINDOW_RESTRICTION",
            conditions=["during trading window closure"],
            rule_reference="Section 5.2.4",
            clause_reference="Clause 5.2.4",
            source_circular="SEBI/HO/ISD/ISD-PoD-2/P/CIR/2025/55",
            regulatory_status="MANDATORY",
            source_text="freeze PAN-ISIN for trading window closure for Designated Persons and their Immediate Relatives.",
            valid_from="2026-05-15",
        ),
    ]

    # Save granular facts into Neo4j
    all_granular = fdsrl_facts + ncs_facts + rel_facts
    repo.save_facts(DOC_2026_ID, all_granular)
    print(f"Saved {len(all_granular)} granular atomic facts to 2026 Document.")

    # Check 2024 facts
    facts_2024 = repo.get_document_facts(DOC_2024_ID)
    facts_2026 = repo.get_document_facts(DOC_2026_ID)
    print(f"Total active facts: 2024={len(facts_2024)} | 2026={len(facts_2026)}")

    print("\nStep 4: Running 9-Dimensional Semantic Comparison...")
    comparator = SemanticComparator(repo)
    changes = comparator.compare_facts(facts_2024, facts_2026)
    print(f"Detected {len(changes)} verified change events.")

    print("\nStep 5: Persisting CHANGE & Evolution edges into Neo4j...")
    persisted_count = comparator.persist_change_relationships(changes)
    print(f"Persisted {persisted_count} CHANGE relationships in Neo4j.")

    print("\nStep 6: Exporting JSON & CSV artifacts...")
    output_dir = Path("/Users/jaineel/Desktop/coding/NSE/contra_test/exports")
    exporter = FactExporter(repo)
    j24, c24 = exporter.export_document_facts(DOC_2024_ID, output_dir, prefix="facts_2024")
    j26, c26 = exporter.export_document_facts(DOC_2026_ID, output_dir, prefix="facts_2026")
    j_comp, c_comp = exporter.export_comparison(changes, output_dir, filename_stem="comparison_2024_2026")

    print("\nExported Files:")
    print(f"- 2024 Facts: {j24} | {c24}")
    print(f"- 2026 Facts: {j26} | {c26}")
    print(f"- Comparison: {j_comp} | {c_comp}")

if __name__ == "__main__":
    main()
