"""Tests for Regulatory Query Analyzer, Cypher Generator, Validator, and Query API endpoints."""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.models.query import QueryType, RegulatoryQueryPlan, RegulatoryQueryRequest
from app.services.cypher_generator import CypherGenerator
from app.services.cypher_validator import CypherSecurityError, CypherValidator
from app.services.regulatory_analyzer import RegulatoryQueryAnalyzer
from app.services.subgraph_builder import SubgraphBuilder
from app.services.answer_synthesizer import RegulatoryAnswerSynthesizer


@pytest.fixture
def client():
    return TestClient(app)


# ==================== 1. REGULATORY QUERY ANALYZER TESTS ====================

@pytest.mark.asyncio
async def test_analyzer_current_query():
    analyzer = RegulatoryQueryAnalyzer()
    req = RegulatoryQueryRequest(query="What are the current client margin reporting requirements for Trading Members?")
    plan = await analyzer.analyze_query(req)

    assert plan.query_type == QueryType.CURRENT
    assert "Trading Member" in plan.entities
    assert "REQUIRES" in plan.predicates


@pytest.mark.asyncio
async def test_analyzer_historical_query():
    analyzer = RegulatoryQueryAnalyzer()
    req = RegulatoryQueryRequest(query="What were the margin rules as of 2022-01-07 for Trading Members?")
    plan = await analyzer.analyze_query(req)

    assert plan.query_type == QueryType.HISTORICAL
    assert plan.temporal_scope.get("as_of_date") == "2022-01-07"
    assert "Trading Member" in plan.entities


@pytest.mark.asyncio
async def test_analyzer_evolution_query():
    analyzer = RegulatoryQueryAnalyzer()
    req = RegulatoryQueryRequest(query="How has Rule 8(1)(f) on Trading Member business evolved over time?")
    plan = await analyzer.analyze_query(req)

    assert plan.query_type == QueryType.EVOLUTION
    assert "Trading Member" in plan.entities


@pytest.mark.asyncio
async def test_analyzer_amendment_query():
    analyzer = RegulatoryQueryAnalyzer()
    req = RegulatoryQueryRequest(query="What changes and amendments were introduced by Circular 2026/amendment?")
    plan = await analyzer.analyze_query(req)

    assert plan.query_type == QueryType.AMENDMENT


@pytest.mark.asyncio
async def test_analyzer_master_to_latest_query():
    analyzer = RegulatoryQueryAnalyzer()
    req = RegulatoryQueryRequest(query="Trace master circular 2022 to the latest active requirements.")
    plan = await analyzer.analyze_query(req)

    assert plan.query_type == QueryType.MASTER_TO_LATEST


@pytest.mark.asyncio
async def test_analyzer_odr_topic_query():
    analyzer = RegulatoryQueryAnalyzer()
    req = RegulatoryQueryRequest(
        query="How has the Online Resolution of Disputes framework evolved from the July 31, 2023 circular to the August 4, 2023 Corrigendum cum Amendment?"
    )
    plan = await analyzer.analyze_query(req)

    assert plan.query_type == QueryType.EVOLUTION
    assert any("Dispute" in kw or "Online Resolution of Disputes" in kw or "ODR" in kw for kw in plan.keywords)
    assert plan.temporal_scope.get("as_of_date") == "2023-08-04"
    assert "2023-07-31" in plan.temporal_scope.get("dates_mentioned", [])
    assert "2023-08-04" in plan.temporal_scope.get("dates_mentioned", [])


@pytest.mark.asyncio
async def test_analyzer_manual_override():
    analyzer = RegulatoryQueryAnalyzer()
    req = RegulatoryQueryRequest(
        query="Tell me about Trading Members",
        query_type_override=QueryType.EVOLUTION,
    )
    plan = await analyzer.analyze_query(req)

    assert plan.query_type == QueryType.EVOLUTION
    assert "manually overridden" in plan.reasoning


# ==================== 2. CYPHER VALIDATOR & GUARDRAILS TESTS ====================

def test_cypher_validator_allows_safe_queries():
    safe_query = "MATCH (d:Document) RETURN d.title, d.reference_number LIMIT 10"
    validated = CypherValidator.validate_and_sanitize(safe_query)
    assert validated == safe_query


def test_cypher_validator_blocks_mutating_create():
    unsafe_query = "CREATE (d:Document {title: 'Fake'}) RETURN d"
    with pytest.raises(CypherSecurityError) as exc_info:
        CypherValidator.validate_and_sanitize(unsafe_query)
    assert "restricted keywords" in str(exc_info.value)


def test_cypher_validator_blocks_mutating_delete():
    unsafe_query = "MATCH (d:Document) DELETE d"
    with pytest.raises(CypherSecurityError) as exc_info:
        CypherValidator.validate_and_sanitize(unsafe_query)
    assert "restricted keywords" in str(exc_info.value)


def test_cypher_validator_blocks_mutating_set():
    unsafe_query = "MATCH (f:Fact) SET f.status = 'ACTIVE' RETURN f"
    with pytest.raises(CypherSecurityError) as exc_info:
        CypherValidator.validate_and_sanitize(unsafe_query)
    assert "restricted keywords" in str(exc_info.value)


def test_cypher_validator_blocks_semicolon_chaining():
    unsafe_query = "MATCH (d:Document) RETURN d; DROP CONSTRAINT something;"
    with pytest.raises(CypherSecurityError) as exc_info:
        CypherValidator.validate_and_sanitize(unsafe_query)
    assert "Multiple Cypher statements" in str(exc_info.value)


def test_cypher_validator_blocks_admin_procedures():
    unsafe_query = "CALL dbms.security.createUser('hacker', 'password')"
    with pytest.raises(CypherSecurityError) as exc_info:
        CypherValidator.validate_and_sanitize(unsafe_query)
    assert "system procedures" in str(exc_info.value)


# ==================== 3. CYPHER GENERATOR TESTS ====================

def test_cypher_generator_all_archetypes():
    gen = CypherGenerator()

    for qtype in [QueryType.CURRENT, QueryType.HISTORICAL, QueryType.EVOLUTION, QueryType.AMENDMENT, QueryType.MASTER_TO_LATEST]:
        plan = RegulatoryQueryPlan(
            query_type=qtype,
            raw_query="Test query",
            entities=["Trading Member"],
            target_circulars=["50957"],
            temporal_scope={"as_of_date": "2022-01-01"},
            reasoning="Test",
        )
        cypher, params = gen.generate_cypher(plan)
        assert isinstance(cypher, str)
        assert "MATCH" in cypher
        assert "RETURN" in cypher
        assert params["limit"] == 50


# ==================== 4. SUBGRAPH BUILDER TESTS ====================

def test_subgraph_builder_current_records():
    builder = SubgraphBuilder()
    records = [
        {
            "document_id": "doc-1",
            "reference_number": "NSE/COMP/50957",
            "doc_title": "Clarification Circular",
            "document_date": "2022-01-07",
            "fact_id": "fact-1",
            "subject": "Trading Member",
            "predicate": "REQUIRES",
            "object": "Submit quarterly report",
            "status": "ACTIVE",
            "source_text": "Trading members shall submit quarterly reports.",
            "subject_entity": "Trading Member",
        }
    ]
    subgraph = builder.build_subgraph(records, QueryType.CURRENT)
    assert subgraph.total_nodes >= 2
    assert subgraph.total_edges >= 1
    node_types = {n.type for n in subgraph.nodes}
    assert "Document" in node_types
    assert "Fact" in node_types


def test_subgraph_builder_evolution_records():
    builder = SubgraphBuilder()
    records = [
        {
            "new_fact_id": "fact-2026",
            "new_subject": "Trading Member",
            "new_predicate": "REQUIRES",
            "new_object": "Auto-sync by 10 AM",
            "new_status": "ACTIVE",
            "old_fact_id": "fact-2022",
            "old_subject": "Trading Member",
            "old_predicate": "REQUIRES",
            "old_object": "Manual upload by 5 PM",
            "old_status": "MODIFIED",
            "evolution_types": ["MODIFIES"],
            "new_doc_ref": "NSE/INSP/2026/01",
            "old_doc_ref": "NSE/COMP/50957",
        }
    ]
    subgraph = builder.build_subgraph(records, QueryType.EVOLUTION)
    assert subgraph.total_nodes >= 2
    assert subgraph.total_edges >= 1
    edge_types = {e.type for e in subgraph.edges}
    assert "MODIFIES" in edge_types or "AMENDS" in edge_types


# ==================== 5. ANSWER SYNTHESIZER TESTS ====================

@pytest.mark.asyncio
async def test_answer_synthesizer():
    synthesizer = RegulatoryAnswerSynthesizer()
    plan = RegulatoryQueryPlan(
        query_type=QueryType.CURRENT,
        raw_query="What are the margin requirements?",
        reasoning="Test",
    )
    records = [
        {
            "document_id": "doc-1",
            "reference_number": "NSE/COMP/50957",
            "document_date": "2022-01-07",
            "fact_id": "fact-1",
            "subject": "Trading Member",
            "predicate": "REQUIRES",
            "object": "Report client margins",
            "status": "ACTIVE",
            "source_text": "Trading members must report client margins daily.",
        }
    ]
    curr, hist, markdown, sources = await synthesizer.synthesize_response(plan, records)

    assert curr is not None
    assert curr.status == "ACTIVE"
    assert "Trading Member" in curr.summary
    assert len(sources) == 1
    assert "Trading members must report client margins daily." in markdown


# ==================== 6. API ENDPOINT INTEGRATION TESTS ====================

def test_api_query_archetypes(client):
    response = client.get("/api/query/archetypes")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 5
    types = [item["type"] for item in data]
    assert "CURRENT" in types
    assert "EVOLUTION" in types
    assert "MASTER_TO_LATEST" in types


def test_api_query_plan_endpoint(client):
    payload = {
        "query": "How has Rule 8(1)(f) evolved over time?",
    }
    response = client.post("/api/query/plan", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "plan" in data
    assert data["plan"]["query_type"] == "EVOLUTION"
    assert "cypher_query" in data
    assert "MATCH" in data["cypher_query"]


def test_api_query_execute_endpoint(client):
    payload = {
        "query": "What are the current rules for Trading Members?",
        "include_subgraph": True,
    }
    response = client.post("/api/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "plan" in data
    assert "cypher_query" in data
    assert "llm_answer" in data
    assert "subgraph" in data


def test_api_diagnostics_endpoint(client):
    response = client.get("/api/query/diagnostics")
    assert response.status_code == 200
    data = response.json()
    assert "documents_count" in data
    assert "chunks_count" in data
    assert "facts_count" in data
    assert "active_facts_count" in data
    assert "evolution_relationships_count" in data or "fact_evolution_count" in data


# ==================== 7. BOILERPLATE FILTERING & RELATIONSHIP TESTS ====================

def test_boilerplate_chunk_filter():
    from app.services.fact_extractor import is_boilerplate_chunk
    from app.models.chunk import Chunk

    # 1. Header chunk (should be filtered)
    header_chunk = Chunk(
        document_id="doc-1",
        content="NATIONAL STOCK EXCHANGE OF INDIA LIMITED\nCIRCULAR Ref. No: 32/2026\nDownload Ref No: NSE/INSP/74836\nDate: June 23, 2026",
    )
    is_bp, reason = is_boilerplate_chunk(header_chunk)
    assert is_bp is True
    assert "Header" in reason

    # 2. Signature chunk (should be filtered)
    sig_chunk = Chunk(
        document_id="doc-1",
        content="Yours faithfully,\nFor National Stock Exchange of India Ltd.\nJohn Doe\nChief Manager",
    )
    is_bp, reason = is_boilerplate_chunk(sig_chunk)
    assert is_bp is True
    assert "Signature" in reason

    # 3. Address salutation chunk (should be filtered)
    addr_chunk = Chunk(
        document_id="doc-1",
        content="To All Trading Members,\nDear Sir / Madam,",
    )
    is_bp, reason = is_boilerplate_chunk(addr_chunk)
    assert is_bp is True
    assert "Address" in reason

    # 4. Operative regulatory chunk (should NOT be filtered)
    reg_chunk = Chunk(
        document_id="doc-1",
        content="Point No. 1. Issuing Corporate Guarantees towards credit facilities availed by any entity is strictly prohibited.",
        heading_path=["Illustrative List", "Point 1"],
    )
    is_bp, reason = is_boilerplate_chunk(reg_chunk)
    assert is_bp is False
    assert reg_chunk.path == "Illustrative List > Point 1"


def test_save_change_event_active_verb_mapping():
    from unittest.mock import MagicMock
    from app.graph.repository import GraphRepository
    from app.models.change import ChangeEvent, ChangeType

    mock_client = MagicMock()
    repo = GraphRepository(mock_client)

    event = ChangeEvent(
        source_document_id="doc-2026",
        target_document_id="doc-2022",
        source_fact_id="fact-2026-1",
        target_fact_id="fact-2022-1",
        change_type=ChangeType.MODIFIED,
        description="Modified Point 1",
    )
    repo.save_change_event(event)

    # Verify Cypher query contains MODIFIES relationship
    executed_queries = [call.args[0] for call in mock_client.execute_query.call_args_list]
    rel_queries = [q for q in executed_queries if "MERGE (new_fact)-[r:MODIFIES]->(old_fact)" in q]
    assert len(rel_queries) == 1

