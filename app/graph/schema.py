"""Neo4j schema constraints and index initialization."""
from app.core.logging import logger
from app.graph.neo4j_client import Neo4jClient


CONSTRAINTS = [
    ("constraint_doc_id", "CREATE CONSTRAINT constraint_doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.document_id IS UNIQUE"),
    ("constraint_section_id", "CREATE CONSTRAINT constraint_section_id IF NOT EXISTS FOR (s:Section) REQUIRE s.id IS UNIQUE"),
    ("constraint_clause_id", "CREATE CONSTRAINT constraint_clause_id IF NOT EXISTS FOR (c:Clause) REQUIRE c.id IS UNIQUE"),
    ("constraint_chunk_id", "CREATE CONSTRAINT constraint_chunk_id IF NOT EXISTS FOR (ch:Chunk) REQUIRE ch.chunk_id IS UNIQUE"),
    ("constraint_fact_id", "CREATE CONSTRAINT constraint_fact_id IF NOT EXISTS FOR (f:Fact) REQUIRE f.fact_id IS UNIQUE"),
    ("constraint_entity_id", "CREATE CONSTRAINT constraint_entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE"),
    ("constraint_change_id", "CREATE CONSTRAINT constraint_change_id IF NOT EXISTS FOR (ce:ChangeEvent) REQUIRE ce.id IS UNIQUE"),
]

INDEXES = [
    ("index_doc_ref_num", "CREATE INDEX index_doc_ref_num IF NOT EXISTS FOR (d:Document) ON (d.reference_number)"),
    ("index_doc_date", "CREATE INDEX index_doc_date IF NOT EXISTS FOR (d:Document) ON (d.document_date)"),
    ("index_fact_doc_id", "CREATE INDEX index_fact_doc_id IF NOT EXISTS FOR (f:Fact) ON (f.document_id)"),
    ("index_fact_status", "CREATE INDEX index_fact_status IF NOT EXISTS FOR (f:Fact) ON (f.status)"),
    ("index_fact_valid_from", "CREATE INDEX index_fact_valid_from IF NOT EXISTS FOR (f:Fact) ON (f.valid_from)"),
    ("index_fact_valid_to", "CREATE INDEX index_fact_valid_to IF NOT EXISTS FOR (f:Fact) ON (f.valid_to)"),
    ("index_entity_name", "CREATE INDEX index_entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)"),
    ("index_entity_type", "CREATE INDEX index_entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)"),
    ("index_entity_norm_name", "CREATE INDEX index_entity_norm_name IF NOT EXISTS FOR (e:Entity) ON (e.normalized_name)"),
]


def initialize_schema(client: Neo4jClient) -> None:
    """Apply Neo4j constraints and indexes at startup."""
    logger.info("Initializing Neo4j schema constraints and indexes...")
    for name, cypher in CONSTRAINTS:
        try:
            client.execute_query(cypher)
            logger.debug("Applied constraint: %s", name)
        except Exception as e:
            logger.warning("Failed to apply constraint %s: %s", name, e)

    for name, cypher in INDEXES:
        try:
            client.execute_query(cypher)
            logger.debug("Applied index: %s", name)
        except Exception as e:
            logger.warning("Failed to apply index %s: %s", name, e)

    logger.info("Neo4j schema initialization complete.")
