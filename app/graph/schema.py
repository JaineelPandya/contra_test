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
    """Apply Neo4j constraints, indexes, and schema integrity migrations at startup."""
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

    # Schema integrity & bidirectional relationship backfills for existing graphs
    migrations = [
        # 1. Backfill SUPPORTED_BY from CONTAINS_FACT
        (
            "backfill_supported_by",
            "MATCH (c:Chunk)-[:CONTAINS_FACT]->(f:Fact) MERGE (f)-[:SUPPORTED_BY]->(c)",
        ),
        # 2. Backfill Chunk.path from Chunk.heading_path
        (
            "backfill_chunk_path",
            """
            MATCH (c:Chunk)
            WHERE c.path IS NULL OR c.path = ''
            SET c.path = CASE
                WHEN c.heading_path IS NOT NULL AND size(c.heading_path) > 0
                THEN reduce(s = '', x in c.heading_path | CASE WHEN s = '' THEN x ELSE s + ' > ' + x END)
                ELSE 'Section'
            END
            """,
        ),
        # 3. Canonicalize any passive verb fact relationships to active verbs
        (
            "canonicalize_fact_modifies",
            "MATCH (f1:Fact)-[r:MODIFIED]->(f2:Fact) MERGE (f1)-[:MODIFIES]->(f2)",
        ),
        (
            "canonicalize_fact_supersedes",
            "MATCH (f1:Fact)-[r:SUPERSEDED]->(f2:Fact) MERGE (f1)-[:SUPERSEDES]->(f2)",
        ),
        (
            "canonicalize_fact_clarifies",
            "MATCH (f1:Fact)-[r:CLARIFIED]->(f2:Fact) MERGE (f1)-[:CLARIFIES]->(f2)",
        ),
        (
            "canonicalize_fact_deletes",
            "MATCH (f1:Fact)-[r:DELETED]->(f2:Fact) MERGE (f1)-[:DELETES]->(f2)",
        ),
        # 4. Purge junk metadata facts
        (
            "purge_metadata_facts",
            """
            MATCH (f:Fact)
            WHERE f.source_text =~ '(?i)^(Date:|DEPARTMENT:|Download Ref|Circular Ref|NATIONAL STOCK|Page \\\\d+|CIN|Tel No|To All Members|Yours faithfully|Regd\\\\.? Office|CIRCULAR Ref).*'
               OR (f.object IS NOT NULL AND f.object =~ '(?i)^(Date:|DEPARTMENT:|Download Ref|Circular Ref|NATIONAL STOCK|Page \\\\d+|CIN|Tel No|To All Members|Yours faithfully).*')
            DETACH DELETE f
            """,
        ),
        # 5. Purge false Cartesian AMENDS relationships that link unrelated facts or lack evidence
        (
            "purge_false_amends_edges",
            """
            MATCH (f1:Fact)-[r:MODIFIES|SUPERSEDES|CLARIFIES|AMENDS]->(f2:Fact)
            WHERE (f1.document_id = f2.document_id)
               OR (f1.source_text =~ '(?i)^(Date:|DEPARTMENT:|Download Ref|Circular Ref|NATIONAL STOCK|Page \\\\d+).*')
               OR (f2.source_text =~ '(?i)^(Date:|DEPARTMENT:|Download Ref|Circular Ref|NATIONAL STOCK|Page \\\\d+).*')
            DELETE r
            """,
        ),
    ]

    for mig_name, mig_cypher in migrations:
        try:
            client.execute_query(mig_cypher)
            logger.debug("Applied schema migration: %s", mig_name)
        except Exception as mig_err:
            logger.warning("Schema migration %s failed: %s", mig_name, mig_err)

    logger.info("Neo4j schema initialization and integrity migrations complete.")


