"""Graph Query Planner and Parameterized Cypher Generator for all query archetypes."""
from typing import Any, Dict, Optional, Tuple
from app.core.logging import logger
from app.models.query import QueryType, RegulatoryQueryPlan
from app.services.cypher_validator import CypherValidator


class CypherGenerator:
    """Generates secure, schema-aware parameterized Cypher queries based on RegulatoryQueryPlan."""

    def __init__(self) -> None:
        self.validator = CypherValidator()

    def generate_cypher(self, plan: RegulatoryQueryPlan, limit: int = 50) -> Tuple[str, Dict[str, Any]]:
        """Generate Cypher query string and parameter dict corresponding to the query plan."""
        logger.info("Generating Cypher query for archetype: %s", plan.query_type.value)

        entity_param = plan.entities[0] if plan.entities else None
        circular_param = plan.target_circulars[0] if plan.target_circulars else None
        as_of_date_param = plan.temporal_scope.get("as_of_date")
        keyword_param = plan.keywords[0] if plan.keywords else None

        params: Dict[str, Any] = {
            "entity": entity_param,
            "circular": circular_param,
            "keyword": keyword_param,
            "limit": limit,
        }

        if plan.query_type == QueryType.CURRENT:
            cypher = self._generate_current_query()
        elif plan.query_type == QueryType.HISTORICAL:
            params["as_of_date"] = as_of_date_param
            cypher = self._generate_historical_query()
        elif plan.query_type == QueryType.EVOLUTION:
            cypher = self._generate_evolution_query()
        elif plan.query_type == QueryType.AMENDMENT:
            cypher = self._generate_amendment_query()
        elif plan.query_type == QueryType.MASTER_TO_LATEST:
            cypher = self._generate_master_to_latest_query()
        elif plan.query_type == QueryType.CHANGELOG:
            cypher = self._generate_changelog_query()
        else:
            cypher = self._generate_current_query()

        # Validate query safety with guardrails
        validated_cypher = self.validator.validate_and_sanitize(cypher)
        return validated_cypher, params

    def _generate_current_query(self) -> str:
        return """
        MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)<-[:SUPPORTED_BY]-(f:Fact)
        WHERE f.status = 'ACTIVE'
          AND ($circular IS NULL OR coalesce(d.reference_number, '') CONTAINS $circular OR coalesce(d.circular_number, '') CONTAINS $circular OR coalesce(d.filename, '') CONTAINS $circular)
          AND (
            ($entity IS NULL AND $keyword IS NULL)
            OR ($entity IS NOT NULL AND (
                coalesce(f.subject, '') =~ ('(?i).*' + $entity + '.*')
                OR coalesce(f.rule_reference, '') =~ ('(?i).*' + $entity + '.*')
                OR coalesce(f.clause_reference, '') =~ ('(?i).*' + $entity + '.*')
                OR coalesce(f.source_text, '') =~ ('(?i).*' + $entity + '.*')
                OR coalesce(d.title, '') =~ ('(?i).*' + $entity + '.*')
            ))
            OR ($keyword IS NOT NULL AND (
                coalesce(f.source_text, '') =~ ('(?i).*' + $keyword + '.*')
                OR coalesce(f.object, '') =~ ('(?i).*' + $keyword + '.*')
                OR coalesce(f.subject, '') =~ ('(?i).*' + $keyword + '.*')
                OR coalesce(f.rule_reference, '') =~ ('(?i).*' + $keyword + '.*')
                OR coalesce(f.clause_reference, '') =~ ('(?i).*' + $keyword + '.*')
                OR coalesce(d.title, '') =~ ('(?i).*' + $keyword + '.*')
            ))
          )
        OPTIONAL MATCH (f)-[:SUBJECT]->(sub:Entity)
        OPTIONAL MATCH (f)-[:OBJECT]->(obj:Entity)
        RETURN f.fact_id AS fact_id,
               f.subject AS subject,
               f.predicate AS predicate,
               f.object AS object,
               f.status AS status,
               f.valid_from AS valid_from,
               f.valid_to AS valid_to,
               f.source_text AS source_text,
               f.confidence AS confidence,
               d.document_id AS document_id,
               d.title AS doc_title,
               d.reference_number AS reference_number,
               d.circular_number AS circular_number,
               d.document_date AS document_date,
               c.chunk_id AS chunk_id,
               coalesce(c.path, '') AS clause_path,
               sub.name AS subject_entity,
               obj.name AS object_entity
        ORDER BY d.document_date DESC, f.created_at DESC
        LIMIT $limit
        """.strip()

    def _generate_historical_query(self) -> str:
        return """
        MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)<-[:SUPPORTED_BY]-(f:Fact)
        WHERE ($as_of_date IS NULL OR f.valid_from IS NULL OR f.valid_from <= $as_of_date)
          AND ($as_of_date IS NULL OR f.valid_to IS NULL OR f.valid_to > $as_of_date)
          AND ($circular IS NULL OR coalesce(d.reference_number, '') CONTAINS $circular OR coalesce(d.circular_number, '') CONTAINS $circular OR coalesce(d.filename, '') CONTAINS $circular)
          AND (
            ($entity IS NULL AND $keyword IS NULL)
            OR ($entity IS NOT NULL AND (
                coalesce(f.subject, '') =~ ('(?i).*' + $entity + '.*')
                OR coalesce(f.rule_reference, '') =~ ('(?i).*' + $entity + '.*')
                OR coalesce(f.clause_reference, '') =~ ('(?i).*' + $entity + '.*')
                OR coalesce(f.source_text, '') =~ ('(?i).*' + $entity + '.*')
                OR coalesce(d.title, '') =~ ('(?i).*' + $entity + '.*')
            ))
            OR ($keyword IS NOT NULL AND (
                coalesce(f.source_text, '') =~ ('(?i).*' + $keyword + '.*')
                OR coalesce(f.object, '') =~ ('(?i).*' + $keyword + '.*')
                OR coalesce(f.subject, '') =~ ('(?i).*' + $keyword + '.*')
                OR coalesce(f.rule_reference, '') =~ ('(?i).*' + $keyword + '.*')
                OR coalesce(f.clause_reference, '') =~ ('(?i).*' + $keyword + '.*')
                OR coalesce(d.title, '') =~ ('(?i).*' + $keyword + '.*')
            ))
          )
        OPTIONAL MATCH (f)-[:SUBJECT]->(sub:Entity)
        OPTIONAL MATCH (f)-[:OBJECT]->(obj:Entity)
        RETURN f.fact_id AS fact_id,
               f.subject AS subject,
               f.predicate AS predicate,
               f.object AS object,
               f.status AS status,
               f.valid_from AS valid_from,
               f.valid_to AS valid_to,
               f.source_text AS source_text,
               f.confidence AS confidence,
               d.document_id AS document_id,
               d.title AS doc_title,
               d.reference_number AS reference_number,
               d.circular_number AS circular_number,
               d.document_date AS document_date,
               c.chunk_id AS chunk_id,
               coalesce(c.path, '') AS clause_path,
               sub.name AS subject_entity,
               obj.name AS object_entity
        ORDER BY d.document_date DESC
        LIMIT $limit
        """.strip()

    def _generate_evolution_query(self) -> str:
        return """
        MATCH path = (new_fact:Fact)-[r:MODIFIES|SUPERSEDES|CLARIFIES|DELETES|CONTRADICTS*1..5]->(old_fact:Fact)
        MATCH (new_doc:Document {document_id: new_fact.document_id})
        MATCH (old_doc:Document {document_id: old_fact.document_id})
        WHERE NOT (coalesce(new_fact.source_text, '') =~ '(?i)^(Date:|DEPARTMENT:|Download Ref|Circular Ref|NATIONAL STOCK|Page [0-9]+|CIN).*')
          AND NOT (coalesce(old_fact.source_text, '') =~ '(?i)^(Date:|DEPARTMENT:|Download Ref|Circular Ref|NATIONAL STOCK|Page [0-9]+|CIN).*')
          AND ($circular IS NULL OR coalesce(new_doc.reference_number, '') CONTAINS $circular OR coalesce(old_doc.reference_number, '') CONTAINS $circular OR coalesce(new_doc.filename, '') CONTAINS $circular OR coalesce(old_doc.filename, '') CONTAINS $circular)
          AND (
            ($entity IS NULL AND $keyword IS NULL)
            OR ($entity IS NOT NULL AND (
                coalesce(new_fact.subject, '') =~ ('(?i).*' + $entity + '.*')
                OR coalesce(old_fact.subject, '') =~ ('(?i).*' + $entity + '.*')
                OR coalesce(new_fact.rule_reference, '') =~ ('(?i).*' + $entity + '.*')
                OR coalesce(old_fact.rule_reference, '') =~ ('(?i).*' + $entity + '.*')
                OR coalesce(new_fact.source_text, '') =~ ('(?i).*' + $entity + '.*')
                OR coalesce(old_fact.source_text, '') =~ ('(?i).*' + $entity + '.*')
                OR coalesce(new_doc.title, '') =~ ('(?i).*' + $entity + '.*')
                OR coalesce(old_doc.title, '') =~ ('(?i).*' + $entity + '.*')
            ))
            OR ($keyword IS NOT NULL AND (
                coalesce(new_fact.source_text, '') =~ ('(?i).*' + $keyword + '.*')
                OR coalesce(old_fact.source_text, '') =~ ('(?i).*' + $keyword + '.*')
                OR coalesce(new_fact.object, '') =~ ('(?i).*' + $keyword + '.*')
                OR coalesce(old_fact.object, '') =~ ('(?i).*' + $keyword + '.*')
                OR coalesce(new_fact.subject, '') =~ ('(?i).*' + $keyword + '.*')
                OR coalesce(old_fact.subject, '') =~ ('(?i).*' + $keyword + '.*')
                OR coalesce(new_fact.rule_reference, '') =~ ('(?i).*' + $keyword + '.*')
                OR coalesce(old_fact.rule_reference, '') =~ ('(?i).*' + $keyword + '.*')
                OR coalesce(new_doc.title, '') =~ ('(?i).*' + $keyword + '.*')
                OR coalesce(old_doc.title, '') =~ ('(?i).*' + $keyword + '.*')
            ))
          )
        OPTIONAL MATCH (new_fact)-[:SUPPORTED_BY]->(nc:Chunk)
        OPTIONAL MATCH (old_fact)-[:SUPPORTED_BY]->(oc:Chunk)
        RETURN new_fact.fact_id AS new_fact_id,
               new_fact.subject AS new_subject,
               new_fact.predicate AS new_predicate,
               new_fact.object AS new_object,
               new_fact.status AS new_status,
               new_fact.valid_from AS new_valid_from,
               new_fact.source_text AS new_source_text,
               new_fact.rule_reference AS new_rule_ref,
               new_fact.clause_reference AS new_clause_ref,
               old_fact.fact_id AS old_fact_id,
               old_fact.subject AS old_subject,
               old_fact.predicate AS old_predicate,
               old_fact.object AS old_object,
               old_fact.status AS old_status,
               old_fact.valid_from AS old_valid_from,
               old_fact.valid_to AS old_valid_to,
               old_fact.source_text AS old_source_text,
               old_fact.rule_reference AS old_rule_ref,
               old_fact.clause_reference AS old_clause_ref,
               [rel in relationships(path) | type(rel)] AS evolution_types,
               new_doc.document_id AS new_doc_id,
               new_doc.reference_number AS new_doc_ref,
               new_doc.document_date AS new_doc_date,
               old_doc.document_id AS old_doc_id,
               old_doc.reference_number AS old_doc_ref,
               old_doc.document_date AS old_doc_date,
               coalesce(nc.path, '') AS new_clause_path,
               coalesce(oc.path, '') AS old_clause_path
        ORDER BY new_doc.document_date DESC, old_doc.document_date ASC
        LIMIT $limit
        """.strip()

    def _generate_amendment_query(self) -> str:
        return """
        MATCH (ce:ChangeEvent)
        MATCH (src_doc:Document {document_id: ce.source_document_id})
        MATCH (tgt_doc:Document {document_id: ce.target_document_id})
        WHERE ($circular IS NULL OR coalesce(src_doc.reference_number, '') CONTAINS $circular OR coalesce(src_doc.circular_number, '') CONTAINS $circular OR coalesce(src_doc.filename, '') CONTAINS $circular OR coalesce(tgt_doc.reference_number, '') CONTAINS $circular OR coalesce(tgt_doc.circular_number, '') CONTAINS $circular OR coalesce(tgt_doc.filename, '') CONTAINS $circular)
          AND (
            ($entity IS NULL AND $keyword IS NULL)
            OR ($entity IS NOT NULL AND (
                coalesce(ce.description, '') =~ ('(?i).*' + $entity + '.*')
                OR coalesce(ce.target_point_or_clause, '') =~ ('(?i).*' + $entity + '.*')
                OR coalesce(src_doc.title, '') =~ ('(?i).*' + $entity + '.*')
                OR coalesce(tgt_doc.title, '') =~ ('(?i).*' + $entity + '.*')
            ))
            OR ($keyword IS NOT NULL AND (
                coalesce(ce.description, '') =~ ('(?i).*' + $keyword + '.*')
                OR coalesce(ce.target_point_or_clause, '') =~ ('(?i).*' + $keyword + '.*')
                OR coalesce(src_doc.title, '') =~ ('(?i).*' + $keyword + '.*')
                OR coalesce(tgt_doc.title, '') =~ ('(?i).*' + $keyword + '.*')
            ))
          )
        OPTIONAL MATCH (ce)-[:AFFECTS_FACT]->(f:Fact)
        RETURN ce.id AS change_id,
               ce.change_type AS change_type,
               ce.target_point_or_clause AS target_clause,
               ce.description AS description,
               ce.effective_date AS effective_date,
               src_doc.document_id AS source_doc_id,
               src_doc.reference_number AS source_ref_num,
               src_doc.document_date AS source_date,
               tgt_doc.document_id AS target_doc_id,
               tgt_doc.reference_number AS target_ref_num,
               tgt_doc.document_date AS target_date,
               f.fact_id AS affected_fact_id,
               f.subject AS affected_subject,
               f.object AS affected_object,
               f.status AS affected_status,
               f.source_text AS affected_source_text
        ORDER BY src_doc.document_date DESC, ce.created_at DESC
        LIMIT $limit
        """.strip()

    def _generate_master_to_latest_query(self) -> str:
        return """
        MATCH (root_doc:Document)
        WHERE ($circular IS NULL OR coalesce(root_doc.reference_number, '') CONTAINS $circular OR coalesce(root_doc.circular_number, '') CONTAINS $circular OR coalesce(root_doc.filename, '') CONTAINS $circular)
          AND ($keyword IS NULL OR coalesce(root_doc.title, '') =~ ('(?i).*' + $keyword + '.*'))
        OPTIONAL MATCH (latest_doc:Document)-[:AMENDS|SUPERSEDES|MODIFIES|CLARIFIES|REFERENCES_DOC*1..5]->(root_doc)
        OPTIONAL MATCH (latest_doc)-[:HAS_CHUNK]->(:Chunk)<-[:SUPPORTED_BY]-(active_fact:Fact {status: 'ACTIVE'})
        OPTIONAL MATCH (root_doc)-[:HAS_CHUNK]->(:Chunk)<-[:SUPPORTED_BY]-(root_fact:Fact)
        RETURN root_doc.document_id AS root_doc_id,
               root_doc.reference_number AS root_ref_num,
               root_doc.document_date AS root_date,
               root_doc.title AS root_title,
               latest_doc.document_id AS latest_doc_id,
               latest_doc.reference_number AS latest_ref_num,
               latest_doc.document_date AS latest_date,
               active_fact.fact_id AS active_fact_id,
               active_fact.subject AS active_subject,
               active_fact.predicate AS active_predicate,
               active_fact.object AS active_object,
               active_fact.status AS active_status,
               active_fact.source_text AS active_source_text,
               root_fact.fact_id AS root_fact_id,
               root_fact.subject AS root_subject,
               root_fact.status AS root_status,
               root_fact.source_text AS root_source_text
        ORDER BY root_doc.document_date ASC, latest_doc.document_date DESC
        LIMIT $limit
        """.strip()

    def _generate_changelog_query(self) -> str:
        """Fetch ALL ChangeEvents across all documents with full circular, clause, and fact context.
        Used for cross-circular 'list all changes / amendments / deletions' queries.
        Supports optional entity/keyword scoping.
        """
        return """
        MATCH (ce:ChangeEvent)
        MATCH (src_doc:Document {document_id: ce.source_document_id})
        MATCH (tgt_doc:Document {document_id: ce.target_document_id})
        WHERE (
            ($entity IS NULL AND $keyword IS NULL)
            OR ($entity IS NOT NULL AND (
                coalesce(ce.description, '')               =~ ('(?i).*' + $entity + '.*')
                OR coalesce(ce.target_point_or_clause, '') =~ ('(?i).*' + $entity + '.*')
                OR coalesce(src_doc.title, '')             =~ ('(?i).*' + $entity + '.*')
                OR coalesce(tgt_doc.title, '')             =~ ('(?i).*' + $entity + '.*')
            ))
            OR ($keyword IS NOT NULL AND (
                coalesce(ce.description, '')               =~ ('(?i).*' + $keyword + '.*')
                OR coalesce(ce.target_point_or_clause, '') =~ ('(?i).*' + $keyword + '.*')
                OR coalesce(src_doc.title, '')             =~ ('(?i).*' + $keyword + '.*')
                OR coalesce(tgt_doc.title, '')             =~ ('(?i).*' + $keyword + '.*')
            ))
        )
        OPTIONAL MATCH (ce)-[:AFFECTS_FACT]->(affected_fact:Fact)
        RETURN ce.id                              AS change_id,
               ce.change_type                    AS change_type,
               ce.target_point_or_clause         AS target_clause,
               ce.description                    AS description,
               ce.effective_date                 AS effective_date,
               src_doc.document_id               AS source_doc_id,
               src_doc.title                     AS source_doc_title,
               src_doc.reference_number          AS source_ref_num,
               src_doc.circular_number           AS source_circular_num,
               src_doc.document_date             AS source_date,
               tgt_doc.document_id               AS target_doc_id,
               tgt_doc.title                     AS target_doc_title,
               tgt_doc.reference_number          AS target_ref_num,
               tgt_doc.circular_number           AS target_circular_num,
               tgt_doc.document_date             AS target_date,
               affected_fact.fact_id             AS affected_fact_id,
               affected_fact.subject             AS affected_subject,
               affected_fact.predicate           AS affected_predicate,
               affected_fact.object              AS affected_object,
               affected_fact.status              AS affected_status,
               affected_fact.clause_reference    AS affected_clause_ref,
               affected_fact.source_text         AS affected_source_text
        ORDER BY src_doc.document_date DESC, ce.change_type ASC, ce.target_point_or_clause ASC
        LIMIT $limit
        """.strip()
