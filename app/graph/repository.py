"""Neo4j repository for executing parameterized Cypher queries."""
from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Optional
from app.core.logging import logger
from app.graph.neo4j_client import Neo4jClient
from app.models.document import Document, ProcessingStatus, ProcessingStage
from app.models.hierarchy import HierarchyNode, HierarchyNodeType
from app.models.chunk import Chunk
from app.models.fact import Fact, FactStatus
from app.models.entity import Entity
from app.models.reference import DocumentReference
from app.models.change import ChangeEvent, ChangeType


class GraphRepository:
    """Repository handling all Neo4j graph operations with parameterized Cypher."""

    def __init__(self, client: Neo4jClient) -> None:
        self.client = client

    # ==================== DOCUMENT OPERATIONS ====================

    def save_document(self, doc: Document) -> None:
        """Create or update a Document node in Neo4j."""
        query = """
        MERGE (d:Document {document_id: $document_id})
        SET d.filename = $filename,
            d.file_path = $file_path,
            d.source = $source,
            d.department = $department,
            d.reference_number = $reference_number,
            d.circular_number = $circular_number,
            d.document_date = $document_date,
            d.title = $title,
            d.total_pages = $total_pages,
            d.status = $status,
            d.current_stage = $current_stage,
            d.pages_processed = $pages_processed,
            d.sections_created = $sections_created,
            d.chunks_created = $chunks_created,
            d.facts_extracted = $facts_extracted,
            d.errors = $errors,
            d.created_at = $created_at,
            d.updated_at = $updated_at
        """
        params = {
            "document_id": doc.document_id,
            "filename": doc.filename,
            "file_path": doc.file_path,
            "source": doc.source,
            "department": doc.department,
            "reference_number": doc.reference_number,
            "circular_number": doc.circular_number,
            "document_date": doc.document_date,
            "title": doc.title,
            "total_pages": doc.total_pages,
            "status": doc.status.value,
            "current_stage": doc.current_stage.value,
            "pages_processed": doc.pages_processed,
            "sections_created": doc.sections_created,
            "chunks_created": doc.chunks_created,
            "facts_extracted": doc.facts_extracted,
            "errors": doc.errors,
            "created_at": doc.created_at,
            "updated_at": doc.updated_at,
        }
        self.client.execute_query(query, params)

    def update_document_status(
        self,
        document_id: str,
        status: Optional[ProcessingStatus] = None,
        stage: Optional[ProcessingStage] = None,
        pages_processed: Optional[int] = None,
        total_pages: Optional[int] = None,
        sections_created: Optional[int] = None,
        chunks_created: Optional[int] = None,
        facts_extracted: Optional[int] = None,
        errors: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update processing progress and status of a Document."""
        set_clauses = ["d.updated_at = $updated_at"]
        params: Dict[str, Any] = {
            "document_id": document_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        if status is not None:
            set_clauses.append("d.status = $status")
            params["status"] = status.value
        if stage is not None:
            set_clauses.append("d.current_stage = $current_stage")
            params["current_stage"] = stage.value
        if pages_processed is not None:
            set_clauses.append("d.pages_processed = $pages_processed")
            params["pages_processed"] = pages_processed
        if total_pages is not None:
            set_clauses.append("d.total_pages = $total_pages")
            params["total_pages"] = total_pages
        if sections_created is not None:
            set_clauses.append("d.sections_created = $sections_created")
            params["sections_created"] = sections_created
        if chunks_created is not None:
            set_clauses.append("d.chunks_created = $chunks_created")
            params["chunks_created"] = chunks_created
        if facts_extracted is not None:
            set_clauses.append("d.facts_extracted = $facts_extracted")
            params["facts_extracted"] = facts_extracted
        if errors is not None:
            set_clauses.append("d.errors = $errors")
            params["errors"] = errors
        if metadata is not None:
            for key in ["source", "department", "reference_number", "circular_number", "document_date", "title"]:
                if key in metadata and metadata[key] is not None:
                    set_clauses.append(f"d.{key} = ${key}")
                    params[key] = metadata[key]

        query = f"""
        MATCH (d:Document {{document_id: $document_id}})
        SET {', '.join(set_clauses)}
        """
        self.client.execute_query(query, params)

    def get_document(self, document_id: str) -> Optional[Document]:
        """Fetch a single Document by its UUID."""
        query = """
        MATCH (d:Document {document_id: $document_id})
        RETURN d
        """
        records = self.client.execute_query(query, {"document_id": document_id})
        if not records:
            return None
        data = records[0]["d"]
        return Document(
            document_id=data["document_id"],
            filename=data.get("filename", ""),
            file_path=data.get("file_path", ""),
            source=data.get("source"),
            department=data.get("department"),
            reference_number=data.get("reference_number"),
            circular_number=data.get("circular_number"),
            document_date=data.get("document_date"),
            title=data.get("title"),
            total_pages=data.get("total_pages", 1),
            status=ProcessingStatus(data.get("status", "UPLOADED")),
            current_stage=ProcessingStage(data.get("current_stage", "UPLOADED")),
            pages_processed=data.get("pages_processed", 0),
            sections_created=data.get("sections_created", 0),
            chunks_created=data.get("chunks_created", 0),
            facts_extracted=data.get("facts_extracted", 0),
            errors=data.get("errors", []),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )

    def delete_document(self, document_id: str) -> bool:
        """Delete a Document and its associated hierarchy nodes, chunks, facts, and references."""
        query = """
        MATCH (d:Document {document_id: $document_id})
        OPTIONAL MATCH (d)-[:HAS_SECTION|HAS_SUBSECTION|HAS_CLAUSE*1..5]->(hn)
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(ch:Chunk)
        OPTIONAL MATCH (d)-[:HAS_FACT]->(f:Fact)
        OPTIONAL MATCH (ch)-[:CONTAINS_FACT]->(cf:Fact)
        OPTIONAL MATCH (ce:ChangeEvent) WHERE ce.source_document_id = $document_id OR ce.target_document_id = $document_id
        DETACH DELETE d, hn, ch, f, cf, ce
        RETURN count(d) as deleted_count
        """
        records = self.client.execute_query(query, {"document_id": document_id})
        return bool(records and records[0].get("deleted_count", 0) > 0)

    def list_documents(self) -> List[Document]:
        """List all Document nodes."""
        query = """
        MATCH (d:Document)
        RETURN d
        ORDER BY d.created_at DESC
        """
        records = self.client.execute_query(query)
        docs: List[Document] = []
        for rec in records:
            data = rec["d"]
            docs.append(
                Document(
                    document_id=data["document_id"],
                    filename=data.get("filename", ""),
                    file_path=data.get("file_path", ""),
                    source=data.get("source"),
                    department=data.get("department"),
                    reference_number=data.get("reference_number"),
                    circular_number=data.get("circular_number"),
                    document_date=data.get("document_date"),
                    title=data.get("title"),
                    total_pages=data.get("total_pages", 1),
                    status=ProcessingStatus(data.get("status", "UPLOADED")),
                    current_stage=ProcessingStage(data.get("current_stage", "UPLOADED")),
                    pages_processed=data.get("pages_processed", 0),
                    sections_created=data.get("sections_created", 0),
                    chunks_created=data.get("chunks_created", 0),
                    facts_extracted=data.get("facts_extracted", 0),
                    errors=data.get("errors", []),
                    created_at=data.get("created_at", ""),
                    updated_at=data.get("updated_at", ""),
                )
            )
        return docs

    def find_document_by_reference(
        self,
        reference_number: Optional[str] = None,
        circular_number: Optional[str] = None,
        date: Optional[str] = None,
    ) -> Optional[Document]:
        """Search existing documents matching exact or normalized reference_number, circular_number, or filename."""
        from app.services.reference_detector import normalize_reference_number

        norm_ref = normalize_reference_number(reference_number) if reference_number else None
        ref_suffix = None
        if norm_ref:
            m = re.search(r"([0-9]{4,6})", norm_ref)
            if m:
                ref_suffix = m.group(1)

        norm_circ = circular_number.replace(" ", "").strip() if circular_number else None

        logger.info(
            "Neo4j Document lookup | raw_ref='%s' | norm_ref='%s' | ref_suffix='%s' | circular_num='%s'",
            reference_number,
            norm_ref,
            ref_suffix,
            norm_circ,
        )

        query = """
        MATCH (d:Document)
        WHERE (
            $norm_ref IS NOT NULL AND (
                toLower(d.reference_number) = toLower($norm_ref)
                OR replace(toLower(d.reference_number), ' ', '') = replace(toLower($norm_ref), ' ', '')
                OR ($ref_suffix IS NOT NULL AND (
                    d.reference_number CONTAINS $ref_suffix
                    OR d.filename CONTAINS $ref_suffix
                ))
            )
        ) OR (
            $circular_num IS NOT NULL AND (
                d.circular_number = $circular_num
                OR replace(toLower(d.circular_number), ' ', '') = replace(toLower($circular_num), ' ', '')
            )
        )
        RETURN d
        LIMIT 1
        """
        params = {
            "norm_ref": norm_ref,
            "ref_suffix": ref_suffix,
            "circular_num": norm_circ,
        }
        records = self.client.execute_query(query, params)
        if not records:
            logger.info("Neo4j Document lookup returned NO MATCH for params=%s", params)
            return None

        data = records[0]["d"]
        logger.info(
            "Neo4j Document lookup MATCHED: document_id=%s, reference_number=%s, circular_number=%s, filename=%s",
            data.get("document_id"),
            data.get("reference_number"),
            data.get("circular_number"),
            data.get("filename"),
        )
        return Document(
            document_id=data["document_id"],
            filename=data.get("filename", ""),
            file_path=data.get("file_path", ""),
            source=data.get("source"),
            department=data.get("department"),
            reference_number=data.get("reference_number"),
            circular_number=data.get("circular_number"),
            document_date=data.get("document_date"),
            title=data.get("title"),
            total_pages=data.get("total_pages", 1),
            status=ProcessingStatus(data.get("status", "UPLOADED")),
            current_stage=ProcessingStage(data.get("current_stage", "UPLOADED")),
        )

    # ==================== HIERARCHY OPERATIONS ====================

    def save_hierarchy_nodes(self, document_id: str, nodes: List[HierarchyNode]) -> None:
        """Store all hierarchy nodes and link them according to their tree structure."""
        if not nodes:
            return

        for node in nodes:
            # Map node type to primary Neo4j Label (Section, Clause, FAQ, etc.)
            labels = ["HierarchyNode"]
            if node.node_type in [HierarchyNodeType.SECTION, HierarchyNodeType.ANNEXURE]:
                labels.append("Section")
            elif node.node_type in [HierarchyNodeType.SUBSECTION]:
                labels.append("Section")
                labels.append("Subsection")
            elif node.node_type in [HierarchyNodeType.CLAUSE, HierarchyNodeType.POINT, HierarchyNodeType.LIST_ITEM]:
                labels.append("Clause")
            elif node.node_type == HierarchyNodeType.FAQ:
                labels.append("FAQ")
            elif node.node_type == HierarchyNodeType.QUESTION:
                labels.append("Question")
            elif node.node_type == HierarchyNodeType.ANSWER:
                labels.append("Answer")
            else:
                labels.append("Section")

            label_str = ":".join(labels)

            query = f"""
            MERGE (n:{label_str} {{id: $id}})
            SET n.document_id = $document_id,
                n.node_type = $node_type,
                n.title = $title,
                n.content = $content,
                n.level = $level,
                n.order = $order,
                n.page_start = $page_start,
                n.page_end = $page_end,
                n.parent_id = $parent_id
            """
            self.client.execute_query(
                query,
                {
                    "id": node.id,
                    "document_id": document_id,
                    "node_type": node.node_type.value,
                    "title": node.title,
                    "content": node.content,
                    "level": node.level,
                    "order": node.order,
                    "page_start": node.page_start,
                    "page_end": node.page_end,
                    "parent_id": node.parent_id,
                },
            )

            # Link to Document or Parent
            if node.parent_id:
                rel_query = """
                MATCH (p {id: $parent_id})
                MATCH (c {id: $child_id})
                MERGE (p)-[:HAS_SUBSECTION]->(c)
                """
                if node.node_type in [HierarchyNodeType.CLAUSE, HierarchyNodeType.POINT, HierarchyNodeType.LIST_ITEM]:
                    rel_query = """
                    MATCH (p {id: $parent_id})
                    MATCH (c {id: $child_id})
                    MERGE (p)-[:HAS_CLAUSE]->(c)
                    """
                self.client.execute_query(rel_query, {"parent_id": node.parent_id, "child_id": node.id})
            else:
                doc_rel_query = """
                MATCH (d:Document {document_id: $document_id})
                MATCH (c {id: $child_id})
                MERGE (d)-[:HAS_SECTION]->(c)
                """
                self.client.execute_query(doc_rel_query, {"document_id": document_id, "child_id": node.id})

    def get_document_hierarchy(self, document_id: str) -> List[Dict[str, Any]]:
        """Retrieve complete hierarchy structure for a document."""
        query = """
        MATCH (d:Document {document_id: $document_id})
        OPTIONAL MATCH p = (d)-[:HAS_SECTION|HAS_SUBSECTION|HAS_CLAUSE*1..5]->(n)
        RETURN n, relationships(p) as rels
        ORDER BY n.level ASC, n.order ASC
        """
        records = self.client.execute_query(query, {"document_id": document_id})
        nodes: List[Dict[str, Any]] = []
        seen_ids = set()
        for rec in records:
            node_data = rec.get("n")
            if node_data and node_data.get("id") not in seen_ids:
                seen_ids.add(node_data.get("id"))
                nodes.append(node_data)
        return nodes

    # ==================== CHUNKING OPERATIONS ====================

    def save_chunks(self, document_id: str, chunks: List[Chunk]) -> None:
        """Store chunk nodes, link them to the Document/Clause, and create NEXT/PREVIOUS sequential chains."""
        if not chunks:
            return

        for chunk in chunks:
            chunk_query = """
            MERGE (c:Chunk {chunk_id: $chunk_id})
            SET c.document_id = $document_id,
                c.content = $content,
                c.heading_path = $heading_path,
                c.parent_id = $parent_id,
                c.previous_chunk_id = $previous_chunk_id,
                c.next_chunk_id = $next_chunk_id,
                c.page_start = $page_start,
                c.page_end = $page_end,
                c.order = $order,
                c.token_count = $token_count
            """
            self.client.execute_query(
                chunk_query,
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": document_id,
                    "content": chunk.content,
                    "heading_path": chunk.heading_path,
                    "parent_id": chunk.parent_id,
                    "previous_chunk_id": chunk.previous_chunk_id,
                    "next_chunk_id": chunk.next_chunk_id,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "order": chunk.order,
                    "token_count": chunk.token_count,
                },
            )

            # Link Document -> Chunk
            doc_link = """
            MATCH (d:Document {document_id: $document_id})
            MATCH (c:Chunk {chunk_id: $chunk_id})
            MERGE (d)-[:HAS_CHUNK]->(c)
            """
            self.client.execute_query(doc_link, {"document_id": document_id, "chunk_id": chunk.chunk_id})

            # Link Parent Clause/Section -> Chunk if parent_id exists
            if chunk.parent_id:
                parent_link = """
                MATCH (p {id: $parent_id})
                MATCH (c:Chunk {chunk_id: $chunk_id})
                MERGE (p)-[:HAS_CHUNK]->(c)
                """
                self.client.execute_query(parent_link, {"parent_id": chunk.parent_id, "chunk_id": chunk.chunk_id})

        # Link sequential NEXT and PREVIOUS relationships
        for i in range(len(chunks) - 1):
            curr_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            seq_query = """
            MATCH (c1:Chunk {chunk_id: $c1_id})
            MATCH (c2:Chunk {chunk_id: $c2_id})
            MERGE (c1)-[:NEXT]->(c2)
            MERGE (c2)-[:PREVIOUS]->(c1)
            """
            self.client.execute_query(
                seq_query,
                {"c1_id": curr_chunk.chunk_id, "c2_id": next_chunk.chunk_id},
            )

    # ==================== ENTITY OPERATIONS ====================

    def save_entities(self, entities: List[Entity]) -> None:
        """Upsert entities by normalized_name and type, updating aliases."""
        if not entities:
            return

        for entity in entities:
            query = """
            MERGE (e:Entity {normalized_name: $normalized_name, type: $type})
            ON CREATE SET 
                e.entity_id = $entity_id,
                e.name = $name,
                e.aliases = $aliases,
                e.description = $description
            ON MATCH SET 
                e.aliases = [x IN apoc.coll.toSet(e.aliases + $aliases) WHERE x IS NOT NULL]
            """
            # Fallback if APOC is not available: simple merge
            fallback_query = """
            MERGE (e:Entity {normalized_name: $normalized_name, type: $type})
            ON CREATE SET 
                e.entity_id = $entity_id,
                e.name = $name,
                e.aliases = $aliases,
                e.description = $description
            ON MATCH SET
                e.name = $name
            """
            try:
                self.client.execute_query(
                    query,
                    {
                        "entity_id": entity.entity_id,
                        "name": entity.name,
                        "type": entity.type.value,
                        "normalized_name": entity.normalized_name,
                        "aliases": entity.aliases,
                        "description": entity.description,
                    },
                )
            except Exception:
                # Use fallback if APOC is not installed
                self.client.execute_query(
                    fallback_query,
                    {
                        "entity_id": entity.entity_id,
                        "name": entity.name,
                        "type": entity.type.value,
                        "normalized_name": entity.normalized_name,
                        "aliases": entity.aliases,
                        "description": entity.description,
                    },
                )

    # ==================== FACT OPERATIONS ====================

    def save_facts(self, document_id: str, facts: List[Fact]) -> None:
        """Store atomic facts, link to Chunk with provenance, and link to Subject/Object Entities."""
        if not facts:
            return

        for fact in facts:
            fact_query = """
            MERGE (f:Fact {fact_id: $fact_id})
            SET f.document_id = $document_id,
                f.chunk_id = $chunk_id,
                f.hierarchy_node_id = $hierarchy_node_id,
                f.subject = $subject,
                f.predicate = $predicate,
                f.object = $object,
                f.conditions = $conditions,
                f.exceptions = $exceptions,
                f.regulatory_status = $regulatory_status,
                f.effective_date = $effective_date,
                f.confidence = $confidence,
                f.source_text = $source_text,
                f.extraction_method = $extraction_method,
                f.model = $model,
                f.fallback_reason = $fallback_reason,
                f.status = $status,
                f.valid_from = $valid_from,
                f.valid_to = $valid_to,
                f.created_at = $created_at,
                f.updated_at = $updated_at
            """
            self.client.execute_query(
                fact_query,
                {
                    "fact_id": fact.fact_id,
                    "document_id": document_id,
                    "chunk_id": fact.chunk_id,
                    "hierarchy_node_id": fact.hierarchy_node_id,
                    "subject": fact.subject,
                    "predicate": fact.predicate.value,
                    "object": fact.object,
                    "conditions": fact.conditions,
                    "exceptions": fact.exceptions,
                    "regulatory_status": fact.regulatory_status,
                    "effective_date": fact.effective_date,
                    "confidence": fact.confidence,
                    "source_text": fact.source_text,
                    "extraction_method": fact.extraction_method,
                    "model": fact.model,
                    "fallback_reason": fact.fallback_reason,
                    "status": fact.status.value,
                    "valid_from": fact.valid_from,
                    "valid_to": fact.valid_to,
                    "created_at": fact.created_at,
                    "updated_at": fact.updated_at,
                },
            )

            # Provenance: (Chunk)-[:CONTAINS_FACT]->(Fact)
            prov_query = """
            MATCH (c:Chunk {chunk_id: $chunk_id})
            MATCH (f:Fact {fact_id: $fact_id})
            MERGE (c)-[:CONTAINS_FACT]->(f)
            """
            self.client.execute_query(prov_query, {"chunk_id": fact.chunk_id, "fact_id": fact.fact_id})

            # Direct shortcut: (Document)-[:HAS_FACT]->(Fact) for efficient cross-doc queries
            has_fact_query = """
            MATCH (d:Document {document_id: $document_id})
            MATCH (f:Fact {fact_id: $fact_id})
            MERGE (d)-[:HAS_FACT]->(f)
            """
            self.client.execute_query(has_fact_query, {"document_id": document_id, "fact_id": fact.fact_id})

            # Link Subject Entity
            subject_query = """
            MATCH (f:Fact {fact_id: $fact_id})
            MERGE (e:Entity {normalized_name: toUpper($subject_name)})
            ON CREATE SET e.entity_id = randomUUID(), e.name = $subject_name, e.type = 'ACTIVITY'
            MERGE (f)-[:SUBJECT]->(e)
            """
            self.client.execute_query(subject_query, {"fact_id": fact.fact_id, "subject_name": fact.subject})

            # Link Object Entity
            object_query = """
            MATCH (f:Fact {fact_id: $fact_id})
            MERGE (e:Entity {normalized_name: toUpper($object_name)})
            ON CREATE SET e.entity_id = randomUUID(), e.name = $object_name, e.type = 'REQUIREMENT'
            MERGE (f)-[:OBJECT]->(e)
            """
            self.client.execute_query(object_query, {"fact_id": fact.fact_id, "object_name": fact.object})

    def get_document_facts(self, document_id: str) -> List[Dict[str, Any]]:
        """Retrieve all facts belonging to a document along with provenance and status."""
        query = """
        MATCH (f:Fact {document_id: $document_id})
        OPTIONAL MATCH (c:Chunk)-[:CONTAINS_FACT]->(f)
        OPTIONAL MATCH (f)-[:SUBJECT]->(sub:Entity)
        OPTIONAL MATCH (f)-[:OBJECT]->(obj:Entity)
        RETURN f, c.chunk_id AS chunk_id, c.heading_path AS heading_path,
               sub.name AS subject_entity, obj.name AS object_entity
        ORDER BY f.created_at ASC
        """
        records = self.client.execute_query(query, {"document_id": document_id})
        facts = []
        for rec in records:
            fact_data = rec["f"]
            fact_data["chunk_id"] = rec.get("chunk_id")
            fact_data["heading_path"] = rec.get("heading_path")
            fact_data["subject_entity"] = rec.get("subject_entity")
            fact_data["object_entity"] = rec.get("object_entity")
            facts.append(fact_data)
        return facts

    # ==================== CROSS-DOCUMENT REFERENCES & CHANGE TRACKING ====================

    def save_reference(self, ref: DocumentReference) -> None:
        """Create cross-document and cross-clause reference edges."""
        if not ref.target_document_id:
            return

        rel_type = "AMENDS" if ref.reference_type == "AMENDS" else "REFERENCES"
        doc_ref_query = f"""
        MATCH (source:Document {{document_id: $source_doc_id}})
        MATCH (target:Document {{document_id: $target_doc_id}})
        MERGE (source)-[r:REFERENCES_DOC]->(target)
        SET r.reference_id = $reference_id,
            r.reference_type = $reference_type,
            r.reference_text = $reference_text,
            r.target_point_or_clause = $target_point_or_clause,
            r.confidence = $confidence,
            r.updated_at = timestamp()
        MERGE (source)-[r_typed:{rel_type}]->(target)
        SET r_typed.reference_id = $reference_id,
            r_typed.reference_text = $reference_text,
            r_typed.target_point_or_clause = $target_point_or_clause,
            r_typed.confidence = $confidence
        """
        self.client.execute_query(
            doc_ref_query,
            {
                "source_doc_id": ref.source_document_id,
                "target_doc_id": ref.target_document_id,
                "reference_id": ref.reference_id,
                "reference_type": ref.reference_type,
                "reference_text": ref.reference_text,
                "target_point_or_clause": ref.target_point_or_clause,
                "confidence": ref.confidence,
            },
        )
        logger.info(
            "Persisted Neo4j Document relationship: (Document {id=%s})-[:REFERENCES_DOC]->(Document {id=%s})",
            ref.source_document_id,
            ref.target_document_id,
        )

        # If specific clause references exist, link them
        if ref.source_clause_id and ref.target_clause_id:
            clause_ref_query = f"""
            MATCH (source:Clause {{id: $source_clause_id}})
            MATCH (target:Clause {{id: $target_clause_id}})
            MERGE (source)-[r:{rel_type}]->(target)
            SET r.reference_id = $reference_id,
                r.reference_text = $reference_text
            """
            self.client.execute_query(
                clause_ref_query,
                {
                    "source_clause_id": ref.source_clause_id,
                    "target_clause_id": ref.target_clause_id,
                    "reference_id": ref.reference_id,
                    "reference_text": ref.reference_text,
                },
            )

    def save_change_event(self, event: ChangeEvent) -> None:
        """Record a ChangeEvent node and link affected facts and documents."""
        query = """
        MERGE (ce:ChangeEvent {id: $id})
        SET ce.source_document_id = $source_doc_id,
            ce.target_document_id = $target_doc_id,
            ce.source_fact_id = $source_fact_id,
            ce.target_fact_id = $target_fact_id,
            ce.change_type = $change_type,
            ce.target_point_or_clause = $target_point_or_clause,
            ce.description = $description,
            ce.effective_date = $effective_date,
            ce.created_at = $created_at
        WITH ce
        MATCH (target_doc:Document {document_id: $target_doc_id})
        MERGE (ce)-[:APPLIED_TO_DOC]->(target_doc)
        WITH ce
        MATCH (source_doc:Document {document_id: $source_doc_id})
        MERGE (ce)-[:ORIGINATED_FROM_DOC]->(source_doc)
        MERGE (source_doc)-[:PRODUCED_CHANGE_EVENT]->(ce)
        """
        self.client.execute_query(
            query,
            {
                "id": event.id,
                "source_doc_id": event.source_document_id,
                "target_doc_id": event.target_document_id,
                "source_fact_id": event.source_fact_id,
                "target_fact_id": event.target_fact_id,
                "change_type": event.change_type.value,
                "target_point_or_clause": event.target_point_or_clause,
                "description": event.description,
                "effective_date": event.effective_date,
                "created_at": event.created_at,
            },
        )

        # Link target fact if explicitly specified: (ce:ChangeEvent)-[:AFFECTS_FACT]->(target_fact)
        if event.target_fact_id:
            tgt_fact_query = """
            MATCH (ce:ChangeEvent {id: $id})
            MATCH (tf:Fact {fact_id: $target_fact_id})
            MERGE (ce)-[:AFFECTS_FACT]->(tf)
            """
            self.client.execute_query(tgt_fact_query, {"id": event.id, "target_fact_id": event.target_fact_id})

        # Link source fact if explicitly specified: (ce:ChangeEvent)-[:PRODUCED_BY_FACT]->(source_fact)
        if event.source_fact_id:
            src_fact_query = """
            MATCH (ce:ChangeEvent {id: $id})
            MATCH (sf:Fact {fact_id: $source_fact_id})
            MERGE (ce)-[:PRODUCED_BY_FACT]->(sf)
            """
            self.client.execute_query(src_fact_query, {"id": event.id, "source_fact_id": event.source_fact_id})

        # If source fact and target fact exist, create direct semantic fact-to-fact evolution edge
        if event.source_fact_id and event.target_fact_id:
            rel_name = event.change_type.value
            fact_rel_query = f"""
            MATCH (new_fact:Fact {{fact_id: $new_fact_id}})
            MATCH (old_fact:Fact {{fact_id: $old_fact_id}})
            MERGE (new_fact)-[r:{rel_name}]->(old_fact)
            SET r.change_event_id = $event_id,
                r.effective_date = $effective_date
            """
            self.client.execute_query(
                fact_rel_query,
                {
                    "new_fact_id": event.source_fact_id,
                    "old_fact_id": event.target_fact_id,
                    "event_id": event.id,
                    "effective_date": event.effective_date,
                },
            )

        logger.info(
            "Saved ChangeEvent | id=%s | type=%s | target_clause='%s' | target_doc=%s",
            event.id,
            event.change_type.value,
            event.target_point_or_clause,
            event.target_document_id,
        )

    def update_fact_temporal_status(
        self,
        fact_id: str,
        status: FactStatus,
        valid_to: Optional[str] = None,
    ) -> None:
        """Update temporal status and valid_to timestamp for an evolved or deleted fact."""
        query = """
        MATCH (f:Fact {fact_id: $fact_id})
        SET f.status = $status,
            f.valid_to = CASE WHEN $valid_to IS NOT NULL THEN $valid_to ELSE f.valid_to END,
            f.updated_at = $updated_at
        """
        self.client.execute_query(
            query,
            {
                "fact_id": fact_id,
                "status": status.value,
                "valid_to": valid_to,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def find_target_facts_by_point_or_clause(
        self,
        target_document_id: str,
        point_or_clause: str,
    ) -> List[Dict[str, Any]]:
        """Find facts in target document that belong to a specific point or clause (e.g. 'Point 3', 'Point No. 3')."""
        query = """
        MATCH (d:Document {document_id: $doc_id})
        MATCH (d)-[:HAS_SECTION|HAS_SUBSECTION|HAS_CLAUSE*1..5]->(c)
        WHERE toLower(c.title) CONTAINS toLower($pattern)
           OR any(word IN split($pattern, ' ') WHERE toLower(c.title) CONTAINS toLower(word))
        MATCH (c)-[:HAS_CHUNK]->(ch:Chunk)-[:CONTAINS_FACT]->(f:Fact)
        RETURN DISTINCT f.fact_id AS fact_id, f.subject AS subject, f.predicate AS predicate,
                        f.object AS object, f.status AS status, c.title AS clause_title
        """
        return self.client.execute_query(query, {"doc_id": target_document_id, "pattern": point_or_clause})

    def find_all_facts_by_document(self, document_id: str) -> List[Dict[str, Any]]:
        """Find all facts belonging to a document."""
        query = """
        MATCH (f:Fact {document_id: $doc_id})
        RETURN f.fact_id AS fact_id, f.subject AS subject, f.predicate AS predicate,
               f.object AS object, f.status AS status, f.valid_from AS valid_from, f.valid_to AS valid_to
        """
        return self.client.execute_query(query, {"doc_id": document_id})

    def get_document_changes(self, document_id: str) -> List[Dict[str, Any]]:
        """Fetch all change events initiated by or affecting this document."""
        query = """
        MATCH (ce:ChangeEvent)
        WHERE ce.source_document_id = $doc_id OR ce.target_document_id = $doc_id
        RETURN ce
        ORDER BY ce.created_at DESC
        """
        records = self.client.execute_query(query, {"doc_id": document_id})
        return [dict(rec["ce"]) for rec in records]

    def create_fact_relationship(
        self,
        source_fact_id: str,
        target_fact_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create a semantic relationship edge between two Fact nodes.

        Relationship aliases are normalized to the canonical set:
          AMENDS, EXTENDS  → MODIFIES
          REPLACES         → SUPERSEDES
          EXEMPTS          → EXCEPTION_TO
          APPLIES_TO       → not a fact-to-fact relationship; ignored
          RELATED_TO       → too vague; ignored

        Canonical fact-to-fact relationships:
          Core:     MODIFIES, SUPERSEDES, DELETES, CLARIFIES
          Semantic: REFERENCES, EXCEPTION_TO, DEPENDS_ON, CONTRADICTS
        """
        props = properties or {}
        rel_upper = relationship_type.upper()

        _aliases: Dict[str, str] = {
            "AMENDS": "MODIFIES",
            "EXTENDS": "MODIFIES",
            "REPLACES": "SUPERSEDES",
            "EXEMPTS": "EXCEPTION_TO",
        }
        _discard = {"APPLIES_TO", "RELATED_TO", "NONE"}
        _canonical = {
            "MODIFIES", "SUPERSEDES", "DELETES", "CLARIFIES",
            "REFERENCES", "EXCEPTION_TO", "DEPENDS_ON", "CONTRADICTS",
        }

        if rel_upper in _discard:
            logger.debug("Skipping fact-to-fact relationship '%s' (not meaningful)", rel_upper)
            return

        rel_clean = _aliases.get(rel_upper, rel_upper)
        if rel_clean not in _canonical:
            logger.warning(
                "Unknown fact-to-fact relationship '%s' (aliased from '%s'); skipping",
                rel_clean, rel_upper,
            )
            return

        query = f"""
        MATCH (f1:Fact {{fact_id: $src_id}})
        MATCH (f2:Fact {{fact_id: $tgt_id}})
        MERGE (f1)-[r:{rel_clean}]->(f2)
        SET r.confidence = $confidence,
            r.reason = $reason,
            r.extraction_method = $extraction_method,
            r.updated_at = timestamp()
        """
        self.client.execute_query(
            query,
            {
                "src_id": source_fact_id,
                "tgt_id": target_fact_id,
                "confidence": props.get("confidence", 0.95),
                "reason": props.get("reason", ""),
                "extraction_method": props.get("extraction_method", "groq"),
            },
        )

    def update_facts_status_by_clause(
        self,
        document_id: str,
        clause_identifier: str,
        new_status: FactStatus,
        valid_to: Optional[str] = None,
        change_event_id: Optional[str] = None,
    ) -> int:
        """Update the temporal status of all facts in a document that belong to a given clause.

        Matches facts flexibly by clause number (e.g. 'Point 3' -> 3, '10' -> 10) in source_text,
        object, and chunk heading paths. Returns count of updated facts and links (ce)-[:AFFECTS_FACT]->(f).
        """
        m = re.search(r"(\d+[a-zA-Z]?)", clause_identifier)
        num_clause = m.group(1) if m else clause_identifier.strip()

        query = """
        MATCH (f:Fact {document_id: $document_id})
        WHERE toLower(f.source_text) CONTAINS toLower($clause_id)
           OR toLower(f.object) CONTAINS toLower($clause_id)
           OR f.source_text =~ ('(?i)^\\s*' + $num_clause + '[.:) ].*')
           OR f.source_text =~ ('(?i).*\\b' + $num_clause + '[.:) ].*')
        SET f.status = $status,
            f.valid_to = CASE WHEN $valid_to IS NOT NULL THEN $valid_to ELSE f.valid_to END,
            f.updated_at = $updated_at
        WITH f
        OPTIONAL MATCH (ce:ChangeEvent {id: $change_event_id})
        FOREACH (_ IN CASE WHEN ce IS NOT NULL THEN [1] ELSE [] END |
            MERGE (ce)-[:AFFECTS_FACT]->(f)
        )
        RETURN count(f) AS updated_count, collect(f.fact_id) AS affected_fact_ids
        """
        records = self.client.execute_query(
            query,
            {
                "document_id": document_id,
                "clause_id": clause_identifier,
                "num_clause": num_clause,
                "status": new_status.value,
                "valid_to": valid_to,
                "change_event_id": change_event_id or "",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        count = records[0]["updated_count"] if records else 0
        fact_ids = records[0]["affected_fact_ids"] if records else []
        logger.info(
            "Updated status of %d facts in document %s matching clause '%s' (num='%s') → %s (affected_facts=%s)",
            count,
            document_id,
            clause_identifier,
            num_clause,
            new_status.value,
            fact_ids,
        )
        return count

    def get_fact_evolution(
        self,
        fact_id: str,
        max_hops: int = 10,
    ) -> List[Dict[str, Any]]:
        """Traverse the temporal evolution chain of a fact.

        Returns an ordered list of facts from the given fact_id backwards through
        all MODIFIES / SUPERSEDES / CLARIFIES / DELETES relationships.
        """
        query = """
        MATCH path = (start:Fact {fact_id: $fact_id})
              <-[:MODIFIES|SUPERSEDES|CLARIFIES|DELETES*0..$max_hops]-(ancestor:Fact)
        WITH nodes(path) AS chain_nodes, relationships(path) AS chain_rels, length(path) AS depth
        UNWIND range(0, size(chain_nodes) - 1) AS i
        WITH chain_nodes[i] AS fact_node,
             CASE WHEN i < size(chain_rels) THEN type(chain_rels[i]) ELSE null END AS change_rel,
             i AS position
        RETURN fact_node.fact_id AS fact_id,
               fact_node.subject AS subject,
               fact_node.predicate AS predicate,
               fact_node.object AS object,
               fact_node.status AS status,
               fact_node.valid_from AS valid_from,
               fact_node.valid_to AS valid_to,
               fact_node.document_id AS document_id,
               change_rel AS changed_by_relationship,
               position
        ORDER BY position ASC
        """
        return self.client.execute_query(
            query,
            {"fact_id": fact_id, "max_hops": max_hops},
        )

    # ==================== DEAD-LETTER / FAILED JOB OPERATIONS ====================

    def save_dead_letter_job(self, job_dict: Dict[str, Any]) -> None:
        """Persist a failed LLM job for later background reprocessing."""
        query = """
        MERGE (j:DeadLetterJob {job_id: $job_id})
        SET j.document_id = $document_id,
            j.job_type = $job_type,
            j.status = $status,
            j.attempt_count = $attempt_count,
            j.error_code = $error_code,
            j.error_message = $error_message,
            j.payload_json = $payload_json,
            j.created_at = $created_at,
            j.updated_at = $updated_at
        """
        import json
        self.client.execute_query(
            query,
            {
                "job_id": job_dict.get("job_id"),
                "document_id": job_dict.get("document_id"),
                "job_type": str(job_dict.get("job_type", "FACT_EXTRACTION")),
                "status": str(job_dict.get("status", "FAILED_RETRYABLE")),
                "attempt_count": int(job_dict.get("attempt_count", 1)),
                "error_code": job_dict.get("error_code"),
                "error_message": job_dict.get("error_message", ""),
                "payload_json": json.dumps(job_dict.get("payload", {})),
                "created_at": job_dict.get("created_at", ""),
                "updated_at": job_dict.get("updated_at", ""),
            },
        )

    def get_dead_letter_jobs(self, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve all failed retryable jobs."""
        import json
        if document_id:
            query = """
            MATCH (j:DeadLetterJob {document_id: $document_id})
            RETURN j
            ORDER BY j.created_at DESC
            """
            records = self.client.execute_query(query, {"document_id": document_id})
        else:
            query = """
            MATCH (j:DeadLetterJob)
            RETURN j
            ORDER BY j.created_at DESC
            """
            records = self.client.execute_query(query)

        jobs = []
        for rec in records:
            data = dict(rec["j"])
            payload_str = data.pop("payload_json", "{}")
            try:
                data["payload"] = json.loads(payload_str)
            except Exception:
                data["payload"] = {}
            jobs.append(data)
        return jobs

    def update_dead_letter_job_status(self, job_id: str, status: str) -> None:
        """Update the status of a dead-letter job."""
        query = """
        MATCH (j:DeadLetterJob {job_id: $job_id})
        SET j.status = $status,
            j.updated_at = timestamp()
        """
        self.client.execute_query(query, {"job_id": job_id, "status": status})
