"""Change detection, candidate generation, and temporal fact evolution service."""
import re
from typing import Any, Callable, Dict, List, Optional
from app.core.logging import logger
from app.graph.repository import GraphRepository
from app.llm.base import BaseLLMProvider
from app.models.change import ChangeEvent, ChangeType
from app.models.chunk import Chunk
from app.models.document import DocumentMetadata
from app.models.fact import Fact, FactStatus
from app.models.reference import DocumentReference
from app.services.candidate_generator import CandidateGenerator


class ChangeDetector:
    """Detects regulatory modifications, deletions, supersessions, and updates temporal states."""

    def __init__(self, repository: GraphRepository, llm_provider: BaseLLMProvider) -> None:
        self.repository = repository
        self.llm_provider = llm_provider
        self.candidate_generator = CandidateGenerator()

    async def detect_and_apply_changes(
        self,
        source_document_id: str,
        references: List[DocumentReference],
        chunks: List[Chunk],
        new_facts: List[Fact],
        metadata: Optional[DocumentMetadata] = None,
        on_dead_letter: Optional[Callable[[int, str, Dict[str, Any]], None]] = None,
    ) -> List[ChangeEvent]:
        """Analyze changes against referenced documents, record ChangeEvents, and evolve fact temporal states."""
        logger.info(
            "Running cross-document change detection for source_document_id=%s | total_refs=%d | new_facts=%d",
            source_document_id,
            len(references),
            len(new_facts),
        )
        all_change_events: List[ChangeEvent] = []
        effective_date = metadata.document_date if metadata else None

        valid_refs = [r for r in references if r.target_document_id]
        if not valid_refs:
            logger.info("No referenced target documents found in Neo4j for document %s. Skipping change evolution.", source_document_id)
            return []

        full_amending_text = "\n\n".join([c.content for c in chunks])

        for ref in valid_refs:
            target_doc_id = ref.target_document_id
            if not target_doc_id:
                continue

            logger.info(
                "Processing referenced target document: target_doc_id=%s | ref_type=%s | ref_text='%s'",
                target_doc_id,
                ref.reference_type,
                ref.reference_text[:120].replace("\n", " "),
            )

            target_facts_raw = self.repository.get_document_facts(target_doc_id)
            target_facts = [
                Fact(
                    fact_id=tf["fact_id"],
                    document_id=target_doc_id,
                    chunk_id=tf.get("chunk_id", ""),
                    subject=tf.get("subject", ""),
                    predicate=tf.get("predicate", "REQUIRES"),
                    object=tf.get("object", ""),
                    source_text=tf.get("source_text", ""),
                    status=tf.get("status", "ACTIVE"),
                )
                for tf in target_facts_raw
            ]
            logger.info("Loaded %d existing facts from target document %s for comparison", len(target_facts), target_doc_id)

            # 1. Deterministic range deletion & modification detection
            detected_changes = self._detect_deterministic_changes(
                source_doc_id=source_document_id,
                target_doc_id=target_doc_id,
                text=full_amending_text,
                effective_date=effective_date,
            )
            logger.info("Deterministic change detection found %d change events", len(detected_changes))

            # 2. Candidate generation & batched semantic relationship verification
            if new_facts and target_facts:
                candidates = self.candidate_generator.generate_candidates(
                    source_facts=new_facts,
                    target_facts=target_facts,
                    references=[ref],
                    source_chunks=chunks,
                )

                if candidates:
                    logger.info("Sending %d top-K candidate pairs for batched relationship classification", len(candidates))
                    classifications = await self.llm_provider.classify_relationships_batch(
                        candidate_pairs=candidates,
                        on_dead_letter=on_dead_letter,
                    )

                    for cls_res in classifications:
                        rel = cls_res.get("relationship", "NONE")
                        if rel in {"MODIFIES", "DELETES", "SUPERSEDES", "CLARIFIES", "AMENDS", "REPLACES"}:
                            src_id = cls_res.get("source_id")
                            tgt_id = cls_res.get("target_id")

                            ctype_map = {
                                "MODIFIES": ChangeType.MODIFIED,
                                "REPLACES": ChangeType.MODIFIED,
                                "AMENDS": ChangeType.MODIFIED,
                                "DELETES": ChangeType.DELETED,
                                "SUPERSEDES": ChangeType.SUPERSEDED,
                                "CLARIFIES": ChangeType.CLARIFIED,
                            }
                            ctype = ctype_map.get(rel, ChangeType.MODIFIED)

                            change_event = ChangeEvent(
                                source_document_id=source_document_id,
                                target_document_id=target_doc_id,
                                source_fact_id=src_id,
                                target_fact_id=tgt_id,
                                change_type=ctype,
                                description=cls_res.get("reason", f"Fact {src_id} {rel} fact {tgt_id}"),
                                effective_date=effective_date,
                            )
                            detected_changes.append(change_event)

                            # Connect semantic fact-to-fact relationship in Neo4j
                            if src_id and tgt_id:
                                self.repository.create_fact_relationship(
                                    source_fact_id=src_id,
                                    target_fact_id=tgt_id,
                                    relationship_type=rel,
                                    properties={
                                        "confidence": cls_res.get("confidence", 0.95),
                                        "reason": cls_res.get("reason", ""),
                                        "extraction_method": cls_res.get("extraction_method", "groq"),
                                    },
                                )
                                logger.info(
                                    "Created semantic fact relationship: (Fact {id=%s})-[:%s]->(Fact {id=%s})",
                                    src_id, rel, tgt_id,
                                )

            # 3. Apply changes to Neo4j graph & evolve temporal fact states
            for change in detected_changes:
                self._apply_single_change(change, new_facts, target_doc_id, effective_date)
                all_change_events.append(change)

        logger.info("Recorded and applied %d total ChangeEvents across %d referenced documents", len(all_change_events), len(valid_refs))
        return all_change_events

    def _detect_deterministic_changes(
        self,
        source_doc_id: str,
        target_doc_id: str,
        text: str,
        effective_date: Optional[str],
    ) -> List[ChangeEvent]:
        """Detect standard legal amendment formulas (e.g. 'Points 3 to 12 stand deleted')."""
        changes: List[ChangeEvent] = []
        seen_keys = set()

        # 1. Multi-point range deletions (e.g. "Point No. 3 to 12 ... stand deleted", "Points 3-12 are deleted")
        range_del_matches = re.finditer(
            r"Point(?:s| No\.)?\s*(\d+)\s*(?:to|-)\s*(\d+)[\s\S]{0,250}?(?:stands?\s*|are\s*|hereby\s*)?deleted",
            text,
            re.IGNORECASE,
        )
        for m in range_del_matches:
            start_pt = int(m.group(1))
            end_pt = int(m.group(2))
            for pt_num in range(start_pt, end_pt + 1):
                key = (ChangeType.DELETED, f"Point {pt_num}")
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                changes.append(
                    ChangeEvent(
                        source_document_id=source_doc_id,
                        target_document_id=target_doc_id,
                        change_type=ChangeType.DELETED,
                        target_point_or_clause=f"Point {pt_num}",
                        description=f"Point {pt_num} deleted pursuant to range repeal in amendment.",
                        effective_date=effective_date,
                    )
                )
                logger.info(
                    "Deterministic change candidate: DELETED | target_clause='Point %d' | source_doc=%s | target_doc=%s",
                    pt_num, source_doc_id, target_doc_id,
                )

        # 2. Multi-point modifications (e.g. "Point No. 1 and 2 of illustrative list ... are modified as below")
        mod_multi = re.finditer(
            r"(?:Point|Clause|Section)(?:s| No\.)?\s*(\d+[a-zA-Z]?)\s*(?:and|,|&)\s*(\d+[a-zA-Z]?)[\s\S]{0,250}?(?:stands?\s*|is\s*|are\s*)?(?:modified|amended|substituted|revised)",
            text,
            re.IGNORECASE,
        )
        for m in mod_multi:
            for group_idx in (1, 2):
                if m.group(group_idx):
                    target_pt = f"Point {m.group(group_idx)}"
                    key = (ChangeType.MODIFIED, target_pt)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        changes.append(
                            ChangeEvent(
                                source_document_id=source_doc_id,
                                target_document_id=target_doc_id,
                                change_type=ChangeType.MODIFIED,
                                target_point_or_clause=target_pt,
                                description=f"{target_pt} modified / substituted by new circular.",
                                effective_date=effective_date,
                            )
                        )
                        logger.info(
                            "Deterministic change candidate: MODIFIED | target_clause='%s' | source_doc=%s | target_doc=%s",
                            target_pt, source_doc_id, target_doc_id,
                        )

        # 3. Single clause deletions: "Point No. 4 stands deleted"
        single_del_matches = re.finditer(
            r"(?:Clause|Point|Section|Para(?:graph)?)\s*(?:No\.?)?\s*(\d+[a-zA-Z]?|\([a-zA-Z0-9]+\))[\s\S]{0,100}?(?:stands?\s*|is\s*)?deleted",
            text,
            re.IGNORECASE,
        )
        for m in single_del_matches:
            target_pt = f"Point {m.group(1)}"
            key = (ChangeType.DELETED, target_pt)
            if key not in seen_keys:
                seen_keys.add(key)
                changes.append(
                    ChangeEvent(
                        source_document_id=source_doc_id,
                        target_document_id=target_doc_id,
                        change_type=ChangeType.DELETED,
                        target_point_or_clause=target_pt,
                        description=f"{target_pt} stands deleted.",
                        effective_date=effective_date,
                    )
                )
                logger.info(
                    "Deterministic change candidate: DELETED | target_clause='%s' | source_doc=%s | target_doc=%s",
                    target_pt, source_doc_id, target_doc_id,
                )

        # 4. Single clause modifications: "Point No. 1 is amended to permit..."
        single_mod_matches = re.finditer(
            r"(?:Clause|Point|Section)\s*(?:No\.?)?\s*(\d+[a-zA-Z]?)[\s\S]{0,100}?(?:stands?\s*|is\s*|are\s*)?(?:modified|amended|substituted|revised)",
            text,
            re.IGNORECASE,
        )
        for m in single_mod_matches:
            target_pt = f"Point {m.group(1)}"
            key = (ChangeType.MODIFIED, target_pt)
            if key not in seen_keys:
                seen_keys.add(key)
                changes.append(
                    ChangeEvent(
                        source_document_id=source_doc_id,
                        target_document_id=target_doc_id,
                        change_type=ChangeType.MODIFIED,
                        target_point_or_clause=target_pt,
                        description=f"{target_pt} modified / substituted by new circular.",
                        effective_date=effective_date,
                    )
                )
                logger.info(
                    "Deterministic change candidate: MODIFIED | target_clause='%s' | source_doc=%s | target_doc=%s",
                    target_pt, source_doc_id, target_doc_id,
                )

        return changes

    def _apply_single_change(
        self,
        change: ChangeEvent,
        new_facts: List[Fact],
        target_doc_id: str,
        effective_date: Optional[str],
    ) -> None:
        """Persist ChangeEvent node and update affected fact statuses in Neo4j."""
        self.repository.save_change_event(change)

        if change.change_type == ChangeType.DELETED:
            if change.target_point_or_clause:
                self.repository.update_facts_status_by_clause(
                    document_id=target_doc_id,
                    clause_identifier=change.target_point_or_clause,
                    new_status=FactStatus.DELETED,
                    valid_to=effective_date,
                    change_event_id=change.id,
                )
        elif change.change_type in (ChangeType.MODIFIED, ChangeType.SUPERSEDED):
            if change.target_point_or_clause:
                self.repository.update_facts_status_by_clause(
                    document_id=target_doc_id,
                    clause_identifier=change.target_point_or_clause,
                    new_status=FactStatus.MODIFIED if change.change_type == ChangeType.MODIFIED else FactStatus.SUPERSEDED,
                    valid_to=effective_date,
                    change_event_id=change.id,
                )
