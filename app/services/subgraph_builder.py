"""Subgraph Serializer for converting Neo4j execution results into interactive visual graphs."""
from typing import Any, Dict, List, Set, Tuple
from app.core.logging import logger
from app.models.query import GraphSubgraph, QueryType, SubgraphEdge, SubgraphNode


class SubgraphBuilder:
    """Builds clean, deduplicated, styled graph structures for frontend visualization."""

    def build_subgraph(self, records: List[Dict[str, Any]], query_type: QueryType) -> GraphSubgraph:
        """Parse query result records and return a fully styled GraphSubgraph."""
        nodes_dict: Dict[str, SubgraphNode] = {}
        edges_dict: Dict[str, SubgraphEdge] = {}

        if not records:
            return GraphSubgraph(nodes=[], edges=[], total_nodes=0, total_edges=0)

        for rec in records:
            if query_type in (QueryType.CURRENT, QueryType.HISTORICAL):
                self._process_current_historical_record(rec, nodes_dict, edges_dict)
            elif query_type == QueryType.EVOLUTION:
                self._process_evolution_record(rec, nodes_dict, edges_dict)
            elif query_type == QueryType.AMENDMENT:
                self._process_amendment_record(rec, nodes_dict, edges_dict)
            elif query_type == QueryType.MASTER_TO_LATEST:
                self._process_master_record(rec, nodes_dict, edges_dict)
            else:
                self._process_current_historical_record(rec, nodes_dict, edges_dict)

        nodes_list = list(nodes_dict.values())
        edges_list = list(edges_dict.values())

        return GraphSubgraph(
            nodes=nodes_list,
            edges=edges_list,
            total_nodes=len(nodes_list),
            total_edges=len(edges_list),
        )

    def _process_current_historical_record(
        self,
        rec: Dict[str, Any],
        nodes: Dict[str, SubgraphNode],
        edges: Dict[str, SubgraphEdge],
    ) -> None:
        doc_id = rec.get("document_id")
        fact_id = rec.get("fact_id")
        sub_entity = rec.get("subject_entity")
        obj_entity = rec.get("object_entity")

        # Document node
        if doc_id and doc_id not in nodes:
            doc_title = rec.get("doc_title") or ""
            ref_num = rec.get("reference_number") or rec.get("circular_number")
            if not ref_num and doc_title:
                ref_num = (doc_title[:32] + "...") if len(doc_title) > 32 else doc_title
            ref_num = ref_num or doc_id[:8]
            nodes[doc_id] = SubgraphNode(
                id=doc_id,
                label=f"📄 {ref_num}",
                type="Document",
                group="document",
                properties={
                    "title": rec.get("doc_title", ""),
                    "reference_number": rec.get("reference_number", ""),
                    "document_date": rec.get("document_date", ""),
                },
            )

        # Fact node
        if fact_id and fact_id not in nodes:
            status = rec.get("status", "ACTIVE")
            subj = rec.get("subject", "Entity")
            pred = rec.get("predicate", "REQUIRES")
            obj = (rec.get("object") or "")[:40]
            label = f"⚖️ [{status}] {subj} {pred}: {obj}..."
            nodes[fact_id] = SubgraphNode(
                id=fact_id,
                label=label,
                type="Fact",
                status=status,
                group="fact",
                properties={
                    "subject": subj,
                    "predicate": pred,
                    "object": rec.get("object", ""),
                    "status": status,
                    "valid_from": rec.get("valid_from", ""),
                    "valid_to": rec.get("valid_to", ""),
                    "source_text": rec.get("source_text", ""),
                    "clause_path": rec.get("clause_path", ""),
                },
            )

            # Edge Document -> Fact
            if doc_id:
                edge_id = f"{doc_id}->{fact_id}"
                if edge_id not in edges:
                    edges[edge_id] = SubgraphEdge(
                        id=edge_id,
                        source=doc_id,
                        target=fact_id,
                        type="CONTAINS_FACT",
                        properties={"clause_path": rec.get("clause_path", "")},
                    )

        # Subject Entity node
        if sub_entity:
            ent_id = f"ent_{sub_entity}"
            if ent_id not in nodes:
                nodes[ent_id] = SubgraphNode(
                    id=ent_id,
                    label=f"🏛️ {sub_entity}",
                    type="Entity",
                    group="entity",
                    properties={"name": sub_entity},
                )
            if fact_id:
                edge_id = f"{fact_id}->{ent_id}"
                if edge_id not in edges:
                    edges[edge_id] = SubgraphEdge(
                        id=edge_id,
                        source=fact_id,
                        target=ent_id,
                        type="SUBJECT",
                    )

    def _process_evolution_record(
        self,
        rec: Dict[str, Any],
        nodes: Dict[str, SubgraphNode],
        edges: Dict[str, SubgraphEdge],
    ) -> None:
        new_fact_id = rec.get("new_fact_id")
        old_fact_id = rec.get("old_fact_id")
        new_doc_id = rec.get("new_doc_id") or f"doc_{rec.get('new_doc_ref')}"
        old_doc_id = rec.get("old_doc_id") or f"doc_{rec.get('old_doc_ref')}"

        # Documents
        if new_doc_id and new_doc_id not in nodes:
            nodes[new_doc_id] = SubgraphNode(
                id=new_doc_id,
                label=f"📄 {rec.get('new_doc_ref', 'Amending Circular')}",
                type="Document",
                group="document",
                properties={"date": rec.get("new_doc_date")},
            )
        if old_doc_id and old_doc_id not in nodes:
            nodes[old_doc_id] = SubgraphNode(
                id=old_doc_id,
                label=f"📄 {rec.get('old_doc_ref', 'Base Circular')}",
                type="Document",
                group="document",
                properties={"date": rec.get("old_doc_date")},
            )

        # Edge New Doc -> Old Doc (AMENDS)
        if new_doc_id and old_doc_id and new_doc_id != old_doc_id:
            edge_id = f"{new_doc_id}->{old_doc_id}"
            if edge_id not in edges:
                edges[edge_id] = SubgraphEdge(
                    id=edge_id,
                    source=new_doc_id,
                    target=old_doc_id,
                    type="AMENDS",
                    properties={"relationship": "AMENDS"},
                )

        # New Fact Node (Active/Amended)
        if new_fact_id and new_fact_id not in nodes:
            status = rec.get("new_status", "ACTIVE")
            subj = rec.get("new_subject", "Entity")
            pred = rec.get("new_predicate", "REQUIRES")
            obj = (rec.get("new_object") or "")[:40]
            nodes[new_fact_id] = SubgraphNode(
                id=new_fact_id,
                label=f"🟢 [{status}] {subj} {pred}: {obj}...",
                type="Fact",
                status=status,
                group="fact",
                properties={
                    "subject": subj,
                    "predicate": pred,
                    "object": rec.get("new_object", ""),
                    "source_text": rec.get("new_source_text", ""),
                    "valid_from": rec.get("new_valid_from", ""),
                    "status": status,
                },
            )

        # Old Fact Node (Superseded/Deleted/Modified)
        if old_fact_id and old_fact_id not in nodes:
            status = rec.get("old_status", "MODIFIED")
            subj = rec.get("old_subject", "Entity")
            pred = rec.get("old_predicate", "REQUIRES")
            obj = (rec.get("old_object") or "")[:40]
            nodes[old_fact_id] = SubgraphNode(
                id=old_fact_id,
                label=f"🔴 [{status}] {subj} {pred}: {obj}...",
                type="Fact",
                status=status,
                group="fact",
                properties={
                    "subject": subj,
                    "predicate": pred,
                    "object": rec.get("old_object", ""),
                    "source_text": rec.get("old_source_text", ""),
                    "valid_to": rec.get("old_valid_to", ""),
                    "status": status,
                },
            )

        # Fact Evolution Edge: (New Fact) -> (Old Fact)
        if new_fact_id and old_fact_id:
            evo_types = rec.get("evolution_types") or ["MODIFIES"]
            primary_rel = evo_types[0] if isinstance(evo_types, list) and evo_types else "MODIFIES"
            edge_id = f"{new_fact_id}->{old_fact_id}"
            if edge_id not in edges:
                edges[edge_id] = SubgraphEdge(
                    id=edge_id,
                    source=new_fact_id,
                    target=old_fact_id,
                    type=primary_rel,
                    properties={"evolution_chain": evo_types},
                )

    def _process_amendment_record(
        self,
        rec: Dict[str, Any],
        nodes: Dict[str, SubgraphNode],
        edges: Dict[str, SubgraphEdge],
    ) -> None:
        src_id = rec.get("source_doc_id")
        tgt_id = rec.get("target_doc_id")
        change_id = rec.get("change_id")
        fact_id = rec.get("affected_fact_id")

        if src_id and src_id not in nodes:
            nodes[src_id] = SubgraphNode(
                id=src_id,
                label=f"📄 {rec.get('source_ref_num', src_id[:8])}",
                type="Document",
                group="document",
                properties={"date": rec.get("source_date")},
            )

        if tgt_id and tgt_id not in nodes:
            nodes[tgt_id] = SubgraphNode(
                id=tgt_id,
                label=f"📄 {rec.get('target_ref_num', tgt_id[:8])}",
                type="Document",
                group="document",
                properties={"date": rec.get("target_date")},
            )

        if src_id and tgt_id and src_id != tgt_id:
            edge_id = f"{src_id}->{tgt_id}"
            if edge_id not in edges:
                edges[edge_id] = SubgraphEdge(
                    id=edge_id,
                    source=src_id,
                    target=tgt_id,
                    type="AMENDS",
                )

        if change_id and change_id not in nodes:
            chg_type = rec.get("change_type", "MODIFIED")
            clause = rec.get("target_clause", "")
            nodes[change_id] = SubgraphNode(
                id=change_id,
                label=f"⚡ {chg_type}: {clause}",
                type="ChangeEvent",
                group="change",
                status=chg_type,
                properties={
                    "change_type": chg_type,
                    "description": rec.get("description", ""),
                    "effective_date": rec.get("effective_date", ""),
                },
            )
            if src_id:
                edges[f"{src_id}->{change_id}"] = SubgraphEdge(
                    id=f"{src_id}->{change_id}",
                    source=src_id,
                    target=change_id,
                    type="RECORDED_IN",
                )

        if fact_id and fact_id not in nodes:
            status = rec.get("affected_status", "DELETED")
            nodes[fact_id] = SubgraphNode(
                id=fact_id,
                label=f"🔴 [{status}] {rec.get('affected_subject')}: {(rec.get('affected_object') or '')[:30]}...",
                type="Fact",
                status=status,
                group="fact",
                properties={"source_text": rec.get("affected_source_text")},
            )
            if change_id:
                edges[f"{change_id}->{fact_id}"] = SubgraphEdge(
                    id=f"{change_id}->{fact_id}",
                    source=change_id,
                    target=fact_id,
                    type="TARGETS",
                )

    def _process_master_record(
        self,
        rec: Dict[str, Any],
        nodes: Dict[str, SubgraphNode],
        edges: Dict[str, SubgraphEdge],
    ) -> None:
        root_id = rec.get("root_doc_id")
        latest_id = rec.get("latest_doc_id")
        active_fact_id = rec.get("active_fact_id")
        root_fact_id = rec.get("root_fact_id")

        if root_id and root_id not in nodes:
            nodes[root_id] = SubgraphNode(
                id=root_id,
                label=f"📜 Master: {rec.get('root_ref_num', root_id[:8])}",
                type="Document",
                group="document",
                properties={"title": rec.get("root_title"), "date": rec.get("root_date")},
            )

        if latest_id and latest_id not in nodes:
            nodes[latest_id] = SubgraphNode(
                id=latest_id,
                label=f"📄 Latest: {rec.get('latest_ref_num', latest_id[:8])}",
                type="Document",
                group="document",
                properties={"date": rec.get("latest_date")},
            )

        if latest_id and root_id and latest_id != root_id:
            edge_id = f"{latest_id}->{root_id}"
            if edge_id not in edges:
                edges[edge_id] = SubgraphEdge(
                    id=edge_id,
                    source=latest_id,
                    target=root_id,
                    type="AMENDS",
                )

        if active_fact_id and active_fact_id not in nodes:
            nodes[active_fact_id] = SubgraphNode(
                id=active_fact_id,
                label=f"🟢 [ACTIVE] {rec.get('active_subject')}: {(rec.get('active_object') or '')[:30]}...",
                type="Fact",
                status="ACTIVE",
                group="fact",
                properties={"source_text": rec.get("active_source_text")},
            )
            if latest_id:
                edges[f"{latest_id}->{active_fact_id}"] = SubgraphEdge(
                    id=f"{latest_id}->{active_fact_id}",
                    source=latest_id,
                    target=active_fact_id,
                    type="CONTAINS_FACT",
                )

        if root_fact_id and root_fact_id not in nodes:
            status = rec.get("root_status", "MODIFIED")
            nodes[root_fact_id] = SubgraphNode(
                id=root_fact_id,
                label=f"⚖️ [{status}] {rec.get('root_subject')}: {(rec.get('root_object') or '')[:30]}...",
                type="Fact",
                status=status,
                group="fact",
                properties={"source_text": rec.get("root_source_text")},
            )
            if root_id:
                edges[f"{root_id}->{root_fact_id}"] = SubgraphEdge(
                    id=f"{root_id}->{root_fact_id}",
                    source=root_id,
                    target=root_fact_id,
                    type="CONTAINS_FACT",
                )
