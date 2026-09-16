"""Regulatory Answer Synthesizer for generating legal summaries, changelogs, and citations."""
import re
from typing import Any, Dict, List, Optional, Tuple
from app.core.logging import logger
from app.llm.base import BaseLLMProvider
from app.models.fact import FactStatus
from app.models.query import (
    CurrentRegulationSummary,
    QueryType,
    RegulatoryChangeRecord,
    RegulatoryQueryPlan,
    RegulatorySourceCitation,
)


class RegulatoryAnswerSynthesizer:
    """Synthesizes structured regulatory answers, change histories, and citations from graph results."""

    def __init__(self, llm_provider: Optional[BaseLLMProvider] = None) -> None:
        self.llm_provider = llm_provider

    async def synthesize_response(
        self,
        plan: RegulatoryQueryPlan,
        records: List[Dict[str, Any]],
    ) -> Tuple[Optional[CurrentRegulationSummary], List[RegulatoryChangeRecord], str, List[RegulatorySourceCitation]]:
        """Extract structured sections and generate synthesized markdown answer."""
        if not records:
            empty_msg = f"No regulatory facts or changes found matching query: '{plan.raw_query}' (Type: {plan.query_type.value})."
            return None, [], empty_msg, []

        citations = self._extract_citations(records, plan.query_type)
        change_history = self._extract_change_history(records, plan.query_type)
        current_summary = self._extract_current_summary(records, plan.query_type, plan)
        
        # Build synthesis markdown
        llm_answer = self._generate_structured_answer(plan, current_summary, change_history, citations)
        
        return current_summary, change_history, llm_answer, citations

    def _extract_citations(self, records: List[Dict[str, Any]], query_type: QueryType) -> List[RegulatorySourceCitation]:
        citations: List[RegulatorySourceCitation] = []
        seen_texts = set()

        for r in records:
            source_text = r.get("source_text") or r.get("new_source_text") or r.get("old_source_text") or r.get("active_source_text")
            if source_text and source_text not in seen_texts:
                seen_texts.add(source_text)
                citations.append(
                    RegulatorySourceCitation(
                        document_id=r.get("document_id") or r.get("new_doc_id") or r.get("source_doc_id") or r.get("root_doc_id"),
                        circular_number=r.get("circular_number") or r.get("new_doc_ref") or r.get("source_ref_num") or r.get("root_ref_num"),
                        reference_number=r.get("reference_number") or r.get("new_doc_ref") or r.get("source_ref_num") or r.get("root_ref_num"),
                        document_date=r.get("document_date") or r.get("new_doc_date") or r.get("source_date") or r.get("root_date"),
                        clause_path=r.get("clause_path") or r.get("new_clause_path") or r.get("target_clause"),
                        source_text=source_text,
                        status=r.get("status") or r.get("new_status") or "ACTIVE",
                        confidence=float(r.get("confidence", 1.0)),
                    )
                )
        return citations

    def _extract_change_history(self, records: List[Dict[str, Any]], query_type: QueryType) -> List[RegulatoryChangeRecord]:
        history: List[RegulatoryChangeRecord] = []
        seen_changes = set()

        for r in records:
            if r.get("old_fact_id"):
                evo_types = r.get("evolution_types") or ["MODIFIES"]
                rel_type = evo_types[0] if evo_types else "MODIFIES"
                old_txt = r.get("old_object") or r.get("old_source_text") or ""
                new_txt = r.get("new_object") or r.get("new_source_text") or ""
                clause_ref = r.get("old_clause_ref") or r.get("new_clause_ref") or r.get("old_rule_ref") or r.get("new_rule_ref")
                key = f"{r.get('new_fact_id')}->{r.get('old_fact_id')}"
                if key not in seen_changes:
                    seen_changes.add(key)
                    history.append(
                        RegulatoryChangeRecord(
                            date=r.get("new_doc_date") or r.get("old_valid_to"),
                            change_type=rel_type,
                            amending_circular=r.get("new_doc_ref"),
                            target_point_or_clause=clause_ref,
                            description=f"{r.get('new_subject', 'Rule')} {rel_type.lower()} previous rule ({r.get('old_subject', '')}). Old: \"{old_txt[:100]}\" → New: \"{new_txt[:100]}\"",
                            source_fact_id=r.get("new_fact_id"),
                            target_fact_id=r.get("old_fact_id"),
                            effective_date=r.get("new_valid_from"),
                        )
                    )

            if r.get("change_id"):
                cid = r.get("change_id")
                if cid not in seen_changes:
                    seen_changes.add(cid)
                    # Build a rich description including source → target circular context
                    src_title = r.get("source_doc_title") or r.get("source_ref_num") or "Amending Circular"
                    tgt_title = r.get("target_doc_title") or r.get("target_ref_num") or "Affected Circular"
                    base_desc = r.get("description") or "Regulatory change detected"
                    affected_txt = r.get("affected_source_text") or ""
                    affected_status = r.get("affected_status") or ""
                    clause = r.get("target_clause") or r.get("affected_clause_ref") or ""
                    status_badge = f" [{affected_status}]" if affected_status else ""
                    rich_desc = (
                        f"{base_desc}{status_badge}\n"
                        f"  → Amending: **{src_title}**  |  Affected: **{tgt_title}**"
                    )
                    if affected_txt:
                        rich_desc += f"\n  → Original provision: \"{affected_txt[:160]}\""
                    history.append(
                        RegulatoryChangeRecord(
                            date=r.get("effective_date") or r.get("source_date"),
                            change_type=r.get("change_type", "MODIFIED"),
                            amending_circular=r.get("source_ref_num") or src_title,
                            target_point_or_clause=clause,
                            description=rich_desc,
                            target_fact_id=r.get("affected_fact_id"),
                            effective_date=r.get("effective_date"),
                        )
                    )

        return sorted(history, key=lambda x: x.date or "", reverse=True)

    def _extract_current_summary(
        self,
        records: List[Dict[str, Any]],
        query_type: QueryType,
        plan: Optional[RegulatoryQueryPlan] = None,
    ) -> Optional[CurrentRegulationSummary]:
        active_records = [
            r for r in records
            if r.get("status") == "ACTIVE" or r.get("new_status") == "ACTIVE" or r.get("active_status") == "ACTIVE"
        ]

        if not active_records:
            return None

        # Dynamic query-relevance ranking
        query_terms = set()
        if plan:
            for kw in plan.keywords:
                query_terms.update(re.findall(r"\b\w{3,}\b", kw.lower()))
            for ent in plan.entities:
                query_terms.update(re.findall(r"\b\w{3,}\b", ent.lower()))

        def score_record(rec: Dict[str, Any]) -> int:
            score = 0
            text = (rec.get("source_text") or rec.get("new_source_text") or rec.get("object") or "").lower()
            subj = (rec.get("subject") or rec.get("new_subject") or "").lower()
            date = str(rec.get("document_date") or rec.get("new_doc_date") or "")

            # Keyword / Topic relevance match
            if query_terms:
                matched_terms = sum(1 for term in query_terms if term in text or term in subj)
                score += matched_terms * 40

            # Substantive content length & validity
            if len(text) > 30:
                score += 25

            # Recency boost
            if "2026" in date or "2025" in date:
                score += 20
            elif "2024" in date or "2023" in date:
                score += 15

            # Confidence boost
            conf = float(rec.get("confidence", 1.0))
            score += int(conf * 10)

            # Penalize non-substantive header boilerplate
            if re.search(r"^(?:date:|department:|download ref|circular ref|national stock|page \d+|cin)", text.strip(), re.IGNORECASE):
                score -= 100

            return score

        ranked_records = sorted(active_records, key=score_record, reverse=True)
        primary = ranked_records[0]

        subject = primary.get("subject") or primary.get("new_subject") or primary.get("active_subject") or "Regulated Entity"
        predicate = primary.get("predicate") or primary.get("new_predicate") or primary.get("active_predicate") or "GOVERNS"
        obj = primary.get("object") or primary.get("new_object") or primary.get("active_object") or ""
        source_snippet = primary.get("source_text") or primary.get("new_source_text") or obj
        circular = primary.get("reference_number") or primary.get("new_doc_ref") or primary.get("latest_ref_num") or primary.get("circular_number")
        date = primary.get("document_date") or primary.get("new_doc_date") or primary.get("latest_date")

        # Dynamic, generalized summary text
        clean_obj = obj.strip() if obj else ""
        if clean_obj and len(clean_obj) > 10:
            summary_text = f"{subject} is subject to provision ({predicate}): {clean_obj}"
        elif source_snippet:
            clean_snippet = re.sub(r"\s+", " ", source_snippet).strip()
            summary_text = f"{subject} ({predicate}): {clean_snippet[:220]}"
        else:
            summary_text = f"{subject} is governed by active {predicate} requirement."

        entities = list({primary.get("subject_entity"), primary.get("object_entity")} - {None})
        if not entities:
            entities = [subject, "Regulated Intermediary / Market Participant"]

        return CurrentRegulationSummary(
            summary=summary_text,
            status=FactStatus.ACTIVE,
            active_facts_count=len(active_records),
            effective_date=date,
            primary_circular=circular,
            entities_affected=entities,
        )

    def _generate_structured_answer(
        self,
        plan: RegulatoryQueryPlan,
        current: Optional[CurrentRegulationSummary],
        history: List[RegulatoryChangeRecord],
        citations: List[RegulatorySourceCitation],
    ) -> str:
        parts: List[str] = []

        parts.append(f"### Regulatory Analysis: *\"{plan.raw_query}\"*")
        parts.append(f"**Query Classification**: `{plan.query_type.value}` | **Reasoning**: {plan.reasoning}\n")

        # ── CHANGELOG archetype: show full cross-circular change table ──────────
        if plan.query_type == QueryType.CHANGELOG:
            if history:
                # Group by change_type for a clear breakdown
                modified = [h for h in history if h.change_type in ("MODIFIED", "MODIFIES")]
                deleted  = [h for h in history if h.change_type in ("DELETED", "DELETES")]
                superseded = [h for h in history if h.change_type in ("SUPERSEDED", "SUPERSEDES")]
                other    = [h for h in history if h.change_type not in (
                    "MODIFIED", "MODIFIES", "DELETED", "DELETES", "SUPERSEDED", "SUPERSEDES"
                )]

                parts.append(f"#### 📋 Cross-Circular Changelog ({len(history)} total change events)\n")
                parts.append(f"| # | Date | Change Type | Clause | Amending Circular | Description |")
                parts.append(f"|---|------|-------------|--------|-------------------|-------------|")
                for idx, h in enumerate(history[:30], 1):
                    date_str = h.date or "N/A"
                    ctype = h.change_type
                    clause = h.target_point_or_clause or "—"
                    circ = h.amending_circular or "—"
                    # Trim description for table — strip newlines
                    desc_first_line = (h.description or "").split("\n")[0][:100]
                    parts.append(f"| {idx} | {date_str} | `{ctype}` | {clause} | {circ} | {desc_first_line} |")

                parts.append("")
                parts.append(f"**Summary**: {len(modified)} modification(s), {len(deleted)} deletion(s), "
                             f"{len(superseded)} supersession(s), {len(other)} other change(s).\n")

                # Latest change highlight
                latest = history[0]
                parts.append("#### ✅ Most Recent Change")
                parts.append(f"- **Date**: {latest.date or 'N/A'}")
                parts.append(f"- **Type**: `{latest.change_type}`")
                parts.append(f"- **Clause**: {latest.target_point_or_clause or 'N/A'}")
                parts.append(f"- **Amending Circular**: {latest.amending_circular or 'N/A'}")
                parts.append(f"- **Detail**: {(latest.description or '').split(chr(10))[0]}\n")
            else:
                parts.append("#### 📋 Cross-Circular Changelog")
                parts.append("- No change events found in the knowledge graph for this query scope.\n")

        else:
            # ── All other archetypes: standard 3-section layout ─────────────────

            # 1. Current State
            if current:
                parts.append("#### 1. Current Active Regulation (Governing State)")
                parts.append(f"- **Active Legal Rule**: {current.summary}")
                if current.primary_circular:
                    parts.append(f"- **Governing Circular**: `{current.primary_circular}` (Effective Date: {current.effective_date or 'N/A'})")
                if current.entities_affected:
                    parts.append(f"- **Applicable Entities**: {', '.join(current.entities_affected)}")
                parts.append(f"- **Active Graph Provisions**: {current.active_facts_count} verified provision(s)\n")
            else:
                parts.append("#### 1. Current Active Regulation")
                parts.append("- No active `CURRENT` rules found matching this specific query scope.\n")

            # 2. Chronological Changes & Core Shift
            parts.append("#### 2. Key Regulatory Shifts & Chronological Evolution")
            if history:
                parts.append(f"**Recorded Regulatory Transitions ({len(history)} change events)**:\n")
                for idx, h in enumerate(history[:10], 1):
                    date_info = f" *({h.date})*" if h.date else ""
                    circ_info = f" [`{h.amending_circular}`]" if h.amending_circular else ""
                    clause_info = f" in *{h.target_point_or_clause}*" if h.target_point_or_clause else ""
                    parts.append(f"{idx}. **[{h.change_type}]**{date_info}{circ_info}{clause_info}: {h.description}")
                parts.append("")
            elif current:
                circ_label = f"`{current.primary_circular}`" if current.primary_circular else "the governing Circular"
                parts.append(
                    f"- **Current Active Baseline**: Provisions from {circ_label} are active with "
                    f"{current.active_facts_count} verified graph provision(s). No subsequent amending repeal or modification "
                    f"has superseded these provisions in the knowledge graph.\n"
                )
            else:
                parts.append("- No amendment transitions recorded for this query scope.\n")

        # 3. Provenance & Citations (all archetypes)
        if citations:
            parts.append("#### 3. Verbatim Regulatory Citations & Provenance")
            for c in citations[:6]:
                circ_badge = f"`{c.reference_number}`" if c.reference_number else "`Regulatory Notice`"
                date_str = f" ({c.document_date})" if c.document_date else ""
                clause_str = f" > *{c.clause_path}*" if c.clause_path else ""
                parts.append(f"- {circ_badge}{date_str}{clause_str}: *\"{c.source_text}\"* (Status: `{c.status}`)")
            parts.append("")

        return "\n".join(parts)

