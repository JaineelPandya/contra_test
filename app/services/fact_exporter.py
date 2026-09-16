"""Fact and Regulatory Comparison Exporter (JSON and CSV)."""
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from app.core.logging import logger
from app.graph.repository import GraphRepository


FACT_EXPORT_COLUMNS = [
    "fact_id",
    "document_id",
    "document_date",
    "subject",
    "predicate",
    "object",
    "condition",
    "exception",
    "threshold",
    "time",
    "rule_reference",
    "clause_reference",
    "source_circular",
    "regulatory_status",
    "confidence",
    "source_text",
]

COMPARISON_EXPORT_COLUMNS = [
    "change_type",
    "action",
    "old_relation",
    "new_relation",
    "source_circular",
    "reason",
    "confidence",
    "time",
    "threshold",
]


class FactExporter:
    """Exports structured facts and comparison tables to JSON and CSV formats."""

    def __init__(self, repository: GraphRepository) -> None:
        self.repository = repository

    def export_document_facts(
        self,
        document_id: str,
        output_dir: Path,
        prefix: str = "facts",
    ) -> Tuple[Path, Path]:
        """Export all facts for a document to JSON and CSV files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_facts = self.repository.get_document_facts(document_id)

        # Retrieve document date
        doc_query = "MATCH (d:Document {document_id: $doc_id}) RETURN d.document_date AS doc_date, d.title AS title"
        docs = self.repository.client.execute_query(doc_query, {"doc_id": document_id})
        doc_date = docs[0].get("doc_date") if docs else None

        flattened_facts: List[Dict[str, Any]] = []
        for f in raw_facts:
            # Handle condition / exception list formatting
            conds = f.get("conditions") or []
            cond_str = "; ".join(conds) if isinstance(conds, list) else str(conds or "")

            excs = f.get("exceptions") or []
            exc_str = "; ".join(excs) if isinstance(excs, list) else str(excs or "")

            obj_val = f.get("object") or f.get("object_entity") or ""

            item = {
                "fact_id": f.get("fact_id", ""),
                "document_id": document_id,
                "document_date": doc_date,
                "subject": f.get("subject", ""),
                "predicate": f.get("predicate", ""),
                "object": obj_val,
                "condition": cond_str if cond_str else None,
                "exception": exc_str if exc_str else None,
                "threshold": f.get("threshold"),
                "time": f.get("time"),
                "rule_reference": f.get("rule_reference"),
                "clause_reference": f.get("clause_reference"),
                "source_circular": f.get("source_circular"),
                "regulatory_status": f.get("regulatory_status", "ACTIVE"),
                "confidence": f.get("confidence", 1.0),
                "source_text": f.get("source_text", ""),
            }
            flattened_facts.append(item)

        json_path = output_dir / f"{prefix}_{document_id[:8]}.json"
        csv_path = output_dir / f"{prefix}_{document_id[:8]}.csv"

        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(flattened_facts, jf, indent=2)

        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=FACT_EXPORT_COLUMNS)
            writer.writeheader()
            for row in flattened_facts:
                writer.writerow(row)

        logger.info(
            "Exported %d facts for document %s to %s and %s",
            len(flattened_facts),
            document_id,
            json_path,
            csv_path,
        )
        return json_path, csv_path

    def export_comparison(
        self,
        comparison_results: List[Dict[str, Any]],
        output_dir: Path,
        filename_stem: str = "comparison_2024_2026",
    ) -> Tuple[Path, Path]:
        """Export semantic comparison results to JSON and CSV files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        flattened: List[Dict[str, Any]] = []
        for r in comparison_results:
            item = {
                "change_type": r.get("change_type", ""),
                "action": r.get("action", ""),
                "old_relation": r.get("old_relation") or "NONE",
                "new_relation": r.get("new_relation") or "NONE",
                "source_circular": r.get("source_circular") or "NONE",
                "reason": r.get("reason", ""),
                "confidence": r.get("confidence", 1.0),
                "time": r.get("time") or "NONE",
                "threshold": r.get("threshold") or "NONE",
            }
            flattened.append(item)

        json_path = output_dir / f"{filename_stem}.json"
        csv_path = output_dir / f"{filename_stem}.csv"

        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(comparison_results, jf, indent=2)

        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=COMPARISON_EXPORT_COLUMNS)
            writer.writeheader()
            for row in flattened:
                writer.writerow(row)

        logger.info(
            "Exported %d comparison records to %s and %s",
            len(flattened),
            json_path,
            csv_path,
        )
        return json_path, csv_path
