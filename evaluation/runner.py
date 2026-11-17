"""Evaluation runner that invokes the production fact extraction pipeline.

Reuses the production FactExtractor and configured LLM provider without duplication.
Supports caching, resume, and evaluating pre-extracted predictions.
"""
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4
from datetime import datetime, timezone

from app.core.config import get_settings
from app.core.logging import logger
from app.llm.base import BaseLLMProvider
from app.llm.factory import get_llm_provider
from app.models.chunk import Chunk
from app.models.document import DocumentMetadata
from app.models.fact import Fact, FactPredicate
from app.prompts.fact_extraction import BATCH_FACT_EXTRACTION_SYSTEM_PROMPT
from app.services.chunker import HierarchicalChunker
from app.services.fact_extractor import FactExtractor
from app.services.hierarchy_builder import HierarchyBuilder
from app.services.pdf_parser import PDFParser


def compute_prompt_hash() -> str:
    """Compute deterministic SHA-256 hash of current production extraction prompt."""
    return hashlib.sha256(BATCH_FACT_EXTRACTION_SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:16]


class EvaluationRunner:
    """Runs extraction against chunks using production extractor, or loads cached predictions."""

    def __init__(
        self,
        llm_provider: Optional[BaseLLMProvider] = None,
        results_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.settings = get_settings()
        self.llm_provider = llm_provider or get_llm_provider()
        self.extractor = FactExtractor(self.llm_provider)
        self.results_dir = Path(results_dir or "evaluation/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def load_cached_predictions(cls, predictions_path: Union[str, Path]) -> List[Fact]:
        """Load raw predictions from a JSONL or JSON file into Fact instances."""
        path = Path(predictions_path)
        if not path.exists():
            raise FileNotFoundError(f"Predictions file not found at {path}")

        facts: List[Fact] = []
        ext = path.suffix.lower()

        if ext == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    facts.append(cls._dict_to_fact(data))
        elif ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        facts.append(cls._dict_to_fact(item))
                elif isinstance(data, dict) and "facts" in data:
                    for item in data["facts"]:
                        facts.append(cls._dict_to_fact(item))
        else:
            raise ValueError(f"Unsupported predictions format: {ext}. Use .jsonl or .json")

        logger.info("Loaded %d cached predictions from %s", len(facts), path)
        return facts

    @classmethod
    def _dict_to_fact(cls, d: Dict[str, Any]) -> Fact:
        """Robustly parse dict into Fact model, handling conditions/exceptions formatting."""
        # Normalize predicate
        raw_pred = str(d.get("predicate", "REQUIRES")).upper()
        try:
            pred = FactPredicate(raw_pred)
        except ValueError:
            # Map known variants
            mapping = {
                "PERMITTED": FactPredicate.PERMITS,
                "EXCLUDES": FactPredicate.EXCEPTION_TO,
                "EXEMPTS": FactPredicate.EXCEPTION_TO,
                "RESTRICTS": FactPredicate.PROHIBITS,
            }
            pred = mapping.get(raw_pred, FactPredicate.REQUIRES)

        # Parse conditions
        raw_conds = d.get("conditions")
        if raw_conds is None:
            raw_conds = d.get("condition")
        if isinstance(raw_conds, str):
            conds = [c.strip() for c in raw_conds.split(";") if c.strip()]
        elif isinstance(raw_conds, list):
            conds = [str(c).strip() for c in raw_conds if str(c).strip()]
        else:
            conds = []

        # Parse exceptions
        raw_excs = d.get("exceptions")
        if raw_excs is None:
            raw_excs = d.get("exception")
        if isinstance(raw_excs, str):
            excs = [e.strip() for e in raw_excs.split(";") if e.strip()]
        elif isinstance(raw_excs, list):
            excs = [str(e).strip() for e in raw_excs if str(e).strip()]
        else:
            excs = []

        return Fact(
            fact_id=d.get("fact_id") or str(uuid4()),
            document_id=d.get("document_id") or "eval_doc",
            chunk_id=d.get("chunk_id") or "eval_chunk",
            hierarchy_node_id=d.get("hierarchy_node_id"),
            page_number=d.get("page_number") or d.get("pdf_page"),
            subject=str(d.get("subject") or "Unknown Subject"),
            predicate=pred,
            object=str(d.get("object") or d.get("object_or_action") or "Unknown Object"),
            rule_reference=d.get("rule_reference"),
            clause_reference=d.get("clause_reference") or d.get("section_ref"),
            conditions=conds,
            exceptions=excs,
            threshold=d.get("threshold"),
            time=d.get("time"),
            source_circular=d.get("source_circular"),
            regulatory_status=d.get("regulatory_status", "ACTIVE"),
            effective_date=d.get("effective_date"),
            confidence=float(d.get("confidence", 1.0)),
            source_text=str(d.get("source_text") or d.get("evidence") or ""),
            extraction_method=d.get("extraction_method", "eval_extractor"),
            model=d.get("model") or get_settings().GROQ_MODEL,
        )

    def extract_from_chunks(
        self,
        document_id: str,
        chunks: List[Chunk],
        metadata: Optional[DocumentMetadata] = None,
        run_id: Optional[str] = None,
    ) -> Tuple[List[Fact], Path]:
        """Synchronously execute production extraction over chunks and persist predictions."""
        return asyncio.run(self.extract_from_chunks_async(document_id, chunks, metadata, run_id))

    async def extract_from_chunks_async(
        self,
        document_id: str,
        chunks: List[Chunk],
        metadata: Optional[DocumentMetadata] = None,
        run_id: Optional[str] = None,
    ) -> Tuple[List[Fact], Path]:
        """Asynchronously extract facts using production FactExtractor and save raw predictions."""
        active_run_id = run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
        logger.info("Executing evaluation extraction for doc=%s, chunks=%d (run_id=%s)", document_id, len(chunks), active_run_id)

        raw_facts = await self.extractor.extract_all_facts(
            document_id=document_id,
            chunks=chunks,
            metadata=metadata,
        )

        # Save raw predictions
        pred_path = self.results_dir / f"raw_predictions_{active_run_id}.jsonl"
        with open(pred_path, "w", encoding="utf-8") as f:
            for fact in raw_facts:
                f.write(fact.model_dump_json() + "\n")

        # Save run manifest metadata
        meta_path = self.results_dir / f"run_manifest_{active_run_id}.json"
        manifest = {
            "run_id": active_run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "document_id": document_id,
            "chunk_count": len(chunks),
            "prediction_count": len(raw_facts),
            "provider": self.settings.LLM_PROVIDER,
            "model": getattr(self.settings, f"{self.settings.LLM_PROVIDER.upper()}_MODEL", "unknown"),
            "prompt_hash": compute_prompt_hash(),
            "predictions_file": str(pred_path),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info("Saved %d predictions to %s", len(raw_facts), pred_path)
        return raw_facts, pred_path

    def extract_from_pdf(
        self,
        pdf_path: Union[str, Path],
        document_id: Optional[str] = None,
        chunk_limit: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> Tuple[List[Fact], Path]:
        """Extract facts directly from a PDF file using standard parser and chunker."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"Circular PDF not found at {path}")

        doc_id = document_id or str(uuid4())
        parser = PDFParser()
        pages, total_pages = parser.parse_pdf(path)

        hierarchy_builder = HierarchyBuilder()
        nodes = hierarchy_builder.build_hierarchy(doc_id, pages)

        chunker = HierarchicalChunker()
        chunks = chunker.create_chunks(doc_id, nodes)

        if chunk_limit and chunk_limit > 0:
            chunks = chunks[:chunk_limit]

        meta = DocumentMetadata(
            title=path.stem,
            total_pages=total_pages,
            document_date="2026-05-15",
        )

        return self.extract_from_chunks(doc_id, chunks, meta, run_id)
