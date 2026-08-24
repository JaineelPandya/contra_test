"""Abstract Base Class for LLM providers."""
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional
from app.models.chunk import Chunk
from app.models.document import DocumentMetadata
from app.models.fact import Fact
from app.models.entity import Entity
from app.models.reference import DocumentReference
from app.models.change import ChangeEvent


class BaseLLMProvider(ABC):
    """Abstract interface that all LLM providers (Gemini, Ollama, OpenAI, Mock) must implement."""

    @abstractmethod
    async def extract_metadata(self, header_text: str) -> Optional[DocumentMetadata]:
        """Extract structured document metadata from header text using LLM."""
        pass

    @abstractmethod
    async def extract_facts_batch(
        self,
        document_id: str,
        chunks: List[Chunk],
        metadata: Optional[DocumentMetadata] = None,
        on_dead_letter: Optional[Callable[[int, str, Dict[str, Any]], None]] = None,
    ) -> List[Fact]:
        """Extract atomic regulatory facts from a batch of chunks in a single call."""
        pass

    @abstractmethod
    async def classify_relationships_batch(
        self,
        candidate_pairs: List[Dict[str, Any]],
        on_dead_letter: Optional[Callable[[int, str, Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Classify a batch of candidate relationship pairs using controlled vocabulary."""
        pass

    @abstractmethod
    async def detect_references(
        self,
        source_document_id: str,
        text: str,
    ) -> List[DocumentReference]:
        """Detect cross-document references and citations from text."""
        pass
