"""Hierarchical chunking service with path provenance and token management."""
from typing import Dict, List, Optional
from app.core.config import get_settings
from app.core.logging import logger
from app.models.chunk import Chunk
from app.models.hierarchy import HierarchyNode


def approximate_token_count(text: str) -> int:
    """Approximate token count based on whitespace and word splitting (approx 1.3 tokens per word)."""
    words = text.split()
    return int(len(words) * 1.3)


class HierarchicalChunker:
    """Creates sequential atomic chunks mapped to the document hierarchy tree."""

    def __init__(self, max_tokens: Optional[int] = None, overlap_tokens: Optional[int] = None) -> None:
        settings = get_settings()
        self.max_tokens = max_tokens or settings.MAX_CHUNK_TOKENS
        self.overlap_tokens = overlap_tokens or settings.CHUNK_OVERLAP_TOKENS

    def create_chunks(self, document_id: str, nodes: List[HierarchyNode]) -> List[Chunk]:
        """Convert hierarchical nodes into sequential atomic chunks with full heading paths."""
        logger.info("Starting hierarchical chunking for document %s across %d nodes", document_id, len(nodes))
        
        # Build node id to node map for reconstructing heading paths
        node_map: Dict[str, HierarchyNode] = {node.id: node for node in nodes}

        raw_chunks: List[Chunk] = []
        order = 1

        for node in nodes:
            heading_path = self._build_heading_path(node, node_map)
            content = node.content.strip()
            if not content:
                continue

            estimated_tokens = approximate_token_count(content)

            if estimated_tokens <= self.max_tokens:
                # Node fits entirely into a single atomic chunk
                chunk = Chunk(
                    document_id=document_id,
                    content=content,
                    heading_path=heading_path,
                    parent_id=node.id,
                    page_start=node.page_start,
                    page_end=node.page_end,
                    order=order,
                    token_count=estimated_tokens,
                )
                raw_chunks.append(chunk)
                order += 1
            else:
                # Large section: split into smaller token-bounded sub-chunks while keeping heading path
                sub_chunks = self._split_large_content(
                    document_id=document_id,
                    parent_node=node,
                    content=content,
                    heading_path=heading_path,
                    start_order=order,
                )
                raw_chunks.extend(sub_chunks)
                order += len(sub_chunks)

        # Wire up previous_chunk_id and next_chunk_id pointers
        for i in range(len(raw_chunks)):
            if i > 0:
                raw_chunks[i].previous_chunk_id = raw_chunks[i - 1].chunk_id
            if i < len(raw_chunks) - 1:
                raw_chunks[i].next_chunk_id = raw_chunks[i + 1].chunk_id

        logger.info("Generated %d sequential hierarchical chunks for document %s", len(raw_chunks), document_id)
        return raw_chunks

    def _build_heading_path(self, node: HierarchyNode, node_map: Dict[str, HierarchyNode]) -> List[str]:
        """Traverse upwards to root to build full hierarchy path array."""
        path = [node.title]
        curr = node
        while curr.parent_id and curr.parent_id in node_map:
            curr = node_map[curr.parent_id]
            path.insert(0, curr.title)
        return path

    def _split_large_content(
        self,
        document_id: str,
        parent_node: HierarchyNode,
        content: str,
        heading_path: List[str],
        start_order: int,
    ) -> List[Chunk]:
        """Split oversized text into paragraph-aware chunks."""
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        chunks: List[Chunk] = []
        current_paragraphs: List[str] = []
        current_tokens = 0
        order = start_order

        for p in paragraphs:
            p_tokens = approximate_token_count(p)
            if current_tokens + p_tokens > self.max_tokens and current_paragraphs:
                chunk_text = "\n\n".join(current_paragraphs)
                chunks.append(
                    Chunk(
                        document_id=document_id,
                        content=chunk_text,
                        heading_path=heading_path,
                        parent_id=parent_node.id,
                        page_start=parent_node.page_start,
                        page_end=parent_node.page_end,
                        order=order,
                        token_count=current_tokens,
                    )
                )
                order += 1
                current_paragraphs = [p]
                current_tokens = p_tokens
            else:
                current_paragraphs.append(p)
                current_tokens += p_tokens

        if current_paragraphs:
            chunk_text = "\n\n".join(current_paragraphs)
            chunks.append(
                Chunk(
                    document_id=document_id,
                    content=chunk_text,
                    heading_path=heading_path,
                    parent_id=parent_node.id,
                    page_start=parent_node.page_start,
                    page_end=parent_node.page_end,
                    order=order,
                    token_count=current_tokens,
                )
            )

        return chunks
