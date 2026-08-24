"""Hierarchical structure parsing for legal and regulatory documents."""
import re
from typing import List, Optional, Tuple
from app.core.logging import logger
from app.models.hierarchy import HierarchyNode, HierarchyNodeType
from app.services.pdf_parser import PDFPage


# Regular expressions for hierarchy detection
PATTERNS = [
    # Annexures: Annexure A, Annexure 1, Annexure-I
    ("ANNEXURE", re.compile(r"^(?:ANNEXURE|SCHEDULE|APPENDIX)\s*[-–:]?\s*([A-Z0-9]+.*)$", re.IGNORECASE), HierarchyNodeType.ANNEXURE, 1),
    
    # FAQs section header
    ("FAQ_HEADER", re.compile(r"^(?:FAQ|FREQUENTLY ASKED QUESTIONS|CLARIFICATIONS)\s*.*$", re.IGNORECASE), HierarchyNodeType.FAQ, 1),

    # Questions: Question No. 1:, Question 1:, Q1., Q.1:
    ("QUESTION", re.compile(r"^(?:Question\s*(?:No\.?)?\s*(\d+|[A-Z])|Q\.?\s*(\d+))\s*[:\.\-]\s*(.*)$", re.IGNORECASE), HierarchyNodeType.QUESTION, 2),

    # Answers: Answer:, Ans:, Ans.
    ("ANSWER", re.compile(r"^(?:Answer|Ans)\s*[:\.\-]\s*(.*)$", re.IGNORECASE), HierarchyNodeType.ANSWER, 3),

    # Point No. 1, Point 1, Point No. 1.1
    ("POINT", re.compile(r"^(?:Point\s*(?:No\.?)?\s*(\d+(?:\.\d+)*))\s*[:\.\-]?\s*(.*)$", re.IGNORECASE), HierarchyNodeType.POINT, 2),

    # Numbered sections: 1., 2., 3., 1.1, 1.1.1
    ("NUMBERED_SECTION_3", re.compile(r"^(\d+\.\d+\.\d+)\s+([A-Z0-9].*)$"), HierarchyNodeType.SUBSECTION, 3),
    ("NUMBERED_SECTION_2", re.compile(r"^(\d+\.\d+)\s+([A-Z0-9].*)$"), HierarchyNodeType.SUBSECTION, 2),
    ("NUMBERED_SECTION_1", re.compile(r"^(\d+)\.\s+([A-Z0-9].*)$"), HierarchyNodeType.SECTION, 1),

    # Explicit named sections: Section 1, Rule 8, Clause 20
    ("NAMED_SECTION", re.compile(r"^(?:Section|Rule|Clause)\s+(\d+(?:\([a-zA-Z0-9]+\))*)\s*[:\.\-]?\s*(.*)$", re.IGNORECASE), HierarchyNodeType.CLAUSE, 2),

    # Roman numeral list items: (i), (ii), (iii), (iv) or i., ii.
    ("ROMAN_CLAUSE", re.compile(r"^(?:\(([ivxlcdm]+)\)|([ivxlcdm]+)\.)\s+(.*)$", re.IGNORECASE), HierarchyNodeType.CLAUSE, 3),

    # Alphabetic list items: (a), (b), (c) or a., b.
    ("ALPHA_CLAUSE", re.compile(r"^(?:\(([a-z])\)|([a-z])\.)\s+(.*)$", re.IGNORECASE), HierarchyNodeType.LIST_ITEM, 3),
]


class HierarchyBuilder:
    """Builds a hierarchical tree of nodes representing sections, clauses, points, and FAQs."""

    def build_hierarchy(self, document_id: str, pages: List[PDFPage]) -> List[HierarchyNode]:
        """Parse all pages into a structured tree of HierarchyNodes."""
        logger.info("Building document hierarchy for document: %s (%d pages)", document_id, len(pages))
        
        flat_nodes: List[HierarchyNode] = []
        current_parent_by_level: dict[int, HierarchyNode] = {}
        order_counter = 1

        # We buffer content for the currently active node
        active_node: Optional[HierarchyNode] = None

        for page in pages:
            lines = page.text.split("\n")
            for line in lines:
                stripped_line = line.strip()
                if not stripped_line:
                    continue

                # Check if this line matches a new structural heading
                matched = self._match_heading(stripped_line)

                if matched:
                    # Finalize active node page_end if switching
                    if active_node:
                        active_node.page_end = page.page_number

                    node_type, title, level, initial_content = matched
                    
                    # Determine parent node based on level hierarchy
                    parent_node = None
                    for l in range(level - 1, 0, -1):
                        if l in current_parent_by_level:
                            parent_node = current_parent_by_level[l]
                            break

                    new_node = HierarchyNode(
                        document_id=document_id,
                        node_type=node_type,
                        title=title,
                        content=initial_content if initial_content else title,
                        level=level,
                        order=order_counter,
                        page_start=page.page_number,
                        page_end=page.page_number,
                        parent_id=parent_node.id if parent_node else None,
                    )
                    order_counter += 1

                    # Register in tree tracking
                    current_parent_by_level[level] = new_node
                    # Clear deeper levels
                    levels_to_clear = [l for l in current_parent_by_level if l > level]
                    for l in levels_to_clear:
                        del current_parent_by_level[l]

                    if parent_node:
                        parent_node.children.append(new_node)

                    flat_nodes.append(new_node)
                    active_node = new_node
                else:
                    # Regular body content line
                    if active_node:
                        active_node.content += "\n" + stripped_line
                        active_node.page_end = page.page_number
                    else:
                        # Before the first recognized section, create a default Document Overview section
                        new_node = HierarchyNode(
                            document_id=document_id,
                            node_type=HierarchyNodeType.SECTION,
                            title="General / Preamble",
                            content=stripped_line,
                            level=1,
                            order=order_counter,
                            page_start=page.page_number,
                            page_end=page.page_number,
                            parent_id=None,
                        )
                        order_counter += 1
                        current_parent_by_level[1] = new_node
                        flat_nodes.append(new_node)
                        active_node = new_node

        logger.info("Successfully identified %d structural hierarchy nodes", len(flat_nodes))
        return flat_nodes

    def _match_heading(self, line: str) -> Optional[Tuple[HierarchyNodeType, str, int, str]]:
        """Test line against known hierarchy patterns. Returns (NodeType, Title, Level, InitialContent)."""
        # Skip headers / footers that are too short or page markers
        if len(line) < 3 or re.match(r"^Page \d+ of \d+$", line, re.IGNORECASE):
            return None

        for name, pattern, node_type, level in PATTERNS:
            match = pattern.match(line)
            if match:
                groups = match.groups()
                # Determine title
                if name == "ANNEXURE":
                    title = f"Annexure {groups[0]}".strip()
                    content = line
                elif name == "FAQ_HEADER":
                    title = line.strip()
                    content = line
                elif name == "QUESTION":
                    q_num = groups[0] or groups[1]
                    rest = groups[2] if len(groups) > 2 else ""
                    title = f"Question {q_num}" + (f": {rest[:50]}" if rest else "")
                    content = line
                elif name == "ANSWER":
                    title = "Answer"
                    content = line
                elif name == "POINT":
                    p_num = groups[0]
                    rest = groups[1] if len(groups) > 1 else ""
                    title = f"Point {p_num}" + (f": {rest[:40]}" if rest else "")
                    content = line
                elif name.startswith("NUMBERED_SECTION"):
                    sec_num = groups[0]
                    rest = groups[1]
                    title = f"{sec_num} {rest[:50]}".strip()
                    content = line
                elif name == "NAMED_SECTION":
                    sec_num = groups[0]
                    rest = groups[1] if len(groups) > 1 else ""
                    title = f"Clause {sec_num}" + (f": {rest[:40]}" if rest else "")
                    content = line
                elif name in ["ROMAN_CLAUSE", "ALPHA_CLAUSE"]:
                    item_mark = groups[0] or groups[1]
                    rest = groups[2] if len(groups) > 2 else ""
                    title = f"({item_mark})" + (f" {rest[:40]}" if rest else "")
                    content = line
                else:
                    title = line[:60].strip()
                    content = line

                return node_type, title, level, content

        return None
