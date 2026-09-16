"""Deterministic Entity Resolution and normalization service."""
import re
from typing import Any, Dict, List, Optional
from app.core.logging import logger
from app.models.chunk import Chunk
from app.models.entity import Entity, EntityType
from app.models.fact import Fact


# Known regulatory entity definitions and canonical names
CANONICAL_ENTITIES = [
    {
        "name": "Securities and Exchange Board of India",
        "normalized_name": "SECURITIES AND EXCHANGE BOARD OF INDIA",
        "type": EntityType.REGULATOR,
        "aliases": ["SEBI", "The Board", "Securities and Exchange Board of India", "Securities & Exchange Board of India"],
        "pattern": r"\b(SEBI|Securities and Exchange Board of India|Securities & Exchange Board of India|The Board)\b",
    },
    {
        "name": "National Stock Exchange of India Limited",
        "normalized_name": "NATIONAL STOCK EXCHANGE OF INDIA LIMITED",
        "type": EntityType.EXCHANGE,
        "aliases": ["NSE", "NSEIL", "The Exchange", "National Stock Exchange", "National Stock Exchange of India"],
        "pattern": r"\b(NSE|NSEIL|National Stock Exchange|National Stock Exchange of India)\b",
    },
    {
        "name": "Trading Member",
        "normalized_name": "TRADING MEMBER",
        "type": EntityType.TRADING_MEMBER,
        "aliases": ["Trading Member", "Stock Broker", "Broker", "TM", "Members", "Trading Members"],
        "pattern": r"\b(Trading Member|Trading Members|Stock Broker|Stock Brokers|TM|Broker)\b",
    },
    {
        "name": "Clearing Member",
        "normalized_name": "CLEARING MEMBER",
        "type": EntityType.CLEARING_MEMBER,
        "aliases": ["Clearing Member", "Clearing Members", "CM", "Self-Clearing Member"],
        "pattern": r"\b(Clearing Member|Clearing Members|CM)\b",
    },
    {
        "name": "Client",
        "normalized_name": "CLIENT",
        "type": EntityType.CLIENT,
        "aliases": ["Client", "Clients", "Investor", "Investors", "Constituent"],
        "pattern": r"\b(Client|Clients|Investor|Investors|Constituent)\b",
    },
    {
        "name": "Authorized Person",
        "normalized_name": "AUTHORIZED PERSON",
        "type": EntityType.AUTHORIZED_PERSON,
        "aliases": ["Authorized Person", "Authorized Persons", "AP", "Sub-broker"],
        "pattern": r"\b(Authorized Person|Authorized Persons|AP|Sub-broker)\b",
    },
    {
        "name": "Rule 8(1)(f)",
        "normalized_name": "RULE 8(1)(F)",
        "type": EntityType.RULE,
        "aliases": ["Rule 8(1)(f)", "Rule 8 1 f", "Rule 8(1)(f) of SCR Rules, 1957"],
        "pattern": r"\b(Rule\s*8\s*\(\s*1\s*\)\s*\(\s*f\s*\)|SCR\s*Rules|Securities Contracts Regulations Rules)\b",
    },
    {
        "name": "Asset Management Company",
        "normalized_name": "ASSET MANAGEMENT COMPANY",
        "type": EntityType.ORGANIZATION,
        "aliases": ["AMC", "AMCs", "Asset Management Companies", "Asset Management Company"],
        "pattern": r"\b(Asset\s+Management\s+Companies|Asset\s+Management\s+Company|AMCs|AMC)\b",
    },
    {
        "name": "Mutual Fund",
        "normalized_name": "MUTUAL FUND",
        "type": EntityType.ORGANIZATION,
        "aliases": ["Mutual Fund", "Mutual Funds", "MF"],
        "pattern": r"\b(Mutual\s+Funds|Mutual\s+Fund)\b",
    },
]


class EntityResolver:
    """Extracts, normalizes, and resolves entities from extracted facts and chunk texts deterministically."""

    def __init__(self, llm_provider: Optional[Any] = None) -> None:
        self.canonical_entities = CANONICAL_ENTITIES

    def resolve_from_facts(self, facts: List[Fact], chunks: Optional[List[Chunk]] = None) -> List[Entity]:
        """Extract and resolve entities directly from extracted facts and chunk text with zero LLM calls."""
        logger.info("Resolving entities deterministically from %d facts", len(facts))
        entity_map: Dict[str, Entity] = {}

        # 1. Match from pre-defined canonical definitions across all chunks
        text_corpus = ""
        if chunks:
            text_corpus = " ".join(c.content for c in chunks)
        else:
            text_corpus = " ".join(f"{f.subject} {f.object} {f.source_text}" for f in facts)

        for spec in self.canonical_entities:
            if re.search(spec["pattern"], text_corpus, re.IGNORECASE):
                norm_name = spec["normalized_name"]
                if norm_name not in entity_map:
                    entity_map[norm_name] = Entity(
                        name=spec["name"],
                        type=spec["type"],
                        normalized_name=norm_name,
                        aliases=spec["aliases"],
                        description=f"Regulatory entity: {spec['name']}",
                    )

        # 2. Extract and canonicalize entities from Fact subjects and objects
        for fact in facts:
            subject_norm = self._normalize_entity_name(fact.subject)
            if subject_norm and subject_norm not in entity_map:
                entity_map[subject_norm] = Entity(
                    name=fact.subject,
                    type=self._infer_entity_type(fact.subject),
                    normalized_name=subject_norm,
                    aliases=[fact.subject],
                )

        resolved_entities = list(entity_map.values())
        logger.info("Resolved %d unique canonical entities deterministically", len(resolved_entities))
        return resolved_entities

    async def resolve_entities(self, chunks: List[Chunk], facts: Optional[List[Fact]] = None) -> List[Entity]:
        """Backward-compatible entry point for pipeline."""
        return self.resolve_from_facts(facts or [], chunks)

    def _normalize_entity_name(self, name: str) -> str:
        """Strip punctuation and uppercase for canonical matching."""
        if not name:
            return ""
        norm = re.sub(r"[^\w\s]", "", name).strip().upper()
        # Aliasing rules
        if norm in {"SEBI", "THE BOARD", "SECURITIES AND EXCHANGE BOARD OF INDIA"}:
            return "SECURITIES AND EXCHANGE BOARD OF INDIA"
        if norm in {"NSE", "NSEIL", "THE EXCHANGE", "NATIONAL STOCK EXCHANGE"}:
            return "NATIONAL STOCK EXCHANGE OF INDIA LIMITED"
        if norm in {"TM", "TRADING MEMBERS", "STOCK BROKER", "STOCK BROKERS", "BROKER"}:
            return "TRADING MEMBER"
        if norm in {"CM", "CLEARING MEMBERS"}:
            return "CLEARING MEMBER"
        if norm in {"CLIENTS", "INVESTOR", "INVESTORS"}:
            return "CLIENT"
        if norm in {"AMC", "AMCS", "ASSET MANAGEMENT COMPANIES", "ASSET MANAGEMENT COMPANY"}:
            return "ASSET MANAGEMENT COMPANY"
        if norm in {"MF", "MUTUAL FUNDS", "MUTUAL FUND"}:
            return "MUTUAL FUND"
        return norm

    def _infer_entity_type(self, name: str) -> EntityType:
        """Infer EntityType from name string."""
        upper = name.upper()
        if "SEBI" in upper or "BOARD" in upper:
            return EntityType.REGULATOR
        if "EXCHANGE" in upper or "NSE" in upper or "BSE" in upper:
            return EntityType.EXCHANGE
        if "MEMBER" in upper or "BROKER" in upper:
            return EntityType.TRADING_MEMBER
        if "CLIENT" in upper or "INVESTOR" in upper:
            return EntityType.CLIENT
        if "RULE" in upper or "ACT" in upper or "SECTION" in upper or "REGULATION" in upper:
            return EntityType.RULE
        if "AMC" in upper or "ASSET MANAGEMENT" in upper or "MUTUAL FUND" in upper:
            return EntityType.ORGANIZATION
        return EntityType.ORGANIZATION
