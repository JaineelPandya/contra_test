"""Entity extraction and resolution models."""
from enum import Enum
from typing import List, Optional
from uuid import uuid4
from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Entity types relevant to regulatory compliance graphs."""
    REGULATOR = "REGULATOR"
    EXCHANGE = "EXCHANGE"
    ORGANIZATION = "ORGANIZATION"
    REGULATION = "REGULATION"
    RULE = "RULE"
    CLAUSE = "CLAUSE"
    CIRCULAR = "CIRCULAR"
    TRADING_MEMBER = "TRADING_MEMBER"
    CLEARING_MEMBER = "CLEARING_MEMBER"
    AUTHORIZED_PERSON = "AUTHORIZED_PERSON"
    CLIENT = "CLIENT"
    ACTIVITY = "ACTIVITY"
    FINANCIAL_INSTRUMENT = "FINANCIAL_INSTRUMENT"
    REQUIREMENT = "REQUIREMENT"
    PENALTY = "PENALTY"
    DATE = "DATE"


class Entity(BaseModel):
    """Normalized entity with alias tracking."""
    entity_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    type: EntityType
    normalized_name: str
    aliases: List[str] = Field(default_factory=list)
    description: Optional[str] = None


class EntityExtractionResult(BaseModel):
    """LLM extraction response for entities."""
    entities: List[Entity] = Field(default_factory=list)
