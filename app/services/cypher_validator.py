"""Cypher Query Validator and Read-Only Safety Guardrails."""
import re
from typing import Set
from app.core.logging import logger


class CypherSecurityError(ValueError):
    """Raised when a generated Cypher query violates security or read-only constraints."""
    pass


class CypherValidator:
    """Validates Cypher queries for read-only safety, structure, and injection risks."""

    FORBIDDEN_MUTATING_KEYWORDS: Set[str] = {
        "CREATE",
        "MERGE",
        "DELETE",
        "DETACH",
        "SET",
        "REMOVE",
        "DROP",
        "ALTER",
        "GRANT",
        "REVOKE",
        "DENY",
        "RENAME",
        "TERMINATE",
    }

    FORBIDDEN_PROCEDURES_REGEX = re.compile(
        r"\bCALL\s+(dbms\.|apoc\.create|apoc\.merge|apoc\.refactor|apoc\.periodic\.iterate|apoc\.load\.csv|apoc\.export)",
        re.IGNORECASE,
    )

    @classmethod
    def strip_comments_and_strings(cls, cypher: str) -> str:
        """Remove comments and quoted string literals to inspect actual Cypher code tokens."""
        # Remove single line comments // ...
        cleaned = re.sub(r"//.*$", "", cypher, flags=re.MULTILINE)
        # Remove block comments /* ... */
        cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
        # Remove double quoted strings
        cleaned = re.sub(r'"(?:\\.|[^"\\])*"', '""', cleaned)
        # Remove single quoted strings
        cleaned = re.sub(r"'(?:\\.|[^'\\])*'", "''", cleaned)
        # Remove backticked identifiers
        cleaned = re.sub(r"`(?:\\.|[^`\\])*`", "``", cleaned)
        return cleaned

    @classmethod
    def validate_and_sanitize(cls, cypher: str) -> str:
        """Validate that a Cypher query is strictly read-only and structurally safe."""
        if not cypher or not cypher.strip():
            raise CypherSecurityError("Empty Cypher query cannot be executed.")

        raw_query = cypher.strip()

        # Check for multiple statements (semicolons outside strings)
        cleaned = cls.strip_comments_and_strings(raw_query)
        if ";" in cleaned.rstrip(";"):
            raise CypherSecurityError("Multiple Cypher statements separated by semicolons are not permitted.")

        # Check for forbidden mutating keywords
        words = set(re.findall(r"\b[A-Za-z_]+\b", cleaned.upper()))
        found_forbidden = words.intersection(cls.FORBIDDEN_MUTATING_KEYWORDS)
        if found_forbidden:
            raise CypherSecurityError(
                f"Mutating Cypher operations are forbidden. Detected restricted keywords: {', '.join(sorted(found_forbidden))}"
            )

        # Check for dangerous procedures
        if cls.FORBIDDEN_PROCEDURES_REGEX.search(raw_query):
            raise CypherSecurityError("Execution of modifying or administrative system procedures is forbidden.")

        # Check that query contains at least MATCH or RETURN
        if "RETURN" not in words and "MATCH" not in words and "SHOW" not in words:
            raise CypherSecurityError("Invalid Cypher query: Query must contain MATCH and RETURN clauses.")

        logger.debug("Cypher query validated successfully (safe read-only): %s", raw_query[:100].replace("\n", " "))
        return raw_query
