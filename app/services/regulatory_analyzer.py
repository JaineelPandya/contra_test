"""Regulatory Query Analyzer for parsing user intent, entities, and temporal scope."""
import re
from typing import Any, Dict, List, Optional
from app.core.logging import logger
from app.llm.base import BaseLLMProvider
from app.models.query import QueryType, RegulatoryQueryPlan, RegulatoryQueryRequest


# Standard regulatory entity definitions
COMMON_REGULATORY_ENTITIES = [
    "Trading Member",
    "Clearing Member",
    "Depository Participant",
    "Stock Broker",
    "Clearing Corporation",
    "Stock Exchange",
    "Depository",
    "Qualified Institutional Buyer",
    "Retail Investor",
    "Client",
    "Non-Banking Financial Company",
    "Asset Management Company",
    "Merchant Banker",
    "Compliance Officer",
    "Designated Director",
    "Authorized Person",
    "SEBI",
    "NSE",
    "BSE",
    "NISM",
    "Online Dispute Resolution",
    "ODR Portal",
    "Market Infrastructure Institution",
    "MII",
]

PREDICATE_TOPIC_KEYWORDS = {
    "REQUIRES": ["must", "shall", "require", "required", "requirement", "requirements", "obligation", "mandatory", "compulsory", "submit", "report", "reporting", "comply", "compliance", "rule", "rules"],
    "PROHIBITS": ["prohibit", "forbidden", "shall not", "cannot", "must not", "banned", "disallow", "restriction", "bar"],
    "PERMITS": ["permit", "allowed", "may", "authorized", "eligible", "optional", "entitled"],
    "DEFINES": ["definition", "means", "defined as", "meaning", "classified as"],
    "MODIFIES": ["modify", "amend", "amendment", "revise", "revision", "substitute", "change", "update"],
    "DELETES": ["delete", "repeal", "cancel", "rescind", "omit", "withdraw"],
    "SUPERSEDES": ["supersede", "override", "replace", "supplant"],
    "CLARIFIES": ["clarify", "clarification", "faq", "guidance", "interpret", "illustration"],
}

QUERY_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "cannot", "could", "did", "do", "does", "doing", "down", "during", "each", "few",
    "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "me", "more",
    "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some",
    "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we",
    "were", "what", "when", "where", "which", "while", "who", "whom", "why", "with", "would", "you",
    "your", "yours", "yourself", "yourselves",
    # Query intent stopwords (NOTE: "change", "changes", "latest" intentionally NOT included
    # as they are meaningful regulatory intent signals, not noise)
    "show", "tell", "explain", "find", "give", "list", "describe", "detail", "details", "trace",
    "evolve", "evolved", "evolving", "evolution", "time", "rule",
    "circular", "circulars", "requirement", "requirements", "provision", "provisions",
    "framework", "state", "dated", "date", "year", "years", "current", "currently", "cum",
}


class RegulatoryQueryAnalyzer:
    """Analyzes natural language queries to produce structured graph query plans."""

    def __init__(self, llm_provider: Optional[BaseLLMProvider] = None) -> None:
        self.llm_provider = llm_provider

    async def analyze_query(self, request: RegulatoryQueryRequest) -> RegulatoryQueryPlan:
        """Parse query intent, entities, circular numbers, temporal bounds, and archetype."""
        text = request.query.strip()
        logger.info("Analyzing regulatory query: '%s'", text)

        # 1. Check for manual override
        if request.query_type_override:
            plan = self._rule_based_analysis(request)
            plan.query_type = request.query_type_override
            plan.reasoning = f"Query archetype manually overridden to {request.query_type_override.value}"
            return plan

        # 2. Heuristic rule-based analysis (fast, deterministic)
        plan = self._rule_based_analysis(request)
        return plan

    def _rule_based_analysis(self, request: RegulatoryQueryRequest) -> RegulatoryQueryPlan:
        text = request.query.strip()
        lower_text = text.lower()

        # 1. Extract entities
        extracted_entities: List[str] = []
        if request.entity:
            extracted_entities.append(request.entity)

        sorted_entities = sorted(COMMON_REGULATORY_ENTITIES, key=len, reverse=True)
        for entity in sorted_entities:
            pattern = r"\b" + re.escape(entity) + r"(?:s|es)?\b"
            if re.search(pattern, text, re.IGNORECASE):
                if entity not in extracted_entities:
                    extracted_entities.append(entity)

        # Also extract rule citations like Rule 8(1)(f) as specific entities/anchors
        rule_matches = re.findall(r"\b(Rule\s+\d+(?:\([0-9a-zA-Z]+\))*)\b", text, re.IGNORECASE)
        for rm in rule_matches:
            if rm not in extracted_entities:
                extracted_entities.append(rm)

        # 2. Extract circular / reference numbers
        extracted_circulars: List[str] = []
        if request.circular_number:
            extracted_circulars.append(request.circular_number)

        # Matches: NSE/COMP/50957, SEBI/HO/..., 02/2022, 50957, circular_2022_50957, 1691751968746
        circ_patterns = [
            r"\b([A-Z]{2,6}/[A-Z0-9_-]+/\d+)\b",
            r"\b(\d{2,4}/\d{4})\b",
            r"\bcircular[_\s]+(\d+)\b",
            r"\bcircular[_\s]+(\d{4}[_\s]+\d+)\b",
            r"\b(\d{5,15})\b",
        ]
        for pat in circ_patterns:
            matches = re.findall(pat, text, re.IGNORECASE)
            for m in matches:
                m_clean = m if isinstance(m, str) else m[0]
                if m_clean not in extracted_circulars:
                    extracted_circulars.append(m_clean)

        # 3. Extract temporal scope / dates
        temporal_scope: Dict[str, Any] = {}
        if request.as_of_date:
            temporal_scope["as_of_date"] = request.as_of_date

        # Look for explicit ISO dates YYYY-MM-DD or Month DD, YYYY
        iso_date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
        if iso_date_match:
            temporal_scope["as_of_date"] = iso_date_match.group(1)

        # Look for human dates like "July 31, 2023", "August 4, 2023", "January 07, 2022"
        human_dates = re.findall(
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b",
            text,
            re.IGNORECASE,
        )
        if human_dates:
            from app.services.metadata_extractor import parse_date_to_iso
            parsed_dates = []
            for m_name, day, year in human_dates:
                iso_d = parse_date_to_iso(f"{m_name} {day}, {year}")
                if iso_d:
                    parsed_dates.append(iso_d)
            if parsed_dates:
                temporal_scope["dates_mentioned"] = parsed_dates
                if not temporal_scope.get("as_of_date"):
                    temporal_scope["as_of_date"] = parsed_dates[-1]

        year_matches = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
        if year_matches:
            temporal_scope["years_mentioned"] = list(set(year_matches))

        # 4. Extract predicates / topics
        extracted_predicates: List[str] = []
        for pred, keywords in PREDICATE_TOPIC_KEYWORDS.items():
            for kw in keywords:
                if re.search(r"\b" + re.escape(kw) + r"\b", lower_text):
                    if pred not in extracted_predicates:
                        extracted_predicates.append(pred)
                    break

        # 5. Extract significant semantic keywords & topic phrases
        extracted_keywords: List[str] = []

        # Check for multi-word noun phrase patterns (e.g., "Online Resolution of Disputes", "Corrigendum cum Amendment")
        phrase_candidates = [
            "Online Resolution of Disputes",
            "Dispute Resolution",
            "Online Dispute Resolution",
            "ODR",
            "Arbitration",
            "Conciliation",
            "Corporate Guarantee",
            "Bank Overdraft",
            "Proprietary Trading",
            "Securities Lending",
            "Margin Reporting",
            "Margin Loan",
            "Client Brokerage",
            "KYC",
            "Anti-Money Laundering",
            "Algorithmic Trading",
            "Portfolio Management",
            "Derivatives",
            "Corrigendum",
        ]
        for phrase in phrase_candidates:
            if re.search(r"\b" + re.escape(phrase) + r"\b", text, re.IGNORECASE):
                if phrase not in extracted_keywords:
                    extracted_keywords.append(phrase)

        # Extract remaining substantive single words (> 3 chars) not in stopwords or already captured
        words = re.findall(r"\b[A-Za-z]{3,}\b", text)
        for w in words:
            w_lower = w.lower()
            if w_lower not in QUERY_STOPWORDS and len(w) >= 3:
                # Add capitalized or substantive terms
                if w not in extracted_keywords and w.capitalize() not in extracted_keywords:
                    extracted_keywords.append(w)

        # 6. Determine Query Archetype
        query_type = QueryType.CURRENT
        reasoning = "Defaulted to active regulatory requirements (CURRENT)."

        # Check MASTER_TO_LATEST
        if any(k in lower_text for k in ["master circular", "master to latest", "full lineage", "origin to latest", "master lineage"]):
            query_type = QueryType.MASTER_TO_LATEST
            reasoning = "Query requests lineage tracing from master/origin circular to latest."

        # Check CHANGELOG — broad cross-circular "list all changes" queries (no specific circular needed)
        elif any(k in lower_text for k in [
            "all changes", "all amendments", "all modifications", "all regulatory changes",
            "all circulars", "list all", "list changes", "list amendments", "list modifications",
            "what changed", "what was changed", "what has changed", "regulatory changes",
            "amendments made", "all deleted", "all modified", "show changes", "show amendments",
            "which rules changed", "which clauses changed", "across all", "across circulars",
            "show all modifications", "all superseded", "deleted rules", "modified rules",
            "changes and amendments", "amendments and changes",
            "been deleted", "been superseded", "been modified", "have been deleted",
            "have been superseded", "have been modified", "rules deleted", "rules superseded",
        ]):
            query_type = QueryType.CHANGELOG
            reasoning = "Query asks for a comprehensive cross-circular changelog of all amendments, modifications, and deletions."

        # Check EVOLUTION
        elif any(k in lower_text for k in ["evolv", "evolution", "over time", "history of", "how did it change", "how has it changed", "timeline of", "track change", "track how", "from the", "corrigendum"]):
            query_type = QueryType.EVOLUTION
            reasoning = "Query asks for temporal evolution and changes of a regulatory topic over time."

        # Check AMENDMENT (specific circular referenced)
        elif any(k in lower_text for k in ["amend", "amendment in", "what changed in", "what was deleted by", "what was modified in", "repealed by"]):
            if extracted_circulars:
                query_type = QueryType.AMENDMENT
                reasoning = f"Query inquires about specific amendments or deltas introduced by circular {extracted_circulars[0]}."
            else:
                # Amendment keywords but no specific circular → treat as CHANGELOG
                query_type = QueryType.CHANGELOG
                reasoning = "Query asks about amendments without specifying a circular — returning full cross-circular changelog."

        # Check HISTORICAL
        elif temporal_scope.get("as_of_date") or any(k in lower_text for k in ["as of", "in the year", "prior to", "before", "historical", "was the rule", "on that date"]):
            query_type = QueryType.HISTORICAL
            date_str = temporal_scope.get("as_of_date", temporal_scope.get("years_mentioned", ["historical date"])[0] if temporal_scope.get("years_mentioned") else "specified date")
            reasoning = f"Query requests point-in-time regulatory state as of {date_str}."

        # Check CURRENT
        else:
            query_type = QueryType.CURRENT
            reasoning = "Query asks for currently active rules, permissions, or prohibitions."

        return RegulatoryQueryPlan(
            query_type=query_type,
            raw_query=text,
            entities=extracted_entities,
            keywords=extracted_keywords,
            predicates=extracted_predicates,
            target_circulars=extracted_circulars,
            temporal_scope=temporal_scope,
            reasoning=reasoning,
        )
