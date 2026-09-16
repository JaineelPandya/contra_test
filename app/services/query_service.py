"""Orchestration service for regulatory query analysis, Cypher generation, execution, and synthesis."""
from typing import Any, Dict, List, Optional
from app.core.logging import logger
from app.graph.repository import GraphRepository
from app.llm.base import BaseLLMProvider
from app.models.query import (
    GraphSubgraph,
    QueryType,
    RegulatoryQueryPlan,
    RegulatoryQueryRequest,
    RegulatoryQueryResponse,
)
from app.services.answer_synthesizer import RegulatoryAnswerSynthesizer
from app.services.cypher_generator import CypherGenerator
from app.services.cypher_validator import CypherValidator
from app.services.regulatory_analyzer import RegulatoryQueryAnalyzer
from app.services.subgraph_builder import SubgraphBuilder


class RegulatoryQueryService:
    """End-to-end service coordinating NLP analysis, Cypher execution, answer synthesis, and subgraph generation."""

    def __init__(self, repository: GraphRepository, llm_provider: Optional[BaseLLMProvider] = None) -> None:
        self.repository = repository
        self.llm_provider = llm_provider
        self.analyzer = RegulatoryQueryAnalyzer(llm_provider)
        self.generator = CypherGenerator()
        self.validator = CypherValidator()
        self.subgraph_builder = SubgraphBuilder()
        self.synthesizer = RegulatoryAnswerSynthesizer(llm_provider)

    async def plan_query(self, request: RegulatoryQueryRequest) -> Dict[str, Any]:
        """Generate query plan and preview Cypher query without executing."""
        plan = await self.analyzer.analyze_query(request)
        cypher, params = self.generator.generate_cypher(plan)
        return {
            "plan": plan.model_dump(),
            "cypher_query": cypher,
            "cypher_params": params,
        }

    async def execute_query(self, request: RegulatoryQueryRequest) -> RegulatoryQueryResponse:
        """Run full end-to-end pipeline: Analyze -> Plan -> Cypher -> Validate -> Neo4j -> Synthesize + Subgraph."""
        logger.info("Executing regulatory query request: '%s'", request.query)

        # 1. Analyze query and classify archetype
        plan = await self.analyzer.analyze_query(request)

        # 2. Generate parameterized Cypher query
        cypher_query, cypher_params = self.generator.generate_cypher(plan)

        # 3. Validate read-only safety guardrails
        validated_cypher = self.validator.validate_and_sanitize(cypher_query)

        # 4. Execute on Neo4j Graph
        records: List[Dict[str, Any]] = []
        effective_query_type = plan.query_type
        try:
            records = self.repository.client.execute_query(validated_cypher, cypher_params)
            logger.info("Neo4j query executed successfully. Retrieved %d records.", len(records))
        except Exception as e:
            logger.error("Failed to execute Cypher query: %s | Query: %s", e, validated_cypher)
            records = []

        effective_plan = plan
        if not records and plan.query_type == QueryType.EVOLUTION:
            logger.info("No direct fact-evolution paths found for '%s'. Checking ChangeEvent amendments.", request.query)
            amend_plan = plan.model_copy()
            amend_plan.query_type = QueryType.AMENDMENT
            amend_cypher, amend_params = self.generator.generate_cypher(amend_plan)
            try:
                amend_records = self.repository.client.execute_query(amend_cypher, amend_params)
                if amend_records:
                    records = amend_records
                    effective_query_type = QueryType.AMENDMENT
                    effective_plan = amend_plan
                    validated_cypher = amend_cypher
                    cypher_params = amend_params
                    logger.info("Found %d amendment records via ChangeEvents.", len(records))
            except Exception as ae:
                logger.error("Amendment fallback query failed: %s", ae)

        # CHANGELOG fallback: if no change events returned, try broad AMENDMENT then CURRENT
        if not records and plan.query_type == QueryType.CHANGELOG:
            logger.info("No ChangeEvents found for CHANGELOG query. Falling back to AMENDMENT.", )
            amend_plan = plan.model_copy()
            amend_plan.query_type = QueryType.AMENDMENT
            amend_cypher, amend_params = self.generator.generate_cypher(amend_plan)
            try:
                amend_records = self.repository.client.execute_query(amend_cypher, amend_params)
                if amend_records:
                    records = amend_records
                    effective_query_type = QueryType.AMENDMENT
                    effective_plan = amend_plan
                    validated_cypher = amend_cypher
                    cypher_params = amend_params
                    logger.info("Found %d records via AMENDMENT fallback for CHANGELOG query.", len(records))
            except Exception as ae:
                logger.error("CHANGELOG→AMENDMENT fallback failed: %s", ae)

        if not records and plan.query_type in (QueryType.EVOLUTION, QueryType.AMENDMENT, QueryType.MASTER_TO_LATEST, QueryType.CHANGELOG):
            logger.info("No evolution/amendment/changelog records found for '%s'. Falling back to active provisions.", request.query)
            fallback_plan = plan.model_copy()
            fallback_plan.query_type = QueryType.CURRENT
            fallback_cypher, fallback_params = self.generator.generate_cypher(fallback_plan)
            try:
                fallback_records = self.repository.client.execute_query(fallback_cypher, fallback_params)
                if fallback_records:
                    records = fallback_records
                    effective_query_type = QueryType.CURRENT
                    effective_plan = fallback_plan
                    validated_cypher = fallback_cypher
                    cypher_params = fallback_params
                    logger.info("Found %d active regulation records via fallback.", len(records))
            except Exception as fe:
                logger.error("Fallback query failed: %s", fe)


        # 5. Dual-Path Branch A: Answer Synthesis (Current State, Changelog, Citations, Markdown)
        current_summary, change_history, llm_answer, citations = await self.synthesizer.synthesize_response(
            plan=effective_plan,
            records=records,
        )

        # 6. Dual-Path Branch B: Subgraph Builder for Dashboard Visualization
        subgraph: Optional[GraphSubgraph] = None
        if request.include_subgraph:
            subgraph = self.subgraph_builder.build_subgraph(records, effective_query_type)

        return RegulatoryQueryResponse(
            query=request.query,
            plan=plan,
            cypher_query=validated_cypher,
            cypher_params=cypher_params,
            current_regulation=current_summary,
            change_history=change_history,
            llm_answer=llm_answer,
            sources=citations,
            subgraph=subgraph,
        )
