"""Neo4j driver and connection management."""
from typing import Any, Dict, List, Optional
from neo4j import GraphDatabase, Driver
from app.core.config import get_settings
from app.core.logging import logger


class Neo4jClient:
    """Thread-safe Neo4j driver connection wrapper."""

    def __init__(self) -> None:
        self._driver: Optional[Driver] = None

    def connect(self) -> None:
        """Initialize Neo4j driver connection."""
        if self._driver is not None:
            return

        settings = get_settings()
        try:
            self._driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60.0,
            )
            self._driver.verify_connectivity()
            logger.info("Connected successfully to Neo4j at %s", settings.NEO4J_URI)
        except Exception as e:
            logger.error("Failed to connect to Neo4j at %s: %s", settings.NEO4J_URI, e)
            self._driver = None
            raise

    def close(self) -> None:
        """Close Neo4j driver connection."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j driver closed.")

    def is_connected(self) -> bool:
        """Verify Neo4j connectivity."""
        if self._driver is None:
            return False
        try:
            self._driver.verify_connectivity()
            return True
        except Exception:
            return False

    @property
    def driver(self) -> Driver:
        """Get the active driver instance, connecting if needed."""
        if self._driver is None:
            self.connect()
        return self._driver

    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a parameterized read/write Cypher query and return results as dictionaries."""
        settings = get_settings()
        db = database or settings.NEO4J_DATABASE
        driver = self.driver
        with driver.session(database=db) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def execute_write_transaction(
        self,
        func: Any,
        *args: Any,
        database: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute a unit of work within a managed write transaction."""
        settings = get_settings()
        db = database or settings.NEO4J_DATABASE
        driver = self.driver
        with driver.session(database=db) as session:
            return session.execute_write(func, *args, **kwargs)


# Global client singleton
neo4j_client = Neo4jClient()
