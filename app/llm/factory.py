"""Factory function for instantiating LLM providers."""
from app.core.config import get_settings
from app.core.logging import logger
from app.llm.base import BaseLLMProvider
from app.llm.groq_provider import GroqProvider
from app.llm.gemini_provider import GeminiProvider
from app.llm.ollama_provider import OllamaProvider
from app.llm.openai_provider import OpenAICompatibleProvider
from app.llm.mock_provider import MockProvider


def get_llm_provider() -> BaseLLMProvider:
    """Return configured LLM provider instance."""
    settings = get_settings()
    provider_name = settings.LLM_PROVIDER.lower()

    if provider_name == "groq":
        logger.info("Initializing Groq LLM provider (model: %s)", settings.GROQ_MODEL)
        return GroqProvider()
    elif provider_name == "gemini":
        logger.info("Initializing Google Gemini LLM provider (model: %s)", settings.GEMINI_MODEL)
        return GeminiProvider()
    elif provider_name == "ollama":
        logger.info("Initializing Ollama LLM provider (model: %s, url: %s)", settings.OLLAMA_MODEL, settings.OLLAMA_BASE_URL)
        return OllamaProvider()
    elif provider_name in ["openai", "openai_compatible"]:
        logger.info("Initializing OpenAI LLM provider (model: %s, url: %s)", settings.OPENAI_MODEL, settings.OPENAI_BASE_URL)
        return OpenAICompatibleProvider()
    elif provider_name == "mock":
        logger.info("Initializing Mock LLM provider")
        return MockProvider()
    else:
        logger.warning("Unknown LLM provider '%s', defaulting to GroqProvider", settings.LLM_PROVIDER)
        return GroqProvider()
