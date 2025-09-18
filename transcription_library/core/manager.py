# transcription_library/core/manager.py
import logging
from pathlib import Path
from typing import Dict, List, Optional, Type, Any

from .interfaces import ITranscriptionProvider, TranscriptionResult
from .config import settings

logger = logging.getLogger(__name__)

class TranscriptionManager:
    """
    Gerenciador central para provedores de transcrição.
    Permite registrar e selecionar dinamicamente diferentes provedores
    e gerencia a lógica de fallback.
    """
    def __init__(self):
        self._providers: Dict[str, ITranscriptionProvider] = {}
        self._initialized_providers: Dict[str, bool] = {}
        logger.info("TranscriptionManager initialized.")

    def register_provider(self, name: str, provider_instance: ITranscriptionProvider):
        """
        Registra uma instância de um provedor de transcrição na biblioteca.
        Args:
            name: O nome único para referenciar este provedor (e.g., "distil-whisper-pt").
            provider_instance: Uma instância da classe que implementa ITranscriptionProvider.
        """
        if not isinstance(provider_instance, ITranscriptionProvider):
            raise TypeError(f"Provider must implement ITranscriptionProvider interface. Got {type(provider_instance)}")

        if name in self._providers:
            logger.warning(f"Provider '{name}' already registered. Overwriting with new instance.")

        self._providers[name] = provider_instance
        self._initialized_providers[name] = False # Marca como não inicializado inicialmente
        logger.info(f"Provider '{name}' registered: {type(provider_instance).__name__}")

    async def _ensure_provider_initialized(self, name: str) -> bool:
        """Garante que um provedor específico seja inicializado."""
        if not self._providers.get(name):
            logger.error(f"Attempted to initialize unregistered provider: '{name}'")
            return False

        if not self._initialized_providers.get(name, False):
            logger.info(f"Initializing provider '{name}'...")
            success = await self._providers[name].initialize()
            if success:
                self._initialized_providers[name] = True
                logger.info(f"Provider '{name}' initialized successfully.")
            else:
                logger.error(f"Provider '{name}' failed to initialize.")
            return success
        return True # Já inicializado

    async def transcribe_audio(self, audio_path: Path, language: str = "pt") -> TranscriptionResult:
        """
        Transcreve um arquivo de áudio usando o provedor primário, com fallback se necessário.
        A lógica de seleção e fallback é configurada em `settings`.
        """
        provider_chain = [settings.PRIMARY_PROVIDER] + settings.FALLBACK_PROVIDERS

        for provider_name in provider_chain:
            provider = self._providers.get(provider_name)
            if not provider:
                logger.warning(f"Configured provider '{provider_name}' is not registered. Skipping.")
                continue

            if not await self._ensure_provider_initialized(provider_name):
                logger.warning(f"Provider '{provider_name}' failed to initialize. Skipping to next.")
                continue

            logger.info(f"Attempting transcription with provider: '{provider_name}'")
            result = await provider.transcribe(audio_path, language)

            if result.error_message:
                logger.warning(f"Transcription with '{provider_name}' failed: {result.error_message}. Trying fallback.")
                continue # Tenta o próximo provedor em caso de erro

            if result.confidence < settings.CONFIDENCE_THRESHOLD:
                logger.warning(
                    f"Transcription with '{provider_name}' had low confidence ({result.confidence:.2f} "
                    f"< {settings.CONFIDENCE_THRESHOLD:.2f}). Trying fallback."
                )
                continue # Tenta o próximo provedor se a confiança for baixa

            logger.info(f"Transcription successful with '{provider_name}'.")
            return result # Retorna o primeiro resultado bem-sucedido e com confiança adequada

        # Se todos os provedores falharem
        error_msg = "All configured transcription providers failed or returned low confidence."
        logger.error(error_msg)
        return TranscriptionResult(
            text="", confidence=0.0, processing_time=0.0,
            model_used="None", language=language, error_message=error_msg
        )

    def get_provider_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Retorna o status de um provedor específico."""
        provider = self._providers.get(name)
        if provider:
            return provider.get_status()
        logger.warning(f"Provider '{name}' not found.")
        return None

    def get_all_providers_status(self) -> Dict[str, Dict[str, Any]]:
        """Retorna o status de todos os provedores registrados."""
        all_status = {}
        for name, provider in self._providers.items():
            all_status[name] = provider.get_status()
        return all_status

    def clear_all_caches(self) -> None:
        """Limpa o cache de todos os provedores registrados."""
        for name, provider in self._providers.items():
            provider.clear_cache()
            logger.info(f"Cache cleared for provider: {name}")

    async def shutdown(self):
        """Método para realizar o desligamento gracioso (se necessário, por exemplo, para liberar recursos)."""
        logger.info("Shutting down TranscriptionManager.")
        self.clear_all_caches()
        # Outras lógicas de shutdown, se necessário, como liberar modelos da memória, etc.
        # Para modelos transformers, isso geralmente é gerenciado pelo Python GC.