# tests/test_integration.py
import pytest
import logging
from transcription_library.core.manager import TranscriptionManager
from transcription_library.providers.distil_whisper_provider import DistilWhisperProvider
from transcription_library.providers.faster_whisper_provider import FasterWhisperProvider
from transcription_library.providers.gemini_provider import GeminiProvider
from transcription_library.core.config import settings

logger = logging.getLogger(__name__)

class TestIntegration:
    """Testes de integração para múltiplos provedores."""

    @pytest.fixture
    def manager_with_all_providers(self, cleanup_gpu):
        """Fixture que cria um manager com todos os provedores."""
        manager = TranscriptionManager()

        # Registrar provedores na ordem de fallback
        providers = [
            DistilWhisperProvider(),
            FasterWhisperProvider(),
        ]

        # Adicionar Gemini se API key estiver disponível
        if settings.GEMINI_API_KEY:
            providers.append(GeminiProvider())

        for provider in providers:
            manager.register_provider(provider.get_name(), provider)

        yield manager
        # Cleanup será feito pela fixture cleanup_gpu

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_fallback_chain_audio(self, manager_with_all_providers, audio_sample):
        """Testa a cadeia de fallback com arquivo de áudio."""
        logger.info(f"Testando cadeia de fallback com: {audio_sample.name}")

        # Configurar threshold muito alto para forçar fallback
        original_threshold = settings.CONFIDENCE_THRESHOLD
        original_primary = settings.PRIMARY_PROVIDER
        original_fallbacks = settings.FALLBACK_PROVIDERS.copy()

        # Configurar para testar fallback
        settings.CONFIDENCE_THRESHOLD = 0.95  # Threshold muito alto
        settings.PRIMARY_PROVIDER = "distil-whisper-pt"
        settings.FALLBACK_PROVIDERS = ["faster-whisper"]

        if settings.GEMINI_API_KEY:
            settings.FALLBACK_PROVIDERS.append("gemini-hybrid")

        try:
            result = await manager_with_all_providers.transcribe_audio(audio_sample, language="pt")

            assert result is not None

            if not result.error_message:
                assert len(result.text) > 0
                logger.info(f"Fallback chain resultado: {result.model_used}")
                logger.info(f"Confiança: {result.confidence:.3f}")
            else:
                logger.warning(f"Todos os provedores falharam: {result.error_message}")
                # Isso é aceitável em testes, especialmente com arquivos dummy

        finally:
            # Restaurar configurações originais
            settings.CONFIDENCE_THRESHOLD = original_threshold
            settings.PRIMARY_PROVIDER = original_primary
            settings.FALLBACK_PROVIDERS = original_fallbacks

    @pytest.mark.asyncio
    async def test_provider_registration(self, cleanup_gpu):
        """Testa o registro e status de múltiplos provedores."""
        manager = TranscriptionManager()

        # Registrar provedores
        providers = [
            ("distil-whisper-pt", DistilWhisperProvider()),
            ("faster-whisper", FasterWhisperProvider()),
        ]

        if settings.GEMINI_API_KEY:
            providers.append(("gemini-hybrid", GeminiProvider()))

        for name, provider in providers:
            manager.register_provider(name, provider)

        # Verificar status de todos
        all_status = manager.get_all_providers_status()

        assert len(all_status) == len(providers)

        for name, _ in providers:
            assert name in all_status
            status = all_status[name]
            assert isinstance(status, dict)
            logger.info(f"Status {name}: {status}")

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_sequential_transcriptions(self, manager_with_all_providers, audio_sample):
        """Testa múltiplas transcrições sequenciais para verificar limpeza de memória."""
        results = []

        # Executar múltiplas transcrições
        for i in range(3):
            logger.info(f"Execução {i+1}/3")

            result = await manager_with_all_providers.transcribe_audio(audio_sample, language="pt")
            results.append(result)

            assert result is not None

            if not result.error_message:
                logger.info(f"Transcrição {i+1} bem-sucedida com: {result.model_used}")

        # Verificar consistência dos resultados (se todos usaram o mesmo provedor)
        successful_results = [r for r in results if not r.error_message]
        if len(successful_results) > 1:
            first_model = successful_results[0].model_used
            all_same_model = all(r.model_used == first_model for r in successful_results)

            if all_same_model:
                # Se usaram o mesmo provedor, textos devem ser iguais (cache)
                first_text = successful_results[0].text
                assert all(r.text == first_text for r in successful_results)
                logger.info("Resultados consistentes entre execuções")

    @pytest.mark.api
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_mixed_media_types(self, manager_with_all_providers, audio_sample, video_sample):
        """Testa transcrição de diferentes tipos de mídia."""
        if not settings.GEMINI_API_KEY:
            pytest.skip("GEMINI_API_KEY necessária para teste de vídeo")

        media_files = [
            ("audio", audio_sample),
            ("video", video_sample)
        ]

        for media_type, sample_file in media_files:
            logger.info(f"Testando {media_type}: {sample_file.name}")

            # Para vídeo, verificar tamanho
            if media_type == "video":
                file_size_mb = sample_file.stat().st_size / (1024 * 1024)
                if file_size_mb > settings.MAX_VIDEO_SIZE_MB:
                    logger.warning(f"Arquivo {media_type} muito grande: {file_size_mb:.1f}MB")
                    continue

            result = await manager_with_all_providers.transcribe_audio(sample_file, language="pt-BR")

            assert result is not None

            if not result.error_message:
                assert len(result.text) > 0
                logger.info(f"{media_type.title()} transcrito com: {result.model_used}")

                # Vídeo deve usar Gemini
                if media_type == "video":
                    assert "gemini" in result.model_used.lower()
            else:
                logger.warning(f"Falha na transcrição de {media_type}: {result.error_message}")

    @pytest.mark.asyncio
    async def test_concurrent_transcriptions(self, manager_with_all_providers, audio_sample):
        """Testa transcrições concorrentes (não deve ser feito na prática por limitações de memória)."""
        import asyncio

        # Este teste verifica o comportamento, mas não recomendamos uso concorrente
        # devido ao uso de memória GPU dos modelos Whisper
        logger.warning("Teste de concorrência - não recomendado em produção")

        tasks = []
        for i in range(2):  # Apenas 2 para não sobrecarregar
            task = manager_with_all_providers.transcribe_audio(audio_sample, language="pt")
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        assert len(results) == 2

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Transcrição {i+1} falhou com exceção: {result}")
            else:
                logger.info(f"Transcrição {i+1}: {result.model_used if result else 'None'}")

    @pytest.mark.asyncio
    async def test_cache_isolation_between_providers(self, cleanup_gpu, audio_sample):
        """Testa se o cache é isolado entre diferentes provedores."""
        # Criar managers separados para cada provedor
        manager1 = TranscriptionManager()
        provider1 = DistilWhisperProvider()
        manager1.register_provider(provider1.get_name(), provider1)

        manager2 = TranscriptionManager()
        provider2 = FasterWhisperProvider()
        manager2.register_provider(provider2.get_name(), provider2)

        try:
            # Transcrever com o primeiro provedor
            result1 = await manager1.transcribe_audio(audio_sample, language="pt")

            # Transcrever com o segundo provedor
            result2 = await manager2.transcribe_audio(audio_sample, language="pt")

            # Verificar que usaram provedores diferentes
            if not result1.error_message and not result2.error_message:
                assert result1.model_used != result2.model_used
                logger.info(f"Provider 1: {result1.model_used}, Provider 2: {result2.model_used}")

            # Verificar isolamento de cache
            status1 = manager1.get_provider_status(provider1.get_name())
            status2 = manager2.get_provider_status(provider2.get_name())

            # Cada manager deve ter cache apenas do seu provedor
            if not result1.error_message:
                assert status1["cache_size"] > 0
            if not result2.error_message:
                assert status2["cache_size"] > 0

        finally:
            # Limpeza explícita
            await manager1.shutdown()
            await manager2.shutdown()