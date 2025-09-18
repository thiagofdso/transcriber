# tests/test_gemini.py
import pytest
import logging
from transcription_library.core.manager import TranscriptionManager
from transcription_library.providers.gemini_provider import GeminiProvider
from transcription_library.core.config import settings

logger = logging.getLogger(__name__)

class TestGeminiProvider:
    """Testes para o provedor Gemini."""

    @pytest.fixture
    def manager_with_gemini(self, cleanup_gpu, gemini_api_key):
        """Fixture que cria um manager com apenas o provedor Gemini."""
        manager = TranscriptionManager()
        provider = GeminiProvider()
        manager.register_provider(provider.get_name(), provider)
        yield manager
        # Cleanup será feito pela fixture cleanup_gpu

    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_gemini_initialization(self, manager_with_gemini, gemini_api_key):
        """Testa a inicialização do provedor Gemini."""
        provider_name = "gemini-hybrid"
        status = manager_with_gemini.get_provider_status(provider_name)

        assert status is not None
        assert "model_name" in status
        assert "api_key_configured" in status
        assert status["api_key_configured"] is True

        logger.info(f"Status Gemini: {status}")

    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_gemini_audio_transcription(self, manager_with_gemini, audio_sample, gemini_api_key):
        """Testa transcrição de áudio com Gemini."""
        logger.info(f"Testando Gemini (áudio) com: {audio_sample.name}")

        result = await manager_with_gemini.transcribe_audio(audio_sample, language="pt-BR")

        assert result is not None
        assert result.model_used == "gemini-hybrid-audio"

        if result.error_message:
            logger.warning(f"Gemini (áudio) falhou: {result.error_message}")
            # Não falha o teste se for erro de API ou inicialização
            if any(error in result.error_message.lower() for error in ["not initialized", "api", "timeout"]):
                pytest.skip(f"Erro de API Gemini: {result.error_message}")
            else:
                pytest.fail(f"Erro inesperado: {result.error_message}")
        else:
            assert len(result.text) > 0, "Texto transcrito não pode estar vazio"
            assert result.confidence > 0, "Confiança deve ser maior que 0"
            assert result.processing_time > 0, "Tempo de processamento deve ser maior que 0"
            assert result.language == "pt-BR"

            logger.info(f"Transcrição bem-sucedida: {result.text[:100]}...")
            logger.info(f"Confiança: {result.confidence:.3f}")
            logger.info(f"Tempo: {result.processing_time:.2f}s")

    @pytest.mark.api
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_gemini_video_transcription(self, manager_with_gemini, valid_video_sample, gemini_api_key):
        """Testa transcrição de vídeo com Gemini."""
        logger.info(f"Testando Gemini (vídeo) com: {valid_video_sample.name}")

        # Verificar tamanho do arquivo
        file_size_mb = valid_video_sample.stat().st_size / (1024 * 1024)
        logger.info(f"Tamanho do arquivo: {file_size_mb:.2f}MB")

        result = await manager_with_gemini.transcribe_audio(valid_video_sample, language="pt-BR")

        assert result is not None
        assert result.model_used == "gemini-hybrid-video"

        if result.error_message:
            logger.warning(f"Gemini (vídeo) falhou: {result.error_message}")
            # Não falha o teste se for erro de API, upload ou processamento
            if any(error in result.error_message.lower() for error in ["not initialized", "api", "timeout", "upload", "processing"]):
                pytest.skip(f"Erro de API/Upload Gemini: {result.error_message}")
            else:
                pytest.fail(f"Erro inesperado: {result.error_message}")
        else:
            assert len(result.text) > 0, "Texto transcrito não pode estar vazio"
            assert result.confidence > 0, "Confiança deve ser maior que 0"
            assert result.processing_time > 0, "Tempo de processamento deve ser maior que 0"
            assert result.language == "pt-BR"

            # Verificar se o resultado contém os marcadores esperados do Gemini
            expected_markers = ["[FALA]:", "[TEXTO]:", "[DESCRIÇÃO VISUAL]:"]
            has_markers = any(marker in result.text for marker in expected_markers)

            logger.info(f"Transcrição bem-sucedida: {result.text[:100]}...")
            logger.info(f"Confiança: {result.confidence:.3f}")
            logger.info(f"Tempo: {result.processing_time:.2f}s")
            logger.info(f"Contém marcadores estruturados: {has_markers}")

    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_gemini_cache(self, manager_with_gemini, audio_sample, gemini_api_key):
        """Testa o cache do Gemini."""
        logger.info(f"Testando cache do Gemini com: {audio_sample.name}")

        # Primeira transcrição
        result1 = await manager_with_gemini.transcribe_audio(audio_sample, language="pt-BR")

        if result1.error_message:
            pytest.skip(f"Primeira transcrição falhou: {result1.error_message}")

        # Segunda transcrição (deve usar cache)
        result2 = await manager_with_gemini.transcribe_audio(audio_sample, language="pt-BR")

        assert result2 is not None
        assert not result2.error_message
        assert result1.text == result2.text
        # Cache do Gemini pode não ser significativamente mais rápido devido à rede
        # Mas deve retornar o mesmo resultado

        logger.info(f"Cache funcionando - Tempo 1: {result1.processing_time:.2f}s, Tempo 2: {result2.processing_time:.2f}s")

    @pytest.mark.asyncio
    async def test_gemini_without_api_key(self, cleanup_gpu):
        """Testa comportamento do Gemini sem API key."""
        # Temporariamente limpar a API key
        original_key = settings.GEMINI_API_KEY
        settings.GEMINI_API_KEY = None

        try:
            manager = TranscriptionManager()
            provider = GeminiProvider()
            manager.register_provider(provider.get_name(), provider)

            status = manager.get_provider_status("gemini-hybrid")
            assert status["api_key_configured"] is False

            logger.info("Comportamento sem API key testado com sucesso")

        finally:
            # Restaurar API key original
            settings.GEMINI_API_KEY = original_key

    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_gemini_unsupported_file(self, manager_with_gemini, gemini_api_key, tmp_path):
        """Testa comportamento com arquivo não suportado."""
        # Criar arquivo com extensão não suportada
        unsupported_file = tmp_path / "sample.txt"
        unsupported_file.write_text("Este é um arquivo de texto")

        result = await manager_with_gemini.transcribe_audio(unsupported_file, language="pt-BR")

        assert result is not None
        assert result.model_used == "gemini-hybrid"
        assert result.error_message is not None
        assert "unsupported file type" in result.error_message.lower()
        assert len(result.text) == 0
        assert result.confidence == 0.0

        logger.info(f"Erro esperado para arquivo não suportado: {result.error_message}")

    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_gemini_confidence_calculation(self, manager_with_gemini, audio_sample, gemini_api_key):
        """Testa o cálculo heurístico de confiança do Gemini."""
        result = await manager_with_gemini.transcribe_audio(audio_sample, language="pt-BR")

        if result.error_message:
            pytest.skip(f"Transcrição falhou: {result.error_message}")

        # Confiança deve estar entre 0.1 e 0.95 (limites do cálculo)
        assert 0.1 <= result.confidence <= 0.95

        logger.info(f"Confiança calculada: {result.confidence:.3f}")

    @pytest.mark.asyncio
    async def test_gemini_clear_cache(self, manager_with_gemini, gemini_api_key):
        """Testa a limpeza de cache do Gemini."""
        provider_name = "gemini-hybrid"

        # Verificar status inicial
        initial_status = manager_with_gemini.get_provider_status(provider_name)
        assert initial_status is not None

        # Limpar cache
        manager_with_gemini.clear_all_caches()

        # Verificar se cache foi limpo
        final_status = manager_with_gemini.get_provider_status(provider_name)
        assert final_status["cache_size"] == 0

        logger.info("Cache limpo com sucesso")