# tests/test_distil_whisper.py
import pytest
import logging
from transcription_library.core.manager import TranscriptionManager
from transcription_library.providers.distil_whisper_provider import DistilWhisperProvider

logger = logging.getLogger(__name__)

class TestDistilWhisperProvider:
    """Testes para o provedor Distil-Whisper."""

    @pytest.fixture
    def manager_with_distil(self, cleanup_gpu):
        """Fixture que cria um manager com apenas o provedor Distil-Whisper."""
        manager = TranscriptionManager()
        provider = DistilWhisperProvider()
        manager.register_provider(provider.get_name(), provider)
        yield manager
        # Cleanup será feito pela fixture cleanup_gpu

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_distil_whisper_initialization(self, manager_with_distil):
        """Testa a inicialização do provedor Distil-Whisper."""
        provider_name = "distil-whisper-pt"
        status = manager_with_distil.get_provider_status(provider_name)

        assert status is not None
        assert "model_id" in status
        assert status["model_id"] == "freds0/distil-whisper-large-v3-ptbr"

        logger.info(f"Status Distil-Whisper: {status}")

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_distil_whisper_transcription(self, manager_with_distil, audio_sample):
        """Testa transcrição de áudio com Distil-Whisper."""
        logger.info(f"Testando Distil-Whisper com: {audio_sample.name}")

        result = await manager_with_distil.transcribe_audio(audio_sample, language="pt")

        assert result is not None
        assert result.model_used == "distil-whisper-pt"

        if result.error_message:
            logger.warning(f"Distil-Whisper falhou: {result.error_message}")
            # Não falha o teste se for erro de modelo não disponível
            if "initialization failed" in result.error_message.lower():
                pytest.skip("Modelo Distil-Whisper não disponível")
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

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_distil_whisper_cache(self, manager_with_distil, audio_sample):
        """Testa o cache do Distil-Whisper."""
        logger.info(f"Testando cache do Distil-Whisper com: {audio_sample.name}")

        # Primeira transcrição
        result1 = await manager_with_distil.transcribe_audio(audio_sample, language="pt")

        if result1.error_message:
            pytest.skip(f"Primeira transcrição falhou: {result1.error_message}")

        # Segunda transcrição (deve usar cache)
        result2 = await manager_with_distil.transcribe_audio(audio_sample, language="pt")

        assert result2 is not None
        assert not result2.error_message
        assert result1.text == result2.text
        assert result2.processing_time < result1.processing_time  # Cache deve ser mais rápido

        logger.info(f"Cache funcionando - Tempo 1: {result1.processing_time:.2f}s, Tempo 2: {result2.processing_time:.2f}s")

    @pytest.mark.asyncio
    async def test_distil_whisper_clear_cache(self, manager_with_distil):
        """Testa a limpeza de cache do Distil-Whisper."""
        provider_name = "distil-whisper-pt"

        # Verificar status inicial
        initial_status = manager_with_distil.get_provider_status(provider_name)
        assert initial_status is not None

        # Limpar cache
        manager_with_distil.clear_all_caches()

        # Verificar se cache foi limpo
        final_status = manager_with_distil.get_provider_status(provider_name)
        assert final_status["cache_size"] == 0

        logger.info("Cache limpo com sucesso")

    @pytest.mark.asyncio
    async def test_distil_whisper_invalid_file(self, manager_with_distil, tmp_path):
        """Testa comportamento com arquivo inválido."""
        # Criar arquivo inválido
        invalid_file = tmp_path / "invalid_sample.wav"
        invalid_file.write_text("Este não é um arquivo de áudio válido")

        result = await manager_with_distil.transcribe_audio(invalid_file, language="pt")

        assert result is not None
        assert result.model_used == "distil-whisper-pt"
        assert result.error_message is not None
        assert len(result.text) == 0
        assert result.confidence == 0.0

        logger.info(f"Erro esperado para arquivo inválido: {result.error_message}")

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_distil_whisper_different_languages(self, manager_with_distil, audio_sample):
        """Testa transcrição com diferentes idiomas."""
        languages = ["pt", "en", "es"]

        for language in languages:
            logger.info(f"Testando Distil-Whisper com idioma: {language}")

            result = await manager_with_distil.transcribe_audio(audio_sample, language=language)

            assert result is not None
            assert result.model_used == "distil-whisper-pt"

            if not result.error_message:
                # Como o modelo é otimizado para PT-BR, deve sempre retornar pt-BR
                assert result.language == "pt-BR"
                logger.info(f"Idioma {language} -> resultado: {result.language}")
            else:
                logger.warning(f"Falha com idioma {language}: {result.error_message}")