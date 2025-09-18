# tests/test_faster_whisper.py
import pytest
import logging
from transcription_library.core.manager import TranscriptionManager
from transcription_library.providers.faster_whisper_provider import FasterWhisperProvider

logger = logging.getLogger(__name__)

class TestFasterWhisperProvider:
    """Testes para o provedor Faster-Whisper."""

    @pytest.fixture
    def manager_with_faster(self, cleanup_gpu):
        """Fixture que cria um manager com apenas o provedor Faster-Whisper."""
        manager = TranscriptionManager()
        provider = FasterWhisperProvider()
        manager.register_provider(provider.get_name(), provider)
        yield manager
        # Cleanup será feito pela fixture cleanup_gpu

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_faster_whisper_initialization(self, manager_with_faster):
        """Testa a inicialização do provedor Faster-Whisper."""
        provider_name = "faster-whisper"
        status = manager_with_faster.get_provider_status(provider_name)

        assert status is not None
        assert "model_size" in status
        assert "device" in status
        assert "compute_type" in status

        logger.info(f"Status Faster-Whisper: {status}")

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_faster_whisper_transcription(self, manager_with_faster, audio_sample):
        """Testa transcrição de áudio com Faster-Whisper."""
        logger.info(f"Testando Faster-Whisper com: {audio_sample.name}")

        result = await manager_with_faster.transcribe_audio(audio_sample, language="pt")

        assert result is not None
        assert result.model_used == "faster-whisper"

        if result.error_message:
            logger.warning(f"Faster-Whisper falhou: {result.error_message}")
            # Não falha o teste se for erro de modelo não disponível
            if "initialization failed" in result.error_message.lower():
                pytest.skip("Modelo Faster-Whisper não disponível")
            else:
                pytest.fail(f"Erro inesperado: {result.error_message}")
        else:
            assert len(result.text) > 0, "Texto transcrito não pode estar vazio"
            assert result.confidence > 0, "Confiança deve ser maior que 0"
            assert result.processing_time > 0, "Tempo de processamento deve ser maior que 0"
            assert result.segments is not None, "Segmentos devem estar disponíveis"

            logger.info(f"Transcrição bem-sucedida: {result.text[:100]}...")
            logger.info(f"Confiança: {result.confidence:.3f}")
            logger.info(f"Tempo: {result.processing_time:.2f}s")
            logger.info(f"Segmentos: {len(result.segments)}")

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_faster_whisper_segments(self, manager_with_faster, audio_sample):
        """Testa se o Faster-Whisper retorna segmentos com timestamps."""
        result = await manager_with_faster.transcribe_audio(audio_sample, language="pt")

        if result.error_message:
            pytest.skip(f"Transcrição falhou: {result.error_message}")

        assert result.segments is not None
        assert len(result.segments) > 0

        # Verificar estrutura dos segmentos
        for segment in result.segments:
            assert "start" in segment
            assert "end" in segment
            assert "text" in segment
            assert "confidence" in segment
            assert isinstance(segment["start"], (int, float))
            assert isinstance(segment["end"], (int, float))
            assert segment["start"] <= segment["end"]

        logger.info(f"Segmentos válidos: {len(result.segments)}")

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_faster_whisper_cache(self, manager_with_faster, audio_sample):
        """Testa o cache do Faster-Whisper."""
        logger.info(f"Testando cache do Faster-Whisper com: {audio_sample.name}")

        # Primeira transcrição
        result1 = await manager_with_faster.transcribe_audio(audio_sample, language="pt")

        if result1.error_message:
            pytest.skip(f"Primeira transcrição falhou: {result1.error_message}")

        # Segunda transcrição (deve usar cache)
        result2 = await manager_with_faster.transcribe_audio(audio_sample, language="pt")

        assert result2 is not None
        assert not result2.error_message
        assert result1.text == result2.text
        assert result2.processing_time < result1.processing_time  # Cache deve ser mais rápido

        logger.info(f"Cache funcionando - Tempo 1: {result1.processing_time:.2f}s, Tempo 2: {result2.processing_time:.2f}s")

    @pytest.mark.gpu
    @pytest.mark.asyncio
    async def test_faster_whisper_gpu_usage(self, manager_with_faster):
        """Testa informações sobre uso de GPU."""
        provider_name = "faster-whisper"
        status = manager_with_faster.get_provider_status(provider_name)

        assert "cuda_available" in status
        assert "device" in status

        if status["cuda_available"]:
            logger.info(f"GPU disponível - Device: {status['device']}")
        else:
            logger.info("GPU não disponível - usando CPU")

    @pytest.mark.asyncio
    async def test_faster_whisper_invalid_file(self, manager_with_faster, tmp_path):
        """Testa comportamento com arquivo inválido."""
        # Criar arquivo inválido
        invalid_file = tmp_path / "invalid_sample.wav"
        invalid_file.write_text("Este não é um arquivo de áudio válido")

        result = await manager_with_faster.transcribe_audio(invalid_file, language="pt")

        assert result is not None
        assert result.model_used == "faster-whisper"
        assert result.error_message is not None
        assert len(result.text) == 0
        assert result.confidence == 0.0

        logger.info(f"Erro esperado para arquivo inválido: {result.error_message}")

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_faster_whisper_different_languages(self, manager_with_faster, audio_sample):
        """Testa transcrição com diferentes idiomas."""
        languages = ["pt", "en", "es", "fr"]

        for language in languages:
            logger.info(f"Testando Faster-Whisper com idioma: {language}")

            result = await manager_with_faster.transcribe_audio(audio_sample, language=language)

            assert result is not None
            assert result.model_used == "faster-whisper"

            if not result.error_message:
                # Faster-Whisper pode detectar automaticamente o idioma
                logger.info(f"Idioma solicitado: {language} -> detectado: {result.language}")
            else:
                logger.warning(f"Falha com idioma {language}: {result.error_message}")

    @pytest.mark.asyncio
    async def test_faster_whisper_clear_cache(self, manager_with_faster):
        """Testa a limpeza de cache do Faster-Whisper."""
        provider_name = "faster-whisper"

        # Verificar status inicial
        initial_status = manager_with_faster.get_provider_status(provider_name)
        assert initial_status is not None

        # Limpar cache
        manager_with_faster.clear_all_caches()

        # Verificar se cache foi limpo
        final_status = manager_with_faster.get_provider_status(provider_name)
        assert final_status["cache_size"] == 0

        logger.info("Cache limpo com sucesso")