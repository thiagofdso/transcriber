# transcription_library/providers/faster_whisper_provider.py
import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from faster_whisper import WhisperModel
from numba import cuda # Para verificar disponibilidade de CUDA

from ..core.interfaces import ITranscriptionProvider, TranscriptionResult
from ..core.utils import get_audio_duration, get_file_hash
from ..core.config import settings

logger = logging.getLogger(__name__)

class FasterWhisperProvider(ITranscriptionProvider):
    def __init__(self):
        self.model_size = settings.FASTER_WHISPER_MODEL_SIZE
        self.cache_dir = Path(settings.FASTER_WHISPER_CACHE_DIR)
        self.force_cpu = settings.FASTER_WHISPER_FORCE_CPU

        self.model: Optional[WhisperModel] = None
        self.device: str = "cpu"
        self.compute_type: str = "int8" # Padrão para CPU ou menor consumo

        self.transcription_cache: Dict[str, TranscriptionResult] = {}
        self.initialization_complete = False

        # Determinar dispositivo e tipo de computação
        if self.force_cpu:
            self.device = "cpu"
            self.compute_type = "auto"
            logger.info("FasterWhisperProvider: Forçando uso de CPU.")
        elif cuda.is_available():
            self.device = "cuda"
            self.compute_type = "float16" # float16 para CUDA geralmente
            logger.info("FasterWhisperProvider: CUDA disponível. Usando GPU.")
        else:
            self.device = "cpu"
            self.compute_type = "auto"
            logger.info("FasterWhisperProvider: CUDA não disponível. Usando CPU.")

    async def initialize(self) -> bool:
        if self.initialization_complete:
            return True

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Loading Faster-Whisper model: {self.model_size} on {self.device} ({self.compute_type})")
            start_time = time.time()

            loop = asyncio.get_event_loop()

            self.model = await loop.run_in_executor(
                None,
                lambda: WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    download_dir=str(self.cache_dir)
                )
            )

            load_time = time.time() - start_time
            self.initialization_complete = True

            logger.info(f"Faster-Whisper loaded successfully in {load_time:.2f}s")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize Faster-Whisper: {e}")
            self.initialization_complete = False
            return False

    async def transcribe(self, audio_path: Path, language: str = "pt") -> TranscriptionResult:
        if not self.initialization_complete:
            await self.initialize() # Tenta inicializar se ainda não o fez

        if not self.initialization_complete:
            return TranscriptionResult(
                text="", confidence=0.0, processing_time=0.0,
                model_used=self.get_name(), language=language,
                error_message="Model initialization failed or not complete"
            )

        file_hash = await get_file_hash(audio_path)
        if file_hash in self.transcription_cache:
            cached_result = self.transcription_cache[file_hash]
            logger.info(f"Using cached {self.get_name()} transcription for {audio_path.name}")
            return cached_result

        start_time = time.time()

        try:
            logger.info(f"Starting {self.get_name()} transcription for: {audio_path.name}")

            loop = asyncio.get_event_loop()

            # Faster-Whisper retorna segmentos e informações
            segments_generator, info = await loop.run_in_executor(
                None,
                lambda: self.model.transcribe(
                    str(audio_path),
                    language=language, # Use o idioma passado
                    beam_size=5,       # Parâmetro otimizado para Faster-Whisper
                    vad_filter=True    # Filtragem de voz para melhor qualidade
                )
            )

            full_text_segments = []
            segments_metadata = []

            for segment in segments_generator:
                full_text_segments.append(segment.text)
                segments_metadata.append({
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": segment.text,
                    "confidence": 0.8 # Faster-Whisper não oferece confiança de segmento, usar padrão
                })

            transcribed_text = " ".join(full_text_segments).strip()
            processing_time = time.time() - start_time

            # A confiança para Faster-Whisper é heurística ou baseada no modelo
            # Podemos usar info.language_probability para uma estimativa
            confidence = info.language_probability if info.language_probability is not None else 0.8

            transcription_result = TranscriptionResult(
                text=transcribed_text,
                confidence=confidence,
                processing_time=processing_time,
                model_used=self.get_name(),
                language=info.language if info.language else language,
                segments=segments_metadata
            )

            self.transcription_cache[file_hash] = transcription_result

            logger.info(
                f"{self.get_name()} transcription completed for {audio_path.name}: "
                f"{len(transcribed_text)} chars in {processing_time:.2f}s "
                f"(confidence: {confidence:.3f})"
            )

            return transcription_result

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"{self.get_name()} transcription failed for {audio_path.name}: {str(e)}"
            logger.error(error_msg)

            return TranscriptionResult(
                text="",
                confidence=0.0,
                processing_time=processing_time,
                model_used=self.get_name(),
                language=language,
                error_message=error_msg
            )

    def get_name(self) -> str:
        return "faster-whisper"

    def get_status(self) -> Dict[str, Any]:
        return {
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "initialized": self.initialization_complete,
            "cuda_available": cuda.is_available(),
            "cache_size": len(self.transcription_cache)
        }

    def clear_cache(self) -> None:
        self.transcription_cache.clear()
        logger.info(f"{self.get_name()} cache cleared")