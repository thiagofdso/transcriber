# transcription_library/providers/distil_whisper_provider.py
import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline

from ..core.interfaces import ITranscriptionProvider, TranscriptionResult
from ..core.utils import get_file_hash, get_audio_duration
from ..core.config import settings

logger = logging.getLogger(__name__)

class DistilWhisperProvider(ITranscriptionProvider):
    def __init__(self):
        self.model_id = settings.DISTIL_WHISPER_MODEL_ID
        self.force_cpu = settings.FORCE_CPU_FOR_DISTIL_WHISPER
        self.cache_dir = Path(settings.DISTIL_WHISPER_CACHE_DIR)

        self.device = "cuda" if torch.cuda.is_available() and not self.force_cpu else "cpu"
        self.compute_type = torch.float16 if self.device == "cuda" else torch.float32

        self.model = None
        self.processor = None
        self.pipeline_instance = None # Renomeado para evitar conflito com 'pipeline' importado

        self.transcription_cache: Dict[str, TranscriptionResult] = {}
        self.initialization_complete = False

        logger.info(f"DistilWhisperProvider initialized for device: {self.device}")

    async def initialize(self) -> bool:
        if self.initialization_complete:
            return True

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Loading Distil-Whisper Portuguese model: {self.model_id} to {self.device}")
            start_time = time.time()

            loop = asyncio.get_event_loop()

            self.processor = await loop.run_in_executor(
                None,
                lambda: WhisperProcessor.from_pretrained(
                    self.model_id,
                    cache_dir=str(self.cache_dir)
                )
            )

            # Usar 'accelerate' para otimização de GPU, se disponível
            use_accelerate = self.device == "cuda" and torch.cuda.is_available()

            if use_accelerate:
                self.model = await loop.run_in_executor(
                    None,
                    lambda: WhisperForConditionalGeneration.from_pretrained(
                        self.model_id,
                        dtype=self.compute_type,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                        use_safetensors=True,
                        cache_dir=str(self.cache_dir)
                    )
                )
            else:
                self.model = await loop.run_in_executor(
                    None,
                    lambda: WhisperForConditionalGeneration.from_pretrained(
                        self.model_id,
                        dtype=self.compute_type,
                        device_map="auto" if self.device == "cuda" else None,
                        cache_dir=str(self.cache_dir)
                    )
                )

            self.pipeline_instance = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                dtype=self.compute_type,
                chunk_length_s=25,
                stride_length_s=5,
                return_timestamps=True
            )

            load_time = time.time() - start_time
            self.initialization_complete = True

            logger.info(f"Distil-Whisper loaded successfully on {self.device} in {load_time:.2f}s")
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize Distil-Whisper: {e}")
            self.initialization_complete = False
            return False

    async def transcribe(self, audio_path: Path, language: str = "pt") -> TranscriptionResult:
        if not self.initialization_complete:
            await self.initialize() # Tenta inicializar se ainda não o fez

        if not self.initialization_complete:
            return TranscriptionResult(
                text="", confidence=0.0, processing_time=0.0,
                model_used=self.get_name(), language="pt-BR",
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

            # Adaptação da chamada do pipeline
            result = await loop.run_in_executor(
                None,
                lambda: self.pipeline_instance(
                    str(audio_path),
                    generate_kwargs={
                        "language": "portuguese", # Força português brasileiro
                        "task": "transcribe"
                    }
                )
            )

            processing_time = time.time() - start_time

            if not result or "text" not in result:
                raise ValueError(f"Invalid transcription result from {self.get_name()} pipeline")

            transcribed_text = result["text"].strip()
            if not transcribed_text:
                raise ValueError(f"Empty transcription result from {self.get_name()}")

            confidence = self._calculate_confidence(result)
            segments = self._extract_segments(result)

            transcription_result = TranscriptionResult(
                text=transcribed_text,
                confidence=confidence,
                processing_time=processing_time,
                model_used=self.get_name(),
                language="pt-BR",
                segments=segments
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
                language="pt-BR",
                error_message=error_msg
            )

    def get_name(self) -> str:
        return "distil-whisper-pt"

    def get_status(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "device": self.device,
            "initialized": self.initialization_complete,
            "cuda_available": torch.cuda.is_available(),
            "cache_size": len(self.transcription_cache),
            "model_parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0
        }

    def clear_cache(self) -> None:
        self.transcription_cache.clear()
        logger.info(f"{self.get_name()} cache cleared")

    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """Calcula a confiança média a partir dos chunks do pipeline."""
        if "chunks" in result and result["chunks"]:
            confidences = []
            for chunk in result["chunks"]:
                if isinstance(chunk, dict) and "confidence" in chunk:
                    confidences.append(chunk["confidence"])
                else:
                    confidences.append(0.85) # Valor padrão se a confiança não estiver disponível

            if confidences:
                return sum(confidences) / len(confidences)
        return 0.87 # Confiança padrão alta para o modelo Distil-Whisper PT

    def _extract_segments(self, result: Dict[str, Any]) -> Optional[List[Dict]]:
        """Extrai os segmentos com timestamp do resultado do pipeline."""
        segments = []
        if "chunks" in result and result["chunks"]:
            for chunk in result["chunks"]:
                if isinstance(chunk, dict) and "timestamp" in chunk:
                    segments.append({
                        "start": chunk["timestamp"][0] if chunk["timestamp"][0] else 0.0,
                        "end": chunk["timestamp"][1] if chunk["timestamp"][1] else 0.0,
                        "text": chunk.get("text", ""),
                        "confidence": chunk.get("confidence", 0.85)
                    })
        return segments if segments else None