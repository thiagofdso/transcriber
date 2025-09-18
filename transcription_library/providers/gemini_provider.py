# transcription_library/providers/gemini_provider.py
import asyncio
import logging
import time
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import mimetypes

import google.generativeai as genai

from ..core.interfaces import ITranscriptionProvider, TranscriptionResult
from ..core.utils import get_file_hash
from ..core.config import settings

logger = logging.getLogger(__name__)

class GeminiProvider(ITranscriptionProvider):
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.model_name = settings.GEMINI_MODEL_NAME
        self.video_timeout = settings.GEMINI_VIDEO_TIMEOUT
        self.max_video_size_mb = settings.MAX_VIDEO_SIZE_MB

        self.gemini_model: Optional[genai.GenerativeModel] = None
        self.transcription_cache: Dict[str, TranscriptionResult] = {}
        self.initialization_complete = False

        if not self.api_key:
            logger.warning("Gemini API key is not configured. GeminiProvider will not function.")

    async def initialize(self) -> bool:
        if self.initialization_complete:
            return True

        if not self.api_key:
            logger.error("Gemini API key is missing. Cannot initialize GeminiProvider.")
            self.initialization_complete = False
            return False

        try:
            genai.configure(api_key=self.api_key)
            self.gemini_model = genai.GenerativeModel(self.model_name)
            self.initialization_complete = True
            logger.info(f"GeminiProvider initialized with model: {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize GeminiProvider: {e}")
            self.initialization_complete = False
            return False

    async def transcribe(self, file_path: Path, language: str = "pt-BR") -> TranscriptionResult:
        if not self.initialization_complete:
            await self.initialize()

        if not self.initialization_complete:
            return TranscriptionResult(
                text="", confidence=0.0, processing_time=0.0,
                model_used=self.get_name(), language=language,
                error_message="GeminiProvider not initialized due to missing API key or error."
            )

        # Usar o hash do arquivo para o cache
        file_hash = await get_file_hash(file_path)
        if file_hash in self.transcription_cache:
            cached_result = self.transcription_cache[file_hash]
            logger.info(f"Using cached Gemini transcription for {file_path.name}")
            return cached_result

        # Determinar se é áudio ou vídeo baseado na extensão
        mime_type = mimetypes.guess_type(file_path)[0]
        if mime_type and (mime_type.startswith('audio') or mime_type.startswith('video')):
            if mime_type.startswith('audio'):
                result = await self._transcribe_audio_with_gemini(file_path, language)
            else: # Considerar como vídeo se não for áudio e for um tipo de mídia conhecido
                result = await self._transcribe_video_with_gemini(file_path, language)
        else:
            error_msg = f"Unsupported file type for Gemini transcription: {file_path.suffix}"
            logger.error(error_msg)
            result = TranscriptionResult(
                text="", confidence=0.0, processing_time=0.0,
                model_used=self.get_name(), language=language,
                error_message=error_msg
            )

        # Armazenar em cache o resultado bem-sucedido
        if not result.error_message:
            self.transcription_cache[file_hash] = result

        return result

    async def _transcribe_audio_with_gemini(self, audio_path: Path, language: str) -> TranscriptionResult:
        """Transcreve áudio usando a API Gemini."""
        start_time = time.time()
        try:
            with open(audio_path, 'rb') as audio_file:
                audio_data = audio_file.read()

            audio_part = {
                "mime_type": self._get_mime_type(audio_path),
                "data": audio_data
            }

            prompt = f"""Transcreva este áudio em português brasileiro com precisão.
            Retorne apenas o texto transcrito sem comentários adicionais."""

            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.gemini_model.generate_content([prompt, audio_part])
                ),
                timeout=self.video_timeout # Reutiliza o timeout de vídeo, pode ser ajustado
            )

            processing_time = time.time() - start_time
            if response and response.text:
                transcribed_text = response.text.strip()
                confidence = self._calculate_gemini_confidence(transcribed_text)
                logger.info(f"Gemini audio transcription completed for {audio_path.name} (confidence: {confidence:.3f})")
                return TranscriptionResult(
                    text=transcribed_text,
                    confidence=confidence,
                    processing_time=processing_time,
                    model_used=self.get_name() + "-audio",
                    language=language
                )
            else:
                raise ValueError("No text response from Gemini API for audio.")
        except asyncio.TimeoutError:
            error_msg = f"Gemini audio transcription timed out after {self.video_timeout}s for {audio_path.name}"
            logger.error(error_msg)
            return TranscriptionResult(
                text="", confidence=0.0, processing_time=time.time() - start_time,
                model_used=self.get_name() + "-audio", language=language, error_message=error_msg
            )
        except Exception as e:
            error_msg = f"Gemini audio transcription failed for {audio_path.name}: {str(e)}"
            logger.error(error_msg)
            return TranscriptionResult(
                text="", confidence=0.0, processing_time=time.time() - start_time,
                model_used=self.get_name() + "-audio", language=language, error_message=error_msg
            )

    async def _transcribe_video_with_gemini(self, video_path: Path, language: str) -> TranscriptionResult:
        """Transcreve vídeo usando a API Gemini, incluindo reconhecimento de texto visual."""
        start_time = time.time()
        try:
            if not await self._validate_video_file(video_path):
                raise ValueError("Video file validation failed.")

            video_file_part = await self._upload_video_to_gemini(video_path)
            if not video_file_part:
                raise Exception("Failed to upload video to Gemini API.")

            prompt = f"""Transcreva este vídeo em português. Caso não tenha som liste os textos exibidos na tela em ordem. Também forneça uma descrição do que o vídeo apresenta (mesmo que tenha áudio).

Instruções específicas:
- Se houver fala, transcreva com precisão em português brasileiro
- Se não houver fala mas houver texto na tela, liste todos os textos visíveis em ordem cronológica
- Se não houver fala nem texto, descreva o conteúdo visual do vídeo.
- Se houver tanto fala quanto texto, inclua ambos separadamente
- Mantenha a formatação e pontuação adequadas
- Ignore música de fundo, concentre-se na fala e textos

Formato de resposta:
[FALA]: (transcrição da fala, se houver)
[TEXTO]: (textos visíveis na tela, se houver)
[DESCRIÇÃO VISUAL]: (descrição detalhada do que ocorre no vídeo, se não houver fala nem texto visível, ou um resumo adicional)
"""
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.gemini_model.generate_content([prompt, video_file_part])
                ),
                timeout=self.video_timeout
            )

            processing_time = time.time() - start_time
            if response and response.text:
                transcribed_text = response.text.strip()
                confidence = self._calculate_gemini_confidence(transcribed_text)
                logger.info(f"Gemini video transcription completed for {video_path.name} (confidence: {confidence:.3f})")
                return TranscriptionResult(
                    text=transcribed_text,
                    confidence=confidence,
                    processing_time=processing_time,
                    model_used=self.get_name() + "-video",
                    language=language
                )
            else:
                raise ValueError("No text response from Gemini API for video.")
        except asyncio.TimeoutError:
            error_msg = f"Gemini video transcription timed out after {self.video_timeout}s for {video_path.name}"
            logger.error(error_msg)
            return TranscriptionResult(
                text="", confidence=0.0, processing_time=time.time() - start_time,
                model_used=self.get_name() + "-video", language=language, error_message=error_msg
            )
        except Exception as e:
            error_msg = f"Gemini video transcription failed for {video_path.name}: {str(e)}"
            logger.error(error_msg)
            return TranscriptionResult(
                text="", confidence=0.0, processing_time=time.time() - start_time,
                model_used=self.get_name() + "-video", language=language, error_message=error_msg
            )

    def get_name(self) -> str:
        return "gemini-hybrid"

    def get_status(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "initialized": self.initialization_complete,
            "api_key_configured": bool(self.api_key),
            "cache_size": len(self.transcription_cache),
            "video_timeout": self.video_timeout,
            "max_video_size_mb": self.max_video_size_mb
        }

    def clear_cache(self) -> None:
        self.transcription_cache.clear()
        logger.info(f"{self.get_name()} cache cleared")

    # --- Funções Auxiliares (adaptadas do seu código original) ---
    def _get_mime_type(self, file_path: Path) -> str:
        """Determina o tipo MIME para um arquivo, priorizando mimetypes padrão."""
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            return mime_type
        # Fallback para extensões comuns se mimetypes não adivinhar
        suffix = file_path.suffix.lower()
        mime_types_map = {
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.m4a': 'audio/mp4',
            '.ogg': 'audio/ogg',
            '.opus': 'audio/opus',
            '.mp4': 'video/mp4',
            '.webm': 'video/webm',
            '.mov': 'video/quicktime',
            '.avi': 'video/x-msvideo',
            '.mkv': 'video/x-matroska'
        }
        return mime_types_map.get(suffix, 'application/octet-stream') # Retorno genérico

    async def _validate_video_file(self, video_path: Path) -> bool:
        """Valida o arquivo de vídeo para processamento Gemini."""
        if not video_path.exists():
            logger.error(f"Video file does not exist: {video_path}")
            return False

        file_size_mb = video_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_video_size_mb:
            logger.error(f"Video file too large: {file_size_mb:.1f}MB > {self.max_video_size_mb}MB")
            return False

        supported_video_formats = [".mp4", ".webm", ".mov", ".avi", ".mkv"]
        if video_path.suffix.lower() not in supported_video_formats:
            logger.error(f"Unsupported video format: {video_path.suffix}")
            return False
        return True

    async def _upload_video_to_gemini(self, video_path: Path):
        """Faz o upload do arquivo de vídeo para a API Gemini."""
        try:
            loop = asyncio.get_event_loop()
            video_file = await loop.run_in_executor(
                None,
                lambda: genai.upload_file(str(video_path))
            )

            while video_file.state.name == "PROCESSING":
                logger.info("Waiting for video processing by Gemini...")
                await asyncio.sleep(2)
                video_file = await loop.run_in_executor(
                    None,
                    lambda: genai.get_file(video_file.name)
                )

            if video_file.state.name == "FAILED":
                raise Exception(f"Video processing failed: {video_file.state}")

            return video_file
        except Exception as e:
            logger.error(f"Error uploading video to Gemini: {e}")
            return None

    def _calculate_gemini_confidence(self, text: str) -> float:
        """Calcula a pontuação de confiança para resultados Gemini usando heurísticas."""
        base_confidence = 0.8
        if "[FALA]:" in text or "[TEXTO]:" in text or "[DESCRIÇÃO VISUAL]:" in text:
            base_confidence += 0.1
        if len(text.strip()) < 50: # Penalidade para respostas muito curtas
            base_confidence -= 0.2
        if "não foi possível" in text.lower() or "erro" in text.lower() or "não detectado" in text.lower():
            base_confidence -= 0.4

        return max(0.1, min(0.95, base_confidence))