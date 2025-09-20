# transcription_library/core/config.py
import os
from typing import Dict, Any, List, Optional

class AppConfig:
    """
    Configurações da aplicação. Carrega de variáveis de ambiente se disponíveis.
    """
    # --- Configurações para Distil-Whisper ---
    DISTIL_WHISPER_MODEL_ID: str = "freds0/distil-whisper-large-v3-ptbr"
    FORCE_CPU_FOR_DISTIL_WHISPER: bool = False
    DISTIL_WHISPER_CACHE_DIR: str = "./transcription_cache/distil_whisper"

    # --- Configurações para Faster-Whisper ---
    FASTER_WHISPER_MODEL_SIZE: str = "medium" # ou "small", "base", "large-v2", "large-v3"
    FASTER_WHISPER_CACHE_DIR: str = "./transcription_cache/faster_whisper"
    FASTER_WHISPER_FORCE_CPU: bool = False # Se True, força uso de CPU para Faster-Whisper

    # --- Configurações para Gemini (Google AI Studio) ---
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL_NAME: str = "gemini-pro" # Ou "gemini-pro-vision" se quiser usar a versão com visão explicitamente
    GEMINI_VIDEO_TIMEOUT: float = 300.0 # Timeout em segundos para operações de vídeo com Gemini
    MAX_VIDEO_SIZE_MB: int = 200 # Tamanho máximo de vídeo em MB para upload no Gemini

    # --- Configurações Gerais do Gerenciador ---
    PRIMARY_PROVIDER: str = "distil-whisper-pt" # Provedor padrão
    # Ordem de fallback: se o primário falhar, tenta o próximo na lista
    FALLBACK_PROVIDERS: List[str] = ["faster-whisper", "gemini-hybrid"]
    CONFIDENCE_THRESHOLD: float = 0.6 # Limiar mínimo de confiança para aceitar uma transcrição

    # --- Caminhos de ferramentas externas ---
    # Se ffprobe ou cudnn não estiverem no PATH, podem ser configurados aqui
    FFPROBE_PATH: Optional[str] = None # Ex: "/usr/local/bin/ffprobe"
    CUDNN_PATH: Optional[str] = None # Ex: "/usr/local/cuda/lib64"

    def __init__(self):
        # Carregar variáveis de ambiente (MELHOR PRÁTICA)
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", self.GEMINI_API_KEY)
        self.FFPROBE_PATH = os.getenv("FFPROBE_PATH", self.FFPROBE_PATH)
        self.CUDNN_PATH = os.getenv("CUDNN_PATH", self.CUDNN_PATH)

        # Configurações gerais
        self.PRIMARY_PROVIDER = os.getenv("PRIMARY_PROVIDER", self.PRIMARY_PROVIDER)
        self.CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", str(self.CONFIDENCE_THRESHOLD)))

        # Fallback providers (lista separada por vírgulas)
        fallback_env = os.getenv("FALLBACK_PROVIDERS")
        if fallback_env:
            self.FALLBACK_PROVIDERS = [provider.strip() for provider in fallback_env.split(",")]

        # Faster-Whisper
        self.FASTER_WHISPER_MODEL_SIZE = os.getenv("FASTER_WHISPER_MODEL_SIZE", self.FASTER_WHISPER_MODEL_SIZE)
        self.FASTER_WHISPER_FORCE_CPU = os.getenv("FASTER_WHISPER_FORCE_CPU", "false").lower() == "true"

        # Distil-Whisper
        self.DISTIL_WHISPER_CACHE_DIR = os.getenv("DISTIL_WHISPER_CACHE_DIR", self.DISTIL_WHISPER_CACHE_DIR)
        self.FORCE_CPU_FOR_DISTIL_WHISPER = os.getenv("FORCE_CPU_FOR_DISTIL_WHISPER", "false").lower() == "true"

        # Gemini Video
        self.MAX_VIDEO_SIZE_MB = int(os.getenv("MAX_VIDEO_SIZE_MB", str(self.MAX_VIDEO_SIZE_MB)))
        self.GEMINI_VIDEO_TIMEOUT = float(os.getenv("GEMINI_VIDEO_TIMEOUT", str(self.GEMINI_VIDEO_TIMEOUT)))

# Instância de configuração global
settings = AppConfig()