# transcription_library/core/interfaces.py
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

@dataclass
class TranscriptionResult:
    """
    Estrutura padrão para o resultado de uma transcrição.
    Garante consistência na saída de diferentes provedores.
    """
    text: str
    confidence: float
    processing_time: float # Tempo total gasto na transcrição
    model_used: str        # Nome do modelo/serviço utilizado (e.g., "distil-whisper-pt", "faster-whisper")
    language: str
    segments: Optional[List[Dict]] = None # Segmentos com timestamp e texto
    error_message: Optional[str] = None   # Mensagem de erro, se houver

class ITranscriptionProvider(ABC):
    """
    Interface (Classe Abstrata) para todos os provedores de transcrição.
    Qualquer nova implementação de transcrição deve herdar desta classe
    e implementar seus métodos abstratos.
    """
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Inicializa o provedor de transcrição, carregando modelos ou configurando APIs.
        Deve ser idempotente (seguro para ser chamado múltiplas vezes).
        Retorna True se a inicialização for bem-sucedida, False caso contrário.
        """
        pass

    @abstractmethod
    async def transcribe(self, audio_path: Path, language: str = "pt") -> TranscriptionResult:
        """
        Realiza a transcrição de um arquivo de áudio.
        Args:
            audio_path: Caminho para o arquivo de áudio a ser transcrito.
            language: Código do idioma para a transcrição (padrão: "pt").
        Returns:
            Um objeto TranscriptionResult contendo o texto, confiança e metadados.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Retorna o nome único do provedor (e.g., "distil-whisper", "faster-whisper")."""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Retorna informações de status do provedor (e.g., modelo carregado, uso de GPU)."""
        pass

    @abstractmethod
    def clear_cache(self) -> None:
        """Limpa qualquer cache interno mantido pelo provedor."""
        pass