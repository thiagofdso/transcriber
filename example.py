# example.py
import asyncio
import logging
from pathlib import Path
import os

# Configuração básica de logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Importar o gerenciador e os provedores
from transcription_library.core.manager import TranscriptionManager
from transcription_library.providers.distil_whisper_provider import DistilWhisperProvider
from transcription_library.providers.faster_whisper_provider import FasterWhisperProvider
from transcription_library.providers.gemini_provider import GeminiProvider
from transcription_library.core.config import settings

# --- Exemplo de Configuração de Variável de Ambiente (simulação) ---
# Em um ambiente real, você definiria isso como uma variável de ambiente
# Ex: export GEMINI_API_KEY="SUA_CHAVE_API_GEMINI"
# Para testar, você pode descomentar e definir diretamente (NÃO FAZER ISSO EM PRODUÇÃO)
# os.environ["GEMINI_API_KEY"] = "SUA_CHAVE_API_GEMINI_AQUI"
# settings.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Recarregar a chave na instância de settings

async def main():
    logger = logging.getLogger("main_app")
    logger.info("Iniciando aplicação de transcrição plugável.")

    manager = TranscriptionManager()

    # --- 1. Registrar os Provedores de Transcrição ---
    distil_provider = DistilWhisperProvider()
    manager.register_provider(distil_provider.get_name(), distil_provider)

    faster_provider = FasterWhisperProvider()
    manager.register_provider(faster_provider.get_name(), faster_provider)

    gemini_provider = GeminiProvider()
    manager.register_provider(gemini_provider.get_name(), gemini_provider)

    # --- 2. Preparar arquivos de mídia para transcrição ---
    # Substitua pelos caminhos reais dos seus arquivos para testar!
    # Lembre-se que Gemini exige uma chave de API válida para funcionar.

    # Exemplo de arquivo de áudio (MP3 ou WAV)
    audio_file_path = Path("caminho/para/seu/audio_portugues.mp3")

    # Exemplo de arquivo de vídeo (MP4, WebM)
    video_file_path = Path("caminho/para/seu/video_portugues.mp4")

    # Criar arquivos dummy se não existirem para evitar FileNotFoundError imediato
    if not audio_file_path.exists():
        logger.warning(f"Arquivo de áudio dummy não encontrado em '{audio_file_path}'. "
                       "Crie um arquivo de áudio real para testes eficazes.")
        audio_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(audio_file_path, "w") as f: f.write("dummy audio content")

    if not video_file_path.exists():
        logger.warning(f"Arquivo de vídeo dummy não encontrado em '{video_file_path}'. "
                       "Crie um arquivo de vídeo real para testes eficazes.")
        video_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(video_file_path, "w") as f: f.write("dummy video content")

    logger.info(f"Tentando transcrever o arquivo de áudio: {audio_file_path.name}")
    audio_result = await manager.transcribe_audio(audio_file_path, language="pt")

    if audio_result.error_message:
        logger.error(f"A transcrição de áudio falhou: {audio_result.error_message}")
    else:
        logger.info(f"Transcrição de áudio bem-sucedida usando: {audio_result.model_used}")
        logger.info(f"Confiança: {audio_result.confidence:.2f}")
        print("\n--- Transcrição de Áudio Completa ---\n")
        print(audio_result.text)
        print("\n--------------------------------------\n")

    # --- Teste de Transcrição de Vídeo com Gemini ---
    if settings.GEMINI_API_KEY:
        logger.info(f"Tentando transcrever o arquivo de vídeo: {video_file_path.name} com Gemini")
        video_result = await manager.transcribe_audio(video_file_path, language="pt-BR") # Manager aceita qualquer Path

        if video_result.error_message:
            logger.error(f"A transcrição de vídeo falhou: {video_result.error_message}")
        else:
            logger.info(f"Transcrição de vídeo bem-sucedida usando: {video_result.model_used}")
            logger.info(f"Confiança: {video_result.confidence:.2f}")
            print("\n--- Transcrição de Vídeo (Gemini) Completa ---\n")
            print(video_result.text)
            print("\n-----------------------------------------------\n")
    else:
        logger.warning("GEMINI_API_KEY não configurada. Pulando o teste de transcrição de vídeo.")


    # --- 3. Obter Status dos Provedores ---
    logger.info("\n--- Status dos Provedores Registrados ---")
    all_status = manager.get_all_providers_status()
    for name, status_info in all_status.items():
        logger.info(f"Status para '{name}': {status_info}")
    logger.info("------------------------------------------\n")

    await manager.shutdown()
    logger.info("Aplicação de transcrição finalizada.")

if __name__ == "__main__":
    asyncio.run(main())