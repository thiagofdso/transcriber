# tests/conftest.py
import asyncio
import gc
import logging
import pytest
import torch
from pathlib import Path
from typing import List, Dict

from transcription_library.core.config import settings

# Configuração de logging para testes
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

@pytest.fixture(scope="session")
def event_loop():
    """Cria um event loop para toda a sessão de testes."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="function")
def cleanup_gpu():
    """Fixture para limpeza de memória GPU após cada teste."""
    yield
    # Limpeza após o teste
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

@pytest.fixture(scope="session")
def sample_files():
    """
    Descobre arquivos de sample em vários formatos de áudio e vídeo.
    Executa uma vez por sessão de testes.
    """
    sample_files = {
        'audio': [],
        'video': []
    }

    # Formatos suportados
    audio_extensions = ['.wav', '.mp3', '.m4a', '.ogg', '.opus', '.flac']
    video_extensions = ['.mp4', '.webm', '.mov', '.avi', '.mkv']

    # Diretórios para procurar arquivos de sample
    search_dirs = [
        Path('.'),
        Path('./samples'),
        Path('./test_files'),
        Path('./data'),
        Path.home() / 'Downloads',
    ]

    logger = logging.getLogger('sample_discovery')

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        logger.info(f"Procurando arquivos sample em: {search_dir}")

        # Procurar arquivos que contenham "sample" no nome
        for file_path in search_dir.rglob('*sample*'):
            if file_path.is_file():
                suffix = file_path.suffix.lower()

                if suffix in audio_extensions:
                    sample_files['audio'].append(file_path)
                    logger.info(f"Arquivo de áudio encontrado: {file_path}")
                elif suffix in video_extensions:
                    sample_files['video'].append(file_path)
                    logger.info(f"Arquivo de vídeo encontrado: {file_path}")

    # Criar arquivos dummy se nenhum for encontrado
    if not sample_files['audio'] and not sample_files['video']:
        logger.warning("Nenhum arquivo sample encontrado. Criando arquivos dummy...")
        _create_dummy_samples()

        # Re-executar busca
        for search_dir in [Path('./samples')]:
            if search_dir.exists():
                for file_path in search_dir.rglob('*sample*'):
                    if file_path.is_file():
                        suffix = file_path.suffix.lower()
                        if suffix in audio_extensions:
                            sample_files['audio'].append(file_path)
                        elif suffix in video_extensions:
                            sample_files['video'].append(file_path)

    logger.info(f"Total encontrado - Áudio: {len(sample_files['audio'])}, Vídeo: {len(sample_files['video'])}")
    return sample_files

def _create_dummy_samples():
    """Cria arquivos dummy para teste se nenhum sample for encontrado."""
    samples_dir = Path('./samples')
    samples_dir.mkdir(exist_ok=True)

    dummy_files = [
        'sample.wav',
        'sample.mp3',
        'sample_portuguese.wav',
        'sample.mp4',
        'sample_video.webm'
    ]

    for filename in dummy_files:
        filepath = samples_dir / filename
        if not filepath.exists():
            with open(filepath, 'wb') as f:
                f.write(b'DUMMY_SAMPLE_FILE_FOR_TESTING')

@pytest.fixture
def audio_sample(sample_files):
    """Retorna um arquivo de áudio sample para teste."""
    if not sample_files['audio']:
        pytest.skip("Nenhum arquivo de áudio sample encontrado")
    return sample_files['audio'][0]

@pytest.fixture
def video_sample(sample_files):
    """Retorna um arquivo de vídeo sample para teste."""
    if not sample_files['video']:
        pytest.skip("Nenhum arquivo de vídeo sample encontrado")
    return sample_files['video'][0]

@pytest.fixture
def gemini_api_key():
    """Verifica se a API key do Gemini está configurada."""
    if not settings.GEMINI_API_KEY:
        pytest.skip("GEMINI_API_KEY não configurada")
    return settings.GEMINI_API_KEY

@pytest.fixture
def valid_video_sample(sample_files):
    """
    Retorna um arquivo de vídeo válido para Gemini (dentro do limite de tamanho).
    """
    if not sample_files['video']:
        pytest.skip("Nenhum arquivo de vídeo sample encontrado")

    for video_file in sample_files['video']:
        file_size_mb = video_file.stat().st_size / (1024 * 1024)
        if file_size_mb <= settings.MAX_VIDEO_SIZE_MB:
            return video_file

    pytest.skip(f"Nenhum arquivo de vídeo dentro do limite de {settings.MAX_VIDEO_SIZE_MB}MB")

# Marcadores pytest personalizados
def pytest_configure(config):
    """Configurações personalizadas do pytest."""
    config.addinivalue_line("markers", "slow: marca testes como lentos")
    config.addinivalue_line("markers", "gpu: marca testes que requerem GPU")
    config.addinivalue_line("markers", "api: marca testes que requerem API externa")

# Hooks do pytest
def pytest_collection_modifyitems(config, items):
    """Modifica itens de teste baseado em marcadores."""
    # Marcar testes que usam Gemini como API
    for item in items:
        if "gemini" in item.name.lower():
            item.add_marker(pytest.mark.api)

        # Marcar testes que podem ser lentos
        if any(provider in item.name.lower() for provider in ["distil", "faster"]):
            item.add_marker(pytest.mark.slow)