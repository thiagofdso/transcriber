# transcription_library/core/utils.py
import asyncio
import hashlib
import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

async def get_file_hash(file_path: Path) -> str:
    """Gera um hash MD5 de um arquivo para uso em cache."""
    hash_md5 = hashlib.md5()
    loop = asyncio.get_event_loop()

    def _hash_file_sync():
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    return await loop.run_in_executor(None, _hash_file_sync)

async def get_audio_duration(audio_path: Path) -> float:
    """Obtém a duração de um arquivo de áudio usando ffprobe."""
    try:
        command = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(audio_path)
        ]
        loop = asyncio.get_event_loop()
        process = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                encoding="utf-8"
            )
        )

        metadata = json.loads(process.stdout)

        if 'format' in metadata and 'duration' in metadata['format']:
            return float(metadata['format']['duration'])

        # Fallback para streams, se a duração não estiver no formato
        if 'streams' in metadata:
            for stream in metadata['streams']:
                if stream.get('codec_type') == 'audio' and 'duration' in stream:
                    return float(stream['duration'])

        logger.warning(f"Could not get audio duration for: {audio_path}")
        return 0.0

    except Exception as e:
        logger.error(f"Error getting audio duration for {audio_path}: {e}")
        return 0.0