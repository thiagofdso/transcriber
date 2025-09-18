# Biblioteca de Transcrição Pluggável

Uma biblioteca Python flexível e modular para transcrição de áudio e vídeo usando múltiplos provedores de IA. Projetada com arquitetura pluggável que permite fácil extensão e troca entre diferentes modelos de transcrição.

## 🚀 Funcionalidades

- **🔌 Arquitetura Pluggável**: Troque facilmente entre diferentes provedores de transcrição
- **🔄 Fallback Inteligente**: Configure uma cadeia de provedores de fallback automático
- **💾 Cache Automático**: Evita reprocessamento de arquivos já transcritos
- **🎬 Suporte Multimídia**: Transcrição de arquivos de áudio e vídeo
- **⚡ Processamento Assíncrono**: Operações não-bloqueantes para melhor performance
- **📊 Metadados Detalhados**: Retorna confiança, tempo de processamento e segmentos com timestamp
- **🌐 Configuração Flexível**: Via variáveis de ambiente e arquivos de configuração

## 🤖 Provedores Suportados

### Distil-Whisper (PT-BR)
- **Modelo**: `freds0/distil-whisper-large-v3-ptbr`
- **Especialidade**: Otimizado para Português do Brasil
- **Suporte**: GPU/CPU, cache local

### Faster-Whisper
- **Implementação**: Otimizada do Whisper para alta performance
- **Modelos**: small, medium, large-v2, large-v3
- **Suporte**: GPU/CPU, múltiplos idiomas

### Gemini (Google AI)
- **Funcionalidades**: Transcrição de áudio e vídeo
- **Especialidades**: Reconhecimento de texto visual em vídeos
- **Requisitos**: API Key do Google AI Studio

## 📁 Estrutura do Projeto

```
transcription-library/
├── transcription_library/          # Pacote principal
│   ├── __init__.py                # Exportações principais
│   ├── core/                      # Módulos centrais
│   │   ├── interfaces.py          # Interface base e tipos
│   │   ├── manager.py             # Gerenciador de provedores
│   │   ├── config.py              # Configurações
│   │   └── utils.py               # Utilitários compartilhados
│   └── providers/                 # Implementações dos provedores
│       ├── distil_whisper_provider.py
│       ├── faster_whisper_provider.py
│       └── gemini_provider.py
├── tests/                         # Testes unitários e integração
│   ├── conftest.py               # Configurações pytest
│   ├── test_distil_whisper.py    # Testes Distil-Whisper
│   ├── test_faster_whisper.py    # Testes Faster-Whisper
│   ├── test_gemini.py            # Testes Gemini
│   └── test_integration.py       # Testes de integração
├── example.py                     # Exemplo de uso
├── pyproject.toml                 # Configuração do pacote
├── requirements.txt               # Dependências
├── pytest.ini                    # Configuração de testes
└── README.md                      # Esta documentação
```

## 🛠️ Instalação

### Pré-requisitos

1. **Python 3.8+**

2. **FFmpeg** (para análise de arquivos de mídia):
   ```bash
   # Ubuntu/Debian
   sudo apt install ffmpeg

   # macOS
   brew install ffmpeg

   # Windows
   # Baixar de https://ffmpeg.org/download.html
   ```

3. **PyTorch** (recomendado instalar antes para controlar versão GPU/CPU):
   ```bash
   # CPU apenas
   pip install torch

   # GPU (CUDA 12.1)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Chave API Gemini** (opcional):
   ```bash
   export GEMINI_API_KEY="sua_chave_api_aqui"
   ```

### Instalação da Biblioteca

#### Via GitHub (Recomendado):
```bash
pip install git+https://github.com/seu_usuario/transcription-library.git
```

#### Para Desenvolvimento:
```bash
git clone https://github.com/seu_usuario/transcription-library.git
cd transcription-library
pip install -r requirements.txt
```

## 🎯 Uso Rápido

### Exemplo Básico

```python
import asyncio
from pathlib import Path
from transcription_library.core.manager import TranscriptionManager
from transcription_library.providers.distil_whisper_provider import DistilWhisperProvider

async def main():
    # Criar manager e registrar provedor
    manager = TranscriptionManager()
    provider = DistilWhisperProvider()
    manager.register_provider(provider.get_name(), provider)

    # Transcrever arquivo
    audio_file = Path("meu_audio.wav")
    result = await manager.transcribe_audio(audio_file, language="pt")

    if result.error_message:
        print(f"Erro: {result.error_message}")
    else:
        print(f"Transcrição: {result.text}")
        print(f"Confiança: {result.confidence:.2f}")
        print(f"Modelo usado: {result.model_used}")

asyncio.run(main())
```

### Exemplo com Múltiplos Provedores e Fallback

```python
import asyncio
from transcription_library.core.manager import TranscriptionManager
from transcription_library.providers.distil_whisper_provider import DistilWhisperProvider
from transcription_library.providers.faster_whisper_provider import FasterWhisperProvider
from transcription_library.providers.gemini_provider import GeminiProvider
from transcription_library.core.config import settings

async def main():
    manager = TranscriptionManager()

    # Registrar múltiplos provedores
    manager.register_provider("distil-whisper-pt", DistilWhisperProvider())
    manager.register_provider("faster-whisper", FasterWhisperProvider())
    manager.register_provider("gemini-hybrid", GeminiProvider())

    # Configurar fallback
    settings.PRIMARY_PROVIDER = "distil-whisper-pt"
    settings.FALLBACK_PROVIDERS = ["faster-whisper", "gemini-hybrid"]
    settings.CONFIDENCE_THRESHOLD = 0.7

    # Transcrever (usará fallback se necessário)
    result = await manager.transcribe_audio("audio.mp3", language="pt")
    print(f"Resultado: {result.text}")
    print(f"Provedor usado: {result.model_used}")

asyncio.run(main())
```

### Transcrição de Vídeo com Gemini

```python
import asyncio
from transcription_library.core.manager import TranscriptionManager
from transcription_library.providers.gemini_provider import GeminiProvider

async def main():
    manager = TranscriptionManager()
    gemini = GeminiProvider()
    manager.register_provider(gemini.get_name(), gemini)

    # Transcrever vídeo (detecta automaticamente áudio vs vídeo)
    result = await manager.transcribe_audio("video.mp4", language="pt-BR")

    if not result.error_message:
        print("=== Resultado da Transcrição de Vídeo ===")
        print(result.text)

        # Gemini inclui marcadores estruturados:
        # [FALA]: transcrição da fala
        # [TEXTO]: textos visíveis na tela
        # [DESCRIÇÃO VISUAL]: descrição do conteúdo

asyncio.run(main())
```

## ⚙️ Configuração

### Variáveis de Ambiente

```bash
# Gemini API
export GEMINI_API_KEY="sua_chave_aqui"

# Caminhos personalizados (opcional)
export FFPROBE_PATH="/usr/local/bin/ffprobe"
export CUDNN_PATH="/usr/local/cuda/lib64"
```

### Configuração Programática

```python
from transcription_library.core.config import settings

# Configurar provedores
settings.PRIMARY_PROVIDER = "faster-whisper"
settings.FALLBACK_PROVIDERS = ["distil-whisper-pt", "gemini-hybrid"]
settings.CONFIDENCE_THRESHOLD = 0.8

# Configurar modelos
settings.FASTER_WHISPER_MODEL_SIZE = "large-v3"
settings.DISTIL_WHISPER_CACHE_DIR = "./meu_cache"
settings.MAX_VIDEO_SIZE_MB = 500
```

## 🧪 Testes

A biblioteca inclui uma suíte completa de testes unitários e de integração.

### Preparação para Testes

1. **Instalar dependências de teste:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Criar arquivos de sample:**
   - Coloque arquivos de áudio/vídeo com "sample" no nome
   - Formatos suportados: `.wav`, `.mp3`, `.m4a`, `.mp4`, `.webm`, etc.
   - Locais: `./samples/`, `./test_files/`, `./data/`, Downloads

3. **Configurar API Gemini (opcional):**
   ```bash
   export GEMINI_API_KEY="sua_chave"
   ```

### Executar Testes

```bash
# Todos os testes
pytest

# Testes específicos por provedor
pytest tests/test_distil_whisper.py
pytest tests/test_faster_whisper.py
pytest tests/test_gemini.py

# Testes de integração
pytest tests/test_integration.py

# Apenas testes rápidos (excluir lentos)
pytest -m "not slow"

# Apenas testes que não precisam de API
pytest -m "not api"

# Testes com relatório detalhado
pytest -v --tb=short
```

### Marcadores de Teste

- `@pytest.mark.slow` - Testes que podem demorar >30s
- `@pytest.mark.gpu` - Testes que verificam GPU/CUDA
- `@pytest.mark.api` - Testes que requerem API externa
- `@pytest.mark.integration` - Testes de integração

### Limpeza de Memória

Os testes incluem limpeza automática de memória GPU entre execuções:

```python
@pytest.fixture
def cleanup_gpu():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
```

## 📊 Monitoramento e Status

```python
# Verificar status de todos os provedores
all_status = manager.get_all_providers_status()
for name, status in all_status.items():
    print(f"{name}: {status}")

# Status específico
distil_status = manager.get_provider_status("distil-whisper-pt")
print(f"GPU Memory: {distil_status.get('cuda_available')}")
print(f"Cache Size: {distil_status.get('cache_size')}")

# Limpar caches
manager.clear_all_caches()
```

## 🔧 Desenvolvimento

### Adicionando Novos Provedores

1. **Implemente a interface:**
   ```python
   from transcription_library.core.interfaces import ITranscriptionProvider

   class MeuProvedor(ITranscriptionProvider):
       async def initialize(self) -> bool:
           # Carregar modelo/configurar API

       async def transcribe(self, audio_path, language) -> TranscriptionResult:
           # Implementar transcrição

       def get_name(self) -> str:
           return "meu-provedor"
   ```

2. **Registre o provedor:**
   ```python
   provider = MeuProvedor()
   manager.register_provider(provider.get_name(), provider)
   ```

### Estrutura de Resultado

```python
@dataclass
class TranscriptionResult:
    text: str                          # Texto transcrito
    confidence: float                  # Confiança (0.0-1.0)
    processing_time: float             # Tempo em segundos
    model_used: str                    # Nome do modelo
    language: str                      # Idioma detectado/usado
    segments: Optional[List[Dict]]     # Segmentos com timestamp
    error_message: Optional[str]       # Mensagem de erro
```

## 🐛 Problemas Comuns

### Erro de Memória GPU
```python
# Usar CPU forçadamente
settings.FORCE_CPU_FOR_DISTIL_WHISPER = True
settings.FASTER_WHISPER_FORCE_CPU = True
```

### FFprobe não encontrado
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# Ou configurar caminho manualmente
export FFPROBE_PATH="/usr/local/bin/ffprobe"
```

### Erro na API Gemini
```bash
# Verificar chave
echo $GEMINI_API_KEY

# Verificar limites de arquivo
# Máximo padrão: 200MB para vídeos
```

## 📝 Contribuição

1. Fork o repositório
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Add nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

### Executar Testes Antes de Contribuir

```bash
# Testes rápidos para desenvolvimento
pytest -m "not slow and not api" -x

# Suite completa (se tiver recursos)
pytest -v
```

## 📄 Licença

Este projeto está licenciado sob a [Licença MIT](LICENSE).

## 🙏 Agradecimentos

- [Hugging Face Transformers](https://github.com/huggingface/transformers) - Distil-Whisper
- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) - Implementação otimizada
- [Google AI](https://ai.google.dev/) - API Gemini
- [OpenAI](https://openai.com/) - Modelo Whisper original

---

**Feito com ❤️ para a comunidade de desenvolvedores**