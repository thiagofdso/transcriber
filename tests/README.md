# Testes da Biblioteca de Transcrição

Este diretório contém testes unitários e de integração para a biblioteca de transcrição pluggável.

## Estrutura dos Testes

```
tests/
├── __init__.py                 # Inicialização do módulo de testes
├── conftest.py                 # Configurações e fixtures do pytest
├── test_distil_whisper.py      # Testes específicos do Distil-Whisper
├── test_faster_whisper.py      # Testes específicos do Faster-Whisper
├── test_gemini.py              # Testes específicos do Gemini
├── test_integration.py         # Testes de integração
└── README.md                   # Este arquivo
```

## Executando os Testes

### Pré-requisitos

1. **Instalar dependências:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Arquivos de Sample:**
   - Coloque arquivos de áudio/vídeo com "sample" no nome
   - Suportados: `.wav`, `.mp3`, `.m4a`, `.ogg`, `.opus`, `.flac` (áudio)
   - Suportados: `.mp4`, `.webm`, `.mov`, `.avi`, `.mkv` (vídeo)
   - Locais: `./samples/`, `./test_files/`, `./data/`, Downloads

3. **API Key Gemini (opcional):**
   ```bash
   export GEMINI_API_KEY="sua_chave_aqui"
   ```

### Comandos de Teste

#### Executar todos os testes:
```bash
pytest
```

#### Executar testes específicos:
```bash
# Apenas Distil-Whisper
pytest tests/test_distil_whisper.py

# Apenas Faster-Whisper
pytest tests/test_faster_whisper.py

# Apenas Gemini
pytest tests/test_gemini.py

# Apenas integração
pytest tests/test_integration.py
```

#### Executar por marcadores:
```bash
# Apenas testes rápidos (excluir lentos)
pytest -m "not slow"

# Apenas testes que não precisam de API
pytest -m "not api"

# Apenas testes de GPU
pytest -m "gpu"

# Apenas testes de integração
pytest -m "integration"
```

#### Executar com mais detalhes:
```bash
# Modo verboso
pytest -v

# Mostrar saída de logs
pytest -s

# Parar no primeiro erro
pytest -x

# Executar em paralelo (cuidado com GPU!)
pytest -n auto
```

## Marcadores de Teste

- **`@pytest.mark.slow`**: Testes que podem demorar mais de 30s
- **`@pytest.mark.gpu`**: Testes que verificam funcionalidades de GPU
- **`@pytest.mark.api`**: Testes que requerem API externa (Gemini)
- **`@pytest.mark.integration`**: Testes de integração entre componentes

## Fixtures Disponíveis

### Descoberta de Arquivos
- **`sample_files`**: Descobre todos os arquivos sample disponíveis
- **`audio_sample`**: Retorna um arquivo de áudio para teste
- **`video_sample`**: Retorna um arquivo de vídeo para teste
- **`valid_video_sample`**: Retorna vídeo válido para Gemini (tamanho OK)

### Configuração
- **`cleanup_gpu`**: Limpa memória GPU/VRAM após cada teste
- **`gemini_api_key`**: Verifica se API key do Gemini está disponível

### Provedores
- **`manager_with_distil`**: Manager com apenas Distil-Whisper
- **`manager_with_faster`**: Manager com apenas Faster-Whisper
- **`manager_with_gemini`**: Manager com apenas Gemini
- **`manager_with_all_providers`**: Manager com todos os provedores

## Limpeza de Memória

Os testes são projetados para limpar automaticamente a memória GPU entre execuções:

```python
def _cleanup_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
```

## Arquivos Dummy

Se nenhum arquivo sample for encontrado, o sistema criará automaticamente arquivos dummy em `./samples/` para permitir que os testes sejam executados (embora falhem na transcrição real).

## Problemas Comuns

### GPU Out of Memory
```bash
# Executar apenas testes rápidos
pytest -m "not slow"

# Executar um teste por vez
pytest -x
```

### API do Gemini
```bash
# Pular testes de API
pytest -m "not api"

# Verificar se API key está configurada
echo $GEMINI_API_KEY
```

### Arquivos não encontrados
```bash
# Criar pasta samples e adicionar arquivos
mkdir samples
# Copiar arquivos com "sample" no nome
```

## Exemplo de Execução

```bash
# Teste completo com relatório detalhado
pytest -v --tb=short

# Teste rápido para desenvolvimento
pytest -m "not slow and not api" -x

# Teste de produção (todos exceto lentos)
pytest -m "not slow" --maxfail=3
```