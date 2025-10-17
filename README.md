# RAG Workshop - Complete Setup Guide

## ✅ Installation Complete!

Your environment has been successfully set up with **241 packages** including PyTorch with CUDA support!

## Quick Start

```bash
# 1. Sync dependencies (creates venv and installs packages)
uv sync

# 2. Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows

# 3. Configure Azure OpenAI
cp .env.example .env
nano .env  # Add your credentials

# 4. Start Jupyter
jupyter notebook
```

## What's Installed

### Core Framework
- **LlamaIndex 0.14.5** - Complete RAG framework
- **PyTorch 2.9.0+cu128** - Deep learning with CUDA support ✅
- **Sentence Transformers 5.1.1** - Embeddings & cross-encoders
- **RAGAS 0.3.7** - RAG evaluation metrics

### Azure Integration
- `llama-index-llms-azure-openai` - Azure OpenAI LLM
- `llama-index-embeddings-azure-openai` - Azure embeddings

### Advanced Features
- `llama-index-retrievers-bm25` - Keyword search
- `duckduckgo-search` - Web search
- `arxiv` - Academic paper search
- `datasets` - Dataset management

### Data & Visualization
- `pandas`, `numpy` - Data analysis
- `matplotlib` - Plotting
- `jupyter`, `jupyterlab` - Interactive notebooks

## Environment Configuration

Create a `.env` file with your Azure OpenAI credentials:

```env
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
```

## Workshop Demos

Navigate to `RAG_hf_v4/` for 10 advanced RAG demos:

1. **demo_01** - HyDE Query Enhancement
2. **demo_02** - Multi-Query Decomposition
3. **demo_03** - Hybrid Search (Semantic + BM25)
4. **demo_04** - Hierarchical Retrieval
5. **demo_05** - Cross-Encoder Reranking
6. **demo_06** - Context Compression
7. **demo_07** - Corrective RAG
8. **demo_08** - Agentic RAG
9. **demo_09** - Embedding Fine-Tuning
10. **demo_10** - RAG Evaluation

## Verify Installation

```bash
source .venv/bin/activate
python -c "
import torch
from llama_index.llms.azure_openai import AzureOpenAI
print('✅ Installation verified!')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

## Common Commands

```bash
# Sync/update all dependencies
uv sync

# Add a new package
uv pip install package-name

# List installed packages
uv pip list

# Start Jupyter Notebook
jupyter notebook

# Start JupyterLab
jupyter lab

# Deactivate environment
deactivate
```

## Troubleshooting

### Import Errors
Make sure the virtual environment is activated:
```bash
source .venv/bin/activate
which python  # Should show: .venv/bin/python
```

### CUDA Not Available
If you need CPU-only PyTorch:
```bash
uv pip uninstall torch
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Azure OpenAI Connection
- Verify `.env` file has correct credentials
- Check endpoint URL includes trailing slash
- Ensure deployment names match your Azure resource

## Project Structure

```
RAG-Workshop/
├── pyproject.toml              # Dependencies
├── .env.example                # Environment template
├── .venv/                      # Virtual environment
├── setup.sh / setup.bat        # Automated setup
├── RAG_hf_v4/                  # Workshop notebooks
│   ├── demo_01_*.ipynb
│   ├── demo_02_*.ipynb
│   └── ...
└── RAG_v2/data/                # Sample documents
```

## Documentation

- **INSTALLATION_SUCCESS.md** - Detailed installation report
- **QUICKSTART.md** - Ultra-fast start guide
- **SETUP.md** - Complete setup instructions
- **UV_REFERENCE.md** - UV command reference

## Why UV?

- ⚡ **10-100x faster** than pip
- 🔒 **Deterministic** dependency resolution
- 💾 **Efficient** shared package cache
- 🎯 **Compatible** drop-in replacement for pip

## System Requirements

- **Python**: 3.10+ (currently using 3.13.3)
- **Disk Space**: ~4GB for all packages
- **RAM**: 4-8GB recommended (for models)
- **GPU**: Optional but recommended (CUDA detected ✅)

## Getting Help

1. Check documentation files (SETUP.md, QUICKSTART.md)
2. Review individual notebook cells for package usage
3. Verify environment activation: `which python`
4. Check `.env` file for correct Azure credentials

## License

MIT License - See individual packages for their licenses.

---

**Ready to build advanced RAG systems? Start with `uv sync` and explore the demos! 🚀**
