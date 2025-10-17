# RAG Workshop - Environment Setup Summary

## Files Created

### 1. `pyproject.toml` ✅
The main project configuration file with all dependencies:

**Key Dependencies Included:**
- Core: `llama-index`, `llama-index-core`
- Azure Integration: `llama-index-llms-azure-openai`, `llama-index-embeddings-azure-openai`
- Models: `sentence-transformers`, `torch`
- Retrievers: `llama-index-retrievers-bm25`
- Postprocessors: `llama-index-postprocessor-longcontextreorder`
- Evaluation: `ragas`, `datasets`
- Web Tools: `duckduckgo-search`, `arxiv`
- Data: `pandas`, `numpy`, `matplotlib`
- Utilities: `python-dotenv`
- Jupyter: `ipykernel`, `jupyter`, `notebook`

### 2. `.env.example` ✅
Template for environment variables with all required Azure OpenAI settings.

### 3. `SETUP.md` ✅
Comprehensive setup instructions including:
- UV installation instructions
- Virtual environment creation
- Dependency installation
- Environment configuration
- Troubleshooting guide
- Project structure overview

### 4. `setup.sh` ✅ (Linux/macOS)
Automated setup script that:
- Checks for UV installation
- Installs UV if needed
- Creates virtual environment
- Installs all dependencies
- Sets up .env file

### 5. `setup.bat` ✅ (Windows)
Windows version of the automated setup script.

### 6. `UV_REFERENCE.md` ✅
Quick reference guide for UV commands including:
- Common commands
- Dependency management
- Troubleshooting
- Best practices

## How to Use

### Option 1: Automated Setup (Recommended)

**Linux/macOS:**
```bash
cd /home/alibina/repo/ITZ/RAG-Workshop
./setup.sh
```

**Windows:**
```cmd
cd \path\to\RAG-Workshop
setup.bat
```

### Option 2: Manual Setup

```bash
# 1. Install UV (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create virtual environment
uv venv

# 3. Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows

# 4. Install dependencies
uv pip install -e .

# 5. Configure environment
cp .env.example .env
nano .env  # Add your Azure OpenAI credentials

# 6. Start Jupyter
jupyter notebook
```

## Next Steps

1. **Edit `.env`** with your Azure OpenAI credentials:
   - AZURE_OPENAI_API_KEY
   - AZURE_OPENAI_ENDPOINT
   - AZURE_OPENAI_DEPLOYMENT_NAME
   - AZURE_OPENAI_EMBEDDING_DEPLOYMENT

2. **Navigate to RAG_hf_v4/** to access the workshop demos:
   - demo_01: HyDE Query Enhancement
   - demo_02: Multi-Query Decomposition
   - demo_03: Hybrid Search
   - demo_04: Hierarchical Retrieval
   - demo_05: Reranking with Cross-Encoders
   - demo_06: Context Compression
   - demo_07: Corrective RAG
   - demo_08: Agentic RAG
   - demo_09: Embedding Fine-tuning
   - demo_10: RAG Evaluation

3. **Run notebooks** in order or jump to specific techniques you want to explore.

## Testing Your Setup

Run this in a Python shell after activating the environment:

```python
# Test imports
import llama_index
import sentence_transformers
import torch
import ragas
import pandas as pd
import numpy as np

print("✅ All core packages imported successfully")

# Check PyTorch device
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Benefits of This Setup

✅ **Fast Installation**: UV is 10-100x faster than pip
✅ **Reproducible**: All dependencies specified in pyproject.toml
✅ **Automated**: Setup scripts handle everything
✅ **Cross-Platform**: Works on Windows, Linux, and macOS
✅ **Well-Documented**: Multiple reference documents
✅ **Production-Ready**: Includes all necessary packages

## Support

- Refer to `SETUP.md` for detailed instructions
- Check `UV_REFERENCE.md` for command reference
- Review individual notebook cells for package-specific issues

## All Dependencies Covered

This setup includes everything needed for all 10 workshop demos:
- ✅ HyDE query enhancement
- ✅ Multi-query decomposition  
- ✅ Hybrid search (semantic + BM25)
- ✅ Hierarchical retrieval
- ✅ Cross-encoder reranking
- ✅ Context compression
- ✅ Corrective RAG with web search
- ✅ Agentic RAG
- ✅ Embedding fine-tuning
- ✅ RAG evaluation with RAGAS
