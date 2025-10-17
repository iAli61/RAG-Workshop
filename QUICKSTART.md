# 🚀 Quick Start Guide - RAG Workshop

## TL;DR - Get Started in 3 Commands

```bash
uv sync                       # Install all dependencies
source .venv/bin/activate     # Activate environment
jupyter notebook              # Start Jupyter
```

Then edit `.env` with your Azure OpenAI credentials.

**Alternative:** Use the automated setup script: `./setup.sh`

---

## What Was Created?

### ✅ Core Files
- **`pyproject.toml`** - All dependencies (28 packages)
- **`.env.example`** - Environment variable template
- **`setup.sh`** / **`setup.bat`** - Automated setup scripts

### ✅ Documentation
- **`SETUP.md`** - Complete setup guide
- **`UV_REFERENCE.md`** - UV command reference
- **`ENVIRONMENT_SETUP_COMPLETE.md`** - This setup summary

---

## Installation Methods

### Method 1: Automated (Recommended) ⭐

```bash
cd /home/alibina/repo/ITZ/RAG-Workshop
./setup.sh
```

This will:
1. Install UV (if needed)
2. Create virtual environment
3. Install all 28 packages
4. Create `.env` file from template

### Method 2: Manual

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install
uv venv
source .venv/bin/activate
uv pip install -e .

# Setup environment
cp .env.example .env
```

---

## Configure Azure OpenAI

Edit `.env` with your credentials:

```bash
nano .env
```

Required values:
```env
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
```

---

## Test Your Setup

```bash
python -c "
import llama_index
import sentence_transformers
import torch
import ragas
print('✅ All packages imported successfully!')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
"
```

---

## Start Working

```bash
# Activate environment
source .venv/bin/activate

# Start Jupyter
jupyter notebook

# Navigate to RAG_hf_v4/ folder
# Open any demo notebook (01-10)
```

---

## All Packages Included

✅ **LlamaIndex** - Complete RAG framework
✅ **Sentence Transformers** - Embeddings & cross-encoders
✅ **PyTorch** - Deep learning backend
✅ **RAGAS** - RAG evaluation metrics
✅ **BM25** - Keyword search
✅ **Web Search** - DuckDuckGo & ArXiv
✅ **Azure OpenAI** - LLM & embeddings
✅ **Jupyter** - Interactive notebooks
✅ **Pandas** - Data analysis
✅ **Matplotlib** - Visualization

---

## Workshop Demos Ready to Run

1. **Demo 01** - HyDE Query Enhancement
2. **Demo 02** - Multi-Query Decomposition
3. **Demo 03** - Hybrid Search (Semantic + BM25)
4. **Demo 04** - Hierarchical Retrieval
5. **Demo 05** - Cross-Encoder Reranking
6. **Demo 06** - Context Compression
7. **Demo 07** - Corrective RAG
8. **Demo 08** - Agentic RAG
9. **Demo 09** - Embedding Fine-Tuning
10. **Demo 10** - RAG Evaluation

---

## Common Commands

```bash
# Activate environment
source .venv/bin/activate

# Deactivate
deactivate

# Update packages
uv pip install --upgrade -e .

# Install new package
uv pip install package-name

# Start Jupyter
jupyter notebook

# Start JupyterLab
jupyter lab
```

---

## Troubleshooting

**Import errors?**
```bash
source .venv/bin/activate  # Make sure venv is active
which python  # Should show .venv/bin/python
```

**PyTorch CUDA issues?**
```bash
# Install CPU version
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Azure connection errors?**
- Check `.env` file has correct credentials
- Verify endpoint URL includes trailing slash
- Ensure deployment names match your Azure resource

---

## Need Help?

- 📖 Full guide: `SETUP.md`
- 📋 UV commands: `UV_REFERENCE.md`
- 📝 Complete summary: `ENVIRONMENT_SETUP_COMPLETE.md`

---

## Why This Setup?

⚡ **Fast** - UV is 10-100x faster than pip
🔒 **Reliable** - All dependencies pinned in pyproject.toml
🎯 **Complete** - Includes everything for all 10 demos
🔄 **Reproducible** - Same environment on any machine
📦 **Modern** - Uses latest Python packaging standards

---

**Ready to build advanced RAG systems? Start with `./setup.sh` and you're good to go! 🚀**
