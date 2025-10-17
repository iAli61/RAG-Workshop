# RAG Workshop - Environment Setup

## Quick Start with UV

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

### 1. Install UV (if not already installed)

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Create Virtual Environment and Install Dependencies

```bash
# Create a virtual environment
uv venv

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate

# Install all dependencies from pyproject.toml
uv pip install -e .
```

### 3. Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your Azure OpenAI credentials
nano .env  # or use your preferred editor
```

### 4. Run Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab

# Navigate to RAG_hf_v4/ to access the demo notebooks
```

## Project Structure

```
RAG-Workshop/
├── pyproject.toml          # Project dependencies and configuration
├── .env.example            # Example environment variables
├── RAG_hf_v4/              # Latest workshop demos
│   ├── demo_01_hyde_query_enhancement.ipynb
│   ├── demo_02_multi_query_decomposition.ipynb
│   ├── demo_03_hybrid_search.ipynb
│   ├── demo_04_hierarchical_retrieval.ipynb
│   ├── demo_05_reranking_cross_encoders.ipynb
│   ├── demo_06_context_compression.ipynb
│   ├── demo_07_corrective_rag.ipynb
│   ├── demo_08_agentic_rag.ipynb
│   ├── demo_09_embedding_finetuning.ipynb
│   └── demo_10_rag_evaluation.ipynb
└── RAG_v2/
    └── data/               # Sample documents for testing

```

## Key Dependencies

### Core RAG Framework
- **llama-index**: Complete RAG framework with extensive integrations
- **sentence-transformers**: Embedding models and cross-encoders
- **torch**: Deep learning framework (required for models)

### Azure Integration
- **llama-index-llms-azure-openai**: Azure OpenAI LLM integration
- **llama-index-embeddings-azure-openai**: Azure OpenAI embeddings

### Advanced Retrieval
- **llama-index-retrievers-bm25**: BM25 keyword search
- **llama-index-postprocessor-longcontextreorder**: Context reordering

### Evaluation
- **ragas**: RAG-specific evaluation metrics
- **datasets**: Dataset management for evaluation

### Utilities
- **duckduckgo-search**: Web search capabilities
- **arxiv**: Academic paper search
- **python-dotenv**: Environment variable management

## Alternative: Install with pip

If you prefer using pip instead of uv:

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Development Dependencies

To install development tools (linting, formatting, testing):

```bash
uv pip install -e ".[dev]"
```

## Troubleshooting

### PyTorch Installation Issues

If you encounter PyTorch installation issues, install it separately first:

```bash
# For CPU only
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8
uv pip install torch --index-url https://download.pytorch.org/whl/cu118

# Then install the rest
uv pip install -e .
```

### Azure OpenAI Connection Issues

Ensure your `.env` file has valid credentials:
- API key is active
- Endpoint URL is correct (include the trailing slash)
- Deployment names match your Azure OpenAI resource

### Import Errors

If you see import errors, ensure the virtual environment is activated:

```bash
which python  # Should show .venv/bin/python
```

## Updating Dependencies

To update all dependencies:

```bash
uv pip install --upgrade -e .
```

## License

MIT License - See LICENSE file for details
