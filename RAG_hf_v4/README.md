# Advanced RAG Workshop - Demo Notebooks

Welcome to the **Advanced RAG Workshop**! This directory contains 10 progressive, hands-on Jupyter notebooks that demonstrate state-of-the-art Retrieval-Augmented Generation (RAG) techniques using LlamaIndex and Azure OpenAI.

## 📚 Overview

These demos build from foundational RAG enhancements through advanced self-correcting systems to autonomous agents and systematic evaluation. Each notebook is self-contained with comprehensive explanations, implementation code, and comparative analysis.

## 🎯 Workshop Objectives

By completing these demos, you will:
- Master 10+ advanced RAG techniques with practical implementations
- Understand the modular RAG paradigm (Pre-Retrieval → Retrieval → Post-Retrieval)
- Build self-correcting and agentic RAG systems
- Learn systematic evaluation and optimization strategies
- Gain production-ready code for real-world applications

---

## 📖 Demo Catalog

### **Foundation: Pre-Retrieval Optimization (Demos 1-2)**

#### 🔍 Demo #1: HyDE (Hypothetical Document Embeddings)
**File:** `demo_01_hyde_query_enhancement.ipynb`

Demonstrates how generating a hypothetical answer document before retrieval dramatically improves semantic matching.

**Key Concepts:**
- Query enhancement via hypothetical documents
- Answer-to-answer similarity search paradigm
- Bridging the query-document semantic gap

**What You'll Learn:**
- Generate hypothetical documents with LLMs
- Transform search from query-to-document to answer-to-answer
- Compare baseline vs. HyDE-enhanced retrieval

---

#### 🧩 Demo #2: Multi-Query Decomposition
**File:** `demo_02_multi_query_decomposition.ipynb`

Shows how decomposing complex queries into simpler sub-questions enables comprehensive information synthesis.

**Key Concepts:**
- Sub-query decomposition for multi-hop reasoning
- Parallel retrieval execution
- Context aggregation from multiple passes

**What You'll Learn:**
- Break down complex queries automatically
- Execute parallel retrieval operations
- Synthesize information from multiple sources

---

### **Advanced: Retrieval Enhancement (Demos 3-4)**

#### 🔎 Demo #3: Hybrid Search
**File:** `demo_03_hybrid_search.ipynb`

Combines dense vector search with sparse keyword retrieval (BM25) for improved precision across diverse query types.

**Key Concepts:**
- Hybrid search (dense + sparse vectors)
- BM25 keyword matching
- Reciprocal Rank Fusion (RRF)

**What You'll Learn:**
- Implement BM25 sparse retrieval
- Combine dense and sparse results with RRF
- Optimize for different query types (exact match, conceptual, mixed)

---

#### 📂 Demo #4: Hierarchical Retrieval (Parent-Child Chunking)
**File:** `demo_04_hierarchical_retrieval.ipynb`

Solves the chunking trade-off by retrieving with small, precise chunks while generating with larger, context-rich chunks.

**Key Concepts:**
- Parent-child document hierarchy
- Precision in retrieval, richness in generation
- Solving the chunk size dilemma

**What You'll Learn:**
- Create parent-child document structures
- Build custom hierarchical retrievers
- Balance precision and context effectively

---

### **Refinement: Post-Retrieval Enhancement (Demos 5-6)**

#### ⚡ Demo #5: Re-Ranking with Cross-Encoders
**File:** `demo_05_reranking_cross_encoders.ipynb`

Demonstrates two-stage retrieval: fast bi-encoder for initial retrieval + accurate cross-encoder for re-ranking.

**Key Concepts:**
- Two-stage retrieval architecture
- Bi-encoder vs. cross-encoder comparison
- Post-retrieval precision optimization

**What You'll Learn:**
- Implement cross-encoder re-ranking
- Optimize latency vs. accuracy trade-offs
- Benchmark performance improvements

---

#### 🗜️ Demo #6: Context Compression
**File:** `demo_06_context_compression.ipynb`

Shows how strategic reordering and sentence-level pruning optimize LLM generation by addressing "lost in the middle."

**Key Concepts:**
- "Lost in the middle" problem
- Strategic context reordering
- Extractive sentence-level compression

**What You'll Learn:**
- Implement context reordering strategies
- Build sentence-level compressors
- Achieve 30-40% token reduction with quality preservation

---

### **Frontier: Self-Correction and Autonomy (Demos 7-8)**

#### 🔄 Demo #7: Corrective RAG (CRAG)
**File:** `demo_07_corrective_rag.ipynb`

Implements self-reflective systems that evaluate retrieval quality and trigger corrective actions (web search) when needed.

**Key Concepts:**
- LLM-based retrieval evaluation
- Dynamic routing (high/low/ambiguous confidence)
- Web search fallback integration
- Sentence-level knowledge filtering

**What You'll Learn:**
- Build self-evaluating retrieval systems
- Implement dynamic routing logic
- Integrate external search APIs (DuckDuckGo)
- Filter and refine retrieved knowledge

---

#### 🤖 Demo #8: Agentic RAG
**File:** `demo_08_agentic_rag.ipynb`

Demonstrates autonomous agents that dynamically plan retrieval strategies and select appropriate tools.

**Key Concepts:**
- ReAct framework (Thought → Action → Observation)
- Multi-tool orchestration
- Autonomous query planning
- Multi-step reasoning

**What You'll Learn:**
- Build ReAct agents with LlamaIndex
- Integrate multiple knowledge sources
- Add external tools (web search, arXiv)
- Handle complex multi-hop queries

---

### **Optimization: Fine-Tuning and Evaluation (Demos 9-10)**

#### 🎯 Demo #9: Fine-Tuning Embedding Models
**File:** `demo_09_embedding_finetuning.ipynb`

Shows how domain-specific fine-tuning of embedding models significantly improves retrieval accuracy.

**Key Concepts:**
- Contrastive learning with Triplet Loss
- Domain adaptation for specialized terminology
- Evaluation metrics (Recall@K, MRR)

**What You'll Learn:**
- Create training triplets (query, positive, negative)
- Fine-tune sentence-transformers models
- Evaluate with Recall@K and MRR
- Compare fine-tuned vs. base embeddings in RAG

---

#### 📊 Demo #10: RAG Evaluation with RAGAS
**File:** `demo_10_rag_evaluation.ipynb`

Implements comprehensive evaluation framework with quantitative metrics to systematically measure and improve RAG systems.

**Key Concepts:**
- RAGAS metrics (Context Precision, Context Recall, Faithfulness, Answer Relevancy)
- LLM-as-judge pattern
- Bottleneck identification
- Iterative improvement workflow

**What You'll Learn:**
- Set up RAGAS evaluation framework
- Create test sets with ground truth
- Analyze per-question failure modes
- Map bottlenecks to specific improvements

---

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.9 or higher
- Azure OpenAI account with:
  - GPT-4 deployment (for generation)
  - text-embedding-ada-002 deployment (for embeddings)
- 4GB+ RAM recommended
- Internet connection (for external APIs in demos 7-8)

### Step 1: Clone the Repository

```bash
git clone https://github.com/iAli61/RAG-Workshop.git
cd RAG-Workshop/RAG_hf_v4
```

### Step 2: Install uv (Recommended Package Manager)

`uv` is a fast, modern Python package installer. It's significantly faster than pip and handles virtual environments seamlessly.

**Install uv:**

```bash
# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative: Install via pip
pip install uv
```

**Verify installation:**

```bash
uv --version
```

### Step 3: Create Virtual Environment

**Option A: Using uv (Recommended)**

```bash
# Create and activate virtual environment with uv
uv venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

**Option B: Using standard Python**

```bash
# Create virtual environment
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### Step 4: Install Dependencies

**Option A: Using uv (Recommended - Much Faster)**

**Core Dependencies (Required for all demos):**

```bash
uv pip install llama-index llama-index-llms-azure-openai llama-index-embeddings-azure-openai
uv pip install python-dotenv pandas numpy matplotlib
```

**Demo-Specific Dependencies:**

```bash
# Demo 3: Hybrid Search
uv pip install llama-index-retrievers-bm25 rank-bm25

# Demo 5: Re-Ranking
uv pip install sentence-transformers

# Demo 7: Corrective RAG
uv pip install duckduckgo-search

# Demo 8: Agentic RAG
uv pip install duckduckgo-search arxiv

# Demo 9: Embedding Fine-Tuning
uv pip install sentence-transformers torch scikit-learn

# Demo 10: RAG Evaluation
uv pip install ragas datasets langchain-openai
```

**Install All at Once (Recommended):**

```bash
uv pip install llama-index llama-index-llms-azure-openai llama-index-embeddings-azure-openai \
    llama-index-retrievers-bm25 rank-bm25 \
    python-dotenv pandas numpy matplotlib \
    sentence-transformers torch scikit-learn \
    duckduckgo-search arxiv \
    ragas datasets langchain-openai
```

**Option B: Using standard pip**

```bash
pip install llama-index llama-index-llms-azure-openai llama-index-embeddings-azure-openai \
    llama-index-retrievers-bm25 rank-bm25 \
    python-dotenv pandas numpy matplotlib \
    sentence-transformers torch scikit-learn \
    duckduckgo-search arxiv \
    ragas datasets langchain-openai
```

> **💡 Pro Tip:** `uv` is 10-100x faster than pip and provides better dependency resolution. Highly recommended for workshop setup!

### Step 4: Configure Azure OpenAI

Create a `.env` file in the `RAG_hf_v4` directory:

```bash
# .env
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=your_gpt4_deployment_name
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=your_embedding_deployment_name
```

**How to get these values:**
1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to your Azure OpenAI resource
3. Find "Keys and Endpoint" in the left sidebar
4. Copy the key and endpoint
5. Navigate to "Model deployments" to find your deployment names

### Step 5: Launch Jupyter

```bash
jupyter notebook
```

Or use VS Code with the Jupyter extension.

---

## 🚀 Running the Demos

### Recommended Learning Path

**Beginners:** Start with demos 1-6 in order
- These build foundational understanding progressively
- Each demo introduces 1-2 core concepts
- Clear comparisons with baseline approaches

**Intermediate:** Explore demos 7-8
- Self-correcting and agentic systems
- Integration of multiple techniques
- Complex multi-step reasoning

**Advanced:** Dive into demos 9-10
- Fine-tuning and optimization
- Systematic evaluation frameworks
- Production deployment strategies

### Running Individual Demos

Each notebook is self-contained. Simply:

1. Open the notebook in Jupyter
2. Run all cells from top to bottom (`Cell` → `Run All`)
3. Read the markdown explanations between code cells
4. Experiment with the test queries provided
5. Modify parameters to see how behavior changes

### Expected Runtime

- **Demos 1-6:** 5-10 minutes each
- **Demo 7 (CRAG):** 10-15 minutes (includes web search)
- **Demo 8 (Agentic):** 15-20 minutes (complex reasoning)
- **Demo 9 (Fine-Tuning):** 20-30 minutes (model training)
- **Demo 10 (Evaluation):** 10-15 minutes (multiple metrics)

---

## 📂 Data

The demos use a shared data directory (`../RAG_v2/data/`) containing:

- **ml_concepts/**: Machine learning documentation (5 files)
  - Neural networks, random forests, SVMs, gradient boosting, k-means
- **tech_docs/**: Technical documentation (6 files)
  - BERT, GPT-4, Transformers, Docker, REST APIs, Embeddings
- **finance_docs/**: Financial concepts (5 files)
  - Portfolio management, risk analysis, quantitative trading
- **long_form_docs/**: Extended articles (3 files)
  - RAG guides, embedding deep-dives, chunking strategies

This curated dataset provides diverse content for demonstrating different RAG techniques.

---

## 🎓 Educational Features

Each notebook includes:

✅ **Clear Overview** - Explains the technique and its purpose  
✅ **Key Concepts** - Highlights core ideas being demonstrated  
✅ **Step-by-Step Implementation** - Detailed code with explanations  
✅ **Comparative Analysis** - Shows improvements over baseline  
✅ **Visualizations** - Diagrams, charts, data flow illustrations  
✅ **Citations** - References to academic papers and sources  
✅ **Production Considerations** - Real-world deployment advice  
✅ **Key Takeaways** - Summary of lessons learned

---

## 🔧 Troubleshooting

### Common Issues

**Issue: `ModuleNotFoundError`**
```bash
# Solution: Install missing dependencies
pip install <missing-module>
```

**Issue: Azure OpenAI Authentication Error**
```bash
# Solution: Verify .env file configuration
# Ensure no extra spaces in .env file
# Check that API key is valid and active
```

**Issue: Rate Limiting**
```bash
# Solution: Azure OpenAI has rate limits
# Add time.sleep(1) between API calls if needed
# Or use a higher tier Azure subscription
```

**Issue: Out of Memory**
```bash
# Solution: Reduce batch sizes in fine-tuning (Demo 9)
# Use smaller embedding models
# Close other applications
```

**Issue: Web Search Not Working (Demo 7-8)**
```bash
# Solution: Check internet connection
# DuckDuckGo may occasionally be slow
# Consider using alternative search APIs
```

---

## 📊 Workshop Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    ADVANCED RAG WORKSHOP                        │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────┐
│  FOUNDATION (Demos 1-2)              │
│  • HyDE Query Enhancement            │
│  • Multi-Query Decomposition         │
└──────────────┬───────────────────────┘
               │ Query Optimization
               ▼
┌──────────────────────────────────────┐
│  ADVANCED RETRIEVAL (Demos 3-4)      │
│  • Hybrid Search (Dense + Sparse)    │
│  • Hierarchical Retrieval            │
└──────────────┬───────────────────────┘
               │ Better Search & Chunking
               ▼
┌──────────────────────────────────────┐
│  POST-RETRIEVAL (Demos 5-6)          │
│  • Cross-Encoder Re-Ranking          │
│  • Context Compression               │
└──────────────┬───────────────────────┘
               │ Refinement
               ▼
┌──────────────────────────────────────┐
│  SELF-CORRECTION (Demo 7)            │
│  • Corrective RAG (CRAG)             │
│  • Dynamic Routing & Web Search      │
└──────────────┬───────────────────────┘
               │ Adaptive Systems
               ▼
┌──────────────────────────────────────┐
│  AUTONOMY (Demo 8)                   │
│  • Agentic RAG                       │
│  • Multi-Tool Orchestration          │
└──────────────┬───────────────────────┘
               │ Intelligent Agents
               ▼
┌──────────────────────────────────────┐
│  OPTIMIZATION (Demos 9-10)           │
│  • Embedding Fine-Tuning             │
│  • RAGAS Evaluation Framework        │
└──────────────────────────────────────┘
```

---

## 📚 Additional Resources

### Documentation
- **LlamaIndex Docs:** https://docs.llamaindex.ai/
- **Azure OpenAI:** https://learn.microsoft.com/en-us/azure/ai-services/openai/
- **RAGAS Framework:** https://docs.ragas.io/
- **Sentence Transformers:** https://www.sbert.net/

### Academic Papers

Each demo includes citations to relevant research:
- Demo 1: HyDE - "Precise Zero-Shot Dense Retrieval without Relevance Labels"
- Demo 2: "MultiHop-RAG: Benchmarking Retrieval-Augmented Generation"
- Demo 7: "Corrective Retrieval Augmented Generation" (arXiv:2401.15884)
- Demo 8: "Agentic Retrieval-Augmented Generation: A Survey"
- Demo 10: RAGAS - "RAG Assessment Framework"

### Hugging Face Models

Recommended models for experimentation:
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2, BAAI/bge-base-en-v1.5
- **Rerankers:** cross-encoder/ms-marco-MiniLM-L6-v2
- **LLMs:** microsoft/phi-2, mistralai/Mistral-7B-v0.1 (if not using Azure)

---

## 🤝 Contributing

Found an issue or have suggestions? Contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📄 License

This workshop material is provided for educational purposes. Please refer to the repository's LICENSE file for details.

---

## 🙏 Acknowledgments

This workshop implementation leverages:
- **LlamaIndex** - Powerful RAG framework
- **Azure OpenAI** - State-of-the-art LLMs and embeddings
- **Sentence Transformers** - High-quality embedding models
- **RAGAS** - Comprehensive RAG evaluation framework
- **Academic Research** - Citations throughout demos

---

## 📧 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the individual notebook's markdown explanations
3. Consult the `workshop_demo_plan.md` for implementation details
4. Open an issue in the GitHub repository

---

**Happy Learning! 🚀**

*Build production-ready RAG systems with confidence.*

---

**Last Updated:** October 16, 2025  
**Version:** 1.0  
**Demos:** 10 complete implementations  
**Status:** Ready for workshop delivery
