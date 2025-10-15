# Hugging Face Resources for Advanced RAG Workshop

This document provides a comprehensive collection of models, datasets, papers, and documentation from Hugging Face to enhance the Advanced RAG Workshop.

## Table of Contents
1. [Top Embedding Models](#top-embedding-models)
2. [Cross-Encoder Reranking Models](#cross-encoder-reranking-models)
3. [Essential Research Papers](#essential-research-papers)
4. [Demo-Specific Recommendations](#demo-specific-recommendations)
5. [Datasets for Evaluation](#datasets-for-evaluation)
6. [Documentation and Guides](#documentation-and-guides)

---

## Top Embedding Models

### Production-Ready Models

1. **sentence-transformers/all-MiniLM-L6-v2** ⭐ Most Popular
   - **Downloads**: 114.5M
   - **Use Case**: Fast, efficient embeddings for prototyping
   - **Context Length**: 256 tokens
   - **Link**: https://hf.co/sentence-transformers/all-MiniLM-L6-v2

2. **sentence-transformers/all-mpnet-base-v2**
   - **Downloads**: 17.7M
   - **Use Case**: Higher quality embeddings with better semantic understanding
   - **Context Length**: 384 tokens
   - **Link**: https://hf.co/sentence-transformers/all-mpnet-base-v2

3. **BAAI/bge-base-en-v1.5**
   - **Downloads**: 5.0M
   - **Use Case**: Strong performance on MTEB benchmark
   - **Features**: Optimized for retrieval tasks
   - **Link**: https://hf.co/BAAI/bge-base-en-v1.5

4. **BAAI/bge-m3** ⭐ Hybrid Search
   - **Downloads**: 6.0M
   - **Use Case**: Native support for dense, sparse, and hybrid search
   - **Features**: Multilingual, supports multiple retrieval modes
   - **Link**: https://hf.co/BAAI/bge-m3
   - **Paper**: https://hf.co/papers/2402.03216

5. **jinaai/jina-embeddings-v3**
   - **Downloads**: 5.3M
   - **Use Case**: Multilingual, long context support
   - **Context Length**: Up to 8192 tokens
   - **Link**: https://hf.co/jinaai/jina-embeddings-v3

### Specialized Models

6. **sentence-transformers/multi-qa-mpnet-base-dot-v1**
   - **Downloads**: 2.0M
   - **Use Case**: Optimized for question-answering tasks
   - **Link**: https://hf.co/sentence-transformers/multi-qa-mpnet-base-dot-v1

7. **Alibaba-NLP/gte-large-en-v1.5**
   - **Downloads**: 2.5M
   - **Use Case**: Extended context length (up to 8192 tokens)
   - **Link**: https://hf.co/Alibaba-NLP/gte-large-en-v1.5

---

## Cross-Encoder Reranking Models

### Most Popular Rerankers

1. **cross-encoder/ms-marco-MiniLM-L6-v2** ⭐ Most Popular
   - **Downloads**: 4.9M
   - **Use Case**: Standard reranking for English text
   - **Speed**: Fast inference
   - **Link**: https://hf.co/cross-encoder/ms-marco-MiniLM-L6-v2

2. **BAAI/bge-reranker-v2-m3**
   - **Downloads**: 2.6M
   - **Use Case**: Multilingual reranking
   - **Link**: https://hf.co/BAAI/bge-reranker-v2-m3

3. **jinaai/jina-reranker-v2-base-multilingual**
   - **Downloads**: 1.1M
   - **Use Case**: Multilingual, cross-encoder reranking
   - **Link**: https://hf.co/jinaai/jina-reranker-v2-base-multilingual

4. **Qwen/Qwen3-Reranker-0.6B** ⭐ Modern Architecture
   - **Downloads**: 930.5K
   - **Use Case**: Efficient, modern architecture with strong performance
   - **Link**: https://hf.co/Qwen/Qwen3-Reranker-0.6B

### Faster/Smaller Options

5. **cross-encoder/ms-marco-MiniLM-L4-v2**
   - **Downloads**: 1.3M
   - **Use Case**: Faster variant with 4 layers
   - **Link**: https://hf.co/cross-encoder/ms-marco-MiniLM-L4-v2

6. **mixedbread-ai/mxbai-rerank-xsmall-v1**
   - **Downloads**: 962.9K
   - **Use Case**: Extra small model for efficiency
   - **Link**: https://hf.co/mixedbread-ai/mxbai-rerank-xsmall-v1

---

## Essential Research Papers

### Core RAG Foundations

1. **HyDE: Precise Zero-Shot Dense Retrieval without Relevance Labels** (2022)
   - **arXiv**: 2212.10496
   - **Key Concept**: Hypothetical Document Embeddings
   - **Link**: https://hf.co/papers/2212.10496
   - **Application**: Demo #1 - Query Enhancement

2. **Retrieval Augmented Generation (RAG) and Beyond: Comprehensive Survey** (2024)
   - **arXiv**: 2409.14924
   - **Coverage**: RAG taxonomy, integration strategies, future directions
   - **Link**: https://hf.co/papers/2409.14924

3. **ARAGOG: Advanced RAG Output Grading** (2024)
   - **arXiv**: 2404.01037
   - **Key Finding**: Evaluates HyDE, MMR, and reranking effectiveness
   - **Link**: https://hf.co/papers/2404.01037

### Multi-Hop Reasoning

4. **MultiHop-RAG: Benchmarking RAG for Multi-Hop Queries** (2024)
   - **arXiv**: 2401.15391
   - **Contribution**: Dataset and benchmark for multi-hop reasoning
   - **Link**: https://hf.co/papers/2401.15391
   - **Application**: Demo #2 - Multi-Query Decomposition

5. **BeamAggR: Beam Aggregation Reasoning over Multi-source Knowledge** (2024)
   - **arXiv**: 2406.19820
   - **Key Concept**: Probabilistic aggregation across knowledge sources
   - **Link**: https://hf.co/papers/2406.19820

6. **MINTQA: Multi-hop QA on New and Tail Knowledge** (2024)
   - **arXiv**: 2412.17032
   - **Contribution**: 28K+ QA pairs for evaluating multi-hop reasoning
   - **Link**: https://hf.co/papers/2412.17032

### Chunking Strategies

7. **Is Semantic Chunking Worth the Computational Cost?** (2024)
   - **arXiv**: 2410.13070
   - **Key Finding**: Questions effectiveness of semantic chunking
   - **Link**: https://hf.co/papers/2410.13070
   - **Application**: Demo #4 - Hierarchical Retrieval

8. **Rethinking Chunk Size For Long-Document Retrieval** (2025)
   - **arXiv**: 2505.21700
   - **Key Finding**: Optimal chunk sizes: 64-128 tokens (short), 512-1024 tokens (long)
   - **Link**: https://hf.co/papers/2505.21700

9. **Late Chunking: Contextual Chunk Embeddings** (2024)
   - **arXiv**: 2409.04701
   - **Key Concept**: Apply chunking after transformer encoding
   - **Link**: https://hf.co/papers/2409.04701

10. **ChunkRAG: Novel LLM-Chunk Filtering Method** (2024)
    - **arXiv**: 2410.19572
    - **Key Concept**: Token-aware chunk filtering
    - **Link**: https://hf.co/papers/2410.19572

### Agentic RAG

11. **Agentic Retrieval-Augmented Generation: A Survey** (2025)
    - **arXiv**: 2501.09136
    - **Coverage**: Agentic design patterns, reflection, planning, tool use
    - **Link**: https://hf.co/papers/2501.09136
    - **Application**: Demo #8 - Agentic RAG

12. **GFM-RAG: Graph Foundation Model for RAG** (2025)
    - **arXiv**: 2502.01113
    - **Key Concept**: Graph neural networks for complex query-knowledge relationships
    - **Link**: https://hf.co/papers/2502.01113

### Adaptive and Self-Correcting RAG

13. **CDF-RAG: Causal Dynamic Feedback for Adaptive RAG** (2025)
    - **arXiv**: 2504.12560
    - **Key Concept**: Iterative refinement with causal validation
    - **Link**: https://hf.co/papers/2504.12560
    - **Application**: Demo #7 - Corrective RAG

14. **MBA-RAG: A Bandit Approach for Adaptive RAG** (2024)
    - **arXiv**: 2412.01572
    - **Key Concept**: Reinforcement learning for dynamic retrieval strategy selection
    - **Link**: https://hf.co/papers/2412.01572

15. **Finetune-RAG: Fine-Tuning to Resist Hallucination** (2025)
    - **arXiv**: 2505.10792
    - **Key Concept**: Training to handle imperfect retrieval
    - **Link**: https://hf.co/papers/2505.10792

### Context and Evaluation

16. **LaRA: Benchmarking RAG and Long-Context LLMs** (2025)
    - **arXiv**: 2502.09977
    - **Key Finding**: When to use RAG vs long-context models
    - **Link**: https://hf.co/papers/2502.09977

17. **Quantifying reliance on external information during RAG** (2024)
    - **arXiv**: 2410.00857
    - **Key Finding**: LLMs show "shortcut" bias toward retrieved context
    - **Link**: https://hf.co/papers/2410.00857

18. **OCR Hinders RAG: Evaluating Cascading Impact** (2024)
    - **arXiv**: 2412.02592
    - **Contribution**: OHRBench dataset for OCR-RAG evaluation
    - **Link**: https://hf.co/papers/2412.02592

### Cross-Encoder and Reranking

19. **Comparative Analysis of Optimizers for Cross-Encoder Reranking** (2025)
    - **arXiv**: 2506.18297
    - **Coverage**: Lion vs AdamW for training rerankers
    - **Link**: https://hf.co/papers/2506.18297
    - **Application**: Demo #5 - Reranking

20. **Incorporating Relevance Feedback for Few-Shot Document Re-Ranking** (2022)
    - **arXiv**: 2210.10695
    - **Key Concept**: Few-shot learning for reranking with user feedback
    - **Link**: https://hf.co/papers/2210.10695

---

## Demo-Specific Recommendations

### Demo #1: HyDE (Query Enhancement)
- **Papers**: 
  - HyDE original (2212.10496)
  - ARAGOG evaluation (2404.01037)
- **Models**: 
  - sentence-transformers/all-MiniLM-L6-v2 (fast prototyping)
  - BAAI/bge-base-en-v1.5 (better accuracy)

### Demo #2: Multi-Query Decomposition
- **Papers**:
  - MultiHop-RAG (2401.15391)
  - BeamAggR (2406.19820)
  - MINTQA (2412.17032)
- **Models**:
  - sentence-transformers/multi-qa-mpnet-base-dot-v1

### Demo #3: Hybrid Search
- **Papers**:
  - BAAI/bge-m3 paper (2402.03216)
- **Models**:
  - BAAI/bge-m3 (native hybrid search support)
  - sentence-transformers/all-mpnet-base-v2 (dense vectors)

### Demo #4: Hierarchical Retrieval
- **Papers**:
  - Rethinking Chunk Size (2505.21700)
  - Late Chunking (2409.04701)
  - Is Semantic Chunking Worth It? (2410.13070)
- **Models**:
  - Alibaba-NLP/gte-large-en-v1.5 (8192 token context)

### Demo #5: Reranking
- **Papers**:
  - Cross-Encoder Optimizer Comparison (2506.18297)
  - Relevance Feedback (2210.10695)
- **Models**:
  - cross-encoder/ms-marco-MiniLM-L6-v2 (most popular)
  - BAAI/bge-reranker-v2-m3 (multilingual)
  - Qwen/Qwen3-Reranker-0.6B (modern, efficient)

### Demo #6: Context Compression
- **Papers**:
  - ChunkRAG (2410.19572)
  - LaRA benchmark (2502.09977)
  - Quantifying reliance (2410.00857)

### Demo #7: Corrective RAG
- **Papers**:
  - CDF-RAG (2504.12560)
  - MBA-RAG (2412.01572)
  - Finetune-RAG (2505.10792)

### Demo #8: Agentic RAG
- **Papers**:
  - Agentic RAG Survey (2501.09136)
  - GFM-RAG (2502.01113)
  - BeamAggR (2406.19820)
- **Documentation**:
  - Transformers RAG docs

### Demo #9: Embedding Fine-tuning
- **Base Models**:
  - sentence-transformers/all-MiniLM-L6-v2 (fast training)
  - BAAI/bge-base-en-v1.5 (better performance)
- **Example**:
  - datasocietyco/bge-base-en-v1.5-course-recommender-v5

---

## Datasets for Evaluation

While specific RAG benchmark datasets were limited in our search, the papers above reference several key datasets:

1. **MS MARCO** - Passage ranking and retrieval
2. **MultiHop-RAG Dataset** - Multi-hop reasoning
3. **MINTQA** - New and tail knowledge QA (10K+ questions)
4. **OHRBench** - OCR-RAG evaluation (350 documents)
5. **HotpotQA** - Multi-hop question answering
6. **Natural Questions** - Open-domain QA
7. **E-VQA, InfoSeek** - Knowledge-based visual QA

---

## Documentation and Guides

### Official Hugging Face Documentation

1. **RAG Implementation Guide**
   - Link: https://huggingface.co/docs/transformers/chat_extras
   - Coverage: RAG with transformers, document parameter usage

2. **RAG Model Documentation**
   - Link: https://huggingface.co/docs/transformers/model_doc/rag
   - Coverage: RAG model architecture, DPR integration

3. **Sentence Transformers Hub**
   - Link: https://huggingface.co/docs/hub/sentence-transformers
   - Coverage: Using and training sentence transformers

### Model-Specific Documentation

4. **Cohere Command-R for RAG**
   - Link: https://huggingface.co/docs/transformers/model_doc/cohere
   - Features: Built-in RAG support, grounded generation

5. **Cohere Command-R+ for RAG**
   - Link: https://huggingface.co/docs/transformers/model_doc/cohere2
   - Features: Enhanced RAG capabilities, tool use

---

## Quick Reference: Model Selection Guide

### For Fast Prototyping
- **Embedding**: sentence-transformers/all-MiniLM-L6-v2
- **Reranker**: cross-encoder/ms-marco-MiniLM-L6-v2

### For Best Accuracy
- **Embedding**: BAAI/bge-base-en-v1.5 or sentence-transformers/all-mpnet-base-v2
- **Reranker**: BAAI/bge-reranker-v2-m3 or Qwen/Qwen3-Reranker-0.6B

### For Hybrid Search
- **Model**: BAAI/bge-m3 (supports dense + sparse)

### For Long Context
- **Embedding**: Alibaba-NLP/gte-large-en-v1.5 or jinaai/jina-embeddings-v3

### For Multilingual
- **Embedding**: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
- **Reranker**: BAAI/bge-reranker-v2-m3

---

## Installation Commands

```bash
# Core libraries
pip install transformers sentence-transformers

# For specific demos
pip install llama-index  # RAG framework
pip install duckduckgo-search  # Demo #8: Web search
pip install arxiv  # Demo #8: ArXiv search
pip install ragas  # Demo #10: Evaluation

# For fine-tuning (Demo #9)
pip install datasets torch
```

---

## Additional Resources

### Community Discussions
- Many papers have active community discussions on Hugging Face
- Check the "Community" tab on paper pages for insights and clarifications

### Model Cards
- Each model has detailed cards with:
  - Usage examples
  - Performance benchmarks
  - Training details
  - Limitations

### Collections
- Browse curated collections at https://huggingface.co/collections
- Search for "RAG", "retrieval", "embeddings" for related resources

---

**Last Updated**: Based on search conducted October 15, 2025
**Repository**: RAG-Workshop/RAG_v3
**Maintainer**: Workshop Development Team

For questions or suggestions, please open an issue in the workshop repository.
