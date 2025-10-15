# Advanced RAG Workshop Demo Development Plan

## Overview

This document provides a comprehensive development plan for creating code demos for the "Advanced RAG" workshop. Each demo builds progressively from foundational advanced concepts to sophisticated, multi-layered techniques. All demos are designed as self-contained Python Jupyter Notebooks for intermediate-level AI engineers already familiar with basic RAG principles.

---

## **Demo #1: Query Enhancement with HyDE (Hypothetical Document Embeddings)**

**Objective:** Demonstrate how to bridge the semantic gap between user queries and documents by generating hypothetical ideal answers before retrieval.

**Core Concepts Demonstrated:**
- Hypothetical Document Embeddings (HyDE)
- Query-to-document asymmetry problem
- Pre-retrieval optimization

**Implementation Steps:**

1. **Setup Environment and Data Ingestion**
   - Import required libraries: `langchain`, `openai`, `chromadb`, `sentence-transformers`
   - Load a sample knowledge base (e.g., technical documentation or research papers)
   - Create standard chunking and embedding pipeline using a bi-encoder model

2. **Implement Baseline Naive RAG**
   - Build traditional RAG pipeline with direct query embedding
   - Create retrieval function using cosine similarity search
   - Implement basic generation with retrieved context

3. **Implement HyDE Enhancement**
   - Create prompt template: "Please write a passage that contains the answer to the following question: [user query]"
   - Use LLM to generate hypothetical ideal document
   - Embed the generated hypothetical document instead of the original query
   - Perform retrieval using hypothetical document embedding

4. **Comparative Evaluation**
   - Test both approaches on sample queries
   - Compare retrieval quality and relevance of returned documents
   - Analyze cases where HyDE significantly outperforms direct query embedding

5. **Advanced HyDE Variations**
   - Implement multi-perspective HyDE (generate multiple hypothetical documents)
   - Demonstrate query-specific prompt engineering for different domains

**Relevant Citation(s):** Reference 32 - Advanced RAG: Improving Retrieval-Augmented Generation with Hypothetical Document Embeddings (HyDE)

**Status:** [COMPLETED]  
**File Generated:** demo_01_hyde_query_enhancement.ipynb  
**Completion Date:** 2025-01-15  
**Notes:** Successfully implemented comprehensive HyDE demo including baseline RAG comparison, multi-perspective HyDE variations, domain-specific prompt engineering, and detailed performance analysis. The notebook includes all required sections with extensive documentation and real-world examples. All implementation steps from the plan were followed precisely.

---

## **Demo #2: Multi-Query and Sub-Query Decomposition**

**Objective:** Show how to handle complex, multi-faceted queries by breaking them down into smaller, independent sub-queries for comprehensive information gathering.

**Core Concepts Demonstrated:**
- Sub-query decomposition
- Multi-query generation
- Multi-hop reasoning enablement
- Query complexity handling

**Implementation Steps:**

1. **Environment Setup**
   - Configure LangChain with query decomposition chains
   - Prepare knowledge base with interconnected information requiring multi-hop reasoning
   - Set up vector database with diverse document types

2. **Implement Query Decomposition Logic**
   - Create LLM prompt for breaking complex queries into sub-queries
   - Example: "What are the differences in battery life and camera quality between the latest iPhone and Samsung models?" → 4 sub-queries
   - Implement validation to ensure sub-queries are independent and comprehensive

3. **Build Multi-Query Retrieval Pipeline**
   - Execute each sub-query independently against the retriever
   - Implement result aggregation and deduplication
   - Create ranking mechanism for merged results

4. **Develop Context Synthesis**
   - Aggregate retrieved contexts from all sub-queries
   - Implement context organization by sub-topic
   - Create final prompt that includes all sub-contexts with clear structure

5. **Advanced Multi-Query Techniques**
   - Implement Diverse Multi-Query Rewriting (DMQR-RAG) approach
   - Generate queries at different granularity levels
   - Add dependency detection between sub-queries for sequential processing

**Relevant Citation(s):** Reference 31 - Enhancing Financial RAG with Agentic AI and Multi-HyDE; Reference 38 - DMQR-RAG: Diverse Multi-Query Rewriting for Retrieval-Augmented Generation

**Status:** [COMPLETED]  
**File Generated:** demo_02_multi_query_decomposition.ipynb  
**Completion Date:** 2025-10-15  
**Notes:** Successfully implemented comprehensive multi-query and sub-query decomposition demo including baseline naive RAG comparison, complete multi-query RAG pipeline with sub-query decomposition, DMQR-RAG implementation with diverse query rewriting, context synthesis and aggregation, deduplication logic, and comparative evaluation with three complex test queries. The notebook includes detailed explanations, production considerations, query complexity classification, and extensive documentation. All implementation steps from the plan were followed precisely with additional enhancements for production readiness.

---

## **Demo #3: Hybrid Search Implementation**

**Objective:** Combine semantic search (dense vectors) with keyword search (sparse vectors) to leverage the strengths of both retrieval paradigms.

**Core Concepts Demonstrated:**
- Hybrid search architecture
- BM25 keyword search integration
- Dense vs. sparse vector representations
- Score fusion techniques (Weighted Fusion, Reciprocal Rank Fusion)

**Implementation Steps:**

1. **Setup Dual Search Infrastructure**
   - Install and configure `rank-bm25` for sparse retrieval
   - Set up dense vector search with existing embedding model
   - Prepare test dataset with queries requiring both semantic and exact matching

2. **Implement BM25 Keyword Search**
   - Create BM25 index from document corpus
   - Implement keyword-based retrieval function
   - Test on queries requiring exact term matches (acronyms, proper names, code snippets)

3. **Build Dense Vector Search**
   - Implement semantic search using sentence transformers
   - Test on queries requiring conceptual understanding and synonyms
   - Compare performance against BM25 on different query types

4. **Develop Score Fusion Mechanisms**
   - **Weighted Fusion**: Implement combined_score = α × keyword_score + (1-α) × vector_score
   - **Reciprocal Rank Fusion (RRF)**: Implement rank-based combination method
   - Create hyperparameter tuning for optimal α weighting

5. **Performance Analysis and Optimization**
   - Benchmark hybrid search vs. individual methods
   - Analyze query types where each approach excels
   - Implement adaptive weighting based on query characteristics

**Relevant Citation(s):** Reference 43 - Using Hybrid Search to Deliver Fast and Contextually Relevant Results; Reference 44 - LLM RAG: Improving the retrieval phase with Hybrid Search

---

## **Demo #4: Hierarchical Retrieval with Parent Document Retriever**

**Objective:** Solve the chunking trade-off by implementing a two-tier system where small chunks enable precise retrieval while large parent chunks provide rich context for generation.

**Core Concepts Demonstrated:**
- Parent Document Retriever pattern
- Hierarchical chunking strategy
- Precision vs. context trade-off resolution
- Two-stage retrieval architecture

**Implementation Steps:**

1. **Design Hierarchical Document Structure**
   - Implement document splitting into logical parent chunks (sections/chapters)
   - Further split parent chunks into granular child chunks (paragraphs/sentences)
   - Create mapping system between child and parent chunks

2. **Build Dual Storage System**
   - Index only child chunks in vector database for precise retrieval
   - Store parent chunks in separate document store with chunk ID mapping
   - Implement efficient lookup mechanism between child and parent chunks

3. **Implement Two-Stage Retrieval**
   - **Stage 1**: Perform vector similarity search on child chunk embeddings
   - **Stage 2**: Retrieve corresponding parent chunks for top-K child matches
   - Maintain traceability from parent context back to specific child chunk

4. **Context Assembly and Deduplication**
   - Handle cases where multiple child chunks belong to same parent
   - Implement parent chunk deduplication while preserving relevance scores
   - Create context assembly that highlights relevant child sections within parent context

5. **Comparative Analysis Framework**
   - Compare retrieval precision: child chunks vs. parent chunks vs. fixed-size chunks
   - Measure generation quality with different context sizes
   - Analyze computational overhead vs. performance gains

**Relevant Citation(s):** Reference 46 - Parent-Child Chunking in LangChain for Advanced RAG; Reference 47 - Parent-Child Retriever - GraphRAG

---

## **Demo #5: Re-ranking with Cross-Encoders**

**Objective:** Implement a two-stage retrieval system where fast bi-encoders cast a wide net and slower but more accurate cross-encoders reorder results for optimal precision.

**Core Concepts Demonstrated:**
- Two-stage retrieval architecture (Retrieval + Re-ranking)
- Bi-encoder vs. Cross-encoder architectures
- Recall vs. Precision optimization
- Popular re-ranking models integration

**Implementation Steps:**

1. **Establish Baseline Two-Stage Pipeline**
   - Implement first-stage retrieval with bi-encoder (high recall, top 50-100 documents)
   - Set up evaluation framework for measuring retrieval precision and recall
   - Create test queries with known relevant documents

2. **Integrate Cross-Encoder Re-ranking**
   - Install and configure popular re-ranking models: `bge-rerank`, `mxbai-rerank-v1`
   - Implement cross-encoder scoring for query-document pairs
   - Re-rank first-stage results to improve top-K precision

3. **Compare Re-ranking Model Performance**
   - Benchmark different re-ranking models on domain-specific queries
   - Analyze computational overhead vs. accuracy improvement
   - Test integration with Cohere Rerank API for comparison

4. **Optimize Pipeline Efficiency**
   - Implement caching for frequently accessed documents
   - Add async processing for re-ranking operations
   - Create adaptive top-K selection based on query complexity

5. **Advanced Re-ranking Techniques**
   - Implement ensemble re-ranking with multiple models
   - Add query-dependent re-ranking model selection
   - Create custom fine-tuning pipeline for domain-specific re-ranking

**Relevant Citation(s):** Reference 45 - Advanced RAG Optimization: Prioritize Knowledge with Reranking; Reference 56 - Enhancing RAG Pipelines with Re-Ranking

---

## **Demo #6: Context Compression and Strategic Reordering**

**Objective:** Address the "lost in the middle" problem and context window limitations through intelligent context management and compression techniques.

**Core Concepts Demonstrated:**
- "Lost in the middle" problem mitigation
- Strategic context reordering
- Extractive vs. Abstractive compression
- Context window optimization

**Implementation Steps:**

1. **Demonstrate "Lost in the Middle" Problem**
   - Create test scenarios with long context containing relevant information in middle positions
   - Measure LLM performance degradation with middle-positioned relevant context
   - Establish baseline performance metrics

2. **Implement Strategic Reordering**
   - Place most relevant document at beginning of context
   - Position second most relevant document at the end
   - Fill middle positions with less critical but related information
   - Test performance improvement with strategic positioning

3. **Build Extractive Compression Pipeline**
   - Implement sentence-level relevance scoring against original query
   - Create "knowledge strips" extraction from retrieved documents
   - Filter and retain only high-relevance sentences
   - Maintain source traceability for citations

4. **Develop Abstractive Compression**
   - Integrate lightweight LLM for document summarization
   - Implement chunk-wise and full-context summarization
   - Balance compression ratio with information preservation
   - Create fallback mechanisms for critical information retention

5. **Adaptive Context Management**
   - Implement dynamic context window utilization based on query complexity
   - Create hybrid approach combining extractive and abstractive methods
   - Add real-time token counting and context truncation strategies

**Relevant Citation(s):** Reference 22 - Lost in the middle research; Reference 60 - Mastering Advanced RAG Techniques: A Comprehensive Guide

---

## **Demo #7: Graph-Based Retrieval (GraphRAG)**

**Objective:** Implement knowledge graph-based retrieval to enable multi-hop reasoning and relationship-aware information gathering beyond traditional vector similarity.

**Core Concepts Demonstrated:**
- Knowledge graph construction from documents
- Entity and relationship extraction
- Graph traversal algorithms for retrieval
- Multi-hop reasoning capabilities

**Implementation Steps:**

1. **Knowledge Graph Construction Pipeline**
   - Implement entity extraction using NER models or LLMs
   - Extract relationships between entities from document corpus
   - Build graph database using Neo4j or NetworkX
   - Create entity linking and disambiguation logic

2. **Graph-Based Query Processing**
   - Implement query analysis for entity identification
   - Create graph traversal algorithms (BFS, DFS) for relationship exploration
   - Build path-finding logic for multi-hop reasoning queries
   - Example: "Which movies directed by Christopher Nolan also starred Michael Caine?"

3. **Hybrid Graph-Vector Retrieval**
   - Combine graph traversal results with vector similarity search
   - Implement scoring mechanism for graph-derived vs. vector-derived results
   - Create context assembly from multiple information sources

4. **Advanced Graph Reasoning**
   - Implement complex query decomposition for graph operations
   - Add temporal reasoning for time-sensitive relationships
   - Create confidence scoring for multi-hop inference chains

5. **Performance and Scalability Analysis**
   - Benchmark graph retrieval vs. traditional vector search
   - Analyze computational complexity for different query types
   - Implement caching strategies for frequent graph patterns

**Relevant Citation(s):** Reference 50 - What is GraphRAG?; Reference 36 - GraphRAG Field Guide: Navigating the World of Advanced RAG Patterns

---

## **Demo #8: Self-Correcting RAG (CRAG) with Web Search Fallback**

**Objective:** Implement autonomous quality assessment and corrective retrieval mechanisms that can dynamically switch to alternative information sources when initial retrieval fails.

**Core Concepts Demonstrated:**
- Corrective Retrieval Augmented Generation (CRAG)
- Retrieval quality evaluation
- Web search integration as fallback
- Knowledge refinement and filtering

**Implementation Steps:**

1. **Build Retrieval Evaluator**
   - Implement lightweight model for assessing retrieval relevance
   - Create confidence scoring system for retrieved documents
   - Set up decision thresholds for correct/incorrect/ambiguous classifications

2. **Implement CRAG Workflow Logic**
   - **High confidence**: Pass retrieved documents directly to generation
   - **Low confidence**: Discard internal results and trigger web search
   - **Ambiguous**: Combine internal and web search results

3. **Integrate Web Search Fallback**
   - Configure web search APIs (Tavily, SerpAPI, or Google Search)
   - Implement query reformulation for web search optimization
   - Create result filtering and relevance assessment for web content

4. **Knowledge Refinement Pipeline**
   - Decompose retrieved documents into "knowledge strips"
   - Implement granular relevance scoring for individual sentences
   - Filter and retain only high-quality, relevant information

5. **Self-Reflection and Iteration**
   - Add capability for multiple retrieval iterations
   - Implement query refinement based on previous retrieval failures
   - Create logging and analysis framework for retrieval decision patterns

**Relevant Citation(s):** Reference 67 - Corrective Retrieval Augmented Generation; Reference 16 - Self-RAG with reflection tokens

---

## **Demo #9: Agentic RAG with Tool Selection**

**Objective:** Demonstrate autonomous agent-driven RAG where an LLM agent dynamically selects and orchestrates multiple retrieval tools based on query characteristics and iterative assessment.

**Core Concepts Demonstrated:**
- Agentic RAG architecture
- Tool selection and orchestration
- Thought-Action-Observation loops
- Multi-step reasoning and planning

**Implementation Steps:**

1. **Design Agent Architecture**
   - Implement LLM-powered agent with reasoning capabilities
   - Create tool registry with multiple retrieval mechanisms (vector search, graph search, web search, SQL queries)
   - Build memory system for short-term task context and long-term learning

2. **Implement Tool Selection Logic**
   - Create prompt templates for tool selection decision-making
   - Implement query analysis for determining appropriate tools
   - Build dynamic tool combination strategies for complex queries

3. **Build Thought-Action-Observation Loop**
   - **Thought**: Agent analyzes query and plans retrieval strategy
   - **Action**: Agent selects and executes appropriate tools
   - **Observation**: Agent evaluates results and decides on next steps

4. **Advanced Planning and Decomposition**
   - Implement complex query decomposition into sub-tasks
   - Create dependency management between sub-tasks
   - Add iterative refinement based on intermediate results

5. **Multi-Tool Integration and Coordination**
   - Coordinate results from vector search, graph traversal, and web search
   - Implement result fusion and conflict resolution
   - Create final answer synthesis from multiple information sources

**Relevant Citation(s):** Reference 34 - What is Agentic RAG; Reference 61, 62 - IBM Agentic RAG concepts; Reference 63 - Understanding Agentic RAG

---

## **Demo #10: Fine-Tuning RAG Components for Domain Optimization**

**Objective:** Demonstrate targeted fine-tuning of both retriever (embedding model) and generator (LLM) components for domain-specific performance improvements.

**Core Concepts Demonstrated:**
- Retriever fine-tuning with contrastive loss
- Generator fine-tuning for noise resistance
- Domain-specific dataset curation
- Performance evaluation and comparison

**Implementation Steps:**

1. **Dataset Curation for Retriever Fine-tuning**
   - Create domain-specific triplets: (query, positive_passage, negative_passage)
   - Implement hard negative mining for challenging examples
   - Generate synthetic question-answer pairs using LLMs for data augmentation

2. **Implement Retriever Fine-tuning Pipeline**
   - Set up contrastive loss training objective
   - Fine-tune embedding model using domain-specific dataset
   - Implement evaluation framework for retrieval performance measurement

3. **Dataset Curation for Generator Fine-tuning**
   - Create training examples with mixed-quality context (correct + distractor passages)
   - Design "ideal answers" that use only correct information
   - Implement noise injection strategies for robustness training

4. **Implement Generator Fine-tuning (Finetune-RAG)**
   - Use parameter-efficient fine-tuning methods (LoRA)
   - Train model to ignore distractors and focus on relevant context
   - Implement faithfulness metrics for evaluation

5. **Comprehensive Evaluation Framework**
   - Compare fine-tuned vs. general-purpose models
   - Measure domain-specific performance improvements
   - Analyze cost-benefit trade-offs of fine-tuning approaches
   - Create ablation studies for individual component contributions

**Relevant Citation(s):** Reference 69 - Multi-task retriever fine-tuning for domain-specific RAG; Reference 73 - Finetune-RAG: Fine-Tuning Language Models to Resist Distraction

---

## **Demo #11: Advanced RAG Evaluation and Metrics**

**Objective:** Implement comprehensive evaluation framework for RAG systems covering both retriever and generator performance with automated metrics and LLM-as-a-judge approaches.

**Core Concepts Demonstrated:**
- Retriever-side metrics (Context Precision, Context Recall)
- Generator-side metrics (Faithfulness, Answer Relevance)
- Automated evaluation frameworks (RAGAS)
- LLM-as-a-judge evaluation

**Implementation Steps:**

1. **Implement Retriever Evaluation Metrics**
   - **Context Precision**: Measure relevance ratio of retrieved documents
   - **Context Recall**: Assess completeness of information retrieval
   - Classical IR metrics: Mean Reciprocal Rank (MRR), NDCG

2. **Implement Generator Evaluation Metrics**
   - **Faithfulness**: Measure factual grounding in retrieved context
   - **Answer Relevance**: Assess query-answer alignment
   - Implement LLM-as-a-judge for automated scoring

3. **Integrate RAGAS Evaluation Framework**
   - Set up RAGAS library for automated RAG evaluation
   - Configure evaluation pipeline with multiple metrics
   - Create evaluation datasets for consistent benchmarking

4. **Build Comprehensive Evaluation Suite**
   - Create end-to-end evaluation pipeline
   - Implement A/B testing framework for RAG variants
   - Add performance monitoring and alerting for production systems

5. **Advanced Evaluation Techniques**
   - Implement adversarial testing for robustness evaluation
   - Create domain-specific evaluation benchmarks
   - Add explainability analysis for model decisions

**Relevant Citation(s):** Reference 84 - RAG Evaluation Metrics Guide; Reference 89 - RAGAS evaluation framework; Reference 86 - RAG Evaluation Metrics: Answer Relevancy, Faithfulness

---

## Implementation Guidelines

### **Technical Requirements**
- Python 3.8+
- Jupyter Notebook environment
- Key libraries: `langchain`, `llamaindex`, `openai`, `chromadb`, `sentence-transformers`, `transformers`, `neo4j`, `ragas`
- Access to LLM APIs (OpenAI, Anthropic, or local models)
- Vector databases (Chroma, FAISS, or Pinecone)

### **Data Requirements**
- Sample document collections for each demo domain
- Evaluation datasets with ground truth annotations
- Synthetic query-answer pairs for testing
- Domain-specific vocabularies and terminologies

### **Progressive Complexity**
The demos are sequenced to build understanding progressively:
1. **Demos 1-3**: Pre-retrieval optimization techniques
2. **Demos 4-6**: Advanced retrieval and post-retrieval processing
3. **Demos 7-9**: Autonomous and graph-based systems
4. **Demos 10-11**: Optimization and evaluation frameworks

### **Expected Outcomes**
Each demo will be a fully functional Jupyter Notebook that:
- Demonstrates the specific advanced RAG technique
- Includes comparative analysis with baseline approaches
- Provides clear explanations of trade-offs and use cases
- Contains practical implementation code that can be adapted for production use
- Includes evaluation metrics and performance analysis

This comprehensive demo suite will provide workshop attendees with hands-on experience in implementing state-of-the-art RAG techniques, enabling them to build more robust, accurate, and efficient RAG systems for real-world applications.


#### **Works cited**

1. en.wikipedia.org, accessed October 15, 2025, [https://en.wikipedia.org/wiki/Knowledge_cutoff#:~:text=In%20machine%20learning%2C%20a%20knowledge,the%20model's%20internal%20knowledge%20base.](https://en.wikipedia.org/wiki/Knowledge_cutoff#:~:text=In%20machine%20learning%2C%20a%20knowledge,the%20model's%20internal%20knowledge%20base.)  
2. Knowledge cutoff - Wikipedia, accessed October 15, 2025, [https://en.wikipedia.org/wiki/Knowledge_cutoff](https://en.wikipedia.org/wiki/Knowledge_cutoff)  
3. 10 Biggest Limitations of Large Language Models - ProjectPro, accessed October 15, 2025, [https://www.projectpro.io/article/llm-limitations/1045](https://www.projectpro.io/article/llm-limitations/1045)  
4. What Are the Limitations of Large Language Models (LLMs)? - PromptDrive.ai, accessed October 15, 2025, [https://promptdrive.ai/llm-limitations/](https://promptdrive.ai/llm-limitations/)  
5. All You Need to Know about the Limitations of Large Language Models | by Novita AI, accessed October 15, 2025, [https://medium.com/@marketing_novita.ai/all-you-need-to-know-about-the-limitations-of-large-language-models-568e15f66809](https://medium.com/@marketing_novita.ai/all-you-need-to-know-about-the-limitations-of-large-language-models-568e15f66809)  
6. Knowledge Cutoff Dates of all LLMs (GPT-4o, GPT-4, Gemini) - Otterly.AI, accessed October 15, 2025, [https://otterly.ai/blog/knowledge-cutoff/](https://otterly.ai/blog/knowledge-cutoff/)  
7. [2403.12958] Dated Data: Tracing Knowledge Cutoffs in Large Language Models - arXiv, accessed October 15, 2025, [https://arxiv.org/abs/2403.12958](https://arxiv.org/abs/2403.12958)  
8. Dated Data: Tracing Knowledge Cutoffs in Large Language Models - arXiv, accessed October 15, 2025, [https://arxiv.org/html/2403.12958v1](https://arxiv.org/html/2403.12958v1)  
9. What is Retrieval-Augmented Generation (RAG)? | Google Cloud, accessed October 15, 2025, [https://cloud.google.com/use-cases/retrieval-augmented-generation](https://cloud.google.com/use-cases/retrieval-augmented-generation)  
10. LLM Limitations: When Models and Chatbots Make Mistakes - Learn Prompting, accessed October 15, 2025, [https://learnprompting.org/docs/basics/pitfalls](https://learnprompting.org/docs/basics/pitfalls)  
11. What is RAG? - Retrieval-Augmented Generation AI Explained - AWS - Updated 2025, accessed October 15, 2025, [https://aws.amazon.com/what-is/retrieval-augmented-generation/](https://aws.amazon.com/what-is/retrieval-augmented-generation/)  
12. What is RAG (Retrieval Augmented Generation)? - IBM, accessed October 15, 2025, [https://www.ibm.com/think/topics/retrieval-augmented-generation](https://www.ibm.com/think/topics/retrieval-augmented-generation)  
13. Retrieval Augmented Generation - IBM, accessed October 15, 2025, [https://www.ibm.com/architectures/patterns/genai-rag](https://www.ibm.com/architectures/patterns/genai-rag)  
14. Retrieval-augmented generation - Wikipedia, accessed October 15, 2025, [https://en.wikipedia.org/wiki/Retrieval-augmented_generation](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)  
15. Retrieval Augmented Generation (RAG) and Semantic Search for GPTs, accessed October 15, 2025, [https://help.openai.com/en/articles/8868588-retrieval-augmented-generation-rag-and-semantic-search-for-gpts](https://help.openai.com/en/articles/8868588-retrieval-augmented-generation-rag-and-semantic-search-for-gpts)  
16. 8 Retrieval Augmented Generation (RAG) Architectures You Should Know in 2025, accessed October 15, 2025, [https://humanloop.com/blog/rag-architectures](https://humanloop.com/blog/rag-architectures)  
17. Retrieval Augmented Generation (RAG) for LLMs | Prompt ..., accessed October 15, 2025, [https://www.promptingguide.ai/research/rag](https://www.promptingguide.ai/research/rag)  
18. RAG vs LLM? Understanding the Unique Capabilities and Limitations of Each Approach - Kanerika, accessed October 15, 2025, [https://kanerika.com/blogs/rag-vs-llm/](https://kanerika.com/blogs/rag-vs-llm/)  
19. RAG vs Traditional LLMs: Key Differences - Galileo AI, accessed October 15, 2025, [https://galileo.ai/blog/comparing-rag-and-traditional-llms-which-suits-your-project](https://galileo.ai/blog/comparing-rag-and-traditional-llms-which-suits-your-project)  
20. What is Retrieval Augmented Generation (RAG)? | Databricks, accessed October 15, 2025, [https://www.databricks.com/glossary/retrieval-augmented-generation-rag](https://www.databricks.com/glossary/retrieval-augmented-generation-rag)  
21. RAG vs Long Context Models [Discussion] : r/MachineLearning - Reddit, accessed October 15, 2025, [https://www.reddit.com/r/MachineLearning/comments/1ax6j73/rag_vs_long_context_models_discussion/](https://www.reddit.com/r/MachineLearning/comments/1ax6j73/rag_vs_long_context_models_discussion/)  
22. RAG vs. Long-context LLMs - SuperAnnotate, accessed October 15, 2025, [https://www.superannotate.com/blog/rag-vs-long-context-llms](https://www.superannotate.com/blog/rag-vs-long-context-llms)  
23. RAG vs Long-Context LLMs: Approaches for Real-World Applications - Prem AI, accessed October 15, 2025, [https://blog.premai.io/rag-vs-long-context-llms-which-approach-excels-in-real-world-applications/](https://blog.premai.io/rag-vs-long-context-llms-which-approach-excels-in-real-world-applications/)  
24. What is retrieval-augmented generation? - Red Hat, accessed October 15, 2025, [https://www.redhat.com/en/topics/ai/what-is-retrieval-augmented-generation](https://www.redhat.com/en/topics/ai/what-is-retrieval-augmented-generation)  
25. How to use the Parent Document Retriever | 🦜️ LangChain, accessed October 15, 2025, [https://python.langchain.com/docs/how_to/parent_document_retriever/](https://python.langchain.com/docs/how_to/parent_document_retriever/)  
26. Advanced RAG Implementation using Hybrid Search and Reranking | by Nadika Poudel | Medium, accessed October 15, 2025, [https://medium.com/@nadikapoudel16/advanced-rag-implementation-using-hybrid-search-reranking-with-zephyr-alpha-llm-4340b55fef22](https://medium.com/@nadikapoudel16/advanced-rag-implementation-using-hybrid-search-reranking-with-zephyr-alpha-llm-4340b55fef22)  
27. Similarity Search in RAG Applications: Principles Every Business ..., accessed October 15, 2025, [https://mobian.studio/similarity-search-in-rag-applications-principles-every-business-should-know/](https://mobian.studio/similarity-search-in-rag-applications-principles-every-business-should-know/)  
28. Optimizing RAG with Hybrid Search & Reranking | VectorHub by Superlinked, accessed October 15, 2025, [https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking](https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking)  
29. Rerankers and Two-Stage Retrieval - Pinecone, accessed October 15, 2025, [https://www.pinecone.io/learn/series/rag/rerankers/](https://www.pinecone.io/learn/series/rag/rerankers/)  
30. Retrieval-Augmented Generation for Large Language ... - arXiv, accessed October 15, 2025, [https://arxiv.org/pdf/2312.10997](https://arxiv.org/pdf/2312.10997)  
31. Enhancing Financial RAG with Agentic AI and Multi-HyDE: A Novel Approach to Knowledge Retrieval and Hallucination Reduction - arXiv, accessed October 15, 2025, [https://arxiv.org/html/2509.16369v1](https://arxiv.org/html/2509.16369v1)  
32. Advanced RAG: Improving Retrieval-Augmented Generation with Hypothetical Document Embeddings (HyDE) - Pondhouse Data, accessed October 15, 2025, [https://www.pondhouse-data.com/blog/advanced-rag-hypothetical-document-embeddings](https://www.pondhouse-data.com/blog/advanced-rag-hypothetical-document-embeddings)  
33. Seven Failure Points When Engineering a Retrieval Augmented Generation System - arXiv, accessed October 15, 2025, [https://arxiv.org/html/2401.05856v1](https://arxiv.org/html/2401.05856v1)  
34. What is Agentic RAG | Weaviate, accessed October 15, 2025, [https://weaviate.io/blog/what-is-agentic-rag](https://weaviate.io/blog/what-is-agentic-rag)  
35. Advanced RAG Techniques: What They Are & How to Use Them, accessed October 15, 2025, [https://www.falkordb.com/blog/advanced-rag/](https://www.falkordb.com/blog/advanced-rag/)  
36. GraphRAG Field Guide: Navigating the World of Advanced RAG Patterns - Neo4j, accessed October 15, 2025, [https://neo4j.com/blog/developer/graphrag-field-guide-rag-patterns/](https://neo4j.com/blog/developer/graphrag-field-guide-rag-patterns/)  
37. Advanced RAG: Query Expansion - Haystack, accessed October 15, 2025, [https://haystack.deepset.ai/blog/query-expansion](https://haystack.deepset.ai/blog/query-expansion)  
38. DMQR-RAG: Diverse Multi-Query Rewriting for Retrieval-Augmented Generation - arXiv, accessed October 15, 2025, [https://arxiv.org/html/2411.13154v1](https://arxiv.org/html/2411.13154v1)  
39. arXiv:2404.01037v1 [cs.CL] 1 Apr 2024, accessed October 15, 2025, [https://arxiv.org/pdf/2404.01037](https://arxiv.org/pdf/2404.01037)  
40. Build Advanced Retrieval-Augmented Generation Systems | Microsoft Learn, accessed October 15, 2025, [https://learn.microsoft.com/en-us/azure/developer/ai/advanced-retrieval-augmented-generation](https://learn.microsoft.com/en-us/azure/developer/ai/advanced-retrieval-augmented-generation)  
41. Advanced RAG: Techniques, Architecture, and Best Practices ..., accessed October 15, 2025, [https://www.designveloper.com/blog/advanced-rag/](https://www.designveloper.com/blog/advanced-rag/)  
42. Why Basic Similarity Search Still Powers Many RAGs and AI Agents (And Why It's Changing) | by Kaz Sato | Google Cloud - Medium, accessed October 15, 2025, [https://medium.com/google-cloud/why-basic-similarity-search-still-powers-many-rags-and-ai-agents-and-why-its-changing-e977e0bb1401](https://medium.com/google-cloud/why-basic-similarity-search-still-powers-many-rags-and-ai-agents-and-why-its-changing-e977e0bb1401)  
43. Using Hybrid Search to Deliver Fast and Contextually Relevant Results - Ragie, accessed October 15, 2025, [https://www.ragie.ai/blog/hybrid-search](https://www.ragie.ai/blog/hybrid-search)  
44. LLM RAG: Improving the retrieval phase with Hybrid Search | EDICOM Careers, accessed October 15, 2025, [https://careers.edicomgroup.com/techblog/llm-rag-improving-the-retrieval-phase-with-hybrid-search/](https://careers.edicomgroup.com/techblog/llm-rag-improving-the-retrieval-phase-with-hybrid-search/)  
45. Advanced RAG Optimization: Prioritize Knowledge with Reranking | by Richard Song, accessed October 15, 2025, [https://blog.epsilla.com/advanced-rag-optimization-prioritize-knowledge-with-reranking-3ca36ed33b9c](https://blog.epsilla.com/advanced-rag-optimization-prioritize-knowledge-with-reranking-3ca36ed33b9c)  
46. Parent-Child Chunking in LangChain for Advanced RAG | by ..., accessed October 15, 2025, [https://medium.com/@seahorse.technologies.sl/parent-child-chunking-in-langchain-for-advanced-rag-e7c37171995a](https://medium.com/@seahorse.technologies.sl/parent-child-chunking-in-langchain-for-advanced-rag-e7c37171995a)  
47. Parent-Child Retriever - GraphRAG, accessed October 15, 2025, [https://graphrag.com/reference/graphrag/parent-child-retriever/](https://graphrag.com/reference/graphrag/parent-child-retriever/)  
48. Optimizing RAG Indexing Strategy: Multi-Vector Indexing and Parent Document Retrieval, accessed October 15, 2025, [https://dev.to/jamesli/optimizing-rag-indexing-strategy-multi-vector-indexing-and-parent-document-retrieval-49hf](https://dev.to/jamesli/optimizing-rag-indexing-strategy-multi-vector-indexing-and-parent-document-retrieval-49hf)  
49. Document Hierarchy in RAG: Boosting AI Retrieval Efficiency - Medium, accessed October 15, 2025, [https://medium.com/@nay1228/document-hierarchy-in-rag-boosting-ai-retrieval-efficiency-aa23f21b5fb9](https://medium.com/@nay1228/document-hierarchy-in-rag-boosting-ai-retrieval-efficiency-aa23f21b5fb9)  
50. What is GraphRAG? | IBM, accessed October 15, 2025, [https://www.ibm.com/think/topics/graphrag](https://www.ibm.com/think/topics/graphrag)  
51. Retrieval-Augmented Generation with Graphs (GraphRAG) - arXiv, accessed October 15, 2025, [https://arxiv.org/html/2501.00309v2](https://arxiv.org/html/2501.00309v2)  
52. www.elastic.co, accessed October 15, 2025, [https://www.elastic.co/search-labs/blog/rag-graph-traversal#:~:text=Graph%2Dbased%20RAG%20can%20provide,occur%20in%20the%20same%20document.](https://www.elastic.co/search-labs/blog/rag-graph-traversal#:~:text=Graph%2Dbased%20RAG%20can%20provide,occur%20in%20the%20same%20document.)  
53. Graph RAG: Navigating graphs for Retrieval-Augmented Generation using Elasticsearch, accessed October 15, 2025, [https://www.elastic.co/search-labs/blog/rag-graph-traversal](https://www.elastic.co/search-labs/blog/rag-graph-traversal)  
54. Efficient Knowledge Graph Construction and Retrieval from Unstructured Text for Large-Scale RAG Systems - arXiv, accessed October 15, 2025, [https://arxiv.org/html/2507.03226v2](https://arxiv.org/html/2507.03226v2)  
55. Benchmarking Vector, Graph and Hybrid Retrieval Augmented Generation (RAG) Pipelines for Open Radio Access Networks (ORAN) - arXiv, accessed October 15, 2025, [https://arxiv.org/html/2507.03608v1](https://arxiv.org/html/2507.03608v1)  
56. Enhancing RAG Pipelines with Re-Ranking | NVIDIA Technical Blog, accessed October 15, 2025, [https://developer.nvidia.com/blog/enhancing-rag-pipelines-with-re-ranking/](https://developer.nvidia.com/blog/enhancing-rag-pipelines-with-re-ranking/)  
57. Re-ranking in Retrieval Augmented Generation: How to Use Re-rankers in RAG - Chitika, accessed October 15, 2025, [https://www.chitika.com/re-ranking-in-retrieval-augmented-generation-how-to-use-re-rankers-in-rag/](https://www.chitika.com/re-ranking-in-retrieval-augmented-generation-how-to-use-re-rankers-in-rag/)  
58. How to Select the Best Re-Ranking Model in RAG? - ADaSci, accessed October 15, 2025, [https://adasci.org/how-to-select-the-best-re-ranking-model-in-rag/](https://adasci.org/how-to-select-the-best-re-ranking-model-in-rag/)  
59. Advanced RAG Series: Retrieval - Latest and Greatest - Beehiiv, accessed October 15, 2025, [https://div.beehiiv.com/p/advanced-rag-series-retrieval](https://div.beehiiv.com/p/advanced-rag-series-retrieval)  
60. Mastering Advanced RAG Techniques: A Comprehensive Guide | by Sahin Ahmed, Data Scientist | Medium, accessed October 15, 2025, [https://medium.com/@sahin.samia/mastering-advanced-rag-techniques-a-comprehensive-guide-f0491717998a](https://medium.com/@sahin.samia/mastering-advanced-rag-techniques-a-comprehensive-guide-f0491717998a)  
61. www.ibm.com, accessed October 15, 2025, [https://www.ibm.com/think/topics/agentic-rag#:~:text=Authors&text=Agentic%20RAG%20is%20the%20use,to%20increase%20adaptability%20and%20accuracy.](https://www.ibm.com/think/topics/agentic-rag#:~:text=Authors&text=Agentic%20RAG%20is%20the%20use,to%20increase%20adaptability%20and%20accuracy.)  
62. What is Agentic RAG? | IBM, accessed October 15, 2025, [https://www.ibm.com/think/topics/agentic-rag](https://www.ibm.com/think/topics/agentic-rag)  
63. Understanding Agentic RAG - Arize AI, accessed October 15, 2025, [https://arize.com/blog/understanding-agentic-rag/](https://arize.com/blog/understanding-agentic-rag/)  
64. Agentic RAG: How It Works, Use Cases, Comparison With RAG - DataCamp, accessed October 15, 2025, [https://www.datacamp.com/blog/agentic-rag](https://www.datacamp.com/blog/agentic-rag)  
65. Agentic RAG with Knowledge Graphs for Complex Multi-Hop Reasoning in Real-World Applications - arXiv, accessed October 15, 2025, [https://arxiv.org/html/2507.16507v1](https://arxiv.org/html/2507.16507v1)  
66. Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG - arXiv, accessed October 15, 2025, [https://arxiv.org/html/2501.09136v1](https://arxiv.org/html/2501.09136v1)  
67. [2401.15884] Corrective Retrieval Augmented Generation - arXiv, accessed October 15, 2025, [https://arxiv.org/abs/2401.15884](https://arxiv.org/abs/2401.15884)  
68. What are the common challenges in implementing RAG?, accessed October 15, 2025, [https://www.educative.io/blog/rag-challenges](https://www.educative.io/blog/rag-challenges)  
69. Multi-task retriever fine-tuning for domain-specific and efficient RAG - arXiv, accessed October 15, 2025, [https://arxiv.org/html/2501.04652v2](https://arxiv.org/html/2501.04652v2)  
70. Multi-task retriever fine-tuning for domain-specific and ... - arXiv, accessed October 15, 2025, [https://arxiv.org/abs/2501.04652](https://arxiv.org/abs/2501.04652)  
71. ALoFTRAG: Automatic Local Fine Tuning for Retrieval Augmented Generation - arXiv, accessed October 15, 2025, [https://arxiv.org/html/2501.11929v1](https://arxiv.org/html/2501.11929v1)  
72. Leveraging Fine-Tuned Retrieval-Augmented Generation with Long-Context Support: For 3GPP Standards - arXiv, accessed October 15, 2025, [https://arxiv.org/html/2408.11775v1](https://arxiv.org/html/2408.11775v1)  
73. Finetune-RAG: Fine-Tuning Language Models to Resist ... - arXiv, accessed October 15, 2025, [https://arxiv.org/abs/2505.10792](https://arxiv.org/abs/2505.10792)  
74. A fine-tuning enhanced RAG system with quantized influence measure as AI judge - PMC, accessed October 15, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11551171/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11551171/)  
75. Top 5 RAG Tools to Kickstart your Generative AI Journey - Analytics Vidhya, accessed October 15, 2025, [https://www.analyticsvidhya.com/blog/2024/05/rag-tools/](https://www.analyticsvidhya.com/blog/2024/05/rag-tools/)  
76. LlamaIndex vs. LangChain: Which RAG Tool is Right for You? - n8n Blog, accessed October 15, 2025, [https://blog.n8n.io/llamaindex-vs-langchain/](https://blog.n8n.io/llamaindex-vs-langchain/)  
77. For RAG Devs - langchain or llamaindex? - Reddit, accessed October 15, 2025, [https://www.reddit.com/r/Rag/comments/1g2h7s8/for_rag_devs_langchain_or_llamaindex/](https://www.reddit.com/r/Rag/comments/1g2h7s8/for_rag_devs_langchain_or_llamaindex/)  
78. Top 10 Best Tools and Platforms for Building State-of-the-Art RAG ..., accessed October 15, 2025, [https://www.getmaxim.ai/articles/top-10-best-tools-and-platforms-for-building-state-of-the-art-rag-pipelines-and-applications-a-comprehensive-guide/](https://www.getmaxim.ai/articles/top-10-best-tools-and-platforms-for-building-state-of-the-art-rag-pipelines-and-applications-a-comprehensive-guide/)  
79. Retrieval-Augmented Generation (RAG) using LangChain, LlamaIndex, and OpenAI | by Prasad Mahamulkar | Towards AI, accessed October 15, 2025, [https://pub.towardsai.net/introduction-to-retrieval-augmented-generation-rag-using-langchain-and-lamaindex-bd0047628e2a](https://pub.towardsai.net/introduction-to-retrieval-augmented-generation-rag-using-langchain-and-lamaindex-bd0047628e2a)  
80. The RAG Showdown: LangChain vs. LlamaIndex — Which Tool Reigns Supreme? | by Ajay Verma | Medium, accessed October 15, 2025, [https://medium.com/@ajayverma23/the-rag-showdown-langchain-vs-llamaindex-which-tool-reigns-supreme-f79f6fe80f86](https://medium.com/@ajayverma23/the-rag-showdown-langchain-vs-llamaindex-which-tool-reigns-supreme-f79f6fe80f86)  
81. RAG: Fundamentals, Challenges, and Advanced Techniques | Label Studio, accessed October 15, 2025, [https://labelstud.io/blog/rag-fundamentals-challenges-and-advanced-techniques/](https://labelstud.io/blog/rag-fundamentals-challenges-and-advanced-techniques/)  
82. Top 7 Challenges with Retrieval-Augmented Generation - Valprovia, accessed October 15, 2025, [https://www.valprovia.com/en/blog/top-7-challenges-with-retrieval-augmented-generation](https://www.valprovia.com/en/blog/top-7-challenges-with-retrieval-augmented-generation)  
83. 3 best practices for using retrieval-augmented generation (RAG) - Merge.dev, accessed October 15, 2025, [https://www.merge.dev/blog/rag-best-practices](https://www.merge.dev/blog/rag-best-practices)  
84. RAG Evaluation Metrics Guide: Measure AI Success 2025 - Future AGI, accessed October 15, 2025, [https://futureagi.com/blogs/rag-evaluation-metrics-2025](https://futureagi.com/blogs/rag-evaluation-metrics-2025)  
85. A complete guide to RAG evaluation: metrics, testing and best practices - Evidently AI, accessed October 15, 2025, [https://www.evidentlyai.com/llm-guide/rag-evaluation](https://www.evidentlyai.com/llm-guide/rag-evaluation)  
86. RAG Evaluation Metrics: Assessing Answer ... - Confident AI, accessed October 15, 2025, [https://www.confident-ai.com/blog/rag-evaluation-metrics-answer-relevancy-faithfulness-and-more](https://www.confident-ai.com/blog/rag-evaluation-metrics-answer-relevancy-faithfulness-and-more)  
87. Retrieval-Augmented Generation Metrics - Emergent Mind, accessed October 15, 2025, [https://www.emergentmind.com/topics/retrieval-augmented-generation-rag-metrics](https://www.emergentmind.com/topics/retrieval-augmented-generation-rag-metrics)  
88. RAG Evaluation: Don't let customers tell you first - Pinecone, accessed October 15, 2025, [https://www.pinecone.io/learn/series/vector-databases-in-production-for-busy-engineers/rag-evaluation/](https://www.pinecone.io/learn/series/vector-databases-in-production-for-busy-engineers/rag-evaluation/)  
89. Best 9 RAG Evaluation Tools of 2025 - Deepchecks, accessed October 15, 2025, [https://www.deepchecks.com/best-rag-evaluation-tools/](https://www.deepchecks.com/best-rag-evaluation-tools/)  
90. Top 10 RAG & LLM Evaluation Tools for AI Success - Zilliz Learn, accessed October 15, 2025, [https://zilliz.com/learn/top-ten-rag-and-llm-evaluation-tools-you-dont-want-to-miss](https://zilliz.com/learn/top-ten-rag-and-llm-evaluation-tools-you-dont-want-to-miss)  
91. Top 10 RAG & LLM Evaluation Tools and top 10 vector database - DEV Community, accessed October 15, 2025, [https://dev.to/m_smith_2f854964fdd6/top-10-rag-llm-evaluation-tools-and-top-10-vector-database-4ono](https://dev.to/m_smith_2f854964fdd6/top-10-rag-llm-evaluation-tools-and-top-10-vector-database-4ono)  
92. Top 7 examples of retrieval-augmented generation - Glean, accessed October 15, 2025, [https://www.glean.com/blog/rag-examples](https://www.glean.com/blog/rag-examples)  
93. Top 10 RAG Use Cases and Business Benefits - Uptech, accessed October 15, 2025, [https://www.uptech.team/blog/rag-use-cases](https://www.uptech.team/blog/rag-use-cases)  
94. 10 Real-World Examples of Retrieval Augmented Generation - Signity Solutions, accessed October 15, 2025, [https://www.signitysolutions.com/blog/real-world-examples-of-retrieval-augmented-generation](https://www.signitysolutions.com/blog/real-world-examples-of-retrieval-augmented-generation)  
95. HetaRAG: Hybrid Deep Retrieval-Augmented Generation across Heterogeneous Data Stores - arXiv, accessed October 15, 2025, [https://arxiv.org/html/2509.21336v1](https://arxiv.org/html/2509.21336v1)  
96. HD-RAG: Retrieval-Augmented Generation for Hybrid Documents Containing Text and Hierarchical Tables - arXiv, accessed October 15, 2025, [https://arxiv.org/html/2504.09554v1](https://arxiv.org/html/2504.09554v1)