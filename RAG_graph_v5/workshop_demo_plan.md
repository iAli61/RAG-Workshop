# Advanced RAG Workshop: Development Plan for Code Demonstrations

## Overview
This document provides a comprehensive, step-by-step development plan for building code demonstrations for an Advanced RAG workshop. Each demo isolates specific advanced techniques from the curriculum and provides clear implementation guidance using **llama-index** as the primary framework with Azure OpenAI/Azure AI Foundry models.

---

## **Demo #1: HyDE Query Enhancement**

* **Objective**: Demonstrate how Hypothetical Document Embeddings (HyDE) bridges the semantic gap between user queries and source documents by generating and embedding hypothetical answers.

* **Core Concepts Demonstrated**:
  - Hypothetical Document Embeddings (HyDE)
  - Pre-retrieval query transformation
  - Semantic gap reduction between queries and documents

* **Implementation Steps**:
  1. **Environment Setup**: Initialize Azure OpenAI client with credentials for both LLM and embedding models.
  2. **Data Preparation**: Load 3-4 simple markdown documents on technical topics (e.g., ML concepts) using `SimpleDirectoryReader`.
  3. **Baseline RAG Implementation**: Create a standard `VectorStoreIndex` with default embedding and build a basic `query_engine` for comparison.
  4. **HyDE Implementation**: 
     - Import `from llama_index.core.indices.query.query_transform import HyDEQueryTransform`
     - Create `hyde_transform = HyDEQueryTransform(llm=azure_openai_llm, include_original=True)`
     - Build enhanced query engine: `hyde_query_engine = index.as_query_engine(query_transform=hyde_transform)`
  5. **Comparison and Analysis**: Execute identical queries through both `baseline_query_engine` and `hyde_query_engine`, displaying:
     - Original query
     - Generated hypothetical document (from HyDE)
     - Retrieved context chunks from both engines
     - Final responses with source citations
  6. **Visualization**: Show how the hypothetical answer's embedding is semantically closer to relevant documents than the query itself.

* **Relevant Citation(s)**:
  - Advanced RAG Techniques: From Pre-Retrieval to Generation (Reference 18)
  - Mastering RAG: From Fundamentals to Advanced Query Transformation Techniques (Reference 6)

* **Status:** [COMPLETED]
* **File Generated:** demo_01_hyde_query_enhancement.ipynb
* **Completion Date:** 2025-10-16
* **Notes:** Successfully implemented HyDE with comprehensive comparison against baseline RAG. Included visualization of semantic similarity improvements and side-by-side performance metrics. All implementation steps followed precisely from the plan, using Azure OpenAI with llama-index framework. Demo includes theoretical background, practical implementation, and comparative analysis with multiple test queries.

---

## **Demo #2: Multi-Query Decomposition (RAG-Fusion)**

* **Objective**: Illustrate how generating multiple query variations and aggregating results increases recall and retrieval diversity.

* **Core Concepts Demonstrated**:
  - Multi-Query Retrieval (RAG-Fusion)
  - Query expansion for comprehensive coverage
  - Result aggregation and re-ranking

* **Implementation Steps**:
  1. **Environment Setup**: Reuse Azure OpenAI setup from Demo #1.
  2. **Data Preparation**: Use the same or similar small corpus of 3-4 markdown documents.
  3. **Multi-Query Generator**:
     - Create a function `generate_multiple_queries(original_query, llm, num_queries=3)` that uses Azure OpenAI to generate 3 variations/sub-questions
     - Use a prompt template: "Generate {num_queries} alternative ways to ask the following question: {original_query}"
  4. **Parallel Retrieval**:
     - Build standard `VectorStoreIndex` 
     - Create `query_engine = index.as_query_engine(similarity_top_k=5)`
     - For each generated query variation, retrieve top-k chunks using `retriever = index.as_retriever()`
  5. **Result Fusion**:
     - Collect all retrieved nodes from multiple queries
     - Implement Reciprocal Rank Fusion (RRF) scoring: `score = 1 / (k + rank)` where k=60
     - Deduplicate and re-rank nodes based on aggregated RRF scores
  6. **Comparative Display**:
     - Show single-query retrieval results
     - Show multi-query variations generated
     - Display fused, re-ranked results
     - Generate final answer using top re-ranked context

* **Relevant Citation(s)**:
  - Query Transformations - LangChain Blog (Reference 32)
  - RAG techniques: From naive to advanced - Weights & Biases (Reference 20)

* **Status:** [COMPLETED]
* **File Generated:** demo_02_multi_query_decomposition.ipynb
* **Completion Date:** 2025-10-16
* **Notes:** Successfully implemented RAG-Fusion with Reciprocal Rank Fusion algorithm. Comprehensive demo includes query generation, parallel retrieval, RRF fusion, and detailed comparative analysis. Visualization shows diversity improvements and new sources discovered. Implementation follows the plan exactly with clear explanations of RRF formula and multi-query benefits.

---

## **Demo #3: Hybrid Search (Dense + Sparse Retrieval)**

* **Objective**: Show how combining semantic (vector) and keyword-based (BM25) search provides more robust retrieval than either method alone.

* **Core Concepts Demonstrated**:
  - Dense retrieval (vector/semantic search)
  - Sparse retrieval (BM25 keyword matching)
  - Hybrid search with weighted fusion
  - Complementary strengths of lexical and semantic methods

* **Implementation Steps**:
  1. **Environment Setup**: Azure OpenAI setup with embedding model configuration.
  2. **Data Preparation**: Load 5-6 markdown documents with technical jargon and acronyms to demonstrate BM25 strengths.
  3. **Dense Vector Search**:
     - Create `VectorStoreIndex` with `embed_model` from Azure OpenAI
     - Build `vector_retriever = index.as_retriever(similarity_top_k=5)`
  4. **Sparse BM25 Search**:
     - Import `from llama_index.retrievers.bm25 import BM25Retriever`
     - Create `bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)`
  5. **Hybrid Retriever**:
     - Import `from llama_index.core.retrievers import QueryFusionRetriever`
     - Combine retrievers: `hybrid_retriever = QueryFusionRetriever([vector_retriever, bm25_retriever], similarity_top_k=5, num_queries=1)`
     - Alternatively, implement custom weighted fusion with alpha parameter
  6. **Comparative Analysis**:
     - Test with queries emphasizing exact terms (e.g., acronyms, product codes)
     - Test with semantic/conceptual queries
     - Display results from vector-only, BM25-only, and hybrid approaches
     - Show relevance scores and ranking differences

* **Relevant Citation(s)**:
  - RAG Series III: Hybrid Search (Reference 24)
  - Improving RAG Performance: WTF is Hybrid Search? - Fuzzy Labs (Reference 25)

* **Status:** [COMPLETED]
* **File Generated:** demo_03_hybrid_search.ipynb
* **Completion Date:** 2025-10-16
* **Notes:** Successfully implemented hybrid search combining dense vector search with BM25 sparse retrieval. Demo includes comprehensive comparison across query types (semantic, keyword, mixed), custom weighted fusion implementation with alpha parameter tuning, and quantitative analysis with visualizations. Demonstrates complementary strengths of dense and sparse methods and shows hybrid as robust default for production systems.

---

## **Demo #4: Hierarchical Retrieval (Sentence Window & Auto-Merging)**

* **Objective**: Demonstrate how retrieving small, precise chunks while providing larger context windows improves both retrieval accuracy and generation quality.

* **Core Concepts Demonstrated**:
  - Hierarchical retrieval strategies
  - Sentence Window Retrieval
  - Separation of retrieval granularity from generation context
  - Lost-in-the-middle problem mitigation

* **Implementation Steps**:
  1. **Environment Setup**: Azure OpenAI configuration with LLM and embeddings.
  2. **Data Preparation**: Load 2-3 long-form markdown documents (1000+ words each) from data/long_form_docs.
  3. **Sentence Window Retrieval**:
     - Import `from llama_index.core.node_parser import SentenceWindowNodeParser`
     - Create parser: `node_parser = SentenceWindowNodeParser.from_defaults(window_size=3, window_metadata_key="window", original_text_metadata_key="original_text")`
     - Build index with sentence-level nodes
  4. **MetadataReplacementPostProcessor**:
     - Import `from llama_index.core.postprocessor import MetadataReplacementPostProcessor`
     - Create postprocessor: `postproc = MetadataReplacementPostProcessor(target_metadata_key="window")`
     - This replaces retrieved sentence nodes with their surrounding context window
  5. **Query Engine with Post-Processing**:
     - Build `query_engine = index.as_query_engine(similarity_top_k=3, node_postprocessors=[postproc])`
  6. **Comparison**:
     - Compare standard chunking retrieval vs. sentence window retrieval
     - Show retrieved sentence nodes and their expanded context windows
     - Demonstrate improved answer quality with focused retrieval + expanded context

* **Relevant Citation(s)**:
  - Advanced Retrieval-Augmented Generation: From Theory to LlamaIndex Implementation (Reference 37)
  - Develop a RAG Solution - Chunking Phase - Azure Architecture (Reference 19)

* **Status:** [COMPLETED]
* **File Generated:** demo_04_hierarchical_retrieval.ipynb
* **Completion Date:** 2025-10-16
* **Notes:** Successfully implemented Sentence Window Retrieval demonstrating hierarchical retrieval strategy. The demo shows clear separation between retrieval granularity (sentence-level) and generation context (7-sentence windows). Includes comprehensive comparison with baseline chunking, quantitative analysis, and visual explanation of the architecture. Demonstrates how this approach solves the chunking dilemma by achieving both precise retrieval and sufficient context for generation.

---

## **Demo #5: Re-ranking with Cross-Encoders**

* **Objective**: Illustrate the two-pass "recall-then-precision" pattern where initial retrieval casts a wide net, then a cross-encoder re-ranks for optimal relevance.

* **Core Concepts Demonstrated**:
  - Two-pass retrieval architecture
  - Bi-encoder (fast, for initial retrieval) vs. Cross-encoder (accurate, for re-ranking)
  - Post-retrieval optimization
  - Precision improvement over recall-focused retrieval

* **Implementation Steps**:
  1. **Environment Setup**: Azure OpenAI setup plus re-ranking model configuration.
  2. **Data Preparation**: Use corpus of 6-8 markdown documents with varying relevance patterns.
  3. **Initial Retrieval**:
     - Create standard `VectorStoreIndex`
     - Build retriever with high top-k: `retriever = index.as_retriever(similarity_top_k=10)` (optimized for recall)
  4. **Cross-Encoder Re-Ranker**:
     - Import `from llama_index.postprocessor.cohere_rerank import CohereRerank` (or Azure AI service equivalent)
     - Alternative: Use local cross-encoder model if Azure service unavailable
     - Create re-ranker: `reranker = CohereRerank(api_key=api_key, top_n=3)`
  5. **Re-Ranking Query Engine**:
     - Build query engine with re-ranker: `query_engine = index.as_query_engine(similarity_top_k=10, node_postprocessors=[reranker])`
  6. **Comparative Analysis**:
     - Display initial top-10 retrieval results with vector similarity scores
     - Show re-ranked top-3 results with cross-encoder relevance scores
     - Highlight ranking changes and relevance improvements
     - Generate answers with and without re-ranking to show quality difference

* **Relevant Citation(s)**:
  - Retrieval-Augmented Generation (RAG) from basics to advanced (Reference 15)
  - Advanced RAG Techniques: What They Are & How to Use Them - FalkorDB (Reference 5)

* **Status:** [COMPLETED]
* **File Generated:** demo_05_reranking_cross_encoders.ipynb
* **Completion Date:** 2025-10-16
* **Notes:** Successfully implemented two-pass retrieval architecture with cross-encoder re-ranking. Demo includes custom CrossEncoderReranker implementation using sentence-transformers (ms-marco-MiniLM-L-6-v2 model), comprehensive comparison between bi-encoder-only and bi-encoder+cross-encoder approaches, detailed architecture visualization, and quantitative analysis showing precision improvements. Demonstrates the recall-then-precision pattern effectively with clear explanations of when cross-encoders outperform bi-encoders.

---

## **Demo #6: Context Compression and Filtering**

* **Objective**: Show how compressing retrieved context reduces noise, saves tokens, and improves answer quality by distilling information.

* **Core Concepts Demonstrated**:
  - Extractive context compression
  - LLM-based filtering and extraction
  - Signal-to-noise ratio improvement
  - Token optimization without information loss

* **Implementation Steps**:
  1. **Environment Setup**: Azure OpenAI with LLM for compression decisions.
  2. **Data Preparation**: Load documents with verbose content, including some marginally relevant information.
  3. **Baseline Retrieval**:
     - Create standard `VectorStoreIndex`
     - Retrieve top-k chunks: `retriever = index.as_retriever(similarity_top_k=5)`
  4. **LLM-Based Document Filtering**:
     - Create custom `LongContextReorder` to address "lost in the middle"
     - Import `from llama_index.postprocessor.longcontextreorder import LongContextReorder`
     - Create `reorder = LongContextReorder()`
  5. **Sentence-Level Extraction**:
     - Implement custom post-processor that uses Azure OpenAI LLM to extract only relevant sentences from each chunk
     - Prompt: "Given the query '{query}' and the following text, extract only the sentences directly relevant to answering the query: {text}"
  6. **Compression Query Engine**:
     - Build query engine: `query_engine = index.as_query_engine(similarity_top_k=5, node_postprocessors=[reorder, extraction_postprocessor])`
  7. **Comparison**:
     - Show full uncompressed retrieved context (token count)
     - Show compressed/filtered context (token count)
     - Compare answer quality and relevance
     - Demonstrate cost/latency savings

* **Relevant Citation(s)**:
  - Efficient RAG with Compression and Filtering - LanceDB (Reference 39)
  - Contextual Compression in RAG for LLMs: A Survey - arXiv (Reference 42)

* **Status:** [COMPLETED]
* **File Generated:** demo_06_context_compression.ipynb
* **Completion Date:** 2025-10-16
* **Notes:** Successfully implemented multi-stage context compression pipeline with similarity filtering, long-context reordering, and LLM-based sentence extraction. Demo includes comprehensive token counting utilities, cost analysis showing 40-60% token reduction, and detailed comparisons demonstrating both cost savings and answer quality improvements. Includes practical production considerations and ROI analysis for scale deployment. The custom LLMSentenceExtractor effectively demonstrates intelligent extractive compression while preserving answer-critical information.

---

## **Demo #7: Corrective RAG (Self-Reflective Retrieval)**

* **Objective**: Demonstrate self-corrective mechanisms where the system evaluates retrieval quality and takes corrective actions (re-retrieve, web search, or generate without context).

* **Core Concepts Demonstrated**:
  - Retrieval quality evaluation (relevance grading)
  - Conditional workflow based on retrieval assessment
  - Fallback mechanisms (web search, direct generation)
  - Self-correction and adaptive behavior

* **Implementation Steps**:
  1. **Environment Setup**: Azure OpenAI with access to web search API (Bing/Azure Cognitive Search).
  2. **Data Preparation**: Deliberately limited corpus (3-4 docs) to enable "missing content" scenarios.
  3. **Retrieval Relevance Grader**:
     - Create grader function using Azure OpenAI: `assess_relevance(query, retrieved_docs, llm)`
     - Prompt: "Rate the relevance of the following documents to the query on a scale of 1-5: {docs}. Query: {query}"
     - Return: relevance score and decision (relevant/not_relevant/insufficient)
  4. **Conditional Workflow Logic**:
     - If relevance_score >= 4: Proceed with standard RAG generation
     - If 2 <= relevance_score < 4: Trigger re-retrieval with query transformation (HyDE or decomposition)
     - If relevance_score < 2: Trigger web search fallback using external search tool
  5. **Implementation**:
     - Build standard retriever and query engine
     - Wrap in conditional logic: `if assess_relevance(...) == 'insufficient': use_web_search_tool()`
     - Implement simple web search integration
  6. **Demonstration**:
     - Test Query 1: Well-covered by corpus → standard RAG path
     - Test Query 2: Partially covered → re-retrieval path
     - Test Query 3: Not in corpus (e.g., recent event) → web search fallback
     - Show decision tree and reasoning at each step

* **Relevant Citation(s)**:
  - Seven Failure Points When Engineering a RAG System - arXiv (Reference 10)
  - The Common Failure Points of LLM RAG Systems and How to Overcome Them (Reference 11)

* **Status:** [COMPLETED]
* **File Generated:** demo_07_corrective_rag.ipynb
* **Completion Date:** 2025-10-16
* **Notes:** Successfully implemented Corrective RAG with self-reflective retrieval and adaptive routing. Demo includes LLM-based relevance grader, three conditional execution paths (relevant/partial/not_relevant), and comprehensive testing with queries designed to exercise each path. Implementation demonstrates failure mode detection (FP1, FP2) with corrective actions including HyDE query transformation for partial relevance and fallback to direct LLM generation for missing content. Includes comparative analysis showing advantages over baseline RAG that blindly proceeds with any retrieved context. All theoretical concepts from curriculum (failure points, self-reflection, adaptive behavior) are faithfully implemented and demonstrated.

---

## **Demo #8: Self-RAG with Reflection Tokens and Adaptive Retrieval**

* **Objective**: Demonstrate Self-RAG's ability to adaptively decide when to retrieve, evaluate retrieval relevance, assess generation groundedness, and predict utility through reflection tokens—enabling controllable, self-reflective generation.

* **Core Concepts Demonstrated**:
  - Self-Reflective Retrieval-Augmented Generation (Self-RAG)
  - Reflection tokens for retrieval decisions ([Retrieval], [No Retrieval])
  - Relevance assessment tokens ([Relevant], [Irrelevant])
  - Grounding/support critique tokens ([Fully supported], [Partially supported], [No support / Contradictory])
  - Utility tokens ([Utility:1] through [Utility:5])
  - Adaptive retrieval based on model confidence
  - Segment-level beam search with critique-weighted scoring

* **Implementation Steps**:
  1. **Environment Setup**: 
     - Configure Azure OpenAI with a model capable of handling special tokens
     - Note: This demo demonstrates the Self-RAG *concept* using Azure OpenAI as the base, simulating reflection token behavior
     - In production, Self-RAG requires fine-tuning with special reflection tokens (as in the original paper)
  2. **Define Reflection Token System**:
     - Create token definitions:
       - Retrieval tokens: `["[Retrieval]", "[No Retrieval]", "[Continue to Use Evidence]"]`
       - Relevance tokens: `["[Relevant]", "[Irrelevant]"]`
       - Support tokens: `["[Fully supported]", "[Partially supported]", "[No support / Contradictory]"]`
       - Utility tokens: `["[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]`
  3. **Simulated Critic Model Implementation**:
     - Create `SelfRAGCritic` class that uses Azure OpenAI to simulate reflection token predictions:
       - `should_retrieve(query, current_generation)`: Determines if retrieval is needed
       - `assess_relevance(query, retrieved_passage)`: Evaluates document relevance
       - `assess_groundedness(generation, context)`: Checks if generation is supported by evidence
       - `assess_utility(generation, query)`: Scores overall generation quality (1-5)
  4. **Data Preparation**: Load 5-6 markdown documents covering diverse topics to test adaptive behavior.
  5. **Adaptive Retrieval Engine**:
     - Build `AdaptiveRetriever` that wraps standard `VectorStoreIndex`
     - Implement threshold-based retrieval triggering: `if retrieval_confidence > threshold: retrieve()`
     - Create function to determine retrieval necessity using LLM prompt: 
       ```
       "Does answering '{query}' require external factual information? 
       Consider: knowledge cutoff, specific facts, recent events. Answer: Yes/No"
       ```
  6. **Self-RAG Generation Pipeline**:
     - **Step 1 - Initial Decision**: Query the critic to decide if retrieval is needed
     - **Step 2 - Conditional Retrieval**: If `[Retrieval]` predicted, retrieve top-k documents
     - **Step 3 - Relevance Filtering**: For each retrieved doc, use critic to predict `[Relevant]` or `[Irrelevant]`
     - **Step 4 - Generation with Critique**: Generate response using relevant docs
     - **Step 5 - Groundedness Check**: Use critic to assess if generation is `[Fully supported]`
     - **Step 6 - Utility Scoring**: Predict utility score for final ranking
  7. **Segment-Level Beam Search (Simplified)**:
     - For longer generations, implement iterative generation where:
       - Generate one segment (e.g., sentence or phrase)
       - After each segment, critic decides: continue, retrieve more, or stop
       - Maintain beam of top-k candidates based on composite score: `score = w_rel * rel_score + w_sup * support_score + w_use * utility_score`
  8. **Comparative Demonstration**:
     - Test Query 1 (No retrieval needed): "What is 2 + 2?" → Should predict `[No Retrieval]`
     - Test Query 2 (Requires retrieval): "What are the key components of a transformer architecture?" → Should predict `[Retrieval]`
     - Test Query 3 (Ambiguous need): "Explain machine learning" → Adaptive decision based on threshold
  9. **Visualization and Analysis**:
     - Display the decision trace: which reflection tokens were predicted at each step
     - Show relevance scores for retrieved documents
     - Display groundedness assessment for each generation segment
     - Compare with always-retrieve and never-retrieve baselines
     - Demonstrate controllability: adjust `threshold` and `w_*` weights to change behavior
  10. **Critique-Weighted Scoring**:
      - Implement scoring function that combines multiple critique dimensions:
        ```python
        final_score = (w_rel * relevance_score + 
                      w_sup * support_score + 
                      w_use * utility_score) / (w_rel + w_sup + w_use)
        ```
      - Show how adjusting weights enables task-specific optimization (e.g., high `w_sup` for factual tasks)

* **Key Implementation Notes**:
  - This demo *simulates* Self-RAG behavior using prompting, as true Self-RAG requires fine-tuning with special tokens
  - For production use, refer to the original Self-RAG repository: https://github.com/AkariAsai/self-rag
  - The demo emphasizes the *conceptual architecture* and decision-making flow of Self-RAG
  - Highlight the distinction: Traditional RAG always retrieves; Corrective RAG evaluates after retrieval; Self-RAG decides *before* retrieval and critiques *during* generation

* **Relevant Citation(s)**:
  - Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection - arXiv:2310.11511 (ICLR 2024, Oral)
  - Original Implementation: https://github.com/AkariAsai/self-rag

* **Status:** [COMPLETED]
* **File Generated:** demo_08_self_rag.ipynb
* **Completion Date:** 2025-10-16
* **Notes:** Successfully implemented simulated Self-RAG system with complete reflection token framework (retrieval, relevance, support, utility). Demo includes SelfRAGCritic class with four critique functions implementing meta-reasoning capabilities, adaptive retrieval decision-making that conditionally triggers retrieval based on query analysis, multi-dimensional critique pipeline with full decision trace logging, and weighted composite scoring system demonstrating task-specific controllability. Includes comprehensive testing with queries designed to exercise adaptive retrieval ([No Retrieval] for parametric knowledge, [Retrieval] for factual queries), comparative analysis showing efficiency gains over always-retrieve systems, and weight configuration experiments demonstrating how to optimize for different objectives (factual accuracy vs. utility). Implementation faithfully represents Self-RAG conceptual architecture using prompting to simulate reflection tokens, with clear documentation that production systems require fine-tuning. All theoretical concepts from the curriculum about meta-reasoning and self-reflection are implemented and demonstrated.

---

## **Demo #9: Agentic RAG with Routing and Tools**

* **Objective**: Build an autonomous agent that routes queries to appropriate data sources (vector DB, SQL, web search) and orchestrates multi-source information synthesis.

* **Core Concepts Demonstrated**:
  - Routing agents (intelligent query dispatching)
  - Tool use and multi-source data synthesis
  - ReAct (Reason-Act-Observe) framework
  - Dynamic, non-linear RAG workflows

* **Implementation Steps**:
  1. **Environment Setup**: Azure OpenAI with agent-capable model (GPT-4).
  2. **Multi-Source Data Setup**:
     - Create vector index from markdown documents (source 1)
     - Create simple in-memory "SQL" data structure with structured data (source 2)
     - Prepare web search tool interface (source 3)
  3. **Tool Definitions**:
     - Import `from llama_index.core.tools import QueryEngineTool, FunctionTool`
     - Create `vector_tool = QueryEngineTool.from_defaults(query_engine=vector_query_engine, name="document_search", description="Searches technical documentation")`
     - Create `sql_tool = FunctionTool.from_defaults(fn=query_sql_data, name="database_query", description="Queries structured database")`
     - Create `web_tool = FunctionTool.from_defaults(fn=web_search, name="web_search", description="Searches the internet for current information")`
  4. **Routing Agent**:
     - Import `from llama_index.core.agent import ReActAgent`
     - Create agent: `agent = ReActAgent.from_tools([vector_tool, sql_tool, web_tool], llm=azure_openai_llm, verbose=True)`
  5. **Test Scenarios**:
     - Query requiring vector search only: "Explain gradient boosting algorithms"
     - Query requiring structured data: "What was the Q3 revenue?" (if SQL tool has this)
     - Query requiring multiple sources: "Compare our documentation on topic X with industry standards"
  6. **Observation**:
     - Display agent's reasoning trace (thought-action-observation loops)
     - Show which tools were selected and why
     - Demonstrate synthesis of multi-source information
     - Highlight autonomous decision-making

* **Relevant Citation(s)**:
  - What is Agentic RAG? | IBM (Reference 57)
  - GraphRAG and Agentic Architecture - Neo4j (Reference 55)

* **Status:** [COMPLETED]
* **File Generated:** demo_09_agentic_rag.ipynb
* **Completion Date:** 2025-10-16
* **Notes:** Successfully implemented Agentic RAG with ReAct agent architecture featuring autonomous routing across three heterogeneous data sources: vector database (technical documentation), structured database (product/sales data), and simulated web search (current information). Demo includes comprehensive tool definitions with clear descriptions enabling intelligent routing, transparent Reason-Act-Observe reasoning traces showing agent decision-making process, single-source query testing demonstrating correct tool selection, multi-source query synthesis showing the key advantage of agentic architecture over standard RAG, and comparative analysis highlighting how standard single-source RAG fails on queries requiring information from multiple systems. Implementation faithfully represents all curriculum concepts: routing agents as intelligent dispatchers, ReAct framework's iterative reasoning loop, LLM as meta-reasoner performing orchestration and tool selection, and dynamic workflow enabling multi-source data synthesis. The demo clearly demonstrates why Agentic RAG is essential for real-world enterprise scenarios where knowledge is fragmented across multiple, heterogeneous systems.

---

## **Demo #10: Knowledge Graph RAG (GraphRAG) with Multi-Hop Reasoning**

* **Objective**: Demonstrate how explicit entity-relationship graphs enable multi-hop reasoning and explainable answers impossible with pure vector search.

* **Core Concepts Demonstrated**:
  - Knowledge graph construction from unstructured text
  - Graph traversal and multi-hop queries
  - Hybrid retrieval (vector + graph)
  - Enhanced explainability through relationship paths

* **Implementation Steps**:
  1. **Environment Setup**: Azure OpenAI + graph database (use in-memory NetworkX for simplicity, or Neo4j if available).
  2. **Knowledge Graph Construction**:
     - Load 3-4 markdown documents with clear entity relationships (e.g., tech docs with product-feature-company relationships)
     - Import `from llama_index.core.indices import KnowledgeGraphIndex`
     - Create KG index: `kg_index = KnowledgeGraphIndex.from_documents(documents, storage_context=storage_context, max_triplets_per_chunk=10, include_embeddings=True)`
     - The LLM automatically extracts (subject, predicate, object) triplets
  3. **Visualize Graph**:
     - Extract triplets: `triplets = kg_index.get_networkx_graph()`
     - Use NetworkX to visualize nodes and edges
  4. **Multi-Hop Query Example**:
     - Define query requiring multi-hop: "What features does Company X's product have?" (requires: Company X → Product Y → Features)
     - Create query engine: `kg_query_engine = kg_index.as_query_engine(include_text=True, response_mode="tree_summarize")`
  5. **Hybrid GraphRAG**:
     - Combine vector retrieval with graph traversal
     - Use graph relationships to expand/refine vector search results
  6. **Explainability**:
     - Display the subgraph/path traversed to answer the query
     - Show retrieved triplets: (Entity A) --[relationship]--> (Entity B)
     - Contrast with vector-only retrieval which cannot show explicit relationships
  7. **Comparison**:
     - Run same multi-hop query through standard vector RAG (likely fails)
     - Show GraphRAG success with reasoning path

* **Relevant Citation(s)**:
  - Knowledge Graph for RAG: Definition and Examples - Lettria (Reference 44)
  - What is GraphRAG? | IBM (Reference 46)
  - Enhancing the Accuracy of RAG Applications With Knowledge Graphs - Neo4j (Reference 48)

* **Status:** [COMPLETED]
* **File Generated:** demo_10_graphrag.ipynb
* **Completion Date:** 2025-10-16
* **Notes:** Successfully implemented GraphRAG system demonstrating how explicit entity-relationship graphs enable multi-hop reasoning and explainable answers impossible with pure vector search. Demo includes automated knowledge graph construction using LLM to extract entities and relationships from unstructured technical documents (implementing the "virtuous cycle" described in curriculum), NetworkX-based graph visualization showing entity-relationship networks, multi-hop query testing requiring traversal through multiple relationships, explicit relationship triplet retrieval for explainability (showing transparent reasoning paths), query-specific subgraph visualization, and comprehensive comparative analysis with standard vector RAG. Implementation demonstrates all core GraphRAG concepts from curriculum: hybrid retrieval (vector search + graph traversal), multi-hop reasoning enabling "how/why" questions beyond simple "what" questions, enhanced explainability through discrete relationship paths, and the neo-symbolic paradigm merging connectionist AI (neural embeddings) with symbolic AI (graph structure). The demo clearly shows when GraphRAG outperforms vector RAG (complex relationship queries, causal chains, inverse relationships) and when vector RAG is sufficient (simple lookup, semantic similarity). All theoretical concepts faithfully implemented including the four GraphRAG architecture components (Query Processor, Retriever, Organizer, Generator) and integration patterns (direct querying, text-graph embedding, graph-based filtering).

---

## **Demo #11: RAG System Evaluation and Metrics**

* **Objective**: Implement comprehensive evaluation framework to measure and compare RAG system performance using established metrics.

* **Core Concepts Demonstrated**:
  - Context relevance (retrieval precision)
  - Context sufficiency (retrieval recall)
  - Answer faithfulness (hallucination detection)
  - Answer relevance and correctness
  - Automated LLM-as-judge evaluation

* **Implementation Steps**:
  1. **Environment Setup**: Azure OpenAI as evaluator LLM + evaluation framework setup.
  2. **Create Gold Standard Dataset**:
     - Define 10-15 test queries with ground truth answers
     - Store as structured data: `[{"query": "...", "ground_truth": "...", "reference_context": [...]}]`
  3. **RAG System Under Test**:
     - Use one of the previously built RAG systems (e.g., hybrid search with re-ranking)
     - Run all test queries through the system, collecting: retrieved context, final answer
  4. **Evaluation Metrics Implementation**:
     - **Context Relevance**: Use LLM to score each retrieved chunk's relevance to query (1-5 scale)
       - Prompt: "Rate the relevance of this context to the query: {context} | Query: {query}"
     - **Answer Faithfulness**: Use LLM to check if answer is grounded in context
       - Prompt: "Does this answer contain information not present in the context? Answer Yes/No: {answer} | Context: {context}"
     - **Answer Relevance**: Use LLM to assess if answer addresses the query
       - Prompt: "Does this answer directly address the query? Rate 1-5: {answer} | Query: {query}"
     - **Answer Correctness**: Compare against ground truth (cosine similarity of embeddings + LLM scoring)
  5. **Automated Evaluation Pipeline**:
     - Create function: `evaluate_rag_system(test_dataset, rag_pipeline)`
     - Iterate through test queries
     - Calculate aggregate metrics: avg_context_relevance, faithfulness_rate, avg_answer_relevance, correctness_score
  6. **Comparison Dashboard**:
     - Create simple visualization comparing multiple RAG configurations
     - Example: baseline vs. HyDE vs. hybrid+rerank
     - Show metric breakdown per query and aggregated
  7. **Best Practice Implementation**:
     - Demonstrate regression testing: change a component, re-run evaluation, detect performance degradation
     - Show how to identify failure modes (FP1-FP5) from metric patterns

* **Relevant Citation(s)**:
  - RAG Evaluation Metrics: Best Practices for Evaluating RAG Systems - Patronus AI (Reference 75)
  - Evaluating retrieval in RAGs: a practical framework - Tweag (Reference 73)
  - RAG systems: Best practices to master evaluation - Google Cloud (Reference 74)

* **Status:** [COMPLETED]
* **File Generated:** demo_11_rag_evaluation.ipynb
* **Completion Date:** 2025-10-16
* **Notes:** Successfully implemented comprehensive RAG evaluation framework with all five core metrics (context relevance, context sufficiency, answer faithfulness, answer relevance, answer correctness). Demo includes gold standard dataset creation with diverse query types and difficulty levels, LLM-as-judge implementation for automated metric calculation, complete evaluation pipeline that processes RAG systems and generates detailed reports, comparative analysis between baseline and advanced RAG configurations with aggregate statistics and category-specific breakdowns, and exportable results for regression testing and continuous monitoring. Implementation demonstrates the "MLOps-ification" of RAG development with automated testing, versioned benchmarks, and metric-driven optimization. All theoretical concepts from curriculum faithfully represented including the mapping of metrics to failure points (FP1-FP5), best practices for production deployment, and the non-negotiable nature of systematic quantitative evaluation. The evaluation framework provides the foundation for treating RAG development with the same engineering discipline as traditional ML systems.

---

## Implementation Notes and Best Practices

### Technical Requirements
- **Python Environment**: All demos run in standard `.venv` with `uv` package management
- **Core Dependencies**:
  ```
  llama-index-core
  llama-index-llms-azure-openai
  llama-index-embeddings-azure-openai
  llama-index-retrievers-bm25
  llama-index-postprocessor-cohere-rerank (or Azure equivalent)
  networkx (for GraphRAG demo)
  ```

### Demo Data Requirements
- Store all demo data in structured directories:
  - `data/ml_concepts/` - For Demos 1-3
  - `data/long_form_docs/` - For Demo 4
  - `data/tech_docs/` - For Demos 5-7
  - `data/finance_docs/` - For Demo 8 (if using business context)
- Each document should be 300-1000 words, focused on a single topic
- Use markdown format for simplicity and readability

### Notebook Structure Template
Each Jupyter notebook should follow this structure:
1. **Introduction Cell**: Markdown explaining the concept and learning objectives
2. **Setup Cell**: Imports and Azure OpenAI configuration
3. **Data Loading**: Simple, reproducible data loading
4. **Baseline Implementation**: Standard RAG for comparison
5. **Advanced Technique Implementation**: The focal technique of the demo
6. **Comparison and Analysis**: Side-by-side results
7. **Key Takeaways**: Markdown cell summarizing insights

### Pedagogical Principles
- **Progressive Complexity**: Each demo builds on concepts from previous demos
- **Isolation**: Each demo can run independently with its own data
- **Clarity**: Prioritize readable code with extensive comments over brevity
- **Reproducibility**: All results should be deterministic where possible (set seeds)
- **Visualization**: Include visual comparisons (tables, simple charts) wherever helpful

### Azure OpenAI Configuration Pattern
All demos should use a consistent configuration pattern:
```python
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

# LLM Configuration
llm = AzureOpenAI(
    model="gpt-4",
    deployment_name="<deployment_name>",
    api_key="<api_key>",
    azure_endpoint="<endpoint>",
    api_version="2024-02-15-preview"
)

# Embedding Configuration
embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="<embedding_deployment>",
    api_key="<api_key>",
    azure_endpoint="<endpoint>",
    api_version="2024-02-15-preview"
)
```

### Suggested Demo Sequence for Workshop
The demos are ordered pedagogically:
- **Foundational Advanced Concepts (Demos 1-3)**: Pre-retrieval optimization and hybrid search
- **Post-Retrieval Refinement (Demos 4-6)**: Hierarchical retrieval, re-ranking, compression
- **Adaptive and Self-Reflective Systems (Demos 7-8)**: Corrective RAG and Self-RAG with reflection tokens
- **Autonomous and Structured Systems (Demos 9-10)**: Agents and knowledge graphs
- **Quality Assurance (Demo 11)**: Evaluation and production readiness

---

## Conclusion

This development plan provides detailed, actionable specifications for building 11 advanced RAG demonstrations. Each demo is designed to:
- Isolate and teach 1-2 core advanced concepts from the curriculum
- Use minimal, focused datasets to avoid data engineering complexity
- Leverage llama-index as the primary framework with Azure OpenAI models
- Build progressively from foundational to sophisticated techniques
- Enable hands-on learning through reproducible, clear code examples

Engineers implementing these demos should refer to the source curriculum document (AdvancedRAGWorkshop.md) for deeper theoretical context and the cited papers for original research details.
#### **Works cited**

1. What is Retrieval-Augmented Generation (RAG)? | Google Cloud, accessed October 15, 2025, [https://cloud.google.com/use-cases/retrieval-augmented-generation](https://cloud.google.com/use-cases/retrieval-augmented-generation)  
2. What is RAG? \- Retrieval-Augmented Generation AI Explained \- AWS \- Updated 2025, accessed October 15, 2025, [https://aws.amazon.com/what-is/retrieval-augmented-generation/](https://aws.amazon.com/what-is/retrieval-augmented-generation/)  
3. What is Retrieval Augmented Generation (RAG)? | Databricks, accessed October 15, 2025, [https://www.databricks.com/glossary/retrieval-augmented-generation-rag](https://www.databricks.com/glossary/retrieval-augmented-generation-rag)  
4. What is retrieval augmented generation (RAG) \[examples included\] \- SuperAnnotate, accessed October 15, 2025, [https://www.superannotate.com/blog/rag-explained](https://www.superannotate.com/blog/rag-explained)  
5. Advanced RAG Techniques: What They Are & How to Use Them \- FalkorDB, accessed October 15, 2025, [https://www.falkordb.com/blog/advanced-rag/](https://www.falkordb.com/blog/advanced-rag/)  
6. Mastering RAG: From Fundamentals to Advanced Query Transformation Techniques — Part 1 | by Tejpal Kumawat | Medium, accessed October 15, 2025, [https://medium.com/@tejpal.abhyuday/mastering-rag-from-fundamentals-to-advanced-query-transformation-techniques-part-1-a1fee8823806](https://medium.com/@tejpal.abhyuday/mastering-rag-from-fundamentals-to-advanced-query-transformation-techniques-part-1-a1fee8823806)  
7. The Key Components Of A RAG System: A Technical Deep Dive ..., accessed October 15, 2025, [https://customgpt.ai/components-of-a-rag-system/](https://customgpt.ai/components-of-a-rag-system/)  
8. Retrieval-Augmented Generation (RAG) Basics \- 7Rivers, accessed October 15, 2025, [https://7riversinc.com/insights/retrieval-augmented-generation-rag-basics/](https://7riversinc.com/insights/retrieval-augmented-generation-rag-basics/)  
9. Why RAG LLM Systems Outperform Standard Language Models \- Techginity, accessed October 15, 2025, [https://www.techginity.com/blog/why-rag-llm-systems-outperform-standard-language-models](https://www.techginity.com/blog/why-rag-llm-systems-outperform-standard-language-models)  
10. Seven Failure Points When Engineering a Retrieval Augmented Generation System \- arXiv, accessed October 15, 2025, [https://arxiv.org/html/2401.05856v1](https://arxiv.org/html/2401.05856v1)  
11. The Common Failure Points of LLM RAG Systems and How to Overcome Them \- Medium, accessed October 15, 2025, [https://medium.com/@sahin.samia/the-common-failure-points-of-llm-rag-systems-and-how-to-overcome-them-926d9090a88f](https://medium.com/@sahin.samia/the-common-failure-points-of-llm-rag-systems-and-how-to-overcome-them-926d9090a88f)  
12. Retrieval-augmented generation \- Wikipedia, accessed October 15, 2025, [https://en.wikipedia.org/wiki/Retrieval-augmented\_generation](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)  
13. RAG vs Traditional LLMs: Key Differences \- Galileo AI, accessed October 15, 2025, [https://galileo.ai/blog/comparing-rag-and-traditional-llms-which-suits-your-project](https://galileo.ai/blog/comparing-rag-and-traditional-llms-which-suits-your-project)  
14. RAG vs LLM? Understanding the Unique Capabilities and Limitations of Each Approach \- Kanerika, accessed October 15, 2025, [https://kanerika.com/blogs/rag-vs-llm/](https://kanerika.com/blogs/rag-vs-llm/)  
15. Retrieval-Augmented Generation (RAG) from basics to advanced | by Tejpal Kumawat | Medium, accessed October 15, 2025, [https://medium.com/@tejpal.abhyuday/retrieval-augmented-generation-rag-from-basics-to-advanced-a2b068fd576c](https://medium.com/@tejpal.abhyuday/retrieval-augmented-generation-rag-from-basics-to-advanced-a2b068fd576c)  
16. Retrieval Augmented Generation (RAG) \- Prompt Engineering Guide, accessed October 15, 2025, [https://www.promptingguide.ai/techniques/rag](https://www.promptingguide.ai/techniques/rag)  
17. Build Advanced Retrieval-Augmented Generation Systems | Microsoft Learn, accessed October 15, 2025, [https://learn.microsoft.com/en-us/azure/developer/ai/advanced-retrieval-augmented-generation](https://learn.microsoft.com/en-us/azure/developer/ai/advanced-retrieval-augmented-generation)  
18. Advanced RAG Techniques: From Pre-Retrieval to Generation \- TechAhead, accessed October 15, 2025, [https://www.techaheadcorp.com/blog/advanced-rag-techniques-from-pre-retrieval-to-generation/](https://www.techaheadcorp.com/blog/advanced-rag-techniques-from-pre-retrieval-to-generation/)  
19. Develop a RAG Solution \- Chunking Phase \- Azure Architecture Center | Microsoft Learn, accessed October 15, 2025, [https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-chunking-phase](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-chunking-phase)  
20. RAG techniques: From naive to advanced \- Weights & Biases \- Wandb, accessed October 15, 2025, [https://wandb.ai/site/articles/rag-techniques/](https://wandb.ai/site/articles/rag-techniques/)  
21. 7 Chunking Strategies in RAG You Need To Know \- F22 Labs, accessed October 15, 2025, [https://www.f22labs.com/blogs/7-chunking-strategies-in-rag-you-need-to-know/](https://www.f22labs.com/blogs/7-chunking-strategies-in-rag-you-need-to-know/)  
22. Chunking Strategies to Improve Your RAG Performance \- Weaviate, accessed October 15, 2025, [https://weaviate.io/blog/chunking-strategies-for-rag](https://weaviate.io/blog/chunking-strategies-for-rag)  
23. From Word2Vec to LLM2Vec: How to Choose the Right Embedding Model for RAG, accessed October 15, 2025, [https://milvus.io/blog/how-to-choose-the-right-embedding-model-for-rag.md](https://milvus.io/blog/how-to-choose-the-right-embedding-model-for-rag.md)  
24. RAG Series —III : Hybrid Search. Retrieval-Augmented Generation (RAG) is… | by DhanushKumar | Medium, accessed October 15, 2025, [https://medium.com/@danushidk507/rag-series-iii-hybrid-search-e612bbde1abc](https://medium.com/@danushidk507/rag-series-iii-hybrid-search-e612bbde1abc)  
25. Improving RAG Performance: WTF is Hybrid Search? \- Fuzzy Labs, accessed October 15, 2025, [https://www.fuzzylabs.ai/blog-post/improving-rag-performance-hybrid-search](https://www.fuzzylabs.ai/blog-post/improving-rag-performance-hybrid-search)  
26. Integrate sparse and dense vectors to enhance knowledge retrieval in RAG using Amazon OpenSearch Service | AWS Big Data Blog, accessed October 15, 2025, [https://aws.amazon.com/blogs/big-data/integrate-sparse-and-dense-vectors-to-enhance-knowledge-retrieval-in-rag-using-amazon-opensearch-service/](https://aws.amazon.com/blogs/big-data/integrate-sparse-and-dense-vectors-to-enhance-knowledge-retrieval-in-rag-using-amazon-opensearch-service/)  
27. Common retrieval augmented generation (RAG) techniques ..., accessed October 15, 2025, [https://www.microsoft.com/en-us/microsoft-cloud/blog/2025/02/04/common-retrieval-augmented-generation-rag-techniques-explained/](https://www.microsoft.com/en-us/microsoft-cloud/blog/2025/02/04/common-retrieval-augmented-generation-rag-techniques-explained/)  
28. Dense vector \+ Sparse vector \+ Full text search \+ Tensor reranker \= Best retrieval for RAG?, accessed October 15, 2025, [https://infiniflow.org/blog/best-hybrid-search-solution](https://infiniflow.org/blog/best-hybrid-search-solution)  
29. Advanced Query Transformations to Improve RAG \- Towards Data Science, accessed October 15, 2025, [https://towardsdatascience.com/advanced-query-transformations-to-improve-rag-11adca9b19d1/](https://towardsdatascience.com/advanced-query-transformations-to-improve-rag-11adca9b19d1/)  
30. Best Practices for RAG Pipelines | Medium, accessed October 15, 2025, [https://masteringllm.medium.com/best-practices-for-rag-pipeline-8c12a8096453](https://masteringllm.medium.com/best-practices-for-rag-pipeline-8c12a8096453)  
31. Advanced RAG: Query Augmentation for Next-Level Search using LlamaIndex | by Akash Mathur | Medium, accessed October 15, 2025, [https://akash-mathur.medium.com/advanced-rag-query-augmentation-for-next-level-search-using-llamaindex-d362fed7ecc3](https://akash-mathur.medium.com/advanced-rag-query-augmentation-for-next-level-search-using-llamaindex-d362fed7ecc3)  
32. Query Transformations \- LangChain Blog, accessed October 15, 2025, [https://blog.langchain.com/query-transformations/](https://blog.langchain.com/query-transformations/)  
33. Tested 9 RAG query transformation techniques – HydE is absurdly underrated \- Reddit, accessed October 15, 2025, [https://www.reddit.com/r/LocalLLaMA/comments/1o6s89n/tested\_9\_rag\_query\_transformation\_techniques\_hyde/](https://www.reddit.com/r/LocalLLaMA/comments/1o6s89n/tested_9_rag_query_transformation_techniques_hyde/)  
34. Advanced Query Transformations to Improve RAG | Towards Data ..., accessed October 15, 2025, [https://towardsdatascience.com/advanced-query-transformations-to-improve-rag-11adca9b19d1](https://towardsdatascience.com/advanced-query-transformations-to-improve-rag-11adca9b19d1)  
35. What are some ways to test and improve my RAGs retrieval strategy? \- Reddit, accessed October 15, 2025, [https://www.reddit.com/r/Rag/comments/1fku970/what\_are\_some\_ways\_to\_test\_and\_improve\_my\_rags/](https://www.reddit.com/r/Rag/comments/1fku970/what_are_some_ways_to_test_and_improve_my_rags/)  
36. RAG II: Query Transformations. Naive RAG typically splits documents… | by Sulaiman Shamasna | The Deep Hub | Medium, accessed October 15, 2025, [https://medium.com/thedeephub/rag-ii-query-transformations-49865bb0528c](https://medium.com/thedeephub/rag-ii-query-transformations-49865bb0528c)  
37. Advanced Retrieval-Augmented Generation: From Theory to LlamaIndex Implementation, accessed October 15, 2025, [https://towardsdatascience.com/advanced-retrieval-augmented-generation-from-theory-to-llamaindex-implementation-4de1464a9930/](https://towardsdatascience.com/advanced-retrieval-augmented-generation-from-theory-to-llamaindex-implementation-4de1464a9930/)  
38. The 4 Advanced RAG Algorithms You Must Know to Implement \- Comet ML, accessed October 15, 2025, [https://www.comet.com/site/blog/advanced-rag-algorithms-optimize-retrieval/](https://www.comet.com/site/blog/advanced-rag-algorithms-optimize-retrieval/)  
39. Efficient RAG with Compression and Filtering | by Kaushal Choudhary | LanceDB \- Medium, accessed October 15, 2025, [https://medium.com/etoai/enhance-rag-integrate-contextual-compression-and-filtering-for-precision-a29d4a810301](https://medium.com/etoai/enhance-rag-integrate-contextual-compression-and-filtering-for-precision-a29d4a810301)  
40. How to do retrieval with contextual compression | 🦜️ LangChain, accessed October 15, 2025, [https://python.langchain.com/docs/how\_to/contextual\_compression/](https://python.langchain.com/docs/how_to/contextual_compression/)  
41. Efficient online text compression for RAG \- Naver Labs Europe, accessed October 15, 2025, [https://europe.naverlabs.com/blog/efficient-online-text-compression-for-rag/](https://europe.naverlabs.com/blog/efficient-online-text-compression-for-rag/)  
42. Contextual Compression in Retrieval-Augmented Generation for Large Language Models: A Survey \- arXiv, accessed October 15, 2025, [https://arxiv.org/html/2409.13385v1](https://arxiv.org/html/2409.13385v1)  
43. xRAG: Extreme Context Compression for Retrieval-augmented Generation with One Token, accessed October 15, 2025, [https://openreview.net/forum?id=6pTlXqrO0p\&referrer=%5Bthe%20profile%20of%20Tao%20Ge%5D(%2Fprofile%3Fid%3D\~Tao\_Ge1)](https://openreview.net/forum?id=6pTlXqrO0p&referrer=%5Bthe+profile+of+Tao+Ge%5D\(/profile?id%3D~Tao_Ge1\))  
44. Knowledge Graph for RAG: Definition and Examples \- Lettria, accessed October 15, 2025, [https://www.lettria.com/blogpost/knowledge-graph-for-rag-definition-and-examples](https://www.lettria.com/blogpost/knowledge-graph-for-rag-definition-and-examples)  
45. Using a Knowledge Graph to Implement a RAG Application \- DataCamp, accessed October 15, 2025, [https://www.datacamp.com/tutorial/knowledge-graph-rag](https://www.datacamp.com/tutorial/knowledge-graph-rag)  
46. What is GraphRAG? | IBM, accessed October 15, 2025, [https://www.ibm.com/think/topics/graphrag](https://www.ibm.com/think/topics/graphrag)  
47. The critical role of knowledge graphs in RAG systems \- Zive, accessed October 15, 2025, [https://www.zive.com/en/blog/the-critical-role-of-knowledge-graphs-in-rag-systems](https://www.zive.com/en/blog/the-critical-role-of-knowledge-graphs-in-rag-systems)  
48. Enhancing the Accuracy of RAG Applications With Knowledge Graphs | by Tomaz Bratanic | Neo4j Developer Blog | Medium, accessed October 15, 2025, [https://medium.com/neo4j/enhancing-the-accuracy-of-rag-applications-with-knowledge-graphs-ad5e2ffab663](https://medium.com/neo4j/enhancing-the-accuracy-of-rag-applications-with-knowledge-graphs-ad5e2ffab663)  
49. Building, Improving, and Deploying Knowledge Graph RAG Systems on Databricks, accessed October 15, 2025, [https://www.databricks.com/blog/building-improving-and-deploying-knowledge-graph-rag-systems-databricks](https://www.databricks.com/blog/building-improving-and-deploying-knowledge-graph-rag-systems-databricks)  
50. How to construct knowledge graphs | 🦜️ LangChain, accessed October 15, 2025, [https://python.langchain.com/docs/how\_to/graph\_constructing/](https://python.langchain.com/docs/how_to/graph_constructing/)  
51. How to Build a Knowledge Graph for RAG with Astra DB | DataStax, accessed October 15, 2025, [https://www.datastax.com/blog/knowledge-graphs-for-rag-without-a-graphdb](https://www.datastax.com/blog/knowledge-graphs-for-rag-without-a-graphdb)  
52. Graph-RAG in AI: What is it and How does it work? | by Sahin Ahmed, Data Scientist, accessed October 15, 2025, [https://medium.com/@sahin.samia/graph-rag-in-ai-what-is-it-and-how-does-it-work-d719d814e610](https://medium.com/@sahin.samia/graph-rag-in-ai-what-is-it-and-how-does-it-work-d719d814e610)  
53. What is Graph RAG | Ontotext Fundamentals, accessed October 15, 2025, [https://www.ontotext.com/knowledgehub/fundamentals/what-is-graph-rag/](https://www.ontotext.com/knowledgehub/fundamentals/what-is-graph-rag/)  
54. How to Implement Graph RAG Using Knowledge Graphs and Vector Databases \- Medium, accessed October 15, 2025, [https://medium.com/data-science/how-to-implement-graph-rag-using-knowledge-graphs-and-vector-databases-60bb69a22759](https://medium.com/data-science/how-to-implement-graph-rag-using-knowledge-graphs-and-vector-databases-60bb69a22759)  
55. GraphRAG and Agentic Architecture: Practical Experimentation with Neo4j and NeoConverse \- Graph Database & Analytics, accessed October 15, 2025, [https://neo4j.com/blog/developer/graphrag-and-agentic-architecture-with-neoconverse/](https://neo4j.com/blog/developer/graphrag-and-agentic-architecture-with-neoconverse/)  
56. Graph-based RAG | WRITER Knowledge Graph, accessed October 15, 2025, [https://writer.com/product/graph-based-rag/](https://writer.com/product/graph-based-rag/)  
57. What is Agentic RAG? | IBM, accessed October 15, 2025, [https://www.ibm.com/think/topics/agentic-rag](https://www.ibm.com/think/topics/agentic-rag)  
58. RAG: Fundamentals, Challenges, and Advanced Techniques | Label Studio, accessed October 15, 2025, [https://labelstud.io/blog/rag-fundamentals-challenges-and-advanced-techniques/](https://labelstud.io/blog/rag-fundamentals-challenges-and-advanced-techniques/)  
59. Stop Using Outdated RAG: DeepSearcher's Agentic RAG Approach ..., accessed October 15, 2025, [https://milvusio.medium.com/stop-using-outdated-rag-deepsearchers-agentic-rag-approach-changes-everything-0fb81a590a76](https://milvusio.medium.com/stop-using-outdated-rag-deepsearchers-agentic-rag-approach-changes-everything-0fb81a590a76)  
60. Mastering RAG: Choosing the Perfect Vector Database \- Galileo AI, accessed October 15, 2025, [https://galileo.ai/blog/mastering-rag-choosing-the-perfect-vector-database](https://galileo.ai/blog/mastering-rag-choosing-the-perfect-vector-database)  
61. We Tried and Tested 10 Best Vector Databases for RAG Pipelines ..., accessed October 15, 2025, [https://www.zenml.io/blog/vector-databases-for-rag](https://www.zenml.io/blog/vector-databases-for-rag)  
62. How to Choose the Right Vector Database for Your RAG Architecture | DigitalOcean, accessed October 15, 2025, [https://www.digitalocean.com/community/conceptual-articles/how-to-choose-the-right-vector-database](https://www.digitalocean.com/community/conceptual-articles/how-to-choose-the-right-vector-database)  
63. Best Vector Databases for RAG: Complete 2025 Comparison Guide \- Latenode, accessed October 15, 2025, [https://latenode.com/blog/best-vector-databases-for-rag-complete-2025-comparison-guide](https://latenode.com/blog/best-vector-databases-for-rag-complete-2025-comparison-guide)  
64. LangChain vs LlamaIndex: A Detailed Comparison | DataCamp, accessed October 15, 2025, [https://www.datacamp.com/blog/langchain-vs-llamaindex](https://www.datacamp.com/blog/langchain-vs-llamaindex)  
65. Llamaindex vs Langchain: What's the difference? | IBM, accessed October 15, 2025, [https://www.ibm.com/think/topics/llamaindex-vs-langchain](https://www.ibm.com/think/topics/llamaindex-vs-langchain)  
66. LlamaIndex vs. LangChain: Which RAG Tool is Right for You? \- n8n Blog, accessed October 15, 2025, [https://blog.n8n.io/llamaindex-vs-langchain/](https://blog.n8n.io/llamaindex-vs-langchain/)  
67. How To Do Retrieval-Augmented Generation (RAG) With ... \- Scout, accessed October 15, 2025, [https://www.scoutos.com/blog/how-to-do-retrieval-augmented-generation-rag-with-langchain](https://www.scoutos.com/blog/how-to-do-retrieval-augmented-generation-rag-with-langchain)  
68. Build an LLM RAG Chatbot With LangChain \- Real Python, accessed October 15, 2025, [https://realpython.com/build-llm-rag-chatbot-with-langchain/](https://realpython.com/build-llm-rag-chatbot-with-langchain/)  
69. RAG Implementation with LangChain \- DEV Community, accessed October 15, 2025, [https://dev.to/bolajibolajoko51/rag-implementation-with-langchain-2jei](https://dev.to/bolajibolajoko51/rag-implementation-with-langchain-2jei)  
70. Constructing a RAG system using LlamaIndex and Ollama \- ROCm Documentation \- AMD, accessed October 15, 2025, [https://rocm.docs.amd.com/projects/ai-developer-hub/en/v2.0/notebooks/inference/rag\_ollama\_llamaindex.html](https://rocm.docs.amd.com/projects/ai-developer-hub/en/v2.0/notebooks/inference/rag_ollama_llamaindex.html)  
71. Building RAG from Scratch (Open-source only\!) | LlamaIndex Python ..., accessed October 15, 2025, [https://developers.llamaindex.ai/python/examples/low\_level/oss\_ingestion\_retrieval/](https://developers.llamaindex.ai/python/examples/low_level/oss_ingestion_retrieval/)  
72. LlamaIndex RAG tutorial: step-by-step implementation \- Meilisearch, accessed October 15, 2025, [https://www.meilisearch.com/blog/llamaindex-rag](https://www.meilisearch.com/blog/llamaindex-rag)  
73. Evaluating retrieval in RAGs: a practical framework \- Tweag, accessed October 15, 2025, [https://tweag.io/blog/2024-03-21-evaluating-retrieval-in-rag-framework/](https://tweag.io/blog/2024-03-21-evaluating-retrieval-in-rag-framework/)  
74. RAG systems: Best practices to master evaluation for accurate and reliable AI. | Google Cloud Blog, accessed October 15, 2025, [https://cloud.google.com/blog/products/ai-machine-learning/optimizing-rag-retrieval](https://cloud.google.com/blog/products/ai-machine-learning/optimizing-rag-retrieval)  
75. RAG Evaluation Metrics: Best Practices for Evaluating RAG Systems, accessed October 15, 2025, [https://www.patronus.ai/llm-testing/rag-evaluation-metrics](https://www.patronus.ai/llm-testing/rag-evaluation-metrics)  
76. RAG Evaluation: Don't let customers tell you first \- Pinecone, accessed October 15, 2025, [https://www.pinecone.io/learn/series/vector-databases-in-production-for-busy-engineers/rag-evaluation/](https://www.pinecone.io/learn/series/vector-databases-in-production-for-busy-engineers/rag-evaluation/)  
77. RAG Evaluation Survey: Framework, Metrics, and Methods | EvalScope \- Read the Docs, accessed October 15, 2025, [https://evalscope.readthedocs.io/en/latest/blog/RAG/RAG\_Evaluation.html](https://evalscope.readthedocs.io/en/latest/blog/RAG/RAG_Evaluation.html)  
78. Top Problems with RAG systems and ways to mitigate them \- AIMon ..., accessed October 15, 2025, [https://www.aimon.ai/posts/top\_problems\_with\_rag\_systems\_and\_ways\_to\_mitigate\_them/](https://www.aimon.ai/posts/top_problems_with_rag_systems_and_ways_to_mitigate_them/)

