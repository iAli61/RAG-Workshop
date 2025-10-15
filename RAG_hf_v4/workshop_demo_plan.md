# **Advanced RAG Workshop: Demo Development Plan**

## **Workshop Overview**

This development plan outlines a progressive series of hands-on demonstrations for an Advanced RAG workshop. Each demo isolates and teaches 1-2 advanced concepts, building from foundational enhancements to frontier techniques. All implementations use **llama-index** with **Azure OpenAI** models and are designed for reproducibility in a standard `.venv` environment.

---

## **Recommended Hugging Face Resources for RAG Workshop**

### **Embedding Models (Alternative to Azure OpenAI)**
- **sentence-transformers/all-MiniLM-L6-v2** (114.5M downloads) - Fast, efficient embeddings
  - Link: https://hf.co/sentence-transformers/all-MiniLM-L6-v2
- **sentence-transformers/all-mpnet-base-v2** (17.7M downloads) - Higher quality embeddings
  - Link: https://hf.co/sentence-transformers/all-mpnet-base-v2
- **BAAI/bge-base-en-v1.5** (5M downloads) - Strong performance on MTEB benchmark
  - Link: https://hf.co/BAAI/bge-base-en-v1.5
- **BAAI/bge-m3** (6M downloads) - Multilingual, supports hybrid search with dense/sparse vectors
  - Link: https://hf.co/BAAI/bge-m3
- **jinaai/jina-embeddings-v3** (5.3M downloads) - Multilingual, long context support
  - Link: https://hf.co/jinaai/jina-embeddings-v3

### **Cross-Encoder Reranking Models**
- **cross-encoder/ms-marco-MiniLM-L6-v2** (4.9M downloads) - Most popular reranker
  - Link: https://hf.co/cross-encoder/ms-marco-MiniLM-L6-v2
- **BAAI/bge-reranker-v2-m3** (2.6M downloads) - Multilingual reranking
  - Link: https://hf.co/BAAI/bge-reranker-v2-m3
- **jinaai/jina-reranker-v2-base-multilingual** (1.1M downloads)
  - Link: https://hf.co/jinaai/jina-reranker-v2-base-multilingual
- **Qwen/Qwen3-Reranker-0.6B** (930.5K downloads) - Efficient, modern architecture
  - Link: https://hf.co/Qwen/Qwen3-Reranker-0.6B

### **Key Research Papers**
1. **HyDE - Precise Zero-Shot Dense Retrieval** (2212.10496)
   - Original HyDE paper demonstrating hypothetical document embeddings
   - Link: https://hf.co/papers/2212.10496

2. **MultiHop-RAG Benchmark** (2401.15391)
   - Dataset and evaluation for multi-hop reasoning in RAG systems
   - Link: https://hf.co/papers/2401.15391

3. **Agentic RAG Survey** (2501.09136)
   - Comprehensive survey on autonomous agents in RAG pipelines
   - Link: https://hf.co/papers/2501.09136

4. **RAG Comprehensive Survey** (2409.14924)
   - Taxonomy of RAG approaches with integration strategies
   - Link: https://hf.co/papers/2409.14924

5. **ARAGOG: Advanced RAG Output Grading** (2404.01037)
   - Evaluation of HyDE, reranking, and MMR effectiveness
   - Link: https://hf.co/papers/2404.01037

6. **Is Semantic Chunking Worth It?** (2410.13070)
   - Critical analysis of chunking strategies and performance
   - Link: https://hf.co/papers/2410.13070

7. **Late Chunking** (2409.04701)
   - Novel contextual chunk embeddings using long-context models
   - Link: https://hf.co/papers/2409.04701

8. **Rethinking Chunk Size** (2505.21700)
   - Multi-dataset analysis of optimal chunk sizes (64-1024 tokens)
   - Link: https://hf.co/papers/2505.21700

### **Hugging Face Documentation**
- **RAG Implementation Guide**: https://huggingface.co/docs/transformers/chat_extras
- **Sentence Transformers Hub**: https://huggingface.co/docs/hub/sentence-transformers
- **RAG Model Documentation**: https://huggingface.co/docs/transformers/model_doc/rag

---

## **Demo #1: HyDE (Hypothetical Document Embeddings) - Query Enhancement**

* **Objective**: Demonstrate how generating a hypothetical answer document before retrieval can dramatically improve semantic matching between queries and documents.

* **Core Concepts Demonstrated**:
  - Query Enhancement via Hypothetical Document Embeddings (HyDE)
  - Pre-retrieval optimization using LLM-generated context
  - Answer-to-answer similarity search paradigm

* **Implementation Steps**:
  1. **Setup and Data Ingestion**:
     - Create a minimal knowledge base using 3-5 markdown files on a focused technical topic (e.g., machine learning concepts).
     - Use `SimpleDirectoryReader` from `llama_index.core` to load documents.
     - Apply `SentenceSplitter` with chunk_size=512 and chunk_overlap=50 for basic chunking.
  
  2. **Baseline RAG Pipeline**:
     - Initialize Azure OpenAI embedding model using `AzureOpenAIEmbedding` from `llama_index.embeddings.azure_openai`.
     - Configure Azure OpenAI LLM using `AzureOpenAI` from `llama_index.llms.azure_openai`.
     - Create a `VectorStoreIndex` using `SimpleVectorStore` (in-memory).
     - Build a baseline query engine: `baseline_query_engine = index.as_query_engine(llm=azure_llm)`.
     - Execute a test query and record results.
  
  3. **Implement HyDE Enhancement**:
     - Import `HyDEQueryTransform` from `llama_index.core.indices.query.query_transform`.
     - Create a HyDE transformation: `hyde_transform = HyDEQueryTransform(llm=azure_llm, include_original=False)`.
     - Wrap the baseline query engine: `hyde_query_engine = TransformQueryEngine(baseline_query_engine, query_transform=hyde_transform)`.
     - For transparency, add a step to display the generated hypothetical document.
  
  4. **Comparative Evaluation**:
     - Execute the same test query through both engines.
     - Display retrieved chunks from each approach side-by-side.
     - Compare the semantic relevance of retrieved documents.
     - Analyze final generated answers for accuracy and completeness.
  
  5. **Data Flow Visualization**:
     - User Query → LLM generates hypothetical answer → Embed hypothetical answer → Vector search → Retrieved chunks → Final generation

* **Relevant Citation(s)**:
  - HyDE technique as described in the curriculum (Section: Pre-Retrieval Optimization, reference #32: "Advanced RAG: Improving Retrieval-Augmented Generation with Hypothetical Document Embeddings (HyDE)")

* **Recommended HF Resources for Demo #1**:
  - **Paper**: "Precise Zero-Shot Dense Retrieval without Relevance Labels" (HyDE original paper)
    - Link: https://hf.co/papers/2212.10496
  - **Paper**: "ARAGOG: Advanced RAG Output Grading" - Evaluates HyDE effectiveness
    - Link: https://hf.co/papers/2404.01037
  - **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2 for fast prototyping
    - Link: https://hf.co/sentence-transformers/all-MiniLM-L6-v2
  - **Embedding Model**: BAAI/bge-base-en-v1.5 for better accuracy
    - Link: https://hf.co/BAAI/bge-base-en-v1.5

---

## **Demo #2: Multi-Query Decomposition - Complex Query Handling**

* **Objective**: Show how decomposing a complex, multi-faceted query into simpler sub-queries enables comprehensive information retrieval and synthesis.

* **Core Concepts Demonstrated**:
  - Sub-Query Decomposition for multi-hop reasoning
  - Parallel retrieval execution
  - Context aggregation from multiple retrieval passes

* **Implementation Steps**:
  1. **Data Preparation**:
     - Use a knowledge base with 4-6 markdown documents covering distinct but related topics (e.g., different machine learning algorithms).
     - Load using `SimpleDirectoryReader` and chunk with `SentenceSplitter` (chunk_size=512).
     - Create `VectorStoreIndex` with Azure OpenAI embeddings.
  
  2. **Baseline Single-Query Approach**:
     - Configure standard query engine: `baseline_engine = index.as_query_engine(similarity_top_k=3)`.
     - Test with a complex query requiring information synthesis (e.g., "Compare the strengths and weaknesses of gradient boosting and random forests for classification tasks").
     - Observe limitations in coverage and depth.
  
  3. **Implement Sub-Query Decomposition**:
     - Import `SubQuestionQueryEngine` from `llama_index.core.query_engine`.
     - Create query engine tools: `query_engine_tools = [QueryEngineTool(query_engine=index.as_query_engine(), metadata=ToolMetadata(name="ml_algorithms", description="Knowledge base about machine learning algorithms"))]`.
     - Initialize decomposition engine: `subquestion_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=query_engine_tools, llm=azure_llm, verbose=True)`.
     - Enable verbose mode to display generated sub-questions.
  
  4. **Execution and Analysis**:
     - Execute the complex query through the sub-question engine.
     - Display the automatically generated sub-questions.
     - Show retrieved chunks for each sub-question.
     - Present the final synthesized answer.
     - Compare coverage and quality against the baseline approach.
  
  5. **Data Flow Visualization**:
     - Complex Query → LLM decomposes into sub-queries → Each sub-query retrieves independently → Aggregate all contexts → LLM synthesizes comprehensive answer

* **Relevant Citation(s)**:
  - Sub-Query Decomposition technique (Section: Pre-Retrieval Optimization, reference #40: "Build Advanced Retrieval-Augmented Generation Systems")

* **Recommended HF Resources for Demo #2**:
  - **Paper**: "MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries"
    - Link: https://hf.co/papers/2401.15391
  - **Paper**: "BeamAggR: Beam Aggregation Reasoning over Multi-source Knowledge"
    - Link: https://hf.co/papers/2406.19820
  - **Paper**: "MINTQA: Multi-hop Question Answering on New and Tail Knowledge"
    - Link: https://hf.co/papers/2412.17032
  - **Embedding Model**: sentence-transformers/multi-qa-mpnet-base-dot-v1 (optimized for QA)
    - Link: https://hf.co/sentence-transformers/multi-qa-mpnet-base-dot-v1

---

## **Demo #3: Hybrid Search - Combining Semantic and Keyword Retrieval**

* **Objective**: Demonstrate how combining dense vector search with sparse keyword-based retrieval (BM25) improves retrieval precision for diverse query types.

* **Core Concepts Demonstrated**:
  - Hybrid Search (Dense + Sparse vectors)
  - BM25 keyword-based retrieval
  - Reciprocal Rank Fusion (RRF) for result merging

* **Implementation Steps**:
  1. **Data Setup**:
     - Create a dataset with 5-7 documents containing a mix of content: some with specific technical terms/acronyms (e.g., "BERT", "GPT-4", "API"), others with more general conceptual information.
     - Load and chunk documents using `SentenceSplitter`.
  
  2. **Pure Semantic Search Baseline**:
     - Build standard vector index with Azure OpenAI embeddings.
     - Create query engine: `vector_engine = index.as_query_engine(similarity_top_k=5)`.
     - Test with queries requiring exact term matches (e.g., "What is BERT?").
     - Test with conceptual queries (e.g., "How do transformer models work?").
     - Record retrieval performance for both query types.
  
  3. **Implement BM25 Retriever**:
     - Import `BM25Retriever` from `llama_index.retrievers.bm25`.
     - Initialize: `bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)` where `nodes` are the parsed document chunks.
     - Test the same queries with BM25 alone to show its strengths on exact matches and weaknesses on semantic queries.
  
  4. **Create Hybrid Retriever**:
     - Import `QueryFusionRetriever` from `llama_index.core.retrievers`.
     - Combine both retrievers: `hybrid_retriever = QueryFusionRetriever([vector_retriever, bm25_retriever], similarity_top_k=5, num_queries=1, mode="reciprocal_rerank", use_async=False)`.
     - Build query engine from hybrid retriever: `hybrid_engine = RetrieverQueryEngine(retriever=hybrid_retriever, llm=azure_llm)`.
  
  5. **Comparative Evaluation**:
     - Run the same diverse test queries through all three approaches.
     - Create a comparison table showing retrieved chunks and relevance scores.
     - Demonstrate how hybrid search excels on both exact-match and semantic queries.
     - Explain the Reciprocal Rank Fusion mechanism used for merging results.
  
  6. **Data Flow Visualization**:
     - Query → Parallel execution: (Vector Search + BM25 Search) → Two ranked lists → RRF merges ranks → Top-K unified results → LLM generation

* **Relevant Citation(s)**:
  - Hybrid Search technique (Section: Advanced Retrieval Strategies, reference #9: "What is Retrieval-Augmented Generation (RAG)? | Google Cloud", reference #26: "Advanced RAG Implementation using Hybrid Search and Reranking")
  - Reciprocal Rank Fusion (reference #43: "Using Hybrid Search to Deliver Fast and Contextually Relevant Results")

* **Recommended HF Resources for Demo #3**:
  - **Model**: BAAI/bge-m3 - Native support for dense, sparse, and hybrid search
    - Link: https://hf.co/BAAI/bge-m3
    - Paper: https://hf.co/papers/2402.03216
  - **Paper**: "Vietnamese Legal Information Retrieval" - RRF implementation example
    - Link: https://hf.co/papers/2409.13699
  - **Embedding Model**: sentence-transformers/all-mpnet-base-v2 for dense vectors
    - Link: https://hf.co/sentence-transformers/all-mpnet-base-v2

---

## **Demo #4: Hierarchical Retrieval - Parent Document Retriever**

* **Objective**: Solve the chunking trade-off by retrieving with small, precise child chunks while generating with larger, context-rich parent chunks.

* **Core Concepts Demonstrated**:
  - Hierarchical chunking strategy
  - Parent-Child relationship in document storage
  - Precision in retrieval, richness in generation

* **Implementation Steps**:
  1. **Data Preparation with Hierarchy**:
     - Use 3-4 long-form documents (e.g., technical whitepapers or documentation pages).
     - Load documents with `SimpleDirectoryReader`.
  
  2. **Baseline with Single Chunk Size**:
     - Create index with medium-sized chunks (chunk_size=512).
     - Test query engine and observe the trade-off: retrieval may be imprecise, but context is reasonable.
     - Try with small chunks (chunk_size=128) - precise retrieval but insufficient context.
  
  3. **Implement Parent-Child Chunking**:
     - Define parent splitter: `parent_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)`.
     - Define child splitter: `child_splitter = SentenceSplitter(chunk_size=256, chunk_overlap=25)`.
     - Import `SimpleDocumentStore` for parent storage.
     - Parse documents into parent nodes: `parent_nodes = parent_splitter.get_nodes_from_documents(documents)`.
     - For each parent, extract child nodes and link them.
  
  4. **Build Parent Document Retriever**:
     - Index only child nodes in the vector store: `child_index = VectorStoreIndex(child_nodes, embed_model=azure_embed)`.
     - Create custom retriever that retrieves children but returns parents:
       ```python
       from llama_index.core.retrievers import BaseRetriever
       # Implement custom retriever that:
       # 1. Retrieves top-k child nodes
       # 2. Maps each child to its parent_id
       # 3. Fetches parent nodes from document store
       # 4. Returns parent nodes to query engine
       ```
     - Build query engine: `parent_retriever_engine = RetrieverQueryEngine(retriever=parent_retriever, llm=azure_llm)`.
  
  5. **Evaluation and Comparison**:
     - Execute test queries requiring both precision and context.
     - Compare retrieved content across all three approaches (large chunks, small chunks, parent-child).
     - Show how parent-child retrieval finds the right information (via child) and provides sufficient context (via parent).
     - Demonstrate improved answer quality and reduced "keyhole" problem.
  
  6. **Data Flow Visualization**:
     - Query → Embed → Search child embeddings (precise) → Identify top-K children → Lookup parent IDs → Fetch parent chunks (rich context) → LLM generation

* **Relevant Citation(s)**:
  - Parent Document Retriever pattern (Section: Advanced Retrieval Strategies, reference #25: "How to use the Parent Document Retriever | LangChain", reference #46: "Parent-Child Chunking in LangChain for Advanced RAG")

* **Recommended HF Resources for Demo #4**:
  - **Paper**: "Rethinking Chunk Size For Long-Document Retrieval: Multi-Dataset Analysis"
    - Optimal chunk sizes: 64-128 tokens (small), 512-1024 tokens (large)
    - Link: https://hf.co/papers/2505.21700
  - **Paper**: "Late Chunking: Contextual Chunk Embeddings Using Long-Context Models"
    - Alternative approach for hierarchical context
    - Link: https://hf.co/papers/2409.04701
  - **Paper**: "Is Semantic Chunking Worth the Computational Cost?"
    - Analysis of chunking effectiveness
    - Link: https://hf.co/papers/2410.13070
  - **Embedding Model**: Alibaba-NLP/gte-large-en-v1.5 for extended context (8192 tokens)
    - Link: https://hf.co/Alibaba-NLP/gte-large-en-v1.5

---

## **Demo #5: Re-Ranking with Cross-Encoders - Post-Retrieval Refinement**

* **Objective**: Demonstrate how a two-stage retrieval process (fast bi-encoder + accurate cross-encoder) improves precision by re-ordering initial results.

* **Core Concepts Demonstrated**:
  - Two-stage retrieval architecture
  - Bi-encoder vs. Cross-encoder comparison
  - Re-ranking for precision optimization

* **Implementation Steps**:
  1. **Data and Baseline Setup**:
     - Use a moderately sized knowledge base (8-10 documents) with some topically similar but semantically distinct content.
     - Create vector index with Azure OpenAI embeddings.
     - Configure baseline retriever with high top-k: `baseline_retriever = index.as_retriever(similarity_top_k=20)`.
     - Build query engine and test with queries where initial retrieval includes noise.
  
  2. **Analyze Baseline Retrieval Quality**:
     - Display the top-20 retrieved chunks with their similarity scores.
     - Manually annotate or use LLM-as-judge to assess true relevance.
     - Show that relevant documents may not be in the top positions.
  
  3. **Implement Re-Ranking**:
     - For llama-index compatibility, explore available re-ranker options:
       - Option A: Use `CohereRerank` from `llama_index.postprocessor.cohere_rerank` if Cohere API is available.
       - Option B: Integrate an open-source cross-encoder model (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) via custom postprocessor.
     - Configure re-ranker: `reranker = CohereRerank(api_key=cohere_api_key, top_n=5)` or custom implementation.
     - Create query engine with re-ranking: `rerank_engine = index.as_query_engine(similarity_top_k=20, node_postprocessors=[reranker])`.
  
  4. **Execution and Comparison**:
     - Run the same query through both engines.
     - Display the initial top-20 results (bi-encoder scores).
     - Display the re-ranked top-5 results (cross-encoder scores).
     - Create visualization showing rank changes.
     - Compare final generated answers for quality and relevance.
  
  5. **Explain the Architecture**:
     - Illustrate bi-encoder: Query and documents encoded separately → Fast cosine similarity.
     - Illustrate cross-encoder: Query + Document concatenated → Deep attention → Precise relevance score (slower but more accurate).
     - Discuss the trade-off: Use bi-encoder for fast recall, cross-encoder for precise precision.
  
  6. **Data Flow Visualization**:
     - Query → Bi-encoder retrieval (top-20, fast) → Cross-encoder re-ranking (top-5, accurate) → LLM generation with highest-quality context

* **Relevant Citation(s)**:
  - Re-ranking with Cross-Encoders (Section: Post-Retrieval Enhancement, reference #29: "Rerankers and Two-Stage Retrieval - Pinecone", reference #45: "Advanced RAG Optimization: Prioritize Knowledge with Reranking")

* **Recommended HF Resources for Demo #5**:
  - **Reranker Model**: cross-encoder/ms-marco-MiniLM-L6-v2 (4.9M downloads, most popular)
    - Link: https://hf.co/cross-encoder/ms-marco-MiniLM-L6-v2
  - **Reranker Model**: BAAI/bge-reranker-v2-m3 (multilingual, 2.6M downloads)
    - Link: https://hf.co/BAAI/bge-reranker-v2-m3
  - **Reranker Model**: Qwen/Qwen3-Reranker-0.6B (efficient, modern architecture)
    - Link: https://hf.co/Qwen/Qwen3-Reranker-0.6B
  - **Paper**: "Comparative Analysis of Lion and AdamW Optimizers for Cross-Encoder Reranking"
    - Link: https://hf.co/papers/2506.18297
  - **Paper**: "Incorporating Relevance Feedback for Information-Seeking Retrieval"
    - Link: https://hf.co/papers/2210.10695

---

## **Demo #6: Context Compression - Strategic Ordering and Pruning**

* **Objective**: Show how intelligent context management—both reordering to address "lost in the middle" and compression via extractive pruning—optimizes LLM generation.

* **Core Concepts Demonstrated**:
  - "Lost in the Middle" problem
  - Strategic context reordering
  - Extractive context compression (sentence-level pruning)

* **Implementation Steps**:
  1. **Setup with Long Context**:
     - Create a scenario requiring retrieval of many chunks (10-15) to answer a query.
     - Use documents with a mix of relevant and semi-relevant information.
     - Build vector index and retriever with high top-k: `retriever = index.as_retriever(similarity_top_k=15)`.
  
  2. **Demonstrate "Lost in the Middle" Problem**:
     - Create a query engine with standard context ordering: `baseline_engine = RetrieverQueryEngine(retriever=retriever, llm=azure_llm)`.
     - Execute a test query and retrieve 15 chunks.
     - Manually place a critical piece of information in the middle position (position 7-8).
     - Show that the generated answer may miss or underweight this information.
     - Explain the research finding: LLMs pay more attention to beginning and end of context.
  
  3. **Implement Strategic Reordering**:
     - Create a custom node postprocessor that reorders chunks:
       ```python
       from llama_index.core.postprocessor import BaseNodePostprocessor
       class StrategicReorderProcessor(BaseNodePostprocessor):
           # Reorder nodes to place:
           # - Most relevant (rank 1) at the beginning
           # - Second most relevant (rank 2) at the end
           # - Remaining nodes (rank 3-N) in the middle
       ```
     - Apply to query engine: `reordered_engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=[reorder_processor], llm=azure_llm)`.
     - Re-run query and show improved answer quality.
  
  4. **Implement Extractive Compression**:
     - Use `LongContextReorder` from `llama_index.postprocessor.long_context_reorder` for reordering.
     - For compression, implement sentence-level pruning:
       - Import sentence tokenizer: `from llama_index.core.node_parser import SentenceSplitter`.
       - Create postprocessor that:
         1. Splits each retrieved chunk into sentences.
         2. Embeds each sentence.
         3. Scores each sentence's similarity to the query.
         4. Retains only top-scoring sentences (e.g., top 50% or similarity > threshold).
         5. Reconstructs compressed chunks.
     - Alternatively, use `llama_index.core.postprocessor.MetadataReplacementPostProcessor` combined with sentence-level filtering.
  
  5. **Comparative Analysis**:
     - Compare three approaches:
       1. Standard ordering with full context (baseline).
       2. Strategic reordering with full context.
       3. Strategic reordering with compressed context.
     - Measure and display:
       - Total token count sent to LLM.
       - Answer quality/relevance.
       - Latency differences.
     - Show the optimal balance: Reordering + Compression = Best quality with efficiency.
  
  6. **Data Flow Visualization**:
     - Query → Retrieve top-15 → Score each sentence → Prune low-relevance sentences → Reorder (best first, second-best last) → Compressed, strategically-ordered context → LLM generation

* **Relevant Citation(s)**:
  - Lost in the Middle problem (Section: Post-Retrieval Enhancement, reference #22: "RAG vs. Long-context LLMs - SuperAnnotate")
  - Strategic Reordering (reference #59: "Advanced RAG Series: Retrieval - Latest and Greatest")
  - Extractive Compression (Section: Post-Retrieval Enhancement, reference #16: "8 Retrieval Augmented Generation (RAG) Architectures", reference #60: "Mastering Advanced RAG Techniques")

* **Recommended HF Resources for Demo #6**:
  - **Paper**: "ChunkRAG: Novel LLM-Chunk Filtering Method for RAG Systems"
    - Link: https://hf.co/papers/2410.19572
  - **Paper**: "LaRA: Benchmarking RAG and Long-Context LLMs"
    - Evaluates context length vs RAG effectiveness
    - Link: https://hf.co/papers/2502.09977
  - **Paper**: "Quantifying reliance on external information over parametric knowledge during RAG"
    - Link: https://hf.co/papers/2410.00857

---

## **Demo #7: Corrective RAG (CRAG) - Self-Correcting Retrieval**

* **Objective**: Implement a self-reflective system that evaluates retrieval quality and triggers corrective actions (web search) when internal knowledge is insufficient.

* **Core Concepts Demonstrated**:
  - Self-correction and retrieval evaluation
  - Dynamic routing based on confidence scores
  - Fallback to external knowledge sources
  - Knowledge refinement (document grading)

* **Implementation Steps**:
  1. **Setup Limited Internal Knowledge Base**:
     - Create a small, focused knowledge base (4-5 documents on a specific topic, e.g., historical facts about a particular technology).
     - Build vector index with Azure OpenAI embeddings.
     - Deliberately test with queries both within and outside the knowledge base scope.
  
  2. **Implement Retrieval Evaluator**:
     - Create a lightweight evaluator using Azure OpenAI LLM:
       ```python
       def evaluate_retrieval_relevance(query: str, retrieved_chunks: List[str], llm) -> float:
           # Prompt LLM to score relevance on 0-1 scale
           # Prompt: "Given query: {query}, rate the relevance of these documents: {chunks}. Provide a confidence score 0-1."
           # Parse LLM response to extract confidence score
           return confidence_score
       ```
     - Define thresholds: High (>0.7), Ambiguous (0.4-0.7), Low (<0.4).
  
  3. **Implement Web Search Fallback**:
     - For demonstration, use a simple web search API or mock results.
     - If using Azure, integrate with Bing Search API or use a mock function that simulates external retrieval.
     - Create function: `def web_search_fallback(query: str) -> List[str]`.
  
  4. **Build CRAG Query Engine**:
     - Create custom query engine that:
       1. Performs initial retrieval from internal knowledge base.
       2. Calls evaluator to score relevance.
       3. Routes based on score:
          - High: Use internal documents directly.
          - Low: Discard internal results, use web search results.
          - Ambiguous: Merge internal and web search results.
       4. (Optional) Implement knowledge strip grading: break documents into sentences, score each, filter.
       5. Pass refined context to LLM for generation.
     - Implement as custom query engine class or chain of components.
  
  5. **Evaluation Scenarios**:
     - Test with three query types:
       1. **In-domain query**: Knowledge base contains answer → High score → Use internal docs.
       2. **Out-of-domain query**: Knowledge base lacks information → Low score → Trigger web search.
       3. **Ambiguous query**: Partial information in knowledge base → Ambiguous score → Merge sources.
     - Display the decision-making process at each step.
     - Show final answers and their sources.
  
  6. **Data Flow Visualization**:
     - Query → Retrieve from internal KB → Evaluator scores relevance → Decision:
       - IF High: Internal docs → LLM
       - IF Low: Web search → LLM
       - IF Ambiguous: Internal + Web → Merge → LLM
     - (Optional) Pre-LLM: Knowledge strip filtering

* **Relevant Citation(s)**:
  - Corrective RAG (CRAG) (Section: The Frontier of RAG, reference #16: "8 Retrieval Augmented Generation (RAG) Architectures", reference #67: "Corrective Retrieval Augmented Generation - arXiv")

* **Recommended HF Resources for Demo #7**:
  - **Paper**: "CDF-RAG: Causal Dynamic Feedback for Adaptive RAG"
    - Dynamic feedback and validation mechanisms
    - Link: https://hf.co/papers/2504.12560
  - **Paper**: "MBA-RAG: A Bandit Approach for Adaptive RAG"
    - Reinforcement learning for adaptive retrieval
    - Link: https://hf.co/papers/2412.01572
  - **Paper**: "Finetune-RAG: Fine-Tuning to Resist Hallucination in RAG"
    - Handling imperfect retrieval
    - Link: https://hf.co/papers/2505.10792

---

## **Demo #8: Agentic RAG - Autonomous Query Planning and Tool Selection**

* **Objective**: Demonstrate an autonomous agent that dynamically plans retrieval strategies and selects appropriate tools to answer complex queries.

* **Core Concepts Demonstrated**:
  - Agentic workflow (Thought → Action → Observation loop)
  - Dynamic tool selection
  - Multi-step reasoning and iterative retrieval
  - Query decomposition and planning by the agent

* **Implementation Steps**:
  1. **Setup Multiple Knowledge Sources**:
     - Create two distinct knowledge bases:
       1. Technical documentation (5-6 docs on ML concepts).
       2. Financial data or product information (5-6 docs on a different domain).
     - Build separate vector indices for each.
  
  2. **Define Query Engine Tools**:
     - Create tool wrappers for each knowledge base:
       ```python
       from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
       
       ml_tool = QueryEngineTool(
           query_engine=ml_index.as_query_engine(),
           metadata=ToolMetadata(
               name="ml_knowledge",
               description="Expert knowledge about machine learning algorithms, concepts, and techniques."
           )
       )
       
       finance_tool = QueryEngineTool(
           query_engine=finance_index.as_query_engine(),
           metadata=ToolMetadata(
               name="finance_knowledge",
               description="Information about financial products, market analysis, and investment strategies."
           )
       )
       ```
     
     - **Add Internet Search Tool (DuckDuckGo)**:
       ```python
       from duckduckgo_search import DDGS
       
       def internet_search(query: str) -> str:
           """Search the internet for current information using DuckDuckGo."""
           try:
               ddgs = DDGS()
               results = ddgs.text(query, max_results=5)
               formatted_results = "\n\n".join([
                   f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}"
                   for r in results
               ])
               return formatted_results
           except Exception as e:
               return f"Search failed: {str(e)}"
       
       internet_tool = FunctionTool.from_defaults(
           fn=internet_search,
           name="internet_search",
           description="Search the internet for current information, news, and real-time data using DuckDuckGo."
       )
       ```
     
     - **Add arXiv Search Tool**:
       ```python
       import arxiv
       
       def arxiv_search(query: str, max_results: int = 5) -> str:
           """Search arXiv for academic papers related to the query."""
           try:
               search = arxiv.Search(
                   query=query,
                   max_results=max_results,
                   sort_by=arxiv.SortCriterion.Relevance
               )
               results = []
               for paper in search.results():
                   results.append(
                       f"Title: {paper.title}\n"
                       f"Authors: {', '.join([a.name for a in paper.authors])}\n"
                       f"Published: {paper.published.strftime('%Y-%m-%d')}\n"
                       f"arXiv ID: {paper.entry_id.split('/')[-1]}\n"
                       f"Summary: {paper.summary[:300]}...\n"
                   )
               return "\n---\n".join(results) if results else "No papers found."
           except Exception as e:
               return f"arXiv search failed: {str(e)}"
       
       arxiv_search_tool = FunctionTool.from_defaults(
           fn=arxiv_search,
           name="arxiv_search",
           description="Search arXiv for academic papers and research publications. Returns paper titles, authors, IDs, and summaries."
       )
       ```
     
     - **Add arXiv Fetch Tool**:
       ```python
       def arxiv_fetch(arxiv_id: str) -> str:
           """Fetch full details of a specific arXiv paper by its ID."""
           try:
               search = arxiv.Search(id_list=[arxiv_id])
               paper = next(search.results())
               return (
                   f"Title: {paper.title}\n"
                   f"Authors: {', '.join([a.name for a in paper.authors])}\n"
                   f"Published: {paper.published.strftime('%Y-%m-%d')}\n"
                   f"arXiv ID: {paper.entry_id.split('/')[-1]}\n"
                   f"PDF URL: {paper.pdf_url}\n"
                   f"Categories: {', '.join(paper.categories)}\n\n"
                   f"Abstract:\n{paper.summary}\n"
               )
           except Exception as e:
               return f"Failed to fetch paper: {str(e)}"
       
       arxiv_fetch_tool = FunctionTool.from_defaults(
           fn=arxiv_fetch,
           name="arxiv_fetch",
           description="Fetch full details and abstract of a specific arXiv paper using its arXiv ID (e.g., '2301.12345')."
       )
       ```
  
  3. **Create ReAct Agent**:
     - Use llama-index's agent framework:
       ```python
       from llama_index.core.agent import ReActAgent
       
       agent = ReActAgent.from_tools(
           tools=[
               ml_tool, 
               finance_tool, 
               internet_tool, 
               arxiv_search_tool, 
               arxiv_fetch_tool
           ],
           llm=azure_llm,
           verbose=True,
           max_iterations=10  # Increased for multi-step research tasks
       )
       ```
     - Enable verbose mode to display the agent's reasoning process.
     - Note: Install required packages: `uv pip install duckduckgo-search arxiv`
  
  4. **Test with Progressive Query Complexity**:
     - **Simple single-domain query**: "Explain gradient boosting."
       - Agent should identify the correct tool (ml_knowledge) and retrieve.
     
     - **Cross-domain query**: "How can machine learning be applied to stock market prediction?"
       - Agent should use both ml_knowledge and finance_knowledge.
     
     - **Current information query**: "What are the latest developments in large language models in 2025?"
       - Agent should use internet_search for current information not in the knowledge base.
     
     - **Academic research query**: "Find recent papers on retrieval-augmented generation and summarize their key findings."
       - Agent should:
         1. Use arxiv_search to find relevant papers.
         2. Use arxiv_fetch to get detailed abstracts.
         3. Synthesize findings.
     
     - **Multi-hop query with external tools**: "Compare the risk-adjusted returns of portfolio strategies using reinforcement learning vs. traditional diversification, considering the latest research on deep Q-learning."
       - Agent should:
         1. Query ml_knowledge for deep Q-learning basics.
         2. Use arxiv_search for recent research on deep Q-learning in finance.
         3. Use arxiv_fetch to get specific paper details.
         4. Query finance_knowledge for portfolio strategies.
         5. Optionally use internet_search for current market trends.
         6. Synthesize all information.
     
     - **Complex research query**: "What are the latest breakthroughs in transformer architectures according to recent arXiv papers, and how do they relate to the concepts in our ML knowledge base?"
       - Agent should:
         1. Use arxiv_search to find recent transformer papers.
         2. Use arxiv_fetch for detailed abstracts of top papers.
         3. Query ml_knowledge for foundational transformer concepts.
         4. Compare and synthesize.
  
  5. **Visualize Agent Reasoning**:
     - Display the agent's internal loop for each query:
       - **Thought**: "I need information about X. Tool Y seems most appropriate."
       - **Action**: "Query tool Y with refined sub-query Z."
       - **Observation**: "Retrieved information: [summary]."
       - **Thought**: "This partially answers the question. I need additional information about A."
       - (Loop continues until satisfied or max iterations reached)
     - Show final synthesized answer.
  
  6. **Compare with Static RAG**:
     - Attempt the same complex queries with a single, static query engine.
     - Show how the static approach fails or provides incomplete answers.
     - Highlight the agent's adaptability and intelligent tool orchestration.
  
  7. **Data Flow Visualization**:
     - Complex Query → Agent analyzes → Plans sub-tasks → For each sub-task:
       - Select appropriate tool(s):
         * ml_knowledge / finance_knowledge (internal knowledge bases)
         * internet_search (current information)
         * arxiv_search (find relevant papers)
         * arxiv_fetch (get specific paper details)
       - Execute tool
       - Observe results
     - Synthesize all observations → Final comprehensive answer
     
     - Example flow for research query:
       ```
       Query: "Latest RAG techniques with academic papers"
       ├─ Thought: Need recent papers
       ├─ Action: arxiv_search("retrieval augmented generation 2024")
       ├─ Observation: [5 papers found with IDs]
       ├─ Thought: Need details on most relevant paper
       ├─ Action: arxiv_fetch("2401.xxxxx")
       ├─ Observation: [Full abstract retrieved]
       ├─ Thought: Need foundational context
       ├─ Action: ml_knowledge("RAG basics")
       ├─ Observation: [Internal docs retrieved]
       └─ Final Answer: [Synthesized response]
       ```

* **Relevant Citation(s)**:
  - Agentic RAG (Section: The Frontier of RAG, reference #34: "What is Agentic RAG | Weaviate", reference #62: "What is Agentic RAG? | IBM", reference #64: "Agentic RAG: How It Works, Use Cases, Comparison With RAG")

* **Recommended HF Resources for Demo #8**:
  - **Paper**: "Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG"
    - Comprehensive survey on agentic design patterns
    - Link: https://hf.co/papers/2501.09136
  - **Paper**: "GFM-RAG: Graph Foundation Model for RAG"
    - Graph-based knowledge integration for complex reasoning
    - Link: https://hf.co/papers/2502.01113
  - **Paper**: "BeamAggR: Beam Aggregation Reasoning over Multi-source Knowledge"
    - Multi-source knowledge integration techniques
    - Link: https://hf.co/papers/2406.19820
  - **Documentation**: Transformers RAG model documentation
    - Link: https://huggingface.co/docs/transformers/model_doc/rag

---

## **Demo #9: Fine-Tuning the Embedding Model for Domain-Specific Retrieval**

* **Objective**: Show how fine-tuning the embedding model on domain-specific query-passage pairs significantly improves retrieval accuracy.

* **Core Concepts Demonstrated**:
  - Embedding model fine-tuning
  - Contrastive learning with query-positive-negative triplets
  - Domain adaptation for specialized terminology

* **Implementation Steps**:
  1. **Prepare Domain-Specific Dataset**:
     - Create or use a small dataset of 50-100 triplets: (query, positive_passage, negative_passage).
     - Example domain: Medical or Legal terminology where generic embeddings struggle.
     - Format: Each triplet represents a query, a relevant chunk (positive), and an irrelevant but plausible chunk (hard negative).
     - Save as CSV or JSON.
  
  2. **Baseline with Generic Embeddings**:
     - Build RAG system using standard Azure OpenAI embeddings.
     - Test on domain-specific queries.
     - Measure retrieval accuracy (e.g., percentage of queries where the correct document is in top-3).
     - Record baseline metrics.
  
  3. **Fine-Tune Embedding Model**:
     - Note: Fine-tuning Azure OpenAI embeddings directly may not be available. Alternative approaches:
       - Option A: Use an open-source embedding model (e.g., `sentence-transformers/all-MiniLM-L6-v2`) and fine-tune locally.
       - Option B: Use Azure AI Foundry to fine-tune a supported embedding model if available.
       - Option C: Demonstrate the concept with mock fine-tuning (explain the process theoretically and show expected improvements).
     - If using sentence-transformers:
       ```python
       from sentence_transformers import SentenceTransformer, InputExample, losses
       from torch.utils.data import DataLoader
       
       model = SentenceTransformer('all-MiniLM-L6-v2')
       train_examples = [InputExample(texts=[query, pos_passage, neg_passage]) for query, pos_passage, neg_passage in dataset]
       train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
       train_loss = losses.TripletLoss(model)
       model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3)
       model.save('finetuned_embeddings')
       ```
  
  4. **Rebuild RAG with Fine-Tuned Embeddings**:
     - Load fine-tuned model.
     - Re-embed the document chunks using the fine-tuned model.
     - Create new vector index.
     - Build query engine with fine-tuned embeddings.
  
  5. **Comparative Evaluation**:
     - Run the same domain-specific test queries through both systems.
     - Measure:
       - Retrieval accuracy (top-K hit rate).
       - Mean Reciprocal Rank (MRR).
       - Final answer quality.
     - Create visualization showing improvement.
     - Highlight queries where fine-tuning made the critical difference.
  
  6. **Discussion**:
     - Explain the contrastive loss objective: pulling positive pairs together, pushing negatives apart in embedding space.
     - Discuss data requirements and the cost-benefit trade-off.
     - Emphasize that fine-tuning embeddings is often more practical than fine-tuning large generator LLMs.

* **Relevant Citation(s)**:
  - Fine-Tuning the Retriever (Section: Fine-Tuning Components for Domain-Specific Excellence, reference #69: "Multi-task retriever fine-tuning for domain-specific and efficient RAG", reference #71: "ALoFTRAG: Automatic Local Fine Tuning for Retrieval Augmented Generation")

* **Recommended HF Resources for Demo #9**:
  - **Base Models for Fine-tuning**:
    - sentence-transformers/all-MiniLM-L6-v2 (lightweight, fast training)
      - Link: https://hf.co/sentence-transformers/all-MiniLM-L6-v2
    - BAAI/bge-base-en-v1.5 (better performance, MTEB optimized)
      - Link: https://hf.co/BAAI/bge-base-en-v1.5
  - **Documentation**: Sentence Transformers Training Guide
    - Link: https://huggingface.co/docs/hub/sentence-transformers
  - **Example Fine-tuned Models**:
    - datasocietyco/bge-base-en-v1.5-course-recommender-v5 (example domain adaptation)
      - Link: https://hf.co/datasocietyco/bge-base-en-v1.5-course-recommender-v5

---

## **Demo #10: RAG Evaluation - Systematic Metrics and Frameworks**

* **Objective**: Implement a comprehensive evaluation framework to systematically measure and improve RAG system performance using quantitative metrics.

* **Core Concepts Demonstrated**:
  - RAG-specific evaluation metrics (Context Precision, Context Recall, Faithfulness, Answer Relevance)
  - LLM-as-judge evaluation pattern
  - Using RAGAS or similar framework
  - Iterative improvement workflow

* **Implementation Steps**:
  1. **Setup RAG System and Test Set**:
     - Use an existing RAG system from previous demos.
     - Create or use a test dataset with:
       - 10-15 questions
       - Ground truth answers
       - Relevant source documents (reference standard)
     - Format: `{question, ground_truth_answer, ground_truth_contexts: [doc_ids]}`
  
  2. **Install and Configure RAGAS**:
     - Install: `uv pip install ragas`.
     - Import required metrics:
       ```python
       from ragas import evaluate
       from ragas.metrics import (
           context_precision,
           context_recall,
           faithfulness,
           answer_relevance
       )
       ```
     - Configure with Azure OpenAI LLM for LLM-as-judge evaluation.
  
  3. **Run RAG System and Collect Data**:
     - For each test question:
       - Execute query through RAG system.
       - Collect: query, retrieved contexts, generated answer.
     - Store in DataFrame with columns: `question`, `contexts`, `answer`, `ground_truth`.
  
  4. **Calculate Metrics**:
     - Run RAGAS evaluation:
       ```python
       result = evaluate(
           dataset,
           metrics=[
               context_precision,
               context_recall,
               faithfulness,
               answer_relevance
           ],
           llm=azure_llm,
           embeddings=azure_embed
       )
       ```
     - Display metric scores for each question and aggregate statistics.
  
  5. **Analyze and Interpret Results**:
     - **Low Context Precision**: Too much noise in retrieval → Consider re-ranking or better chunking.
     - **Low Context Recall**: Missing relevant documents → Consider query expansion, hybrid search, or larger top-K.
     - **Low Faithfulness**: LLM hallucinating despite good context → Consider fine-tuning generator or better prompting.
     - **Low Answer Relevance**: Answers off-topic → Review prompt engineering or query understanding.
     - Create visualization: Bar charts or radar plots showing metric scores.
  
  6. **Iterative Improvement Workflow**:
     - Based on bottleneck identified, apply one advanced technique (e.g., add re-ranker if precision is low).
     - Re-run evaluation.
     - Compare before/after metrics.
     - Demonstrate the quantitative impact of the improvement.
     - Document findings.
  
  7. **Alternative: Manual Evaluation**:
     - If RAGAS integration is complex, implement simplified metrics manually:
       - **Precision**: % of retrieved chunks that are relevant (manual annotation or LLM-as-judge).
       - **Recall**: % of ground-truth documents retrieved.
       - **Faithfulness**: Use LLM prompt: "Does this answer contradict or add facts not in the context?"
  
  8. **Key Takeaways**:
     - Evaluation must be systematic and metric-driven.
     - Different metrics diagnose different failure modes.
     - RAG optimization is an iterative, data-driven process.

* **Relevant Citation(s)**:
  - RAG Evaluation Metrics (Section: Practical Implementation, reference #84: "RAG Evaluation Metrics Guide", reference #86: "RAG Evaluation Metrics: Assessing Answer Relevancy, Faithfulness")
  - RAGAS Framework (reference #89: "Best 9 RAG Evaluation Tools of 2025")

---

## **Implementation Notes and Best Practices**

### **Environment Setup**

All demos assume the following standardized environment:

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies
uv pip install llama-index llama-index-llms-azure-openai llama-index-embeddings-azure-openai
uv pip install python-dotenv pandas numpy matplotlib

# Additional dependencies per demo:
# Demo 3: uv pip install llama-index-retrievers-bm25 rank-bm25
# Demo 5: uv pip install llama-index-postprocessor-cohere-rerank (if using Cohere)
# Demo 9: uv pip install sentence-transformers torch
# Demo 10: uv pip install ragas datasets
```

### **Azure Configuration**

All demos require Azure OpenAI credentials configured via environment variables:

```python
# .env file
AZURE_OPENAI_API_KEY=<your_key>
AZURE_OPENAI_ENDPOINT=<your_endpoint>
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=<your_gpt4_deployment>
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=<your_embedding_deployment>
```

### **Data Philosophy**

- **Minimalism**: Each demo uses the smallest dataset that effectively illustrates the concept (3-10 documents).
- **Synthetic when appropriate**: Create simple markdown files rather than complex data loading.
- **Focus on technique**: Avoid time spent on data cleaning; pre-process data in setup cells.

### **Code Quality Standards**

- **Clear variable naming**: `hyde_query_engine`, `baseline_retriever`, `reranked_results`.
- **Modular functions**: Each major step (load, chunk, embed, retrieve, generate) is a separate function.
- **Verbose output**: Enable verbose modes and add print statements to show internal processes.
- **Inline documentation**: Each cell has a markdown header explaining its purpose.

### **Progressive Complexity**

Demos are sequenced to build knowledge:
1. **Demos 1-2**: Pre-retrieval enhancements (query optimization)
2. **Demos 3-4**: Retrieval enhancements (search algorithms and chunking)
3. **Demos 5-6**: Post-retrieval enhancements (re-ranking and compression)
4. **Demo 7**: Self-correction (combining multiple concepts)
5. **Demo 8**: Agentic paradigm (highest complexity)
6. **Demos 9-10**: Optimization and evaluation (production readiness)

### **Evaluation Integration**

Each demo (except Demo 10) should include a simple qualitative comparison:
- Side-by-side output display (baseline vs. enhanced)
- Clear explanation of why the advanced technique performs better
- Optional: Simple quantitative metric (e.g., character count of relevant info retrieved)

---

## **Summary**

This development plan provides a complete roadmap for building 10 progressive, hands-on demonstrations of Advanced RAG techniques. Each demo is:
- **Concept-focused**: Isolates 1-2 specific advanced techniques
- **Reproducible**: Uses standard tools and in-memory components
- **Educational**: Includes comparative analysis and clear explanations
- **Practical**: Demonstrates real-world improvements over naive approaches

The progression from query enhancement through autonomous agents provides workshop attendees with a comprehensive understanding of the modular RAG paradigm and the specific techniques available at each stage of the pipeline.

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

