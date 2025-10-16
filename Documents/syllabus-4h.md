Of course. I've updated the syllabus based on your feedback, shortening the GraphRAG section to introduce another advanced RAG architecture and extending the time for the Agentic RAG module. I've also included demo sessions to walk you through some of advanced architectures implementations.

Here is the revised half-day workshop curriculum.

### **Mastering RAG: A Half-Day Intensive Workshop**

**Workshop Duration:** 4 Hours

**Target Audience:** AI developers, data scientists, and software engineers aiming to build, optimize, and deploy robust RAG systems, from foundational to advanced levels.

**Prerequisites:** A foundational understanding of Python and Large Language Models (LLMs) is recommended.

-----

| Time (Approx.) | Module | Topics Covered |
| :--- | :--- | :--- |
| **45 mins** | **Module 1: RAG Fundamentals & The Core Pipeline (Abridged)** | • **What is RAG & Why it Matters:** A quick overview of how RAG solves core LLM limitations like hallucinations and knowledge cutoffs. • **Core Pipeline Overview:** A brief walkthrough of the Ingestion, Retrieval, and Generation stages. • **Focus Area - Chunking Strategies:** A look at key methods for preparing data, including Fixed-Size, Recursive, and Semantic chunking. • **Focus Area - Retrieval Methods:** A comparison of Sparse (BM25), Dense (Vector), and the powerful Hybrid Search approach. |
| **45 mins** | **Module 2: Optimizing the Pipeline & Handling Multimodality** | • **Post-Retrieval Optimization:** Briefly covering the importance of re-ranking to improve precision. • **Context Compression:** A short explanation of its purpose (cost, latency, "lost in the middle") and methods. • **Special Topic - Incorporating Images & Graphs:**   - **Multimodal RAG:** Handling images by generating text summaries or using multimodal embeddings (e.g., CLIP).   - **Introduction to Knowledge Graphs:** A primer on using structured graph data to enhance retrieval, leading into the next module. |
| **15 mins** | *Coffee Break* | |
| **45 mins** | **Module 3: Advanced Architectures I - GraphRAG & Self-Correcting RAG** | • **GraphRAG Overview:** Using Knowledge Graphs to capture explicit relationships, enabling multi-hop reasoning and enhanced explainability. • **New RAG Type - Self-Reflective & Corrective RAG:** Introducing architectures like Self-RAG and CRAG that add a layer of self-evaluation. These systems grade retrieved documents for relevance, critique their own answers, and can trigger corrective actions like web searches if the initial retrieval is poor.  • **Demo** |
| **60 mins** | **Module 4: Advanced Architectures II - Agentic & Deep Research RAG** | • **The Autonomous Frontier:** Defining Agentic RAG as a shift from a static pipeline to a dynamic, LLM-orchestrated process. • **Agentic Patterns & Workflows:** A deeper look at agent roles, including Routing (for multi-source data), Query Planning, and ReAct (Reason-Act). Discussing how frameworks like LangGraph enable these complex, stateful workflows. • **Deep Research Architectures:** How agents enable iterative reasoning to decompose and solve complex questions that are impossible for single-pass RAG systems. • **Demo** |
| **30 mins** | **Module 5: Production-Ready RAG: Evaluation & Best Practices** | • **Evaluation-Driven Development:** The importance of establishing a rigorous, automated testing framework. • **Key RAG Metrics:** Measuring Retrieval (Context Precision/Recall) and Generation (Faithfulness, Answer Relevance). • **Overview of Evaluation Frameworks:** A brief look at popular tools like RAGAS, DeepEval, and TruLens. • **Common Failure Modes & Workshop Wrap-up:** Identifying and mitigating common issues, followed by a final Q\&A session. |