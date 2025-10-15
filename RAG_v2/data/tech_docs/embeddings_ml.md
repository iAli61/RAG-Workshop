# Embeddings in Machine Learning

**Embeddings** are dense vector representations of discrete objects (words, images, users, items) that capture semantic meaning in a continuous, low-dimensional space. They are fundamental to modern machine learning, particularly in natural language processing and recommendation systems.

## Core Concept

Embeddings transform high-dimensional, sparse representations into compact, dense vectors where semantic similarity is reflected by geometric proximity. Objects with similar meanings have similar vector representations.

### Example:
```
"king" - "man" + "woman" ≈ "queen"
```

This famous example demonstrates how embeddings capture semantic relationships through vector arithmetic.

## Types of Embeddings

### Word Embeddings

#### Word2Vec
Developed by Google in 2013, using two architectures:
- **Skip-gram**: Predicts context words from target word
- **CBOW (Continuous Bag of Words)**: Predicts target word from context

Training objective: Maximize probability of context words given target word.

#### GloVe (Global Vectors)
Stanford's approach combining:
- Global matrix factorization
- Local context window methods
- Trains on word co-occurrence statistics

#### FastText
Facebook's extension of Word2Vec:
- Represents words as bags of character n-grams
- Handles out-of-vocabulary words
- Better for morphologically rich languages

### Contextualized Embeddings

Modern approaches generate different vectors for the same word based on context:

#### ELMo (Embeddings from Language Models)
- Bidirectional LSTM-based
- Generates context-dependent representations
- Three layers of representation (character, word, context)

#### BERT Embeddings
- Transformer-based contextual embeddings
- Considers both left and right context
- Pre-trained on massive corpora
- State-of-the-art for many NLP tasks

#### Sentence-BERT
- Specialized for sentence and paragraph embeddings
- Uses Siamese networks for efficiency
- Excellent for semantic similarity tasks

### Other Embedding Types

#### Image Embeddings
- **ResNet, VGG**: CNN-based feature vectors
- **CLIP**: Joint vision-language embeddings
- **Vision Transformers**: Attention-based image representations

#### User/Item Embeddings
- **Matrix Factorization**: For recommendation systems
- **Node2Vec**: Graph-based embeddings
- **LightFM**: Hybrid collaborative and content-based

## Key Properties

### Dimensionality
Typical sizes:
- Word2Vec: 100-300 dimensions
- BERT: 768 (base) or 1024 (large) dimensions
- Sentence-BERT: 384-768 dimensions

Higher dimensions capture more information but increase computational cost.

### Distance Metrics

#### Cosine Similarity
Measures angle between vectors:
```
cosine_sim(A, B) = (A · B) / (||A|| × ||B||)
```
Range: [-1, 1] where 1 = identical direction

#### Euclidean Distance
Straight-line distance in space:
```
euclidean_dist(A, B) = √(Σ(A_i - B_i)²)
```

#### Dot Product
Direct multiplication (used in attention mechanisms):
```
dot_product(A, B) = Σ(A_i × B_i)
```

## Training Embeddings

### Supervised Learning
Train embeddings as part of a larger task:
- Classification models learn task-specific representations
- End-to-end training optimizes for specific objectives

### Self-Supervised Learning
Learn from data structure without labels:
- Masked language modeling (BERT)
- Next sentence prediction
- Contrastive learning (SimCLR, CLIP)

### Transfer Learning
Use pre-trained embeddings:
1. Download pre-trained model
2. Fine-tune on specific task
3. Or use as fixed feature extractors

## Applications

### Natural Language Processing
- **Semantic Search**: Find similar documents/queries
- **Text Classification**: Use embeddings as features
- **Clustering**: Group similar texts
- **Question Answering**: Match questions to answers
- **Translation**: Align words across languages

### Recommendation Systems
- **Collaborative Filtering**: User and item embeddings
- **Cold Start Problem**: Use content-based embeddings
- **Session-Based**: Sequential embedding models

### Computer Vision
- **Image Search**: Find visually similar images
- **Face Recognition**: Embedding-based identification
- **Object Detection**: Feature representations

### Retrieval-Augmented Generation (RAG)
- **Document Retrieval**: Embed documents for semantic search
- **Query-Document Matching**: Find relevant context
- **Hybrid Search**: Combine with keyword methods

## Evaluation Metrics

### Intrinsic Evaluation
- **Word Similarity**: Correlation with human judgments
- **Analogies**: Solve relationship tasks (king:queen :: man:?)
- **Visualization**: t-SNE or UMAP plots

### Extrinsic Evaluation
Measure performance on downstream tasks:
- Classification accuracy
- Retrieval precision/recall
- End-task metrics

## Best Practices

1. **Choose Appropriate Dimension**: Balance expressiveness and efficiency
2. **Normalize Vectors**: For cosine similarity comparisons
3. **Domain Adaptation**: Fine-tune pre-trained embeddings
4. **Handle OOV**: Use subword embeddings (FastText, BPE)
5. **Batch Encoding**: Process multiple items efficiently
6. **Caching**: Store computed embeddings for reuse
7. **Version Control**: Track embedding model versions

## Challenges

- **Bias**: Embeddings can capture and amplify societal biases
- **Polysemy**: Single vector for multiple word meanings (solved by contextualized embeddings)
- **Computational Cost**: Large models require significant resources
- **Interpretability**: Difficult to understand what dimensions represent
