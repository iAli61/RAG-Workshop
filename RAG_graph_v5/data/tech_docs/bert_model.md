# BERT: Bidirectional Encoder Representations from Transformers

**BERT** is a transformer-based language model developed by Google Research in 2018. The acronym stands for **Bidirectional Encoder Representations from Transformers**.

## Key Innovation

BERT's primary innovation is its bidirectional training approach. Unlike previous models that processed text left-to-right or right-to-left, BERT processes text in both directions simultaneously using a masked language modeling (MLM) objective.

## Architecture

BERT is built on the Transformer encoder architecture:
- **Base Model**: 12 encoder layers, 768 hidden dimensions, 12 attention heads (110M parameters)
- **Large Model**: 24 encoder layers, 1024 hidden dimensions, 16 attention heads (340M parameters)

## Training Objectives

### Masked Language Modeling (MLM)
BERT randomly masks 15% of input tokens and trains the model to predict these masked tokens based on surrounding context.

### Next Sentence Prediction (NSP)
The model learns to predict whether two sentences follow each other in the original text, helping with tasks requiring understanding of sentence relationships.

## Pre-training Process

BERT was pre-trained on:
- **BookCorpus**: 800 million words
- **English Wikipedia**: 2.5 billion words

This massive pre-training allows BERT to learn rich language representations that can be fine-tuned for specific downstream tasks.

## Applications

BERT excels at:
- **Question Answering**: Understanding context to extract precise answers
- **Named Entity Recognition (NER)**: Identifying entities in text
- **Sentiment Analysis**: Understanding nuanced opinions and emotions
- **Text Classification**: Categorizing documents with high accuracy
- **Semantic Search**: Finding relevant information based on meaning

## Fine-tuning

BERT can be fine-tuned on task-specific data by adding a simple classification layer on top. This transfer learning approach achieves state-of-the-art results across many NLP benchmarks including:
- SQuAD (Question Answering)
- GLUE (General Language Understanding)
- SWAG (Commonsense Reasoning)

## Impact

BERT revolutionized NLP by demonstrating that large-scale pre-training on unlabeled data followed by task-specific fine-tuning could achieve unprecedented performance. It spawned numerous variants including RoBERTa, ALBERT, and DistilBERT.
