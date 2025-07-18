# Enhanced RAG Features in Infomaid

This document describes the enhanced Retrieval Augmented Generation (RAG) features added to Infomaid, including TF-IDF, BM25, improved document chunking, and hybrid retrieval methods.

## Overview

The enhanced RAG system provides multiple retrieval strategies and advanced document processing to improve the quality and relevance of responses when querying local document collections.

## Key Enhancements

### 1. Multiple Retrieval Methods

#### Vector Similarity (Default)
- Uses semantic embeddings for similarity search
- Best for: General semantic understanding
- Command: `--retrievalmethod vector`

#### TF-IDF Similarity  
- Term Frequency-Inverse Document Frequency with cosine similarity
- Best for: Keyword-based retrieval, technical documents
- Command: `--retrievalmethod tfidf`

#### BM25 Scoring
- Best Match 25 ranking algorithm (used by search engines)
- Best for: Information retrieval, ranking by relevance
- Command: `--retrievalmethod bm25`

#### Hybrid Search
- Combines vector embeddings and TF-IDF scores
- Best for: Balanced results, optimal performance
- Command: `--retrievalmethod hybrid` (default)

### 2. Advanced Document Chunking

#### Semantic Chunking
- Respects sentence boundaries and maintains context
- Uses overlap for continuity between chunks
- Preserves meaning better than character-based splitting

#### Topic-Based Chunking  
- Groups related content using TF-IDF and K-means clustering
- Creates topically coherent chunks
- Automatically determines optimal number of topics

#### Hierarchical Chunking
- Follows document structure (paragraphs → sentences)
- Preserves document organization
- Adapts to content size dynamically

#### Adaptive Chunking
- Adjusts chunk size based on content complexity
- Considers sentence length, punctuation, and structure
- Optimizes for different content types

### 3. Query Enhancement

#### Query Expansion
- Automatically adds related terms using TF-IDF vocabulary
- Improves recall for complex queries
- Uses co-occurrence analysis for term relationships

#### Reranking
- Applies additional scoring after initial retrieval
- Multiple reranking algorithms available
- Improves precision of top results

### 4. Deduplication
- Removes similar or duplicate chunks using cosine similarity
- Configurable similarity threshold
- Keeps longer, more informative chunks

## Installation

Install the additional dependencies for enhanced features:

```bash
poetry add scikit-learn numpy
```

Or update your existing installation:

```bash
poetry install
```

## Usage Examples

### Basic Enhanced RAG

```bash
# Use enhanced RAG with hybrid search (recommended)
poetry run infomaid --useowndata --enhancedrag --prompt "Your question here"
```

### Specific Retrieval Methods

```bash
# Use TF-IDF for keyword-heavy documents
poetry run infomaid --useowndata --enhancedrag --retrievalmethod tfidf --prompt "Find specific technical terms"

# Use BM25 for search engine-like ranking
poetry run infomaid --useowndata --enhancedrag --retrievalmethod bm25 --prompt "Rank by relevance"

# Use pure vector similarity
poetry run infomaid --useowndata --enhancedrag --retrievalmethod vector --prompt "Semantic similarity"
```

### Multiple Results with Enhanced Methods

```bash
# Get 3 results using hybrid approach
poetry run infomaid --useowndata --enhancedrag --retrievalmethod hybrid --count 3 --prompt "Compare different approaches"
```

### Working with Different Document Types

```bash
# For technical documentation
poetry run infomaid --resetdb --usepdf --enhancedrag --retrievalmethod tfidf

# For general documents  
poetry run infomaid --resetdb --usepdf --enhancedrag --retrievalmethod hybrid

# For search/ranking tasks
poetry run infomaid --resetdb --usepdf --enhancedrag --retrievalmethod bm25
```

## Performance Comparison

| Method | Keyword Matching | Semantic Understanding | Ranking Quality | Speed |
|--------|------------------|----------------------|-----------------|-------|
| Vector | Fair | Excellent | Good | Fast |
| TF-IDF | Excellent | Fair | Good | Very Fast |
| BM25 | Excellent | Fair | Excellent | Fast |
| Hybrid | Excellent | Excellent | Excellent | Moderate |

## Configuration Parameters

### Chunking Parameters
- `max_chunk_size`: Maximum characters per chunk (default: 800)
- `min_chunk_size`: Minimum characters per chunk (default: 200)
- `chunk_overlap`: Overlap between chunks (default: 80)

### TF-IDF Parameters
- `max_features`: Maximum vocabulary size (default: 10000)
- `ngram_range`: N-gram range for features (default: 1-3)
- `min_df`: Minimum document frequency (default: 2)
- `max_df`: Maximum document frequency (default: 0.95)

### BM25 Parameters
- `k1`: Term frequency saturation (default: 1.5)
- `b`: Length normalization (default: 0.75)

### Hybrid Search Parameters
- `alpha`: Weight for vector search (default: 0.5)
- Vector weight = alpha, TF-IDF weight = (1 - alpha)

## Best Practices

### When to Use Each Method

1. **Vector Search**: 
   - General purpose queries
   - When semantic meaning is important
   - Cross-language similarity

2. **TF-IDF Search**:
   - Technical documentation
   - Keyword-heavy content
   - When exact term matching is crucial

3. **BM25 Search**:
   - Search engine-like behavior
   - Ranking multiple results
   - Information retrieval tasks

4. **Hybrid Search**:
   - Best overall performance
   - When you need both semantic and keyword matching
   - Complex queries with multiple aspects

### Document Preparation Tips

1. **For Technical Documents**: Use TF-IDF or BM25 methods
2. **For Narrative Content**: Use vector or hybrid methods  
3. **For Mixed Content**: Use hybrid with semantic chunking
4. **For Structured Documents**: Use hierarchical chunking

### Query Optimization

1. **Include key terms** you want to find
2. **Use specific language** from your documents
3. **Try different retrieval methods** for comparison
4. **Use multiple results** (`--count`) to see variations

## Troubleshooting

### Enhanced Features Not Available
```
Error: Enhanced RAG features not available
Solution: Install dependencies with `poetry add scikit-learn numpy`
```

### Poor Retrieval Quality
- Try different retrieval methods
- Check if your query terms appear in the documents
- Use hybrid search for balanced results
- Increase the number of results (`--count`)

### Slow Performance
- Use vector or TF-IDF instead of hybrid
- Reduce the number of chunks in your database
- Use smaller chunk sizes when populating database

## Technical Details

### TF-IDF Implementation
- Uses sklearn's TfidfVectorizer
- Supports n-grams (1-3 terms)
- Filters stop words and rare terms
- Cosine similarity for scoring

### BM25 Implementation
- Classic BM25 formula with tunable parameters
- Document length normalization
- Term frequency saturation
- Inverse document frequency weighting

### Hybrid Scoring
- Linear combination of normalized scores
- Min-max normalization for score alignment
- Configurable weighting between methods

### Chunking Algorithms
- Semantic: Sentence boundary awareness
- Topic: K-means clustering on TF-IDF vectors
- Hierarchical: Document structure preservation
- Adaptive: Content complexity analysis

## Contributing

To contribute enhancements to the RAG system:

1. Focus on improving retrieval quality
2. Add new chunking strategies
3. Implement additional scoring methods
4. Optimize performance for large document collections

## References

- BM25: Robertson, S. & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond
- TF-IDF: Salton, G. & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval
- Vector Embeddings: Modern transformer-based embeddings for semantic similarity
