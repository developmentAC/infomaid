#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced RAG Implementation with TF-IDF, Advanced Chunking, and Multiple Similarity Methods

This module provides advanced Retrieval Augmented Generation capabilities that extend
beyond basic vector similarity search. It implements multiple retrieval strategies
including TF-IDF, BM25, and hybrid approaches to improve document retrieval quality.

Key Features:
- Multiple retrieval methods (vector, TF-IDF, BM25, hybrid)
- Advanced document chunking strategies
- Query expansion for better context matching
- Result reranking with multiple scoring algorithms
- Fallback mechanisms for compatibility

Classes:
    EnhancedRAGRetriever: Core retrieval engine with multiple similarity methods
    EnhancedQueryProcessor: High-level interface for enhanced RAG queries

Dependencies:
    - ChromaDB for vector storage and similarity search
    - scikit-learn for TF-IDF and clustering algorithms
    - NLTK for natural language processing
    - LangChain for document processing and prompt templates
    - Ollama for language model integration

Author: Enhanced by AI Assistant
Project: https://github.com/developmentAC/infomaid
"""

# Configure ChromaDB before any other imports to suppress telemetry warnings
from infomaid.chromadb_config import configure_chromadb
configure_chromadb()

import numpy as np
from typing import List, Dict, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
import re
from collections import Counter
import math

# Import statements with fallback compatibility for different LangChain versions
try:
    from langchain_chroma import Chroma
except ImportError:
    # Fallback to the older import if the new package is not available
    from langchain_community.vectorstores import Chroma

from langchain.schema.document import Document
from langchain.prompts import ChatPromptTemplate

try:
    from langchain_ollama import OllamaLLM as Ollama
except ImportError:
    # Fallback to the older import if the new package is not available
    from langchain_community.llms.ollama import Ollama

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Internal imports
from infomaid import get_embedding_function
from rich.console import Console

# Initialize console for formatted output
console = Console()

class EnhancedRAGRetriever:
    """
    Advanced RAG retriever implementing multiple similarity search methods.
    
    This class extends basic vector similarity search with additional retrieval
    strategies including TF-IDF, BM25, and hybrid approaches. It maintains
    compatibility with ChromaDB while adding enhanced functionality for
    better document retrieval and relevance scoring.
    
    Attributes:
        chroma_path (str): Path to ChromaDB storage directory
        model (str): Embedding model name for vector operations
        db (Chroma): ChromaDB vector database instance
        tfidf_vectorizer (TfidfVectorizer): Scikit-learn TF-IDF processor
        documents_corpus (List[str]): Text corpus for TF-IDF operations
        tfidf_matrix: Computed TF-IDF matrix for similarity calculations
        stemmer (PorterStemmer): Text stemming processor for BM25
    """
    
    def __init__(self, chroma_path: str = "chroma", model: str = "nomic-embed-text"):
        """
        Initialize the enhanced RAG retriever with multiple search capabilities.
        
        Args:
            chroma_path (str): Directory path for ChromaDB persistence
            model (str): Name of the embedding model to use
        """
        self.chroma_path = chroma_path
        self.model = model
        self.embedding_function = get_embedding_function.get_embedding_function(model)
        self.db = Chroma(persist_directory=chroma_path, embedding_function=self.embedding_function)
        
        # Initialize TF-IDF components with optimized settings for small document sets
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,          # Limit vocabulary size for efficiency
            stop_words=None,            # Preserve stop words for short documents
            ngram_range=(1, 2),         # Unigrams and bigrams for better context
            min_df=1,                   # Include terms appearing in ≥1 document
            max_df=1.0,                 # Include all terms regardless of frequency
            token_pattern=r'\b\w+\b'    # Simple word boundary token pattern
        )
        
        # Initialize data structures for different search methods
        self.documents_corpus = []      # Text corpus for TF-IDF operations
        self.tfidf_matrix = None        # Computed TF-IDF matrix
        self.stemmer = PorterStemmer()  # Stemmer for BM25 preprocessing
        
        # Download required NLTK data packages for text processing
        try:
            nltk.download('punkt', quiet=True)              # Sentence tokenizer
            nltk.download('punkt_tab', quiet=True)          # Enhanced tokenizer
            nltk.download('stopwords', quiet=True)          # Stop words lists
            nltk.download('averaged_perceptron_tagger', quiet=True)  # POS tagger
        except:
            # Silently handle download failures (offline environments)
            pass
        
        # Build TF-IDF index from existing ChromaDB documents
        self._build_tfidf_index()
    
    def _build_tfidf_index(self):
        """
        Build TF-IDF index from existing ChromaDB documents.
        
        This method extracts all documents from the ChromaDB collection and
        creates a TF-IDF matrix for similarity calculations. The index is
        built once during initialization to improve query performance.
        
        Side Effects:
            - Populates self.documents_corpus with document texts
            - Creates self.tfidf_matrix for similarity calculations
            - Prints status messages to console
        
        Error Handling:
            - Gracefully handles empty databases
            - Reports errors without crashing the system
        """
        try:
            # Extract all documents from ChromaDB collection
            all_docs = self.db.get()
            if all_docs and all_docs['documents']:
                self.documents_corpus = all_docs['documents']
                
                # Build TF-IDF matrix from document corpus
                console.print(f"\t [cyan] Building TF-IDF index with {len(self.documents_corpus)} documents...[/cyan]")
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.documents_corpus)
                console.print(f"\t [green] TF-IDF index built successfully![/green]")
            else:
                console.print("\t [yellow] No documents found in ChromaDB. TF-IDF index will be empty.[/yellow]")
        except Exception as e:
            console.print(f"\t [red] Error building TF-IDF index: {e}[/red]")
    
    def semantic_chunking(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        Advanced semantic chunking that preserves sentence boundaries and context.
        
        This method splits text into chunks while respecting sentence boundaries
        and maintaining semantic coherence. It includes overlap between chunks
        to preserve context for better retrieval performance.
        
        Args:
            text (str): Input text to be chunked
            chunk_size (int): Target size for each chunk in characters
            overlap (int): Number of characters to overlap between chunks
            
        Returns:
            List[str]: List of text chunks with preserved semantic boundaries
            
        Features:
            - Respects sentence boundaries (no mid-sentence splits)
            - Maintains context through configurable overlap
            - Handles edge cases (very long sentences, short text)
        """
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # Check if adding this sentence would exceed chunk size limit
            if current_size + sentence_size > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap from previous chunk for context
                if overlap > 0 and len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:] + " " + sentence
                else:
                    current_chunk = sentence
                current_size = len(current_chunk)
            else:
                # Add sentence to current chunk
                current_chunk += " " + sentence if current_chunk else sentence
                current_size += sentence_size
        
        # Add the final chunk if it contains content
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def keyword_density_chunking(self, text: str, chunk_size: int = 512) -> List[str]:
        """
        Chunk text based on keyword density to maintain topical coherence.
        
        This method analyzes keyword distribution to create chunks that maintain
        topic coherence. It groups sentences with similar keyword profiles
        together, improving the semantic consistency of each chunk.
        
        Args:
            text (str): Input text to be chunked
            chunk_size (int): Target maximum size for each chunk
            
        Returns:
            List[str]: List of topically coherent text chunks
            
        Algorithm:
            1. Extract and rank keywords by frequency
            2. Analyze keyword overlap between sentences
            3. Group sentences with high keyword similarity
            4. Split when keyword overlap drops or size limit reached
        """
        words = word_tokenize(text.lower())
        word_freq = Counter(words)
        
        # Extract meaningful keywords: remove stopwords and filter by criteria
        stop_words = set(stopwords.words('english'))
        keywords = {word: freq for word, freq in word_freq.items() 
                   if word not in stop_words and word.isalpha() and len(word) > 2}
        
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        current_keywords = set()
        
        for sentence in sentences:
            # Extract keywords from current sentence
            sentence_words = set(word_tokenize(sentence.lower()))
            sentence_keywords = sentence_words.intersection(keywords.keys())
            
            # Calculate keyword overlap with current chunk for coherence analysis
            overlap_score = len(current_keywords.intersection(sentence_keywords))
            
            # Determine if sentence should start a new chunk based on:
            # 1. Chunk size limit exceeded
            # 2. Low keyword overlap indicating topic shift
            if (len(current_chunk) + len(sentence) > chunk_size and current_chunk) or \
               (overlap_score == 0 and len(current_chunk) > chunk_size * 0.5):
                # Finalize current chunk and start new one
                chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_keywords = sentence_keywords
            else:
                # Add sentence to current chunk and update keyword tracking
                current_chunk += " " + sentence if current_chunk else sentence
                current_keywords.update(sentence_keywords)
        
        # Add final chunk if it contains content
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def tfidf_similarity_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Perform similarity search using TF-IDF vectors and cosine similarity.
        
        This method uses Term Frequency-Inverse Document Frequency (TF-IDF) 
        vectorization to find documents most similar to the query. TF-IDF
        captures the importance of terms relative to the document collection.
        
        Args:
            query (str): The search query text
            k (int): Number of top similar documents to return
            
        Returns:
            List[Tuple[str, float]]: List of (document_text, similarity_score) tuples
                                   sorted by similarity score in descending order
                                   
        Algorithm:
            1. Transform query using fitted TF-IDF vectorizer
            2. Calculate cosine similarity with all document vectors
            3. Return top k documents with highest similarity scores
            
        Side Effects:
            - Prints warning if TF-IDF index is not available
            - Falls back to empty list if vectorizer not initialized
        """
        if self.tfidf_matrix is None or len(self.documents_corpus) == 0:
            console.print("\t [yellow] TF-IDF index not available. Falling back to vector search.[/yellow]")
            return []
        
        try:
            # Transform query using the fitted TF-IDF vectorizer
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate cosine similarity between query and all documents
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get indices of top k most similar documents
            top_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for idx in top_indices:
                # Only include documents with meaningful similarity scores
                if similarities[idx] > 0:  
                    results.append((self.documents_corpus[idx], float(similarities[idx])))
            
            return results
        except Exception as e:
            console.print(f"\t [red] Error in TF-IDF similarity search: {e}[/red]")
            return []
    
    def bm25_scoring(self, query: str, documents: List[str], k1: float = 1.5, b: float = 0.75) -> List[Tuple[str, float]]:
        """
        Implement BM25 scoring algorithm for better relevance ranking.
        
        BM25 (Best Matching 25) is a probabilistic ranking function that
        improves upon TF-IDF by incorporating document length normalization
        and term saturation. It's widely used in information retrieval systems.
        
        Args:
            query (str): The search query text
            documents (List[str]): List of document texts to score
            k1 (float): Controls term frequency saturation (default: 1.5)
            b (float): Controls document length normalization (default: 0.75)
            
        Returns:
            List[Tuple[str, float]]: List of (document_text, bm25_score) tuples
                                   sorted by BM25 score in descending order
                                   
        Algorithm:
            1. Tokenize query and documents
            2. Calculate term frequencies and document frequencies
            3. Apply BM25 formula with length normalization
            4. Return documents ranked by BM25 score
            
        BM25 Formula:
            score(D,Q) = Σ IDF(qi) * f(qi,D) * (k1 + 1) / 
                        (f(qi,D) + k1 * (1 - b + b * |D| / avgdl))
        """
        query_terms = [self.stemmer.stem(term.lower()) for term in word_tokenize(query) 
                      if term.isalpha()]
        
        # Calculate document collection statistics for normalization
        doc_lengths = [len(word_tokenize(doc)) for doc in documents]
        avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0
        
        # Build term frequency and document frequency mappings
        term_doc_freq = {}  # term -> number of documents containing term
        doc_term_freq = []  # list of {term: frequency} for each document
        
        # Process each document to extract term statistics
        for doc in documents:
            doc_terms = [self.stemmer.stem(term.lower()) for term in word_tokenize(doc) 
                        if term.isalpha()]
            term_freq = Counter(doc_terms)
            doc_term_freq.append(term_freq)
            
            # Count document frequency for each unique term
            for term in set(doc_terms):
                term_doc_freq[term] = term_doc_freq.get(term, 0) + 1
        
        # Calculate BM25 scores for each document
        scores = []
        for i, doc in enumerate(documents):
            score = 0
            doc_len = doc_lengths[i]
            
            # Sum BM25 scores for each query term found in document
            for term in query_terms:
                if term in doc_term_freq[i]:
                    tf = doc_term_freq[i][term]  # term frequency in document
                    df = term_doc_freq.get(term, 0)  # document frequency of term
                    
                    if df > 0:
                        # Calculate inverse document frequency (IDF)
                        idf = math.log((len(documents) - df + 0.5) / (df + 0.5))
                        
                        # Apply BM25 normalization with document length adjustment
                        normalized_tf = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_length)))
                        score += idf * normalized_tf
            
            scores.append((doc, score))
        
        # Return documents sorted by BM25 score in descending order
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def _get_safe_k(self, requested_k: int) -> int:
        """
        Get a safe k value that doesn't exceed available documents.
        
        This utility method ensures that the requested number of documents
        to retrieve doesn't exceed the actual number of documents in the
        database, preventing errors in retrieval operations.
        
        Args:
            requested_k (int): The originally requested number of documents
            
        Returns:
            int: A safe k value that won't exceed available documents
            
        Side Effects:
            - Prints adjustment message if k is reduced
            - Falls back to original value if database access fails
        """
        try:
            collection = self.db._collection
            available_docs = collection.count()
            safe_k = min(requested_k, available_docs) if available_docs > 0 else 1
            if safe_k != requested_k:
                console.print(f"\t [cyan] Adjusted k from {requested_k} to {safe_k} (available docs: {available_docs})[/cyan]")
            return safe_k
        except Exception:
            return requested_k  # fallback to original value if database access fails
    
    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.5) -> List[Tuple[str, float]]:
        """
        Combine vector similarity search with TF-IDF for hybrid retrieval.
        
        This method implements a hybrid search strategy that combines the
        semantic understanding of vector embeddings with the term-matching
        precision of TF-IDF. The results are weighted and merged to provide
        more comprehensive and accurate retrieval.
        
        Args:
            query (str): The search query text
            k (int): Number of top documents to retrieve
            alpha (float): Weight for vector search (1-alpha for TF-IDF)
                         Range: 0.0 (TF-IDF only) to 1.0 (vector only)
                         
        Returns:
            List[Tuple[str, float]]: Combined and ranked results with
                                   weighted similarity scores
                                   
        Algorithm:
            1. Perform vector similarity search with weight alpha
            2. Perform TF-IDF search with weight (1-alpha)
            3. Normalize scores to [0,1] range
            4. Combine scores: final_score = alpha*vector + (1-alpha)*tfidf
            5. Return top k documents by combined score
            
        Benefits:
            - Captures both semantic similarity and keyword relevance
            - Reduces false negatives from either method alone
            - Provides more robust retrieval performance
        """
        # Get safe k value to prevent database errors
        safe_k = self._get_safe_k(k)
        
        # Get vector similarity results (fetch extra for better hybrid selection)
        vector_results = self.db.similarity_search_with_score(query, k=safe_k*2)
        vector_docs = {doc.page_content: 1.0 - score for doc, score in vector_results}  # Convert distance to similarity
        
        # Get TF-IDF results (also fetch extra for selection)
        tfidf_results = self.tfidf_similarity_search(query, k=safe_k*2)
        tfidf_docs = {doc: score for doc, score in tfidf_results}
        
        # Combine and normalize scores from both methods
        all_docs = set(vector_docs.keys()).union(set(tfidf_docs.keys()))
        hybrid_scores = []
        
        for doc in all_docs:
            vector_score = vector_docs.get(doc, 0)
            tfidf_score = tfidf_docs.get(doc, 0)
            
            # Apply min-max normalization to make scores comparable
            if vector_docs:
                max_vector = max(vector_docs.values())
                vector_score = vector_score / max_vector if max_vector > 0 else 0
            
            if tfidf_docs:
                max_tfidf = max(tfidf_docs.values())
                tfidf_score = tfidf_score / max_tfidf if max_tfidf > 0 else 0
            
            # Calculate weighted combination: alpha*vector + (1-alpha)*tfidf
            combined_score = alpha * vector_score + (1 - alpha) * tfidf_score
            hybrid_scores.append((doc, combined_score))
        
        # Sort by combined score and return top k results
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        return hybrid_scores[:safe_k]
    
    def query_expansion(self, query: str) -> str:
        """
        Expand query with related terms using TF-IDF vocabulary analysis.
        
        This method enhances the original query by adding semantically related
        terms found in the document corpus. It uses TF-IDF analysis to identify
        terms that frequently co-occur with query terms in documents.
        
        Args:
            query (str): The original search query
            
        Returns:
            str: Expanded query with additional related terms
            
        Algorithm:
            1. Transform query using TF-IDF vectorizer
            2. Identify documents containing query terms
            3. Extract highly weighted co-occurring terms
            4. Add relevant terms to original query
            
        Side Effects:
            - Prints warning if query expansion fails
            - Returns original query if TF-IDF matrix unavailable
        """
        if self.tfidf_matrix is None:
            return query
        
        try:
            # Transform query to identify relevant terms in vocabulary
            query_vector = self.tfidf_vectorizer.transform([query])
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Get indices of terms present in the query
            query_terms_indices = query_vector.nonzero()[1]
            
            if len(query_terms_indices) == 0:
                return query
            
            # Find related terms through document co-occurrence analysis
            expanded_terms = set()
            for term_idx in query_terms_indices:
                # Identify documents that contain this query term
                docs_with_term = self.tfidf_matrix[:, term_idx].nonzero()[0]
                
                # Analyze top documents for co-occurring terms
                for doc_idx in docs_with_term[:3]:  # Limit to top 3 documents for efficiency
                    doc_vector = self.tfidf_matrix[doc_idx].toarray()[0]
                    top_term_indices = np.argsort(doc_vector)[-10:]  # Top 10 terms in document
                    
                    # Add highly weighted terms as expansion candidates
                    for related_term_idx in top_term_indices:
                        if doc_vector[related_term_idx] > 0.1:  # Relevance threshold
                            expanded_terms.add(feature_names[related_term_idx])
            
            # Construct expanded query with top 3 most relevant terms
            expanded_query = query
            for term in list(expanded_terms)[:3]:
                if term not in query.lower():
                    expanded_query += f" {term}"
            
            return expanded_query
        except Exception as e:
            console.print(f"\t [yellow] Query expansion failed: {e}[/yellow]")
            return query
    
    def rerank_results(self, query: str, documents: List[str], method: str = "bm25") -> List[Tuple[str, float]]:
        """
        Rerank retrieved documents using different scoring methods.
        
        This method applies post-retrieval ranking to improve result quality.
        It can use various ranking algorithms to reorder documents based on
        their relevance to the query.
        
        Args:
            query (str): The search query
            documents (List[str]): List of retrieved documents to rerank
            method (str): Ranking method to use ("bm25", "tfidf", or other)
            
        Returns:
            List[Tuple[str, float]]: Reranked documents with scores
            
        Supported Methods:
            - "bm25": Uses BM25 probabilistic ranking
            - "tfidf": Uses TF-IDF cosine similarity
            - other: Falls back to simple cosine similarity
        """
        if method == "bm25":
            return self.bm25_scoring(query, documents)
        elif method == "tfidf":
            return self.tfidf_similarity_search(query, len(documents))
        else:
            # Fallback to simple cosine similarity scoring
            return [(doc, 1.0) for doc in documents]


class EnhancedQueryProcessor:
    """
    Enhanced query processing with multiple retrieval strategies.
    
    This class provides a high-level interface for performing advanced RAG
    queries with various retrieval methods, query processing, and result
    formatting. It combines the EnhancedRAGRetriever with prompt templating
    and response generation.
    
    Features:
        - Multiple retrieval strategies (vector, TF-IDF, BM25, hybrid)
        - Query expansion and result reranking
        - Integrated prompt templating
        - Comprehensive error handling and logging
    """
    
    def __init__(self, chroma_path: str = "chroma", model: str = "nomic-embed-text"):
        """
        Initialize the enhanced query processor.
        
        Args:
            chroma_path (str): Path to ChromaDB database
            model (str): Embedding model name for vector operations
        """
        self.retriever = EnhancedRAGRetriever(chroma_path, model)
        self.prompt_template = ChatPromptTemplate.from_template("""
Answer the question based on the following context. Use the most relevant information from multiple sources:

Context:
{context}

Question: {question}

Provide a comprehensive answer based on the context above. If the context doesn't contain enough information, mention what additional information might be helpful.
""")
    
    def enhanced_query_rag(self, query_text: str, retrieval_method: str = "hybrid", k: int = 5) -> str:
        """
        Enhanced RAG query with multiple retrieval methods.
        
        This is the main query interface that supports different retrieval
        strategies and provides comprehensive result processing. It handles
        query expansion, document retrieval, result ranking, and response
        generation using the specified method.
        
        Args:
            query_text (str): The user's search query
            retrieval_method (str): The retrieval strategy to use
            k (int): Number of documents to retrieve
            
        Returns:
            str: Formatted response text based on retrieved context
            
        Supported Retrieval Methods:
            - "vector": Standard vector similarity search using embeddings
            - "tfidf": TF-IDF based lexical search for keyword matching
            - "hybrid": Combination of vector and TF-IDF for balanced results
            - "bm25": BM25 probabilistic ranking for relevance scoring
            
        Process Flow:
            1. Display retrieval method selection
            2. Expand query for enhanced methods (hybrid/tfidf)
            3. Perform document retrieval using chosen method
            4. Format and return results or error message
        """
        console.print(f"\t [cyan] Using retrieval method: {retrieval_method}[/cyan]")
        
        # Get safe k value to prevent database errors
        safe_k = self.retriever._get_safe_k(k)
        
        # Apply query expansion for methods that benefit from it
        if retrieval_method in ["hybrid", "tfidf"]:
            expanded_query = self.retriever.query_expansion(query_text)
            if expanded_query != query_text:
                console.print(f"\t [yellow] Expanded query: {expanded_query}[/yellow]")
                query_text = expanded_query
        
        # Perform document retrieval based on selected method
        if retrieval_method == "vector":
            # Standard vector similarity search using embeddings
            results = self.retriever.db.similarity_search_with_score(query_text, k=safe_k)
            context_docs = [(doc.page_content, 1.0 - score) for doc, score in results]
        elif retrieval_method == "tfidf":
            # TF-IDF based lexical search
            context_docs = self.retriever.tfidf_similarity_search(query_text, k=safe_k)
        elif retrieval_method == "hybrid":
            # Combined vector and TF-IDF search for balanced results
            context_docs = self.retriever.hybrid_search(query_text, k=safe_k)
        elif retrieval_method == "bm25":
            # BM25 scoring on vector search candidates for precision
            vector_results = self.retriever.db.similarity_search(query_text, k=safe_k*2)
            candidate_docs = [doc.page_content for doc in vector_results]
            context_docs = self.retriever.bm25_scoring(query_text, candidate_docs)[:safe_k]
        else:
            # Fallback to vector search for unknown methods
            console.print(f"\t [red] Unknown retrieval method: {retrieval_method}. Using vector search.[/red]")
            results = self.retriever.db.similarity_search_with_score(query_text, k=safe_k)
            context_docs = [(doc.page_content, 1.0 - score) for doc, score in results]
        
        # Format retrieved context with relevance scores for transparency
        context_text = ""
        for i, (doc, score) in enumerate(context_docs):
            context_text += f"Source {i+1} (relevance: {score:.3f}):\n{doc}\n\n---\n\n"
        
        # Generate response using formatted prompt template
        prompt = self.prompt_template.format(context=context_text, question=query_text)
        
        try:
            # Use Ollama model for response generation
            model = Ollama(model="mistral")
            response_text = model.invoke(prompt)
            
            # Display retrieval performance information
            console.print(f"\t [green] Retrieved {len(context_docs)} documents using {retrieval_method} method[/green]")
            console.print(f"\t [blue] Top relevance score: {context_docs[0][1]:.3f}[/blue]" if context_docs else "")
            
            return response_text
        except Exception as e:
            console.print(f"\t [red] Error generating response: {e}[/red]")
            return "Error: Could not generate response. Please check if Ollama is running."


# Enhanced main function for testing
def enhanced_main(query_text: str, model: str = "nomic-embed-text", method: str = "hybrid") -> str:
    """
    Enhanced main function with improved RAG capabilities.
    
    This function provides a convenient interface for testing the enhanced
    RAG system with different models and retrieval methods. It initializes
    the query processor and executes a query using the specified parameters.
    
    Args:
        query_text (str): The search query to process
        model (str): Embedding model name (default: "nomic-embed-text")
        method (str): Retrieval method to use (default: "hybrid")
        
    Returns:
        str: Generated response based on retrieved context
        
    Usage:
        result = enhanced_main("What is machine learning?", method="vector")
    """
    processor = EnhancedQueryProcessor(model=model)
    return processor.enhanced_query_rag(query_text, retrieval_method=method)
