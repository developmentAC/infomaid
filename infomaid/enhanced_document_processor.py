#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced document processing with improved chunking strategies
"""

import os
import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
from collections import Counter

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma

from infomaid import get_embedding_function
from rich.console import Console

console = Console()

class EnhancedDocumentProcessor:
    """Enhanced document processor with multiple chunking strategies."""
    
    def __init__(self, model: str = "nomic-embed-text"):
        self.model = model
        self.embedding_function = get_embedding_function.get_embedding_function(model)
        
        # Initialize TF-IDF for content analysis
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,  # Include terms that appear in at least 1 document (for small datasets)
            max_df=0.8
        )
        
        # Download NLTK data if needed
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True)
        except:
            pass
    
    def semantic_aware_chunking(self, text: str, max_chunk_size: int = 800, min_chunk_size: int = 200) -> List[Document]:
        """
        Create chunks that respect semantic boundaries and maintain context.
        """
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        current_sentences = []
        
        for sentence in sentences:
            # Calculate if adding this sentence would exceed max size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) > max_chunk_size and len(current_chunk) >= min_chunk_size:
                # Finalize current chunk
                chunks.append(Document(
                    page_content=current_chunk.strip(),
                    metadata={
                        "chunk_type": "semantic",
                        "sentence_count": len(current_sentences),
                        "chunk_size": len(current_chunk)
                    }
                ))
                
                # Start new chunk with some overlap for context
                overlap_sentences = current_sentences[-2:] if len(current_sentences) >= 2 else current_sentences
                current_chunk = " ".join(overlap_sentences) + " " + sentence
                current_sentences = overlap_sentences + [sentence]
            else:
                current_chunk = potential_chunk
                current_sentences.append(sentence)
        
        # Add final chunk
        if current_chunk and len(current_chunk) >= min_chunk_size:
            chunks.append(Document(
                page_content=current_chunk.strip(),
                metadata={
                    "chunk_type": "semantic",
                    "sentence_count": len(current_sentences),
                    "chunk_size": len(current_chunk)
                }
            ))
        
        return chunks
    
    def topic_based_chunking(self, text: str, num_topics: int = None) -> List[Document]:
        """
        Chunk documents based on topic clustering using TF-IDF and k-means.
        """
        sentences = sent_tokenize(text)
        
        if len(sentences) < 5:
            # Too few sentences for clustering, use semantic chunking
            return self.semantic_aware_chunking(text)
        
        # Calculate optimal number of topics if not specified
        if num_topics is None:
            num_topics = max(2, min(len(sentences) // 5, 8))
        
        try:
            # Create TF-IDF vectors for sentences
            sentence_vectors = self.tfidf_vectorizer.fit_transform(sentences)
            
            # Perform k-means clustering
            kmeans = KMeans(n_clusters=num_topics, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(sentence_vectors)
            
            # Group sentences by cluster
            clustered_sentences = {}
            for i, cluster_id in enumerate(clusters):
                if cluster_id not in clustered_sentences:
                    clustered_sentences[cluster_id] = []
                clustered_sentences[cluster_id].append((i, sentences[i]))
            
            # Create chunks from clusters
            chunks = []
            for cluster_id, cluster_sentences in clustered_sentences.items():
                # Sort sentences by original order
                cluster_sentences.sort(key=lambda x: x[0])
                
                # Combine sentences in the cluster
                chunk_text = " ".join([sent for _, sent in cluster_sentences])
                
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata={
                        "chunk_type": "topic_based",
                        "topic_id": cluster_id,
                        "sentence_count": len(cluster_sentences),
                        "chunk_size": len(chunk_text)
                    }
                ))
            
            return chunks
            
        except Exception as e:
            console.print(f"[yellow]Topic clustering failed: {e}. Falling back to semantic chunking.[/yellow]")
            return self.semantic_aware_chunking(text)
    
    def sliding_window_chunking(self, text: str, window_size: int = 512, stride: int = 256) -> List[Document]:
        """
        Create overlapping chunks using a sliding window approach.
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), stride):
            chunk_words = words[i:i + window_size]
            chunk_text = " ".join(chunk_words)
            
            if len(chunk_text.strip()) > 50:  # Minimum chunk size
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata={
                        "chunk_type": "sliding_window",
                        "window_start": i,
                        "window_end": i + len(chunk_words),
                        "chunk_size": len(chunk_text)
                    }
                ))
        
        return chunks
    
    def hierarchical_chunking(self, text: str) -> List[Document]:
        """
        Create hierarchical chunks: sections -> paragraphs -> sentences.
        """
        chunks = []
        
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        
        for para_idx, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if len(paragraph) < 50:  # Skip very short paragraphs
                continue
            
            # If paragraph is small enough, use as-is
            if len(paragraph) <= 800:
                chunks.append(Document(
                    page_content=paragraph,
                    metadata={
                        "chunk_type": "hierarchical_paragraph",
                        "paragraph_index": para_idx,
                        "chunk_size": len(paragraph)
                    }
                ))
            else:
                # Split large paragraphs into sentences
                sentences = sent_tokenize(paragraph)
                current_chunk = ""
                sentence_count = 0
                
                for sentence in sentences:
                    if len(current_chunk + sentence) > 800 and current_chunk:
                        chunks.append(Document(
                            page_content=current_chunk.strip(),
                            metadata={
                                "chunk_type": "hierarchical_sentence",
                                "paragraph_index": para_idx,
                                "sentence_count": sentence_count,
                                "chunk_size": len(current_chunk)
                            }
                        ))
                        current_chunk = sentence
                        sentence_count = 1
                    else:
                        current_chunk += " " + sentence if current_chunk else sentence
                        sentence_count += 1
                
                if current_chunk:
                    chunks.append(Document(
                        page_content=current_chunk.strip(),
                        metadata={
                            "chunk_type": "hierarchical_sentence",
                            "paragraph_index": para_idx,
                            "sentence_count": sentence_count,
                            "chunk_size": len(current_chunk)
                        }
                    ))
        
        return chunks
    
    def adaptive_chunking(self, text: str, target_chunk_size: int = 600) -> List[Document]:
        """
        Adaptive chunking that adjusts size based on content density and structure.
        """
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        current_sentences = []
        
        # Calculate sentence complexity scores
        sentence_scores = []
        for sentence in sentences:
            words = word_tokenize(sentence)
            # Score based on length, punctuation, and complexity
            complexity = len(words) + sentence.count(',') * 2 + sentence.count(';') * 3
            sentence_scores.append(complexity)
        
        avg_complexity = np.mean(sentence_scores) if sentence_scores else 0
        
        for i, sentence in enumerate(sentences):
            sentence_complexity = sentence_scores[i]
            
            # Adjust target size based on complexity
            if sentence_complexity > avg_complexity * 1.5:
                # Complex sentences get smaller chunks
                adjusted_target = target_chunk_size * 0.8
            elif sentence_complexity < avg_complexity * 0.5:
                # Simple sentences can have larger chunks
                adjusted_target = target_chunk_size * 1.2
            else:
                adjusted_target = target_chunk_size
            
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) > adjusted_target and current_chunk:
                chunks.append(Document(
                    page_content=current_chunk.strip(),
                    metadata={
                        "chunk_type": "adaptive",
                        "avg_complexity": avg_complexity,
                        "sentence_count": len(current_sentences),
                        "chunk_size": len(current_chunk)
                    }
                ))
                current_chunk = sentence
                current_sentences = [sentence]
            else:
                current_chunk = potential_chunk
                current_sentences.append(sentence)
        
        if current_chunk:
            chunks.append(Document(
                page_content=current_chunk.strip(),
                metadata={
                    "chunk_type": "adaptive",
                    "avg_complexity": avg_complexity,
                    "sentence_count": len(current_sentences),
                    "chunk_size": len(current_chunk)
                }
            ))
        
        return chunks
    
    def process_documents_with_multiple_strategies(self, documents: List[Document], strategies: List[str] = None) -> List[Document]:
        """
        Process documents using multiple chunking strategies and combine results.
        """
        if strategies is None:
            strategies = ["semantic", "topic", "hierarchical"]
        
        all_chunks = []
        
        for doc in documents:
            text = doc.page_content
            
            for strategy in strategies:
                if strategy == "semantic":
                    chunks = self.semantic_aware_chunking(text)
                elif strategy == "topic":
                    chunks = self.topic_based_chunking(text)
                elif strategy == "hierarchical":
                    chunks = self.hierarchical_chunking(text)
                elif strategy == "adaptive":
                    chunks = self.adaptive_chunking(text)
                elif strategy == "sliding":
                    chunks = self.sliding_window_chunking(text)
                else:
                    console.print(f"[yellow]Unknown strategy: {strategy}. Skipping.[/yellow]")
                    continue
                
                # Add source document metadata to chunks
                for chunk in chunks:
                    chunk.metadata.update(doc.metadata)
                    chunk.metadata["chunking_strategy"] = strategy
                
                all_chunks.extend(chunks)
        
        console.print(f"[green]Created {len(all_chunks)} chunks using {len(strategies)} strategies[/green]")
        return all_chunks
    
    def deduplicate_chunks(self, chunks: List[Document], similarity_threshold: float = 0.85) -> List[Document]:
        """
        Remove duplicate or highly similar chunks.
        """
        if len(chunks) <= 1:
            return chunks
        
        # Create TF-IDF vectors for all chunks
        chunk_texts = [chunk.page_content for chunk in chunks]
        
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(chunk_texts)
            
            # Calculate pairwise similarities
            similarities = cosine_similarity(tfidf_matrix)
            
            # Find duplicates
            to_remove = set()
            for i in range(len(chunks)):
                if i in to_remove:
                    continue
                for j in range(i + 1, len(chunks)):
                    if j in to_remove:
                        continue
                    if similarities[i][j] > similarity_threshold:
                        # Keep the longer chunk (more information)
                        if len(chunks[i].page_content) >= len(chunks[j].page_content):
                            to_remove.add(j)
                        else:
                            to_remove.add(i)
                            break
            
            # Return deduplicated chunks
            deduplicated = [chunk for i, chunk in enumerate(chunks) if i not in to_remove]
            
            if len(to_remove) > 0:
                console.print(f"[cyan]Removed {len(to_remove)} duplicate chunks[/cyan]")
            
            return deduplicated
            
        except Exception as e:
            console.print(f"[yellow]Deduplication failed: {e}. Returning original chunks.[/yellow]")
            return chunks
