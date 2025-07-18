#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demo script showcasing Enhanced RAG Features in Infomaid
"""

import os
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add the parent directory to the path to import infomaid modules
sys.path.append(str(Path(__file__).parent.parent))

try:
    from infomaid.enhanced_rag import EnhancedQueryProcessor
    from infomaid.enhanced_document_processor import EnhancedDocumentProcessor
    ENHANCED_AVAILABLE = True
except ImportError as e:
    ENHANCED_AVAILABLE = False
    print(f"Enhanced features not available: {e}")

console = Console()

def demo_enhanced_rag():
    """Demonstrate the enhanced RAG capabilities."""
    
    console.print(Panel.fit(
        "[bold blue]Enhanced RAG Features Demo[/bold blue]\n"
        "This demo showcases improved retrieval methods including:\n"
        "• TF-IDF similarity search\n"
        "• BM25 scoring\n"
        "• Hybrid vector + TF-IDF search\n"
        "• Query expansion\n"
        "• Advanced document chunking strategies",
        border_style="cyan"
    ))
    
    if not ENHANCED_AVAILABLE:
        console.print("[red]Enhanced RAG features are not available. Please install required dependencies:[/red]")
        console.print("poetry add scikit-learn numpy")
        return
    
    # Create a table showing retrieval methods
    table = Table(title="Available Retrieval Methods")
    table.add_column("Method", style="cyan", no_wrap=True)
    table.add_column("Description", style="magenta")
    table.add_column("Best For", style="green")
    
    table.add_row(
        "vector",
        "Standard semantic similarity using embeddings",
        "General semantic understanding"
    )
    table.add_row(
        "tfidf",
        "TF-IDF with cosine similarity",
        "Keyword-based retrieval, technical documents"
    )
    table.add_row(
        "bm25",
        "BM25 ranking algorithm",
        "Information retrieval, search engines"
    )
    table.add_row(
        "hybrid",
        "Combines vector and TF-IDF scores",
        "Best overall performance, balanced results"
    )
    
    console.print(table)
    
    # Demonstrate chunking strategies
    console.print("\n[bold cyan]Document Chunking Strategies:[/bold cyan]")
    
    chunking_table = Table(title="Enhanced Chunking Methods")
    chunking_table.add_column("Strategy", style="cyan")
    chunking_table.add_column("Description", style="magenta")
    chunking_table.add_column("Advantages", style="green")
    
    chunking_table.add_row(
        "Semantic",
        "Respects sentence boundaries and context",
        "Maintains meaning, better coherence"
    )
    chunking_table.add_row(
        "Topic-based",
        "Groups related content using clustering",
        "Topically coherent chunks"
    )
    chunking_table.add_row(
        "Hierarchical",
        "Follows document structure (paragraphs)",
        "Preserves document organization"
    )
    chunking_table.add_row(
        "Adaptive",
        "Adjusts size based on content complexity",
        "Optimized for content type"
    )
    
    console.print(chunking_table)
    
    # Example usage commands
    console.print("\n[bold cyan]Example Usage Commands:[/bold cyan]")
    
    examples = [
        "# Basic enhanced RAG with hybrid search",
        "poetry run infomaid --useowndata --enhancedrag --prompt 'Your question'",
        "",
        "# Use BM25 ranking for better keyword matching",
        "poetry run infomaid --useowndata --enhancedrag --retrievalmethod bm25 --prompt 'Find specific terms'",
        "",
        "# Use TF-IDF for technical documents",
        "poetry run infomaid --useowndata --enhancedrag --retrievalmethod tfidf --prompt 'Technical query'",
        "",
        "# Multiple results with hybrid approach",
        "poetry run infomaid --useowndata --enhancedrag --retrievalmethod hybrid --count 3 --prompt 'Compare approaches'"
    ]
    
    for example in examples:
        if example.startswith("#"):
            console.print(f"[green]{example}[/green]")
        elif example == "":
            console.print()
        else:
            console.print(f"[yellow]{example}[/yellow]")
    
    console.print("\n[bold cyan]Key Improvements:[/bold cyan]")
    improvements = [
        "🎯 Better relevance with multiple scoring methods",
        "📊 TF-IDF for keyword-heavy documents", 
        "🔍 BM25 for search engine-like ranking",
        "🤝 Hybrid approach combines multiple methods",
        "📝 Semantic chunking preserves context",
        "🧠 Topic-based chunking for coherent results",
        "🔄 Query expansion for better coverage",
        "🚫 Deduplication reduces redundant results"
    ]
    
    for improvement in improvements:
        console.print(f"  {improvement}")

def demo_chunking_comparison():
    """Demonstrate different chunking strategies with sample text."""
    
    if not ENHANCED_AVAILABLE:
        console.print("[red]Enhanced features not available[/red]")
        return
    
    console.print("\n[bold cyan]Chunking Strategy Comparison[/bold cyan]")
    
    # Sample text for demonstration
    sample_text = """
    Artificial Intelligence has revolutionized many aspects of modern technology. Machine learning algorithms 
    can now process vast amounts of data and identify patterns that were previously impossible to detect.
    
    Natural Language Processing is a particularly exciting field within AI. It enables computers to understand 
    and generate human language, making interactions between humans and machines more natural and intuitive.
    
    Deep learning models, such as neural networks, have shown remarkable success in tasks like image recognition, 
    speech processing, and language translation. These models can learn complex representations from raw data.
    
    The applications of AI are endless: from autonomous vehicles to medical diagnosis, from financial trading 
    to personalized recommendations. As AI continues to evolve, we can expect even more innovative applications.
    """
    
    processor = EnhancedDocumentProcessor()
    
    # Test different chunking strategies
    strategies = [
        ("semantic", "Semantic Aware"),
        ("topic", "Topic Based"), 
        ("hierarchical", "Hierarchical"),
        ("adaptive", "Adaptive")
    ]
    
    for strategy_name, display_name in strategies:
        console.print(f"\n[bold green]{display_name} Chunking:[/bold green]")
        
        if strategy_name == "semantic":
            chunks = processor.semantic_aware_chunking(sample_text)
        elif strategy_name == "topic":
            chunks = processor.topic_based_chunking(sample_text)
        elif strategy_name == "hierarchical":
            chunks = processor.hierarchical_chunking(sample_text)
        elif strategy_name == "adaptive":
            chunks = processor.adaptive_chunking(sample_text)
        
        for i, chunk in enumerate(chunks):
            console.print(f"  Chunk {i+1}: {len(chunk.page_content)} chars")
            console.print(f"  Preview: {chunk.page_content[:100]}...")
            console.print(f"  Metadata: {chunk.metadata}")
            console.print()

if __name__ == "__main__":
    demo_enhanced_rag()
    
    # Uncomment to see chunking comparison
    # demo_chunking_comparison()
