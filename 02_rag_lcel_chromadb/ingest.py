#!/usr/bin/env python3
"""
Step 2: RAG with LCEL - Ingestion Script

This script demonstrates how to:
1. Load a markdown knowledge base
2. Split it into semantic chunks based on content similarity
3. Generate embeddings using Ollama
4. Store the vectors in SQLite for retrieval

This is the "ingestion phase" of a RAG system, where we prepare our
knowledge base for efficient semantic search.

RAG (Retrieval-Augmented Generation) Overview:
- RAG allows LLMs to access external knowledge without retraining
- Two phases: Ingestion (this script) and Query (query.py)
- Ingestion: Convert documents → chunks → embeddings → vector database
- Query: User question → retrieve relevant chunks → generate answer
"""

from pathlib import Path
import sys

from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


def main():
    print("=" * 60)
    print("Step 2: RAG with LCEL - Knowledge Base Ingestion")
    print("=" * 60)
    print()

    # ========================================
    # STEP 0: Setup - Define file paths
    # ========================================
    script_dir = Path(__file__).parent
    knowledge_base_path = script_dir / "knowledge_base.md"
    # ChromaDB directory is stored at the root of the repository for sharing across demos
    chroma_dir = script_dir.parent / "chroma_db"

    # Validate that the knowledge base file exists
    if not knowledge_base_path.exists():
        print(f"❌ Error: Knowledge base not found at {knowledge_base_path}")
        print("Please ensure knowledge_base.md exists in the same directory.")
        return

    print(f"📄 Loading knowledge base from: {knowledge_base_path}")

    # ========================================
    # STEP 1: Load the Document
    # ========================================
    # TextLoader reads the markdown file and converts it to a LangChain Document object
    # Documents have:
    # - page_content: The actual text content
    # - metadata: Additional info (file path, source, etc.)
    loader = TextLoader(str(knowledge_base_path))
    documents = loader.load()
    print(f"✓ Loaded {len(documents)} document(s)")
    print()

    # ========================================
    # STEP 2: Initialize Embeddings Model
    # ========================================
    # Embeddings convert text into numerical vectors (arrays of numbers)
    # Similar text = similar vectors, which enables semantic search
    #
    # Why "nomic-embed-text"?
    # - Optimized for text embedding tasks
    # - Smaller and faster than general-purpose LLMs
    # - Produces high-quality 768-dimensional vectors
    print("🔗 Initializing embeddings for semantic chunking...")
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",  # Specialized embedding model from Ollama
    )
    print("✓ Embedding model initialized")
    print()

    # ========================================
    # STEP 3: Split Document into Semantic Chunks
    # ========================================
    # Why chunk? Large documents don't fit in LLM context windows, and retrieval
    # is more precise with smaller, focused pieces of text.
    #
    # SemanticChunker is SMART:
    # - Unlike simple character/token splitters, it understands meaning
    # - Groups semantically related sentences together
    # - Splits where topic changes occur
    #
    # How it works:
    # 1. Embeds each sentence
    # 2. Calculates similarity between consecutive sentences
    # 3. Splits where similarity drops below threshold
    print("✂️  Splitting documents into semantic chunks...")
    text_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",  # Use percentile-based splitting
        breakpoint_threshold_amount=50,  # Split at 50th percentile of similarity scores
        # Lower = more splits (more granular chunks)
        # Higher = fewer splits (larger chunks)
        # Default is 95 (very few splits)
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✓ Created {len(chunks)} chunks")
    print()

    # Show a preview of the first chunk to verify chunking worked correctly
    print("Sample chunk:")
    print("-" * 60)
    print(chunks[0].page_content[:200] + "...")
    print("-" * 60)
    print()

    # ========================================
    # STEP 4: Create Vector Store with ChromaDB
    # ========================================
    # A vector store (or vector database) stores embeddings and enables fast
    # similarity search. ChromaDB is a modern vector database designed for AI applications.
    #
    # What happens here:
    # 1. Each chunk is converted to a vector embedding
    # 2. Vectors are stored in ChromaDB
    # 3. An index is created for fast similarity search (k-nearest neighbors)
    print(f"💾 Creating vector store with ChromaDB...")
    print(f"   Database directory: {chroma_dir}")

    # Clean slate: Remove old database if it exists to avoid conflicts
    if chroma_dir.exists():
        print(f"   (Removing existing database)")
        import shutil
        shutil.rmtree(chroma_dir)

    # Create the vector store using Chroma
    # This is a one-line operation that does a lot:
    # - Creates the ChromaDB database
    # - Generates embeddings for all chunks using the embedding model
    # - Stores vectors with metadata in a collection
    # - Creates vector search indexes for fast retrieval
    vectorstore = Chroma.from_documents(
        documents=chunks,  # The semantic chunks we created
        embedding=embeddings,  # The embedding model to use
        collection_name="devfest_knowledge",  # Name of the collection in ChromaDB
        persist_directory=str(chroma_dir),  # Directory for ChromaDB storage
    )

    print(f"✓ Vector store created successfully")
    print()

    # ========================================
    # STEP 5: Test the Vector Store
    # ========================================
    # Verify that semantic search works by running a test query
    # This will:
    # 1. Convert the query to a vector embedding
    # 2. Find the most similar chunk vector in the database
    # 3. Return the corresponding text
    print("🔍 Verifying ingestion with a test query...")
    test_query = "Who is the CEO?"
    results = vectorstore.similarity_search(test_query, k=1)  # Get top 1 result

    if results:
        print(f"✓ Test query successful!")
        print(f"   Query: '{test_query}'")
        print(f"   Top result preview:")
        print("-" * 60)
        print(results[0].page_content[:200] + "...")
        print("-" * 60)
    else:
        print("⚠️  Warning: Test query returned no results")

    print()
    print("=" * 60)
    print("Ingestion Complete!")
    print("=" * 60)
    print()
    print("Summary:")
    print(f"  • Loaded: {len(documents)} document(s)")
    print(f"  • Split into: {len(chunks)} chunks")
    print(f"  • Stored in: {chroma_dir}")
    print(f"  • Collection: devfest_knowledge")
    print()
    print("Next step: Run query.py to ask questions!")


if __name__ == "__main__":
    main()
