#!/usr/bin/env python3
"""
Step 2: RAG with LCEL - Query Script

This script demonstrates:
1. How to retrieve relevant context from a vector store
2. How to build RAG chains using LCEL (LangChain Expression Language)
3. The power of the pipe operator (|) for chaining components
4. Sequential chains for deterministic pipelines

This is the "query phase" of a RAG system, where we answer questions by:
1. Converting the question to a vector embedding
2. Finding similar chunks in our vector store
3. Using those chunks as context for the LLM to generate an answer

LCEL (LangChain Expression Language):
- A declarative way to compose LangChain components
- Uses the pipe operator (|) to chain operations
- Automatically handles data flow, parallelization, and error handling
- Makes complex RAG pipelines readable and maintainable
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_available_model

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def format_docs(docs):
    """
    Format retrieved documents into a single context string.

    This helper function takes a list of Document objects from the vector store
    and combines their content into a single string with double newlines between them.
    This formatted string is then injected into the prompt template.

    Args:
        docs: List of Document objects from similarity search

    Returns:
        A single string with all document contents joined by double newlines
    """
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    # ========================================
    # STEP 0: Parse Command-Line Arguments
    # ========================================
    # Allow users to run in single-question or interactive mode
    parser = argparse.ArgumentParser(description="RAG Query Demo with LCEL")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode (ask multiple questions)"
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Who is the CEO of DevFest Corp?",
        help="Question to ask (default: 'Who is the CEO of DevFest Corp?')"
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Use qwen3:8b thinking model to show reasoning process"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Step 2: RAG with LCEL - Query Demo")
    if args.thinking:
        print("(Using Thinking Model: qwen3:8b)")
    print("=" * 60)
    print()

    # Define paths
    script_dir = Path(__file__).parent
    # ChromaDB directory is stored at the root of the repository for sharing across demos
    chroma_dir = script_dir.parent / "chroma_db"

    # Validate that the database exists (created by ingest.py)
    if not chroma_dir.exists():
        print(f"❌ Error: ChromaDB not found at {chroma_dir}")
        print("Please run ingest.py first to create the knowledge base.")
        return

    print(f"📚 Loading vector store from: {chroma_dir}")

    # ========================================
    # STEP 1: Initialize Embeddings Model
    # ========================================
    # CRITICAL: Must use the SAME embedding model as during ingestion
    # Different models produce incompatible vector spaces
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # ========================================
    # STEP 2: Load the Vector Store
    # ========================================
    # Load the existing ChromaDB vector store
    # ChromaDB automatically handles the connection and indexing
    vectorstore = Chroma(
        collection_name="devfest_knowledge",  # Same collection name from ingestion
        embedding_function=embeddings,         # Same embedding model from ingestion
        persist_directory=str(chroma_dir),     # Directory where ChromaDB is stored
    )
    print("✓ Vector store loaded")
    print()

    # ========================================
    # STEP 3: Initialize the LLM
    # ========================================
    # This is the language model that will generate the final answer
    # based on the retrieved context
    # Choose model based on --thinking flag and availability
    model_name = get_available_model(prefer_thinking=args.thinking)

    print(f"🔗 Connecting to Ollama LLM ({model_name})...")
    llm = ChatOllama(
        model=model_name,
        temperature=0,  # 0 = deterministic, 1 = creative
                        # For factual Q&A, we want deterministic responses
        reasoning=True if args.thinking else False,  # Enable reasoning for thinking models
    )
    print("✓ LLM initialized")
    print()

    # ========================================
    # STEP 4: Create the RAG Prompt Template
    # ========================================
    # The prompt template defines how we'll format the retrieved context
    # and user question for the LLM
    #
    # Key elements:
    # - System instruction (You are a helpful assistant...)
    # - Context placeholder {context} - filled with retrieved chunks
    # - Question placeholder {question} - filled with user's question
    # - Clear instruction to admit when answer isn't in context
    template = """You are a helpful assistant answering questions about DevFest Corp.
Use the following context to answer the question. If you cannot answer based on the context, say so.

Context:
{context}

Question: {question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    print("✓ Prompt template created")
    print()

    # ========================================
    # STEP 5: Build the LCEL Chain
    # ========================================
    # This is where the magic happens! LCEL allows us to compose
    # a complete RAG pipeline in a declarative, readable way.
    #
    # The chain will:
    # 1. Take a question as input
    # 2. Retrieve relevant chunks (retriever)
    # 3. Format them (format_docs)
    # 4. Fill the prompt template (prompt)
    # 5. Send to LLM (llm)
    # 6. Parse the response (StrOutputParser)
    print("⛓️  Building LCEL chain...")
    print()
    print("Chain structure:")
    print("  retriever | format_docs → context")
    print("  RunnablePassthrough() → question")
    print("  ↓")
    print("  prompt | model | output_parser")
    print()

    # Create a retriever from the vector store
    # A retriever is a convenience interface that handles:
    # - Converting query text to embeddings
    # - Finding k most similar vectors
    # - Returning the corresponding documents
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}  # Retrieve top 5 most similar chunks
                                 # Increased from 3 to ensure CEO info is included
    )

    # Build the LCEL chain using the pipe operator (|)
    # This is the heart of LCEL - clean, declarative composition
    #
    # The dictionary syntax runs operations in PARALLEL:
    # - "context": retriever | format_docs  → retrieve & format docs
    # - "question": RunnablePassthrough()   → pass question unchanged
    #
    # Then the pipe operator chains operations SEQUENTIALLY:
    # dict → prompt → llm → parser
    chain = (
        {
            "context": retriever | format_docs,  # Retrieve docs and format them
            "question": RunnablePassthrough()     # Pass question through unchanged
        }
        | prompt                                  # Fill prompt template with context & question
        | llm                                     # Send formatted prompt to LLM
        | StrOutputParser()                       # Parse LLM output to plain string
    )

    print("✓ LCEL chain built successfully")
    print()

    # ========================================
    # STEP 6: Define Question Processing Function
    # ========================================
    # This function handles a single question through the RAG pipeline
    def ask_question(question: str):
        """
        Process a question through the RAG pipeline.

        This function:
        1. Shows what context chunks are retrieved (for transparency)
        2. Executes the LCEL chain to generate an answer
        3. Displays the answer

        Args:
            question: The user's question as a string
        """
        print("=" * 60)
        print(f"Question: {question}")
        print("=" * 60)
        print()

        # For educational purposes, let's show what context is retrieved
        # This helps users understand HOW the RAG system works
        print("🔍 Retrieving relevant context...")
        docs = retriever.invoke(question)  # Retrieve similar chunks
        print(f"   Retrieved {len(docs)} chunks")
        print()
        print("Retrieved context preview:")
        print("-" * 60)
        for i, doc in enumerate(docs, 1):
            preview = doc.page_content[:150].replace("\n", " ")
            print(f"{i}. {preview}...")
        print("-" * 60)
        print()

        # Now execute the full LCEL chain
        # This will:
        # 1. Retrieve docs (again, but cached in production systems)
        # 2. Format them into context string
        # 3. Fill prompt template
        # 4. Send to LLM
        # 5. Parse response
        print("💭 Generating answer using LCEL chain...")
        print()

        # For thinking models, we need to get the full response to access reasoning
        if args.thinking:
            # Build a chain without the output parser to get full response
            chain_with_reasoning = (
                {
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough()
                }
                | prompt
                | llm
            )
            response = chain_with_reasoning.invoke(question)

            # Display the reasoning trace
            reasoning = response.additional_kwargs.get("reasoning_content")
            if reasoning:
                print("### Thinking Trace ###")
                print(reasoning)
                print("\n" + "="*60 + "\n")

            # Display the final answer
            print("### Final Answer ###")
            print(response.content)
        else:
            # The magic happens here - LCEL handles all the data flow automatically
            # No need to manually pass data between components!
            answer = chain.invoke(question)

            print("Answer:")
            print("-" * 60)
            print(answer)
            print("-" * 60)
        print()

    # ========================================
    # STEP 7: Run Query Mode (Interactive or Single)
    # ========================================
    if args.interactive:
        # Interactive mode: Keep asking questions until user quits
        print("🎯 Interactive mode - Type 'quit' or 'exit' to stop")
        print()

        while True:
            try:
                question = input("Your question: ").strip()
                if question.lower() in ["quit", "exit", "q"]:
                    print("\nGoodbye!")
                    break
                if not question:
                    continue

                ask_question(question)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print()

    else:
        # Single question mode (default)
        ask_question(args.question)

    # ========================================
    # Educational Explanation of LCEL
    # ========================================
    print()
    print("=" * 60)
    print("LCEL Explanation")
    print("=" * 60)
    print()
    print("What just happened?")
    print()
    print("1. PIPE OPERATOR (|):")
    print("   The | operator chains components together, passing")
    print("   output from left to right automatically.")
    print("   Example: retriever | format_docs")
    print("   → Retrieves docs, then formats them")
    print()
    print("2. PARALLEL RETRIEVAL:")
    print("   The dictionary syntax runs retriever and passthrough")
    print("   in parallel, then combines results for the prompt.")
    print("   Example: {'context': retriever, 'question': passthrough}")
    print("   → Both run at the same time, results merged into dict")
    print()
    print("3. SEQUENTIAL FLOW:")
    print("   prompt | llm | parser is a sequential chain where")
    print("   each step depends on the previous one.")
    print("   → Can't send to LLM before formatting prompt")
    print("   → Can't parse before LLM responds")
    print()
    print("4. TYPE SAFETY:")
    print("   LCEL ensures each component's output matches the")
    print("   next component's expected input.")
    print("   → Prevents runtime errors from mismatched data types")
    print()
    print("Key benefits of LCEL:")
    print("  ✓ Declarative (expresses what, not how)")
    print("  ✓ Composable (mix and match components)")
    print("  ✓ Optimized (automatic parallelization)")
    print("  ✓ Readable (clear data flow)")
    print("  ✓ Streamable (supports streaming responses)")
    print("  ✓ Async-ready (works with async/await)")
    print()
    print("Compare to traditional code:")
    print("  Traditional: docs = retrieve(q); ctx = format(docs); ...")
    print("  LCEL: chain = retriever | format | prompt | llm | parser")
    print()


if __name__ == "__main__":
    main()
