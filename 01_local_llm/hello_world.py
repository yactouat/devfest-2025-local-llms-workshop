#!/usr/bin/env python3
"""
Step 1: Local LLM Hello World

This script demonstrates a basic interaction with a local LLM through Ollama.
It asks a question that the model cannot answer from its training data,
illustrating the need for RAG (Retrieval-Augmented Generation).

It also demonstrates the use of thinking models (qwen3:8b) which can show
their reasoning process.
"""

import argparse
from langchain_ollama import ChatOllama


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Local LLM Hello World Demo")
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Use qwen3:8b thinking model to show reasoning process"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Step 1: Local LLM Hello World")
    if args.thinking:
        print("(Using Thinking Model: qwen3:8b)")
    print("=" * 60)
    print()

    # Choose model based on flag
    model_name = "qwen3:8b" if args.thinking else "llama3.1:latest"

    # Initialize connection to local Ollama
    # Default endpoint is http://localhost:11434
    print(f"Connecting to Ollama with model: {model_name}...")
    llm = ChatOllama(
        model=model_name,
        temperature=0,  # Deterministic responses
    )
    print("✓ Connected to Ollama")
    print()

    # The question we'll ask
    question = "Who is the CEO of DevFest Corp?"

    print(f"Question: {question}")
    print()
    print("Asking the model...")
    print("-" * 60)

    # Invoke the model
    response = llm.invoke(question)

    # Display the response
    # For thinking models, the response may include reasoning
    if args.thinking:
        print("\n[THINKING MODEL RESPONSE]")
        print("\nThe thinking model processes the query and may show reasoning.")
        print("Note: The model's internal thinking is embedded in the response.\n")

    print(response.content)
    print("-" * 60)
    print()

    print("Observation:")
    print("The model doesn't have information about DevFest Corp since it's")
    print("not in its training data. This is where RAG comes in handy!")

    if args.thinking:
        print()
        print("Thinking Model Note:")
        print("The qwen3:8b model is a thinking model that can expose its reasoning.")
        print("This helps understand how the model arrives at its conclusions.")

    print()
    print("Next step: We'll add a knowledge base to help the model answer")
    print("questions about information it wasn't trained on.")


if __name__ == "__main__":
    main()
