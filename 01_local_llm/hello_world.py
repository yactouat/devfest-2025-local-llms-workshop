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
    # For thinking models, enable reasoning to parse "<think>" blocks
    print(f"Connecting to Ollama with model: {model_name}...")
    llm = ChatOllama(
        model=model_name,
        temperature=0.6,  # Some randomness for more natural responses
        reasoning=True if args.thinking else False,  # Enable reasoning for thinking models
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

    # For thinking models, display the reasoning trace first
    if args.thinking:
        # The Thinking Trace (Reasoning)
        # This is where the model's hidden "thought process" is stored
        reasoning = response.additional_kwargs.get("reasoning_content")
        if reasoning:
            print("### Thinking Trace ###")
            print(reasoning)
            print("\n" + "="*60 + "\n")
        else:
            print("No reasoning trace found (Model might not have generated one).")
            print()

    # The Final Answer
    if args.thinking:
        print("### Final Answer ###")
    print(response.content)
    
    print("-" * 60)
    
    print()

    print("Observation:")
    print("The model doesn't have information about DevFest Corp since it's")
    print("not in its training data. This is where RAG comes in handy!")

    if args.thinking:
        print()
        print("Thinking Model Note:")
        print("The qwen3:8b model is a thinking model that exposes its reasoning process.")
        print("With reasoning=True, LangChain parses '<think>' blocks and moves them")
        print("to response.additional_kwargs['reasoning_content']. This helps understand")
        print("how the model arrives at its conclusions.")
        print()
        print("Try comparing with the non-thinking model:")
        print("  • Without --thinking: python3 01_local_llm/hello_world.py")
        print("  • With --thinking: python3 01_local_llm/hello_world.py --thinking")

    print()
    print("Next step: We'll add a knowledge base to help the model answer")
    print("questions about information it wasn't trained on.")


if __name__ == "__main__":
    main()
