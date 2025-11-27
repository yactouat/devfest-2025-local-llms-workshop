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
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_available_model

from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

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

    # Choose model based on flag and availability
    model_name = get_available_model(prefer_thinking=args.thinking)

    # ! if you need to use a Cloud Model uncomment this line and comment the line above
    # model_name = get_available_model(prefer_thinking=args.thinking, use_cloud=True)

    # Initialize connection to local Ollama
    # Default endpoint is http://localhost:11434
    # For thinking models, enable reasoning to parse "<think>" blocks
    print(f"Connecting to Ollama with model: {model_name}...")
    llm = ChatOllama(
        model=model_name,
        temperature=0.0,  # Some randomness for more natural responses
        reasoning=True if args.thinking else False,  # Enable reasoning for thinking models
    )

    # ! same goes for the Google Cloud Model
    # llm = ChatGoogleGenerativeAI(
    #     model=model_name,
    #     temperature=0,
    # )

    print(f"✓ Connected to {model_name}")
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
