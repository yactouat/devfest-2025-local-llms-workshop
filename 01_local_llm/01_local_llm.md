# Step 1: Local LLM Hello World

## Goal
Verify that Ollama is running and can successfully respond to basic queries. This step demonstrates the fundamental capability of a local LLM without any additional context or knowledge augmentation.

## Prerequisites
Before starting this step, ensure you have:
- Ollama installed and running
- Python virtual environment activated
- Required dependencies installed (`pip install -r requirements.txt`)
- A base model pulled (e.g., `llama3.1:latest` or `qwen3:8b`)

## What You'll Learn
- How to connect to Ollama from Python using LangChain
- How to invoke a local LLM with a simple query
- The limitations of LLMs without external knowledge

## The Scenario

We'll ask the model: **"Who is the CEO of DevFest Corp?"**

Since DevFest Corp is a fictional company (or at minimum, not in the model's training data), the model will not be able to provide a factual answer. This demonstrates a key limitation: LLMs can only answer questions based on their training data.

In later steps, we'll use RAG (Retrieval-Augmented Generation) to provide the model with external knowledge, enabling it to answer such questions accurately.

## Running the Code

1. Make sure Ollama is running:
   ```bash
   ollama serve
   ```
   (In many cases, Ollama runs as a service and this step is not necessary)

2. Run the Python script:
   ```bash
   python3 01_local_llm/hello_world.py
   ```

## Expected Behavior

The script will:
1. Connect to your local Ollama instance
2. Send the query to the model
3. Display the response

Since the model doesn't know about DevFest Corp, it will likely:
- State that it doesn't have information about this company
- Make a general statement about not having access to current information
- Or possibly hallucinate an answer (make something up)

This demonstrates the need for RAG systems, which we'll explore in the next steps!

## Code Explanation

The script uses:
- **`ChatOllama`** from LangChain to interface with the local Ollama service
- A simple invocation pattern to send a message and receive a response
- Default Ollama endpoint (`http://localhost:11434`)

## Thinking Models

The script also demonstrates the use of `qwen3:8b`, which is a **thinking model**. Thinking models have a unique capability: they can show their reasoning process before providing an answer.

### What is a Thinking Model?

Thinking models expose their internal reasoning chain, similar to how a human might "think out loud" while solving a problem. This is particularly useful for:
- Complex reasoning tasks
- Mathematical problems
- Debugging logical issues
- Understanding how the model arrived at its conclusion

### How to Use Thinking Models

When using `qwen3:8b` with the `--verbose` flag, you can see the model's thinking process:

```bash
python3 01_local_llm/hello_world.py --thinking
```

The model's response includes:
- **Thinking process**: The model's internal reasoning (visible when using verbose mode)
- **Final answer**: The actual response to your query

This transparency helps you understand the model's decision-making process and can be valuable for educational purposes or when debugging complex queries.

## Next Steps

Once you've verified that Ollama is working and observed the model's limitation with unknown information, proceed to **Step 2: RAG with LCEL** where we'll add a knowledge base to help the model answer questions it couldn't answer before.
