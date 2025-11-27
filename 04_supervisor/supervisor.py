#!/usr/bin/env python3
"""
Step 4: The Supervisor Pattern

This script demonstrates:
1. How to build a multi-agent system with centralized control
2. How a Supervisor delegates tasks to specialized worker agents
3. How to implement conditional routing based on agent decisions
4. How to coordinate multiple agents with different capabilities

Supervisor Pattern:
- A central Supervisor agent analyzes tasks and delegates to workers
- Workers are specialized agents with specific capabilities
- Workers return results to the Supervisor for final synthesis
- Supervisor decides when to delegate more or provide final answer

Key Concepts:
- Centralized Control: Supervisor coordinates all worker agents
- Task Delegation: Supervisor routes work to appropriate specialists
- Worker Specialization: Each agent has a specific domain of expertise
- Return-to-Supervisor: Workers always report back to central controller
"""

import argparse
from pathlib import Path
import sqlite3
import sys
from typing import Annotated, Literal
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# IMPORTANT: Use pysqlite3 instead of built-in sqlite3
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import sqlite_vss
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import SQLiteVSS
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, ConfigDict

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_available_model, get_google_model, check_google_api_key


# ========================================
# STEP 1: Define Supervisor State
# ========================================
class SupervisorState(BaseModel):
    """
    State for the supervisor multi-agent system.

    Tracks:
    - messages: Full conversation history
    - next_agent: Which agent should handle the next step
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: Annotated[list, add_messages] = Field(
        default_factory=list,
        description="Conversation history including all agent interactions"
    )
    next_agent: str = Field(
        default="supervisor",
        description="The next agent to execute (supervisor, researcher, writer, fact_checker, or FINISH)"
    )


# ========================================
# STEP 2: Global Resources
# ========================================
llm = None
vectorstore = None


# ========================================
# STEP 3: Worker Agent Functions
# ========================================
def researcher_agent(state: SupervisorState) -> dict:
    """
    Researcher Agent - Retrieves information from the knowledge base.

    This agent specializes in:
    - Querying the DevFest Corp knowledge base (SQLite vector store)
    - Finding relevant company information
    - Retrieving policy documents and organizational details

    Args:
        state: Current supervisor state

    Returns:
        State update with research findings
    """
    print("\n🔍 RESEARCHER AGENT working...")

    # Get the user's question from messages
    user_question = None
    for msg in state.messages:
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break

    if not user_question:
        return {
            "messages": [AIMessage(content="[Researcher] No question found to research.")],
            "next_agent": "supervisor"
        }

    # Setup database connection
    script_dir = Path(__file__).parent
    db_path = script_dir.parent / "devfest.db"

    if not db_path.exists():
        return {
            "messages": [AIMessage(
                content="[Researcher] Error: Knowledge base not found. Please run 02_rag_lcel/ingest.py first."
            )],
            "next_agent": "supervisor"
        }

    # Initialize embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Load vector store
    connection = sqlite3.connect(str(db_path), check_same_thread=False)
    connection.enable_load_extension(True)
    connection.row_factory = sqlite3.Row
    sqlite_vss.load(connection)
    connection.enable_load_extension(False)

    vectorstore = SQLiteVSS(
        table="devfest_knowledge",
        embedding=embeddings,
        connection=connection,
    )

    # Search for relevant documents
    results = vectorstore.similarity_search(user_question, k=3)

    if not results:
        research_findings = "[Researcher] No relevant information found in the knowledge base."
    else:
        context = "\n\n".join([doc.page_content for doc in results])
        research_findings = f"[Researcher] I found the following information:\n\n{context}"

    print(f"   ✓ Retrieved {len(results)} relevant documents")

    return {
        "messages": [AIMessage(content=research_findings)],
        "next_agent": "supervisor"
    }


def writer_agent(state: SupervisorState) -> dict:
    """
    Writer Agent - Generates content based on available information.

    This agent specializes in:
    - Creating well-formatted responses
    - Synthesizing information into readable content
    - Crafting professional communications

    Args:
        state: Current supervisor state

    Returns:
        State update with generated content
    """
    print("\n✍️  WRITER AGENT working...")

    # Gather context from previous messages
    context = []
    user_question = None

    for msg in state.messages:
        if isinstance(msg, HumanMessage):
            user_question = msg.content
        elif isinstance(msg, AIMessage) and "[Researcher]" in msg.content:
            context.append(msg.content)

    if not user_question:
        return {
            "messages": [AIMessage(content="[Writer] No question to write about.")],
            "next_agent": "supervisor"
        }

    # Prepare messages for writer
    system_prompt = "You are a professional writer for DevFest Corp. Based on the research provided, write a clear, concise answer to the user's question."
    
    research_context = chr(10).join(context) if context else "No research context provided."
    
    user_prompt = f"""User Question: {user_question}

Research Context:
{research_context}

Write a professional, helpful response:"""

    # Generate content using LLM
    # Use proper message format: system message + user message
    # This ensures both thinking and non-thinking models work correctly
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    written_content = f"[Writer] {response.content}"

    print("   ✓ Content generated")

    return {
        "messages": [AIMessage(content=written_content)],
        "next_agent": "supervisor"
    }


def fact_checker_agent(state: SupervisorState) -> dict:
    """
    Fact Checker Agent - Validates information and checks compliance.

    This agent specializes in:
    - Verifying factual accuracy
    - Checking policy compliance
    - Identifying potential issues or inconsistencies
    - Ensuring information meets quality standards

    NOTE: This is a demonstration agent with mocked output.
    In a production system, this would:
    - Query external fact-checking APIs
    - Verify against trusted data sources
    - Run compliance rule engines
    - Check citations and references

    Args:
        state: Current supervisor state

    Returns:
        State update with fact-checking results
    """
    print("\n✅ FACT CHECKER AGENT working...")

    # Gather content to fact-check
    content_to_check = []

    for msg in state.messages:
        if isinstance(msg, AIMessage):
            if "[Researcher]" in msg.content or "[Writer]" in msg.content:
                content_to_check.append(msg.content)

    if not content_to_check:
        fact_check_result = "[Fact Checker] No content to verify."
    else:
        # DEMO OUTPUT: In a real system, this would perform actual verification
        fact_check_result = """[Fact Checker] Verification Report:

✓ Information Sources: Verified against knowledge base
✓ Policy Compliance: All statements align with DevFest Corp policies
✓ Factual Accuracy: Cross-referenced data points are consistent
✓ Quality Check: Content meets professional standards
✓ Completeness: Response addresses the user's question

Status: APPROVED ✓

Note: This is a demonstration of the fact-checking agent. In production,
this would involve real verification against external sources, compliance
databases, and automated fact-checking services."""

    print("   ✓ Verification completed")

    return {
        "messages": [AIMessage(content=fact_check_result)],
        "next_agent": "supervisor"
    }


# ========================================
# STEP 4: Supervisor Agent
# ========================================
def supervisor_agent(state: SupervisorState) -> dict:
    """
    Supervisor Agent - Orchestrates the multi-agent system.

    The supervisor:
    1. Analyzes the user's request
    2. Determines which worker agents are needed
    3. Delegates tasks to appropriate specialists
    4. Synthesizes results into a final answer
    5. Decides when the task is complete

    Decision Logic:
    - If question needs information: route to researcher
    - If content needs writing: route to writer
    - If answer needs verification: route to fact_checker
    - If task is complete: finish (FINISH)

    Args:
        state: Current supervisor state

    Returns:
        State update with supervisor's decision
    """
    print("\n👔 SUPERVISOR deciding next action...")

    # Analyze conversation state
    has_researcher_output = any("[Researcher]" in msg.content for msg in state.messages if isinstance(msg, AIMessage))
    has_writer_output = any("[Writer]" in msg.content for msg in state.messages if isinstance(msg, AIMessage))
    has_fact_checker_output = any("[Fact Checker]" in msg.content for msg in state.messages if isinstance(msg, AIMessage))

    # Decision tree
    if not has_researcher_output:
        # First step: gather information
        print("   → Delegating to RESEARCHER to gather information")
        return {"next_agent": "researcher"}

    elif has_researcher_output and not has_writer_output:
        # Second step: write the response
        print("   → Delegating to WRITER to craft response")
        return {"next_agent": "writer"}

    elif has_writer_output and not has_fact_checker_output:
        # Third step: verify the response
        print("   → Delegating to FACT CHECKER to verify accuracy")
        return {"next_agent": "fact_checker"}

    else:
        # All workers have completed their tasks
        print("   → All workers completed. Synthesizing final answer...")

        # Extract the writer's content for final response
        writer_content = None
        for msg in state.messages:
            if isinstance(msg, AIMessage) and "[Writer]" in msg.content:
                # Remove the [Writer] prefix for clean output
                writer_content = msg.content.replace("[Writer] ", "")
                break

        final_answer = f"""Based on the work of our specialized team:

{writer_content if writer_content else "Unable to generate response."}

---
(This answer was produced through coordination of Researcher, Writer, and Fact Checker agents)"""

        return {
            "messages": [AIMessage(content=final_answer)],
            "next_agent": "FINISH"
        }


# ========================================
# STEP 5: Routing Function
# ========================================
def route_to_agent(state: SupervisorState) -> Literal["researcher", "writer", "fact_checker", "supervisor", "end"]:
    """
    Conditional edge function that routes to the next agent.

    This function reads the 'next_agent' field from the state
    and returns the appropriate node name for the graph to route to.

    Args:
        state: Current supervisor state

    Returns:
        Name of the next node to execute
    """
    next_agent = state.next_agent

    if next_agent == "FINISH":
        return "end"
    elif next_agent == "researcher":
        return "researcher"
    elif next_agent == "writer":
        return "writer"
    elif next_agent == "fact_checker":
        return "fact_checker"
    else:
        return "supervisor"


# ========================================
# STEP 6: Build the Supervisor Graph
# ========================================
def create_supervisor_graph():
    """
    Creates the LangGraph StateGraph for the supervisor pattern.

    Graph Structure:
        [START] → supervisor → {route_to_agent}
                                ├─ researcher → supervisor (loop)
                                ├─ writer → supervisor (loop)
                                ├─ fact_checker → supervisor (loop)
                                └─ end → [END]

    The supervisor controls the flow, delegating to workers as needed.
    Workers always return to the supervisor.

    Returns:
        Compiled graph ready for execution
    """
    workflow = StateGraph(SupervisorState)

    # Add all nodes
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("writer", writer_agent)
    workflow.add_node("fact_checker", fact_checker_agent)

    # Set entry point
    workflow.set_entry_point("supervisor")

    # Add conditional edges from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "researcher": "researcher",
            "writer": "writer",
            "fact_checker": "fact_checker",
            "supervisor": "supervisor",
            "end": END,
        }
    )

    # Workers return to supervisor
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("writer", "supervisor")
    workflow.add_edge("fact_checker", "supervisor")

    return workflow.compile()


# ========================================
# STEP 7: Main Function
# ========================================
def main():
    global llm

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Supervisor Multi-Agent System")
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
        help="Use qwen3:8b thinking model for supervisor decisions"
    )
    parser.add_argument(
        "--google",
        action="store_true",
        help="Use Google AI instead of local Ollama (requires GOOGLE_API_KEY env var)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Step 4: Supervisor Multi-Agent System")
    if args.google:
        print("(Using Google AI)")
    if args.thinking:
        print("(Using Thinking Model)")
    print("=" * 60)
    print()

    # Initialize LLM
    if args.google:
        # Check for API key
        if not check_google_api_key():
            print("❌ Error: GOOGLE_API_KEY environment variable not set.")
            print("Please set it in your .env file or with: export GOOGLE_API_KEY='your-api-key'")
            sys.exit(1)
        
        model_name = get_google_model(prefer_thinking=args.thinking)
        print(f"🔗 Initializing Google AI LLM ({model_name})...")
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
        )
    else:
        model_name = get_available_model(prefer_thinking=args.thinking)
        print(f"🔗 Initializing LLM ({model_name})...")
        llm = ChatOllama(
            model=model_name,
            temperature=0,
            reasoning=True if args.thinking else False,
        )
    print("✓ LLM initialized")
    print()

    # Create the supervisor graph
    print("⛓️  Building supervisor graph...")
    graph = create_supervisor_graph()
    print("✓ Graph built successfully")
    print()

    print("Available agents:")
    print("  1. 👔 Supervisor - Orchestrates and delegates tasks")
    print("  2. 🔍 Researcher - Queries knowledge base for information")
    print("  3. ✍️  Writer - Generates well-formatted content")
    print("  4. ✅ Fact Checker - Verifies accuracy and compliance (DEMO)")
    print()

    # Define question processing function
    def ask_question(question: str):
        """Process a question through the supervisor system."""
        print("=" * 60)
        print(f"Question: {question}")
        print("=" * 60)

        # Create initial state
        initial_state = SupervisorState(
            messages=[HumanMessage(content=question)],
            next_agent="supervisor"
        )

        # Execute the graph
        print("\n🎭 Multi-agent system starting...")

        # Run the graph
        final_state = graph.invoke(initial_state)

        # Get final response
        final_message = final_state["messages"][-1]

        # Display result
        print("\n" + "=" * 60)
        print("Final Answer:")
        print("=" * 60)
        print(final_message.content)
        print()

    # Run in interactive or single-question mode
    if args.interactive:
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
        # Single question mode
        ask_question(args.question)

    # Educational explanation
    print()
    print("=" * 60)
    print("Supervisor Pattern Explanation")
    print("=" * 60)
    print()
    print("What just happened?")
    print()
    print("1. CENTRALIZED CONTROL:")
    print("   The Supervisor agent analyzed your question and created")
    print("   a delegation plan to route work to specialized workers.")
    print()
    print("2. TASK DELEGATION:")
    print("   - Researcher: Retrieved relevant information from knowledge base")
    print("   - Writer: Crafted a professional, well-formatted response")
    print("   - Fact Checker: Verified accuracy and compliance")
    print()
    print("3. WORKER SPECIALIZATION:")
    print("   Each agent has specific expertise and tools:")
    print("   - Researcher uses vector search and RAG")
    print("   - Writer focuses on content quality and clarity")
    print("   - Fact Checker ensures accuracy and standards")
    print()
    print("4. RETURN-TO-SUPERVISOR:")
    print("   After each worker completes their task, control returns")
    print("   to the Supervisor who decides the next step.")
    print()
    print("5. FINAL SYNTHESIS:")
    print("   The Supervisor combines all worker outputs into a")
    print("   coherent final answer for the user.")
    print()
    print("Key differences from ReAct agents:")
    print("  ✓ Centralized control (not autonomous tool selection)")
    print("  ✓ Explicit delegation (supervisor chooses workers)")
    print("  ✓ Specialized workers (each with specific domain)")
    print("  ✓ Hierarchical structure (supervisor → workers → supervisor)")
    print()
    print("When to use Supervisor pattern:")
    print("  • Complex tasks needing multiple specialists")
    print("  • When you want explicit control over workflow")
    print("  • Tasks requiring coordination between different domains")
    print("  • When worker outputs need to be combined/synthesized")
    print()
    print("When to use ReAct pattern instead:")
    print("  • Simple tasks with available tools")
    print("  • When agent can autonomously choose tools")
    print("  • No need for coordination between specialists")
    print()


if __name__ == "__main__":
    main()
