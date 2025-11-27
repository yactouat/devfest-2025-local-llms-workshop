#!/usr/bin/env python3
"""
Step 3: LangGraph ReAct Agent

This script demonstrates:
1. How to build a ReAct (Reasoning + Acting) agent using LangGraph
2. How to define and use tools for external data access
3. How to implement state-based decision-making with Pydantic
4. The difference between linear chains (LCEL) and cyclic graphs (LangGraph)

ReAct Pattern:
- The agent Reasons about what it needs to do
- Then Acts by choosing appropriate tools
- Observes the results
- Repeats until it can provide a final answer

Key Concepts:
- State Management: Using Pydantic classes to define agent state
- Tools: Functions the agent can call to access external data
- Graph Structure: Nodes (agent, tools) connected by conditional edges
- Decision Logic: Agent decides when to use tools vs. respond directly
"""

import argparse
from pathlib import Path
import sqlite3
import sys
from typing import Annotated
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# IMPORTANT: Use pysqlite3 instead of built-in sqlite3
# pysqlite3 supports the VSS (Vector Similarity Search) extension
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import sqlite_vss
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import SQLiteVSS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, ConfigDict

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_available_model, get_google_model, check_google_api_key


# ========================================
# STEP 1: Define Agent State with Pydantic
# ========================================
class AgentState(BaseModel):
    """
    Pydantic model defining the state of our ReAct agent.

    Using Pydantic provides:
    - Type safety and validation
    - Clear documentation of state structure
    - Automatic serialization/deserialization
    - IDE autocomplete support

    The 'messages' field tracks the entire conversation history,
    including user queries, agent responses, tool calls, and tool outputs.
    """
    # arbitrary_types_allowed = True
    # ==============================
    # This Pydantic configuration setting allows the model to accept fields with types
    # that Pydantic normally doesn't support or validate (like custom classes, complex objects, etc.).
    #
    # WHY WE NEED THIS:
    # - LangChain's message objects (HumanMessage, AIMessage, ToolMessage, etc.) are complex
    #   custom classes that don't follow Pydantic's standard validation patterns
    # - Without this setting, Pydantic would raise validation errors when trying to store
    #   these message objects in our state
    # - This is required for LangGraph state management when using LangChain messages
    #
    # IMPORTANT: This disables some of Pydantic's type checking for these fields,
    # so use it only when necessary (like with LangChain framework objects)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # add_messages Operator
    # ======================
    # The Annotated[list, add_messages] syntax combines type hints with LangGraph-specific behavior:
    #
    # 1. WHAT IS IT?
    #    - 'add_messages' is a special LangGraph reducer/operator function
    #    - It defines HOW state updates are merged when new messages arrive
    #    - Without it, new messages would REPLACE the entire list instead of appending
    #
    # 2. HOW IT WORKS:
    #    - When a node returns {"messages": [new_msg]}, LangGraph needs to know:
    #      Should it replace the old messages or append to them?
    #    - add_messages tells LangGraph: "Append these new messages to the existing list"
    #    - It intelligently handles message deduplication and updates
    #
    # 3. WHY WE NEED IT:
    #    - Preserves conversation history across multiple agent/tool invocations
    #    - Each node can add messages without worrying about losing previous context
    #    - Essential for the ReAct loop where agent → tools → agent needs full history
    #
    # 4. SMART BEHAVIOR:
    #    - If a message with the same ID exists, it gets updated (not duplicated)
    #    - New messages are appended to the end
    #    - Maintains chronological order of the conversation
    #
    # EXAMPLE:
    #   Initial state: messages = [HumanMessage("Hi")]
    #   Node returns: {"messages": [AIMessage("Hello!")]}
    #   Result: messages = [HumanMessage("Hi"), AIMessage("Hello!")]  ← APPENDED, not replaced!
    #
    # WITHOUT add_messages, the result would be:
    #   Result: messages = [AIMessage("Hello!")]  ← REPLACED the entire list!
    messages: Annotated[list, add_messages] = Field(
        default_factory=list,
        description="List of messages in the conversation"
    )


# ========================================
# STEP 2: Define Tools
# ========================================
@tool
def lookup_policy(query: str) -> str:
    """
    Query the DevFest Corp knowledge base (SQLite vector store) for information
    about company policies, people, culture, and other organizational details.

    Use this tool when the user asks about:
    - Company leadership (CEO, executives)
    - Company policies (vacation, remote work, etc.)
    - Company culture and values
    - General information about DevFest Corp

    Args:
        query: The search query to find relevant information

    Returns:
        Relevant information from the knowledge base
    """
    # Setup database connection
    script_dir = Path(__file__).parent
    db_path = script_dir.parent / "devfest.db"

    if not db_path.exists():
        return "Error: Knowledge base not found. Please run 02_rag_lcel/ingest.py first."

    # Initialize embeddings (must match ingestion)
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
    results = vectorstore.similarity_search(query, k=3)

    if not results:
        return "No relevant information found in the knowledge base."

    # Format results
    context = "\n\n".join([doc.page_content for doc in results])
    return f"Retrieved information:\n\n{context}"


@tool
def search_tech_events(query: str) -> str:
    """
    Search for upcoming tech conferences and events. This tool provides information
    about technology conferences, meetups, and other events in the tech community.

    Use this tool when the user asks about:
    - Upcoming tech conferences
    - Technology events
    - AI/ML conferences
    - Developer conferences
    - Tech meetups

    Args:
        query: The search query describing what kind of events to find

    Returns:
        Information about relevant tech events (DEMO CONTENT)
    """
    # This is a fictional tool that returns demo content
    # In a real implementation, this would query an events API or database
    demo_events = """
    === UPCOMING TECH EVENTS (After November 2025) ===

    1. **AI & Machine Learning Summit 2026**
       - Date: January 15-17, 2026
       - Location: San Francisco, CA
       - Topics: Large Language Models, Computer Vision, MLOps
       - Website: ai-ml-summit.example.com
       - Description: Three-day conference featuring the latest advances in AI and ML,
         with hands-on workshops and keynotes from industry leaders.

    2. **DevFest Global 2026**
       - Date: March 10-12, 2026
       - Location: Berlin, Germany (+ Virtual)
       - Topics: Cloud Computing, Web Development, Mobile Apps
       - Website: devfest-global.example.com
       - Description: Annual developer festival showcasing Google technologies
         and community innovations.

    3. **PyData Conference 2026**
       - Date: May 5-8, 2026
       - Location: Austin, TX
       - Topics: Data Science, Python, Analytics
       - Website: pydata-conf.example.com
       - Description: Conference focused on Python tools for data analysis,
         machine learning, and large-scale data processing.

    4. **KubeCon + CloudNativeCon 2026**
       - Date: April 20-23, 2026
       - Location: Amsterdam, Netherlands
       - Topics: Kubernetes, Cloud Native, DevOps
       - Website: kubecon.example.com
       - Description: Premier cloud native computing conference featuring
         Kubernetes, containers, and microservices.

    5. **NeurIPS 2026 (Neural Information Processing Systems)**
       - Date: December 7-13, 2026
       - Location: Vancouver, Canada
       - Topics: Deep Learning, Neural Networks, AI Research
       - Website: neurips.example.com
       - Description: Leading conference in machine learning and computational
         neuroscience, featuring cutting-edge research presentations.

    Note: This is demonstration content for the workshop. In a production system,
    this would query real event databases or APIs. The events and dates shown are
    fictional examples created to demonstrate the multi-tool capabilities of the agent.
    """

    return demo_events


# ========================================
# STEP 3: Initialize Global Resources
# ========================================
# We'll initialize the LLM and tools at module level so they can be
# accessed by the agent and tool nodes

# This will be set in main()
llm_with_tools = None


# ========================================
# STEP 4: Define Graph Nodes
# ========================================
def agent_node(state: AgentState) -> dict:
    """
    The agent node decides what to do next:
    - If it needs more information, it calls a tool
    - If it has enough information, it responds directly

    This is the "Reasoning" part of ReAct.

    Args:
        state: Current agent state (Pydantic model)

    Returns:
        State update with new message
    """
    # Get the latest messages from state
    messages = state.messages

    # Invoke the LLM with tool bindings
    # The LLM will decide whether to:
    # 1. Call a tool (returns tool_call message)
    # 2. Respond directly (returns text message)
    response = llm_with_tools.invoke(messages)

    # Return state update (LangGraph will merge this with existing state)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """
    Conditional edge function that determines the next node.

    This is the decision logic of the graph:
    - If the last message has tool calls, route to "tools" node
    - Otherwise, end the graph (agent is done)

    Args:
        state: Current agent state

    Returns:
        "tools" to execute tools, or "end" to finish
    """
    messages = state.messages
    last_message = messages[-1]

    # Check if the agent wants to use tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"

    # No tools needed, agent is done
    return "end"


# ========================================
# STEP 5: Build the Graph
# ========================================
def create_graph():
    """
    Creates the LangGraph StateGraph for the ReAct agent.

    Graph Structure:
        [START] → agent → {should_continue}
                           ├─ "tools" → tool_node → agent (loop)
                           └─ "end" → [END]

    The agent can loop multiple times, calling different tools
    as needed, until it has enough information to respond.

    Returns:
        Compiled graph ready for execution
    """
    # Create the state graph with our Pydantic state model
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", agent_node)

    # ToolNode is a built-in node that executes tool calls
    # It automatically handles tool invocation and formats results
    tools = [lookup_policy, search_tech_events]
    workflow.add_node("tools", ToolNode(tools))

    # Set entry point
    workflow.set_entry_point("agent")

    # Add conditional edge from agent
    # This is where the ReAct decision-making happens
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",  # If tools needed, go to tools node
            "end": END,        # If done, end the graph
        }
    )

    # Add edge from tools back to agent
    # After executing tools, agent reasons about results
    workflow.add_edge("tools", "agent")

    # Compile the graph
    return workflow.compile()


# ========================================
# STEP 6: Main Function
# ========================================
def main():
    global llm_with_tools

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="ReAct Agent with LangGraph")
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
    parser.add_argument(
        "--google",
        action="store_true",
        help="Use Google AI instead of local Ollama (requires GOOGLE_API_KEY env var)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Step 3: LangGraph ReAct Agent")
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

    # Bind tools to LLM
    tools = [lookup_policy, search_tech_events]
    llm_with_tools = llm.bind_tools(tools)
    print("✓ LLM initialized with tools")
    print()

    # Create the graph
    print("⛓️  Building ReAct agent graph...")
    graph = create_graph()
    print("✓ Graph built successfully")
    print()

    print("Available tools:")
    print("  1. lookup_policy - Query DevFest Corp knowledge base")
    print("  2. search_tech_events - Find upcoming tech conferences")
    print()

    # Define question processing function
    def ask_question(question: str):
        """Process a question through the ReAct agent."""
        print("=" * 60)
        print(f"Question: {question}")
        print("=" * 60)
        print()

        # Create initial state
        initial_state = AgentState(messages=[HumanMessage(content=question)])

        # Execute the graph
        print("🤖 Agent is reasoning and acting...")
        print()

        # Stream the graph execution to see each step
        step = 0
        for output in graph.stream(initial_state):
            step += 1
            print(f"--- Step {step} ---")

            # Show what happened in this step
            for key, value in output.items():
                print(f"Node: {key}")
                if "messages" in value:
                    for msg in value["messages"]:
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            print(f"  Tool Calls: {len(msg.tool_calls)}")
                            for tc in msg.tool_calls:
                                print(f"    - {tc['name']}({tc['args']})")
                        elif hasattr(msg, 'content'):
                            content_preview = str(msg.content)[:100].replace("\n", " ")
                            print(f"  Content: {content_preview}...")
            print()

        # Get final response
        final_state = graph.invoke(initial_state)
        # LangGraph returns a dict, not a Pydantic model
        final_message = final_state["messages"][-1]

        # Display result
        print("=" * 60)
        print("Final Answer:")
        print("=" * 60)

        if args.thinking and not args.google and hasattr(final_message, 'additional_kwargs'):
            reasoning = final_message.additional_kwargs.get("reasoning_content")
            if reasoning:
                print()
                print("### Thinking Trace ###")
                print(reasoning)
                print("\n" + "="*60 + "\n")

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
    print("ReAct Agent Explanation")
    print("=" * 60)
    print()
    print("What just happened?")
    print()
    print("1. REASONING:")
    print("   The agent analyzed your question and decided whether")
    print("   it needed to use tools to gather information.")
    print()
    print("2. ACTING:")
    print("   If tools were needed, the agent called the appropriate")
    print("   tool(s) with relevant arguments.")
    print()
    print("3. OBSERVING:")
    print("   The agent received tool results and incorporated them")
    print("   into its understanding.")
    print()
    print("4. ITERATING:")
    print("   The agent can loop multiple times, calling different")
    print("   tools as needed, until it has enough information.")
    print()
    print("5. RESPONDING:")
    print("   Once the agent has sufficient information, it provides")
    print("   a final answer based on all observations.")
    print()
    print("Key differences from LCEL chains:")
    print("  ✓ Dynamic decision-making (agent chooses tools)")
    print("  ✓ Cyclic flow (can loop back to agent)")
    print("  ✓ State management (tracks conversation history)")
    print("  ✓ Conditional logic (different paths based on state)")
    print()
    print("When to use graphs vs. chains:")
    print("  • Chains: Fixed pipeline, no decisions needed")
    print("  • Graphs: Dynamic flow, tool usage, multi-step reasoning")
    print()


if __name__ == "__main__":
    main()
