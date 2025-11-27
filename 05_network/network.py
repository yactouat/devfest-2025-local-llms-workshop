#!/usr/bin/env python3
"""
Step 5: The Network/Swarm Pattern

This script demonstrates:
1. How to build a decentralized multi-agent system
2. How agents communicate directly with each other through handoffs
3. How to implement peer-to-peer agent collaboration
4. The difference between hierarchical (Supervisor) and network patterns

Network/Swarm Pattern:
- Agents operate as peers without a central controller
- Agents can directly handoff tasks to each other
- Each agent decides independently when to involve others
- Flexible, dynamic collaboration based on task needs

Key Concepts:
- Decentralized Control: No single supervisor coordinating work
- Peer-to-Peer Handoffs: Agents directly transfer control to each other
- Autonomous Decision Making: Each agent chooses when to handoff
- Bidirectional Communication: Agents can call each other as needed
"""

import argparse
from pathlib import Path
import sqlite3
import sys
from typing import Annotated, Literal, Optional
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
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, ConfigDict

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_available_model, get_google_model, check_google_api_key


# ========================================
# STEP 1: Define Network State
# ========================================
class NetworkState(BaseModel):
    """
    State for the network multi-agent system.

    Tracks:
    - messages: Full conversation history including handoffs
    - next_agent: Which agent should execute next
    - handoff_reason: Why the handoff occurred (for transparency)
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: Annotated[list, add_messages] = Field(
        default_factory=list,
        description="Conversation history including all agent interactions and handoffs"
    )
    next_agent: str = Field(
        default="researcher",
        description="The next agent to execute (researcher, writer, fact_checker, or FINISH)"
    )
    handoff_reason: str = Field(
        default="",
        description="Reason for the current handoff"
    )


# ========================================
# STEP 2: Global Resources
# ========================================
llm = None
vectorstore = None


# ========================================
# STEP 3: Define Structured Output Schemas
# ========================================
class HandoffDecision(BaseModel):
    """
    Structured decision for agent handoffs using with_structured_output.
    
    This ensures agents make clean decisions: either handoff OR provide final answer.
    No mixing of formats or unclear responses.
    """
    handoff_to: Optional[Literal["researcher", "writer", "fact_checker"]] = Field(
        default=None,
        description="Which agent to handoff to, or None if providing final answer"
    )
    reason: Optional[str] = Field(
        default=None,
        description="Why you're handing off to this agent (required if handoff_to is set)"
    )
    final_answer: Optional[str] = Field(
        default=None,
        description="The final answer to provide to the user (set this if NOT handing off)"
    )


# ========================================
# STEP 4: Define Handoff Tools
# ========================================
@tool
def transfer_to_researcher(reason: str) -> str:
    """
    Transfer control to the Researcher agent.

    Use this when you need:
    - Information from the DevFest Corp knowledge base
    - Facts about company policies, people, or culture
    - Research on specific topics

    Args:
        reason: Why you're transferring to the Researcher

    Returns:
        Confirmation message
    """
    return f"🔄 Handoff to Researcher: {reason}"


@tool
def transfer_to_writer(reason: str) -> str:
    """
    Transfer control to the Writer agent.

    Use this when you need:
    - Content to be written or formatted
    - A professional response drafted
    - Information synthesized into readable form

    Args:
        reason: Why you're transferring to the Writer

    Returns:
        Confirmation message
    """
    return f"🔄 Handoff to Writer: {reason}"


@tool
def transfer_to_fact_checker(reason: str) -> str:
    """
    Transfer control to the Fact Checker agent.

    Use this when you need:
    - Information to be verified
    - Fact-checking or validation
    - Quality assurance on content

    Args:
        reason: Why you're transferring to the Fact Checker

    Returns:
        Confirmation message
    """
    return f"🔄 Handoff to Fact Checker: {reason}"


# ========================================
# STEP 5: Agent Functions with Structured Output
# ========================================
def researcher_agent(state: NetworkState) -> dict:
    """
    Researcher Agent - Retrieves information and can handoff to others.

    This agent uses with_structured_output to ensure clean decisions:
    - Either provides a final answer OR makes a handoff
    - No mixing of formats or ambiguous responses

    This agent:
    - Queries the DevFest Corp knowledge base
    - Uses structured output to decide on handoffs
    - Decides independently when to handoff

    Args:
        state: Current network state

    Returns:
        State update with research findings and/or handoff decision
    """
    print("\n🔍 RESEARCHER AGENT working...")

    # Check if this is a handoff scenario
    last_message = state.messages[-1] if state.messages else None
    is_handoff = (isinstance(last_message, AIMessage) and
                  last_message.tool_calls and
                  any('transfer_to_researcher' in str(tc) for tc in last_message.tool_calls))

    if is_handoff:
        print(f"   ℹ️  Received handoff: {state.handoff_reason}")

    # Get the user's question from messages
    user_question = None
    for msg in state.messages:
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break

    if not user_question:
        return {
            "messages": [AIMessage(content="[Researcher] No question found to research.")],
            "next_agent": "FINISH"
        }

    # Setup database connection
    script_dir = Path(__file__).parent
    db_path = script_dir.parent / "devfest.db"

    if not db_path.exists():
        return {
            "messages": [AIMessage(
                content="[Researcher] Error: Knowledge base not found. Please run 02_rag_lcel/ingest.py first."
            )],
            "next_agent": "FINISH"
        }

    # Initialize embeddings and load vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
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
        research_findings = "No relevant information found in the knowledge base."
        print("   ✓ No relevant documents found")
    else:
        context = "\n\n".join([doc.page_content for doc in results])
        research_findings = f"I found the following information:\n\n{context}"
        print(f"   ✓ Retrieved {len(results)} relevant documents")

    # Use structured output for clean decision-making
    llm_with_structure = llm.with_structured_output(HandoffDecision)

    # Prepare decision prompt
    system_prompt = """You are the Researcher agent in a network of collaborative agents.

You have just gathered information from the knowledge base. Now you must make ONE of these decisions:

1. If you have good information and it needs to be formatted into a professional response:
   - Set handoff_to to "writer"
   - Set reason explaining what the writer should do
   - Leave final_answer as None

2. If you have information but are unsure about its accuracy:
   - Set handoff_to to "fact_checker"
   - Set reason explaining what needs verification
   - Leave final_answer as None

3. If you found no relevant information or the question is simple enough to answer directly:
   - Set final_answer to your direct response
   - Leave handoff_to as None

CRITICAL: 
- All factual information need to be transferred to the fact checker agent before finalizing the response.
- Even if you are sure about the information you have retrieved, you need to transfer it to the fact checker agent for proofreading.

IMPORTANT: You must provide EITHER a final_answer OR a handoff (handoff_to + reason), never both, never neither."""

    decision_prompt = f"""Research Findings:
{research_findings}

User Question: {user_question}

Based on the research above, make your decision: provide a final answer or handoff to another agent."""

    # Make decision using structured output
    decision: HandoffDecision = llm_with_structure.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=decision_prompt)
    ])

    # Process the structured decision
    if decision.final_answer:
        # Researcher is providing final answer
        print("   ✓ Researcher providing direct response")
        return {
            "messages": [AIMessage(content=f"[Researcher] {decision.final_answer}")],
            "next_agent": "FINISH"
        }
    elif decision.handoff_to and decision.reason:
        # Researcher wants to handoff
        print(f"   → Researcher initiating handoff to: {decision.handoff_to}")
        print(f"   → Reason: {decision.reason}")
        
        # Create the appropriate tool call based on decision
        tool_map = {
            "writer": transfer_to_writer,
            "fact_checker": transfer_to_fact_checker
        }
        
        tool_to_call = tool_map.get(decision.handoff_to)
        if not tool_to_call:
            # Fallback to final answer if invalid handoff
            return {
                "messages": [AIMessage(content=f"[Researcher] {research_findings}")],
                "next_agent": "FINISH"
            }
        
        # Manually create the tool call structure
        tool_call_msg = AIMessage(
            content=f"[Researcher] {research_findings}",
            tool_calls=[{
                "name": f"transfer_to_{decision.handoff_to}",
                "args": {"reason": decision.reason},
                "id": f"call_{decision.handoff_to}"
            }]
        )
        
        return {
            "messages": [tool_call_msg],
        }
    else:
        # Invalid decision - provide default response
        print("   ⚠️  Invalid decision structure, providing research findings")
        return {
            "messages": [AIMessage(content=f"[Researcher] {research_findings}")],
            "next_agent": "FINISH"
        }


def writer_agent(state: NetworkState) -> dict:
    """
    Writer Agent - Generates content and can handoff to others.

    This agent uses with_structured_output to ensure clean decisions:
    - Either provides a final answer OR makes a handoff
    - No mixing of formats or ambiguous responses

    This agent:
    - Creates well-formatted responses
    - Uses structured output to decide on handoffs
    - Decides independently when to handoff

    Args:
        state: Current network state

    Returns:
        State update with generated content and/or handoff decision
    """
    print("\n✍️  WRITER AGENT working...")

    # Check if this is a handoff scenario
    last_message = state.messages[-1] if state.messages else None
    is_handoff = (isinstance(last_message, AIMessage) and
                  last_message.tool_calls and
                  any('transfer_to_writer' in str(tc) for tc in last_message.tool_calls))

    if is_handoff:
        print(f"   ℹ️  Received handoff: {state.handoff_reason}")

    # Gather context from previous messages
    context = []
    user_question = None

    for msg in state.messages:
        if isinstance(msg, HumanMessage):
            user_question = msg.content
        elif isinstance(msg, AIMessage):
            # Include both regular content and tool call messages
            if msg.content and not msg.tool_calls:
                # Regular message with content
                if "[Researcher]" in msg.content or "[Fact Checker]" in msg.content:
                    context.append(msg.content)
            elif msg.content and msg.tool_calls:
                # Message with both content and tool calls (our new format)
                # Extract the content which contains research findings
                if "[Researcher]" in msg.content or "[Fact Checker]" in msg.content:
                    context.append(msg.content)

    if not user_question:
        return {
            "messages": [AIMessage(content="[Writer] No question to write about.")],
            "next_agent": "FINISH"
        }

    research_context = "\n\n".join(context) if context else "No additional context provided."

    # Use structured output for clean decision-making
    llm_with_structure = llm.with_structured_output(HandoffDecision)

    # Generate content
    system_prompt = """You are the Writer agent in a network of collaborative agents.

Your job is to create clear, professional responses based on available information.

You must make ONE of these decisions:

1. If you have enough context, write a complete professional response:
   - Set final_answer to your complete response
   - Leave handoff_to as None

2. If you need more information from the knowledge base:
   - Set handoff_to to "researcher"
   - Set reason explaining what information you need
   - Leave final_answer as None

3. If your content needs fact-checking:
   - Set handoff_to to "fact_checker"
   - Set reason explaining what needs verification
   - Leave final_answer as None

IMPORTANT: You must provide EITHER a final_answer OR a handoff (handoff_to + reason), never both, never neither."""

    user_prompt = f"""User Question: {user_question}

Available Context:
{research_context}

Based on the context above, make your decision: provide a final answer or handoff to another agent."""

    decision: HandoffDecision = llm_with_structure.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    # Process the structured decision
    if decision.final_answer:
        # Writer is providing final answer
        print("   ✓ Writer providing final response")
        return {
            "messages": [AIMessage(content=f"[Writer] {decision.final_answer}")],
            "next_agent": "FINISH"
        }
    elif decision.handoff_to and decision.reason:
        # Writer wants to handoff
        print(f"   → Writer initiating handoff to: {decision.handoff_to}")
        print(f"   → Reason: {decision.reason}")
        
        # Create the appropriate tool call based on decision
        tool_map = {
            "researcher": transfer_to_researcher,
            "fact_checker": transfer_to_fact_checker
        }
        
        tool_to_call = tool_map.get(decision.handoff_to)
        if not tool_to_call:
            # Fallback to final answer if invalid handoff
            return {
                "messages": [AIMessage(content="[Writer] Invalid handoff target. Completing task.")],
                "next_agent": "FINISH"
            }
        
        # Create tool call message
        tools = [transfer_to_researcher, transfer_to_fact_checker]
        tool_node = ToolNode(tools)
        
        # Manually create the tool call structure
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[{
                "name": f"transfer_to_{decision.handoff_to}",
                "args": {"reason": decision.reason},
                "id": f"call_{decision.handoff_to}"
            }]
        )
        
        return {
            "messages": [tool_call_msg],
        }
    else:
        # Invalid decision - provide default response
        print("   ⚠️  Invalid decision structure, providing default response")
        return {
            "messages": [AIMessage(content="[Writer] Unable to process request properly.")],
            "next_agent": "FINISH"
        }


def fact_checker_agent(state: NetworkState) -> dict:
    """
    Fact Checker Agent - Validates information and can handoff to others.

    This agent uses with_structured_output to ensure clean decisions:
    - Either provides a final answer OR makes a handoff
    - No mixing of formats or ambiguous responses

    This agent:
    - Verifies factual accuracy (demonstration mode)
    - Uses structured output to decide on handoffs
    - Decides independently when to handoff

    NOTE: This is a demonstration agent with mocked verification output.
    In production, this would use real fact-checking APIs and data sources.

    Args:
        state: Current network state

    Returns:
        State update with verification results and/or handoff decision
    """
    print("\n✅ FACT CHECKER AGENT working...")

    # Check if this is a handoff scenario
    last_message = state.messages[-1] if state.messages else None
    is_handoff = (isinstance(last_message, AIMessage) and
                  last_message.tool_calls and
                  any('transfer_to_fact_checker' in str(tc) for tc in last_message.tool_calls))

    if is_handoff:
        print(f"   ℹ️  Received handoff: {state.handoff_reason}")

    # Gather content to fact-check
    content_to_check = []
    user_question = None

    for msg in state.messages:
        if isinstance(msg, HumanMessage):
            user_question = msg.content
        elif isinstance(msg, AIMessage):
            # Include messages with content, regardless of whether they have tool calls
            if msg.content and ("[Researcher]" in msg.content or "[Writer]" in msg.content):
                content_to_check.append(msg.content)

    if not content_to_check:
        verification_result = "No content to verify."
    else:
        # DEMO OUTPUT: In a real system, this would perform actual verification
        verification_result = """Verification Report:

✓ Information Sources: Verified against knowledge base
✓ Policy Compliance: All statements align with DevFest Corp policies
✓ Factual Accuracy: Cross-referenced data points are consistent
✓ Quality Check: Content meets professional standards
✓ Completeness: Response addresses the user's question

Status: APPROVED ✓

Note: This is a demonstration of the fact-checking agent."""

    print("   ✓ Verification completed")

    # Use structured output for clean decision-making
    llm_with_structure = llm.with_structured_output(HandoffDecision)

    # Make decision about next steps
    system_prompt = """You are the Fact Checker agent in a network of collaborative agents.

You have just verified the content. Now you must make ONE of these decisions:

1. If verification passed and content is approved:
   - Set final_answer to summarize the approved content for the user
   - Leave handoff_to as None

2. If you need more information to complete verification:
   - Set handoff_to to "researcher"
   - Set reason explaining what information you need
   - Leave final_answer as None

3. If content needs revision OR a nice human-readable response after fact-checking:
   - Set handoff_to to "writer"
   - Set reason explaining what needs to be revised
   - Leave final_answer as None

IMPORTANT: You must provide EITHER a final_answer OR a handoff (handoff_to + reason), never both, never neither."""

    decision_prompt = f"""Verification Result:
{verification_result}

Content Checked:
{chr(10).join(content_to_check)}

Based on the verification above, make your decision: approve with final answer or request changes via handoff."""

    decision: HandoffDecision = llm_with_structure.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=decision_prompt)
    ])

    # Process the structured decision
    if decision.final_answer:
        # Fact checker is approving and providing final answer
        print("   ✓ Fact Checker approving final answer")
        return {
            "messages": [AIMessage(content=f"[Fact Checker] {decision.final_answer}")],
            "next_agent": "FINISH"
        }
    elif decision.handoff_to and decision.reason:
        # Fact checker wants to handoff
        print(f"   → Fact Checker initiating handoff to: {decision.handoff_to}")
        print(f"   → Reason: {decision.reason}")
        
        # Create the appropriate tool call based on decision
        tool_map = {
            "researcher": transfer_to_researcher,
            "writer": transfer_to_writer
        }
        
        tool_to_call = tool_map.get(decision.handoff_to)
        if not tool_to_call:
            # Fallback to final answer if invalid handoff
            return {
                "messages": [AIMessage(content=f"[Fact Checker] {verification_result}")],
                "next_agent": "FINISH"
            }
        
        # Manually create the tool call structure
        tool_call_msg = AIMessage(
            content=f"[Fact Checker] {verification_result}",
            tool_calls=[{
                "name": f"transfer_to_{decision.handoff_to}",
                "args": {"reason": decision.reason},
                "id": f"call_{decision.handoff_to}"
            }]
        )
        
        return {
            "messages": [tool_call_msg],
        }
    else:
        # Invalid decision - provide default response
        print("   ⚠️  Invalid decision structure, approving content")
        return {
            "messages": [AIMessage(content=f"[Fact Checker] {verification_result}")],
            "next_agent": "FINISH"
        }


# ========================================
# STEP 5: Tool Execution Node
# ========================================
def handle_tool_calls(state: NetworkState) -> dict:
    """
    Process tool calls (handoffs) and route to appropriate agent.

    This function:
    - Executes handoff tools
    - Extracts the reason for handoff
    - Routes to the appropriate next agent

    Args:
        state: Current network state

    Returns:
        State update with tool results and next agent routing
    """
    last_message = state.messages[-1]

    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {"next_agent": "FINISH"}

    tool_call = last_message.tool_calls[0]
    tool_name = tool_call['name']

    # Extract handoff reason
    reason = tool_call['args'].get('reason', 'No reason provided')

    print(f"\n🔄 Processing handoff: {tool_name}")
    print(f"   Reason: {reason}")

    # Determine next agent based on tool
    next_agent_map = {
        'transfer_to_researcher': 'researcher',
        'transfer_to_writer': 'writer',
        'transfer_to_fact_checker': 'fact_checker',
    }

    next_agent = next_agent_map.get(tool_name, 'FINISH')

    # Create tool node to execute the handoff
    tools = [transfer_to_researcher, transfer_to_writer, transfer_to_fact_checker]
    tool_node = ToolNode(tools)

    # Execute tool
    tool_result = tool_node.invoke(state)

    return {
        "messages": tool_result['messages'],
        "next_agent": next_agent,
        "handoff_reason": reason
    }


# ========================================
# STEP 6: Routing Function
# ========================================
def route_after_agent(state: NetworkState) -> Literal["tools", "researcher", "writer", "fact_checker", "end"]:
    """
    Conditional edge function that routes based on agent decisions.

    This function checks:
    1. Did the agent make a tool call (handoff)? → Route to tools
    2. Did the agent set next_agent to FINISH? → Route to end
    3. Otherwise, route to the specified next agent

    Args:
        state: Current network state

    Returns:
        Name of the next node to execute
    """
    last_message = state.messages[-1]

    # Check if there's a tool call (handoff request)
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    # Check if task is complete
    if state.next_agent == "FINISH":
        return "end"

    # Route to specified agent
    return state.next_agent


# ========================================
# STEP 7: Build the Network Graph
# ========================================
def create_network_graph():
    """
    Creates the LangGraph StateGraph for the network pattern.

    Graph Structure:
        [START] → researcher → {route_after_agent}
                              ├─ tools → {route to next agent}
                              │         ├─ researcher (loop)
                              │         ├─ writer (loop)
                              │         └─ fact_checker (loop)
                              └─ end → [END]

        writer → {route_after_agent} (same structure)
        fact_checker → {route_after_agent} (same structure)

    Each agent can:
    - Handoff to any other agent via tools
    - Provide a final answer and end the flow
    - No central supervisor - all peer-to-peer

    Returns:
        Compiled graph ready for execution
    """
    workflow = StateGraph(NetworkState)

    # Add agent nodes
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("writer", writer_agent)
    workflow.add_node("fact_checker", fact_checker_agent)
    workflow.add_node("tools", handle_tool_calls)

    # Set entry point (start with researcher by default)
    workflow.set_entry_point("researcher")

    # Add conditional edges from each agent
    for agent in ["researcher", "writer", "fact_checker"]:
        workflow.add_conditional_edges(
            agent,
            route_after_agent,
            {
                "tools": "tools",
                "researcher": "researcher",
                "writer": "writer",
                "fact_checker": "fact_checker",
                "end": END,
            }
        )

    # Route from tools back to agents based on handoff
    workflow.add_conditional_edges(
        "tools",
        lambda state: state.next_agent if state.next_agent != "FINISH" else "end",
        {
            "researcher": "researcher",
            "writer": "writer",
            "fact_checker": "fact_checker",
            "end": END,
        }
    )

    return workflow.compile()


# ========================================
# STEP 8: Main Function
# ========================================
def main():
    global llm

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Network Multi-Agent System")
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
        help="Use qwen3:8b thinking model for agent decisions"
    )
    parser.add_argument(
        "--google",
        action="store_true",
        help="Use Google AI instead of local Ollama (requires GOOGLE_API_KEY env var)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Step 5: Network/Swarm Multi-Agent System")
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

    # Create the network graph
    print("⛓️  Building network graph...")
    graph = create_network_graph()
    print("✓ Graph built successfully")
    print()

    print("Available agents (peer-to-peer network):")
    print("  1. 🔍 Researcher - Queries knowledge base, can handoff to Writer or Fact Checker")
    print("  2. ✍️  Writer - Generates content, can handoff to Researcher or Fact Checker")
    print("  3. ✅ Fact Checker - Verifies accuracy, can handoff to Researcher or Writer")
    print()
    print("Each agent decides independently when to involve others!")
    print()

    # Define question processing function
    def ask_question(question: str):
        """Process a question through the network system."""
        print("=" * 60)
        print(f"Question: {question}")
        print("=" * 60)

        # Create initial state
        initial_state = NetworkState(
            messages=[HumanMessage(content=question)],
            next_agent="researcher"
        )

        # Execute the graph
        print("\n🕸️  Network system starting...")

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
    print("Network/Swarm Pattern Explanation")
    print("=" * 60)
    print()
    print("What just happened?")
    print()
    print("1. DECENTRALIZED COLLABORATION:")
    print("   Unlike the Supervisor pattern, there is NO central controller.")
    print("   Agents communicate directly with each other as peers.")
    print()
    print("2. PEER-TO-PEER HANDOFFS:")
    print("   - Researcher can handoff to Writer or Fact Checker")
    print("   - Writer can handoff to Researcher or Fact Checker")
    print("   - Fact Checker can handoff to Researcher or Writer")
    print("   - Each agent has tools to transfer control")
    print()
    print("3. AUTONOMOUS DECISION MAKING:")
    print("   Each agent independently decides:")
    print("   - When to involve another agent")
    print("   - Which agent to handoff to")
    print("   - When to provide the final answer")
    print()
    print("4. FLEXIBLE COLLABORATION:")
    print("   The workflow emerges naturally from agent interactions,")
    print("   not from a predefined plan.")
    print()
    print("5. STRUCTURED OUTPUT + HANDOFF TOOLS:")
    print("   - Agents use with_structured_output(HandoffDecision) to decide")
    print("   - Decision includes: handoff_to, reason, or final_answer")
    print("   - If handoff chosen, creates tool call message:")
    print("     * transfer_to_researcher(reason)")
    print("     * transfer_to_writer(reason)")
    print("     * transfer_to_fact_checker(reason)")
    print("   - Tools are executed by dedicated node in graph")
    print()
    print("Key differences from Supervisor pattern:")
    print("  ✓ No central controller (decentralized)")
    print("  ✓ Peer-to-peer communication (not hierarchical)")
    print("  ✓ Dynamic workflow (emerges from agent decisions)")
    print("  ✓ Bidirectional handoffs (any agent to any agent)")
    print()
    print("When to use Network pattern:")
    print("  • Tasks requiring flexible, adaptive collaboration")
    print("  • When agents are equal peers (no hierarchy)")
    print("  • Dynamic workflows that vary by task")
    print("  • Scenarios where control should be distributed")
    print()
    print("When to use Supervisor pattern instead:")
    print("  • Need explicit control over workflow")
    print("  • Clear hierarchical structure required")
    print("  • Predictable, consistent task delegation")
    print("  • Central orchestration is beneficial")
    print()


if __name__ == "__main__":
    main()
