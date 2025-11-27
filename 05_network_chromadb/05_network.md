# Step 5: Network/Swarm Multi-Agent Pattern

## Goal
Learn how to build a decentralized multi-agent system where agents communicate directly with each other as peers, without a central supervisor. This step demonstrates peer-to-peer agent collaboration through handoffs, enabling flexible and dynamic workflows.

## Prerequisites
Before starting this step, ensure you have:
- Completed Step 4 (Supervisor Pattern)
- The knowledge base ingested into SQLite (run `02_rag_lcel/ingest.py`)
- Ollama running with models pulled
- Python virtual environment activated

## What You'll Learn
- The difference between **Supervisor pattern** (hierarchical) and **Network pattern** (peer-to-peer)
- How to implement **agent handoffs** using tools
- How agents can **communicate directly** without a central controller
- When to use the **Network pattern** vs. Supervisor pattern
- How to build **decentralized collaboration** systems

## Supervisor Pattern vs. Network Pattern

### Supervisor Pattern (Step 4)
```
┌──────────────────────────────────────────────────────────┐
│          Hierarchical Multi-Agent System                  │
└──────────────────────────────────────────────────────────┘

User Query: "Who is the CEO?"
   │
   ▼
┌─────────────┐
│ SUPERVISOR  │ ← Central controller
│             │   Makes ALL routing decisions
└──────┬──────┘
       │
       ├───────────────┬───────────────┬──────────────┐
       │               │               │              │
       ▼               ▼               ▼              ▼
┌──────────┐   ┌──────────┐   ┌──────────┐
│Researcher│   │  Writer  │   │  Fact    │
│          │   │          │   │ Checker  │
└────┬─────┘   └────┬─────┘   └────┬─────┘
     │              │              │
     └──────────────┴──────────────┘
                    │
            Always return to SUPERVISOR
                    │
                    ▼
          Final Synthesized Response
```

**Characteristics:**
- **Centralized control** through Supervisor
- **Hierarchical structure** (boss → workers)
- **Fixed workflow** determined by Supervisor
- **Workers cannot communicate** with each other directly
- **Best for**: Tasks requiring strict control and predictable workflows

### Network Pattern (Step 5)
```
┌──────────────────────────────────────────────────────────┐
│          Peer-to-Peer Multi-Agent System                  │
└──────────────────────────────────────────────────────────┘

User Query: "Who is the CEO?"
   │
   ▼
┌──────────────┐
│  RESEARCHER  │ ← Entry point (any agent can start)
└──────┬───────┘
       │
       │ [Decides: Need writing?]
       │
       ├─ transfer_to_writer ──▶ ┌──────────────┐
       │                          │    WRITER    │
       │                          └──────┬───────┘
       │                                 │
       │                                 │ [Decides: Need verification?]
       │                                 │
       │           transfer_to_fact_checker ──▶ ┌──────────────┐
       │                                        │ FACT CHECKER │
       │                                        └──────┬───────┘
       │                                               │
       │                                               │ [Decides: All good?]
       │                                               │
       │                                               ▼
       │                                        Final Response
       │
       └─ OR agent can provide direct response at any time

Alternative Flow:
┌──────────────┐  transfer_to_researcher  ┌──────────────┐
│    WRITER    │─────────────────────────▶│  RESEARCHER  │
└──────────────┘                           └──────────────┘

┌──────────────┐  transfer_to_writer  ┌──────────────┐
│ FACT CHECKER │─────────────────────▶│    WRITER    │
└──────────────┘                       └──────────────┘
```

**Characteristics:**
- **Decentralized control** (no supervisor)
- **Peer-to-peer structure** (all agents are equals)
- **Dynamic workflow** (emerges from agent decisions)
- **Agents communicate directly** via handoff tools
- **Best for**: Flexible tasks requiring adaptive collaboration

## The Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              NETWORK MULTI-AGENT SYSTEM                      │
└─────────────────────────────────────────────────────────────┘

USER QUERY: "Who is the CEO of DevFest Corp?"
   │
   ▼
┌────────────────┐
│  RESEARCHER    │ ← Entry point (default)
│                │
│ Available Tools:
│  - transfer_to_writer()
│  - transfer_to_fact_checker()
└────────┬───────┘
         │
         ▼
    [Researcher Decides]
         │
         ├─ Has info, needs writing ──▶ Calls transfer_to_writer()
         │                                      │
         │                                      ▼
         │                              ┌────────────────┐
         │                              │    WRITER      │
         │                              │                │
         │                              │ Available Tools:
         │                              │  - transfer_to_researcher()
         │                              │  - transfer_to_fact_checker()
         │                              └────────┬───────┘
         │                                       │
         │                                       ▼
         │                              [Writer Decides]
         │                                       │
         │                                       ├─ Needs verification ──▶ Calls transfer_to_fact_checker()
         │                                       │                                   │
         │                                       │                                   ▼
         │                                       │                           ┌────────────────┐
         │                                       │                           │ FACT CHECKER   │
         │                                       │                           │                │
         │                                       │                           │ Available Tools:
         │                                       │                           │  - transfer_to_researcher()
         │                                       │                           │  - transfer_to_writer()
         │                                       │                           └────────┬───────┘
         │                                       │                                    │
         │                                       │                                    ▼
         │                                       │                           [Fact Checker Decides]
         │                                       │                                    │
         │                                       │                                    ├─ All verified ──▶ FINISH
         │                                       │                                    │
         │                                       ├─ Content complete ──▶ FINISH       │
         │                                                                            │
         └─ Can answer directly ──▶ FINISH                                           │
                                                                                      │
                                                                                      ▼
                                                                            FINAL RESPONSE

Each agent autonomously decides:
- When to handoff to another agent
- Which agent to handoff to
- When to provide the final answer
```

## The Agents

### 1. Researcher Agent
**Role**: Information gatherer with handoff capabilities

**Specialization:**
- Queries SQLite vector store
- Finds relevant company information
- Retrieves policy documents

**Available Handoff Tools:**
- `transfer_to_writer(reason)`: When information needs to be formatted
- `transfer_to_fact_checker(reason)`: When information needs verification

**Decision Making:**
1. Gathers information from knowledge base
2. Uses structured output to decide: handoff or final answer
3. **Special rule**: All factual information must be transferred to fact checker for proofreading
4. Creates tool call message if handoff needed
5. Otherwise provides final answer (rare - usually handoffs for verification)

**Key Function**: `researcher_agent()` in network.py:171

### 2. Writer Agent
**Role**: Content creator with handoff capabilities

**Specialization:**
- Creates well-formatted responses
- Synthesizes information into readable content
- Maintains professional tone

**Available Handoff Tools:**
- `transfer_to_researcher(reason)`: When more information is needed
- `transfer_to_fact_checker(reason)`: When content needs verification

**Decision Making:**
1. Gathers context from previous agent messages
2. Uses structured output to decide: handoff or final answer
3. Creates tool call message if handoff needed
4. Otherwise provides final formatted response

**Key Function**: `writer_agent()` in network.py:341

### 3. Fact Checker Agent
**Role**: Quality assurance with handoff capabilities

**Specialization:**
- Verifies factual accuracy (demonstration mode)
- Validates content quality
- Ensures compliance

**Available Handoff Tools:**
- `transfer_to_researcher(reason)`: When more information needed for verification
- `transfer_to_writer(reason)`: When content needs revision

**Decision Making:**
1. Performs verification checks (demonstration mode with mock output)
2. Uses structured output to decide: approve or request changes
3. Can handoff to researcher for more info or writer for revision
4. Otherwise approves and provides final answer

**Note**: This is a **demonstration agent** with mocked output.

**Key Function**: `fact_checker_agent()` in network.py:489

## Handoff Mechanism

### How Handoffs Work

Handoffs are implemented using **structured output** for clean decision-making:

```python
class HandoffDecision(BaseModel):
    """Structured decision for agent handoffs."""
    handoff_to: Optional[Literal["researcher", "writer", "fact_checker"]] = Field(
        default=None,
        description="Which agent to handoff to, or None if providing final answer"
    )
    reason: Optional[str] = Field(
        default=None,
        description="Why you're handing off to this agent"
    )
    final_answer: Optional[str] = Field(
        default=None,
        description="The final answer to provide (set this if NOT handing off)"
    )
```

Handoff tools are defined for execution:

```python
@tool
def transfer_to_writer(reason: str) -> str:
    """
    Transfer control to the Writer agent.

    Args:
        reason: Why you're transferring to the Writer

    Returns:
        Confirmation message
    """
    return f"🔄 Handoff to Writer: {reason}"
```

### Agent Decision Process

Each agent uses structured output to make clean decisions:

```python
# In researcher_agent()
llm_with_structure = llm.with_structured_output(HandoffDecision)

# LLM decides using structured format
decision: HandoffDecision = llm_with_structure.invoke([
    SystemMessage(content="You must decide: provide final_answer OR handoff..."),
    HumanMessage(content="Based on research, what should I do?")
])

# Check agent's decision
if decision.final_answer:
    # Agent providing final answer
    return {
        "messages": [AIMessage(content=f"[Researcher] {decision.final_answer}")],
        "next_agent": "FINISH"
    }
elif decision.handoff_to and decision.reason:
    # Agent chose to handoff - create tool call message
    tool_call_msg = AIMessage(
        content=f"[Researcher] {research_findings}",
        tool_calls=[{
            "name": f"transfer_to_{decision.handoff_to}",
            "args": {"reason": decision.reason},
            "id": f"call_{decision.handoff_to}"
        }]
    )
    return {"messages": [tool_call_msg]}
```

### Tool Execution and Routing

The `handle_tool_calls()` function processes handoffs:

```python
def handle_tool_calls(state: NetworkState) -> dict:
    tool_call = state.messages[-1].tool_calls[0]
    tool_name = tool_call['name']
    reason = tool_call['args'].get('reason')

    # Route to appropriate agent
    next_agent_map = {
        'transfer_to_researcher': 'researcher',
        'transfer_to_writer': 'writer',
        'transfer_to_fact_checker': 'fact_checker',
    }

    return {
        "next_agent": next_agent_map[tool_name],
        "handoff_reason": reason
    }
```

**Key Function**: `handle_tool_calls()` in network.py:641

## Workflow Execution

### Example Flow: "Who is the CEO of DevFest Corp?"

**Step 1: Researcher Starts**
```
🔍 RESEARCHER AGENT working...
   ✓ Retrieved 3 relevant documents
   ✓ Found CEO information

Decision (via structured output):
   handoff_to: "writer"
   reason: "Need to format research findings into professional response"
   
Action: Creates tool call message to transfer to Writer
```

**Step 2: Handoff to Writer**
```
🔄 Processing handoff: transfer_to_writer
   Reason: Need to format research findings into professional response

✍️  WRITER AGENT working...
   ℹ️  Received handoff: Need to format research findings into professional response
   ✓ Synthesizing research into readable content
   ✓ Creating professional response

Decision (via structured output):
   handoff_to: "fact_checker"
   reason: "Please verify the CEO information"
   
Action: Creates tool call message to transfer to Fact Checker
```

**Step 3: Handoff to Fact Checker**
```
🔄 Processing handoff: transfer_to_fact_checker
   Reason: Please verify the CEO information

✅ FACT CHECKER AGENT working...
   ℹ️  Received handoff: Please verify the CEO information
   ✓ Verification completed
   ✓ All checks passed

Decision (via structured output):
   final_answer: "Based on verified information, Alex Dupont serves as CEO..."
   handoff_to: None
   
Action: Provides final answer (sets next_agent to "FINISH")
```

**Step 4: Final Response**
```
Final Answer:
[Fact Checker] Based on verified information from the knowledge base, 
Alex Dupont serves as the CEO of DevFest Corp...
```

### Alternative Flow Example

The workflow can vary based on agent decisions:

**Flow 1: Direct Answer**
```
Researcher → (decides info is simple) → Final Answer
```

**Flow 2: Writer Needs More Info**
```
Researcher → Writer → (needs more info) → Researcher → Writer → Final Answer
```

**Flow 3: Multiple Verification Rounds**
```
Researcher → Writer → Fact Checker → (issues found) → Writer → Fact Checker → Final Answer
```

## Implementation Details

### Structured Output for Clean Decisions

The implementation uses `with_structured_output` to ensure agents make clean, unambiguous decisions:

```python
class HandoffDecision(BaseModel):
    """
    Structured decision for agent handoffs.
    
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
```

**Benefits of Structured Output:**
- **Clean separation**: Agent either handoffs OR provides final answer, never both
- **Type safety**: Pydantic validation ensures correct format
- **Clear prompting**: LLM knows exactly what fields to populate
- **Easy processing**: No need to parse free-form text for decisions

**How agents use it:**
```python
# In each agent function
llm_with_structure = llm.with_structured_output(HandoffDecision)

decision: HandoffDecision = llm_with_structure.invoke([
    SystemMessage(content="...decide what to do..."),
    HumanMessage(content="...context...")
])

# Process the structured decision
if decision.final_answer:
    # Provide final answer
    return {
        "messages": [AIMessage(content=f"[Agent] {decision.final_answer}")],
        "next_agent": "FINISH"
    }
elif decision.handoff_to and decision.reason:
    # Create handoff via tool call
    tool_call_msg = AIMessage(
        content=f"[Agent] ...",
        tool_calls=[{
            "name": f"transfer_to_{decision.handoff_to}",
            "args": {"reason": decision.reason},
            "id": f"call_{decision.handoff_to}"
        }]
    )
    return {"messages": [tool_call_msg]}
```

### State Management

```python
class NetworkState(BaseModel):
    """State for the network multi-agent system."""
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
```

**Key Fields:**
- `messages`: Full conversation history including tool calls
- `next_agent`: Which agent should execute next
- `handoff_reason`: Why the handoff occurred (for transparency)

### Routing Logic

```python
def route_after_agent(state: NetworkState) -> Literal["tools", "researcher", "writer", "fact_checker", "end"]:
    """Routes based on agent decisions."""
    last_message = state.messages[-1]

    # Check for tool call (handoff request)
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"  # Process the handoff

    # Check if complete
    if state.next_agent == "FINISH":
        return "end"

    # Route to specified agent
    return state.next_agent
```

**Key Function**: `route_after_agent()` in network.py:696

### Graph Structure

```python
workflow = StateGraph(NetworkState)

# Add agent nodes
workflow.add_node("researcher", researcher_agent)
workflow.add_node("writer", writer_agent)
workflow.add_node("fact_checker", fact_checker_agent)
workflow.add_node("tools", handle_tool_calls)

# Entry point
workflow.set_entry_point("researcher")

# Conditional edges from each agent
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
```

**Key Function**: `create_network_graph()` in network.py:728

## Running the Network System

### Prerequisites
Make sure you've run the ingestion script first:
```bash
python3 02_rag_lcel/ingest.py
```

### Basic Mode (Default Model)
```bash
python3 05_network/network.py
```

### Interactive Mode
```bash
python3 05_network/network.py --interactive
```

### Custom Question
```bash
python3 05_network/network.py --question "What is DevFest Corp's vacation policy?"
```

### Thinking Model Mode
Use `qwen3:8b` to see agent reasoning processes:
```bash
python3 05_network/network.py --thinking
```

Or combine flags:
```bash
python3 05_network/network.py --interactive --thinking
```

## When to Use Network Pattern

### ✅ Use Network Pattern When:

1. **Flexible, Adaptive Workflows**
   - Workflow should vary based on task complexity
   - No single "right" order of operations
   - Example: Creative content that may need multiple revision loops

2. **Equal Peer Collaboration**
   - Agents are equals, not hierarchical
   - No clear "boss" or coordinator role
   - Example: Collaborative research where experts discuss together

3. **Dynamic Decision Making**
   - Each agent should independently decide next steps
   - Workflow emerges naturally from task needs
   - Example: Problem-solving where solution path is unclear

4. **Bidirectional Communication**
   - Agents may need to go back and forth
   - Writer → Researcher → Writer → Researcher (looping)
   - Example: Iterative refinement processes

5. **Distributed Control**
   - No single point of failure
   - System adapts if agents are unavailable
   - Example: Resilient systems where any agent can start/finish

### ❌ Don't Use Network Pattern When:

1. **Fixed Workflows Required**
   - Must follow strict process (e.g., compliance)
   - Predictable execution order needed
   - **Use Supervisor pattern instead**

2. **Need Centralized Oversight**
   - Want single point of control
   - Need to guarantee all steps execute
   - **Use Supervisor pattern instead**

3. **Simple Tasks**
   - Single agent with tools is sufficient
   - **Use ReAct pattern instead** (Step 3)

4. **Performance Critical**
   - Network adds overhead (agent decision-making at each step)
   - **Use simpler patterns for speed**

## Comparing Agent Patterns

| Feature | ReAct Agent | Supervisor | Network |
|---------|-------------|------------|---------|
| **Control** | Autonomous | Centralized | Distributed |
| **Structure** | Single agent | Hierarchical | Peer-to-peer |
| **Decision-Making** | Agent chooses tools | Supervisor delegates | Each agent decides |
| **Communication** | Agent ↔ Tools | Supervisor ↔ Workers | Agent ↔ Agent |
| **Workflow** | Dynamic (tool-based) | Structured (fixed) | Emergent (adaptive) |
| **Flexibility** | Medium | Low | High |
| **Predictability** | Medium | High | Low |
| **Best For** | Simple tasks | Complex workflows | Adaptive collaboration |
| **Overhead** | Low | Medium | Higher |
| **Scalability** | Limited | High | Very High |

## Key Concepts

### 1. Peer-to-Peer Architecture
- All agents are **equal participants**
- No hierarchy or central controller
- Agents make autonomous decisions

### 2. Structured Output for Decisions
- **Uses `with_structured_output` for clean decision-making**
- Each agent returns either `final_answer` OR `handoff_to` + `reason`
- Prevents ambiguous responses and mixed formats
- Type-safe via Pydantic models

### 3. Handoff Tools
- **Tools enable agent-to-agent communication**
- Each handoff includes a `reason` for transparency
- LLM decides when to handoff via structured output
- Tools are executed by the `handle_tool_calls` node

### 4. Emergent Workflows
- **Workflow is not predefined**
- Emerges from agent decisions and task needs
- Different questions may follow different paths

### 5. Autonomous Decision Making
- **Each agent independently decides:**
  - When to involve another agent
  - Which agent to handoff to
  - When to provide final answer
- No external controller makes these decisions

### 6. Bidirectional Handoffs
- **Any agent can handoff to any other agent**
- Researcher ↔ Writer ↔ Fact Checker
- Enables iterative refinement and collaboration

## Advanced Topics

### Adding New Agents to the Network

To add a new agent to the peer network:

1. **Define handoff tool:**
```python
@tool
def transfer_to_analyst(reason: str) -> str:
    """Transfer control to the Analyst agent."""
    return f"🔄 Handoff to Analyst: {reason}"
```

2. **Define agent function:**
```python
def analyst_agent(state: NetworkState) -> dict:
    # Perform analysis
    analysis = perform_analysis(state.messages)

    # Make handoff decision using structured output
    llm_with_structure = llm.with_structured_output(HandoffDecision)

    decision: HandoffDecision = llm_with_structure.invoke([
        SystemMessage(content="Decide if you need to handoff..."),
        HumanMessage(content=f"Analysis: {analysis}\nWhat next?")
    ])

    if decision.final_answer:
        return {
            "messages": [AIMessage(content=f"[Analyst] {decision.final_answer}")],
            "next_agent": "FINISH"
        }
    elif decision.handoff_to and decision.reason:
        tool_call_msg = AIMessage(
            content=f"[Analyst] {analysis}",
            tool_calls=[{
                "name": f"transfer_to_{decision.handoff_to}",
                "args": {"reason": decision.reason},
                "id": f"call_{decision.handoff_to}"
            }]
        )
        return {"messages": [tool_call_msg]}
```

3. **Update HandoffDecision schema:**
```python
class HandoffDecision(BaseModel):
    handoff_to: Optional[Literal["researcher", "writer", "fact_checker", "analyst"]] = Field(
        default=None,
        description="Which agent to handoff to, or None if providing final answer"
    )
    # ... other fields remain the same
```

4. **Add to graph:**
```python
workflow.add_node("analyst", analyst_agent)

# Add conditional edges
workflow.add_conditional_edges(
    "analyst",
    route_after_agent,
    {
        "tools": "tools",
        "researcher": "researcher",
        "writer": "writer",
        "fact_checker": "fact_checker",
        "analyst": "analyst",
        "end": END,
    }
)

# Update tool routing
next_agent_map = {
    # ... existing ...
    'transfer_to_analyst': 'analyst',
}
```

### Preventing Infinite Loops

In network patterns, agents could theoretically loop forever. Strategies to prevent this:

**1. Max Iterations:**
```python
class NetworkState(BaseModel):
    messages: Annotated[list, add_messages]
    next_agent: str
    handoff_reason: str
    iteration_count: int = 0  # Add counter

def route_after_agent(state: NetworkState):
    if state.iteration_count > 10:  # Max 10 iterations
        return "end"
    # ... normal routing ...
```

**2. Track Agent History:**
```python
class NetworkState(BaseModel):
    messages: Annotated[list, add_messages]
    next_agent: str
    agent_history: list = Field(default_factory=list)  # Track agent calls

def route_after_agent(state: NetworkState):
    # Prevent immediate loops (e.g., Writer → Researcher → Writer → Researcher)
    if len(state.agent_history) >= 3:
        if state.agent_history[-3:] == [state.next_agent] * 3:
            return "end"  # Same agent called 3 times in a row
```

**3. Prompt Engineering:**
```python
system_prompt = """You are the Writer agent.

IMPORTANT: Only handoff if absolutely necessary. If you can provide
a complete answer, do so directly. Excessive handoffs waste resources.

You must provide EITHER a final_answer OR a handoff (handoff_to + reason), 
never both, never neither."""
```

### Network vs. Supervisor: When to Choose?

**Choose Network When:**
- Task complexity varies greatly (simple vs. complex questions)
- Need agents to collaborate iteratively
- Want system to adapt dynamically
- Agents should feel like "colleagues" discussing

**Choose Supervisor When:**
- Must ensure all quality steps execute
- Need audit trail of exactly what happened
- Want predictable, repeatable workflows
- Compliance or regulatory requirements

**Real-World Example:**

**Network Pattern** - Creative Writing Assistant:
```
User: "Write a blog post about AI"
Writer: Creates draft
Writer: Calls transfer_to_fact_checker("Check technical accuracy")
Fact Checker: Finds issue
Fact Checker: Calls transfer_to_researcher("Need source for claim X")
Researcher: Finds source
Researcher: Calls transfer_to_writer("Add this citation")
Writer: Updates draft with citation
Writer: Provides final blog post
```
↑ Dynamic, adaptive, iterative

**Supervisor Pattern** - Compliance Document Generation:
```
User: "Generate compliance report"
Supervisor: Routes to Researcher (always first)
Researcher: Gathers data → back to Supervisor
Supervisor: Routes to Writer (always second)
Writer: Creates report → back to Supervisor
Supervisor: Routes to Compliance Checker (always third)
Compliance Checker: Validates → back to Supervisor
Supervisor: Routes to Legal Reviewer (always fourth)
Legal Reviewer: Approves → back to Supervisor
Supervisor: Provides final report
```
↑ Fixed, predictable, auditable

## Code Structure

```
05_network/
├── 05_network.md             # This documentation
└── network.py                 # Network multi-agent implementation
```

## Key Takeaways

1. **Network pattern enables peer-to-peer agent collaboration** without central control
2. **Structured output ensures clean decisions** - agents either handoff OR provide final answer
3. **Handoff tools allow agents to transfer control** to each other dynamically
4. **Each agent autonomously decides** when to handoff and to whom
5. **Workflows emerge naturally** from agent interactions and task needs
6. **More flexible than Supervisor** but less predictable
7. **Use for adaptive, collaborative tasks** where workflow should vary

## Comparison Across All Steps

| Step | Pattern | Control | Communication | Best For |
|------|---------|---------|--------------|----------|
| **Step 1** | Basic LLM | None | Direct Q&A | Simple queries |
| **Step 2** | RAG Chain | Linear | Sequential pipeline | Knowledge retrieval |
| **Step 3** | ReAct Agent | Autonomous | Agent ↔ Tools | Tool-based tasks |
| **Step 4** | Supervisor | Centralized | Hierarchical | Fixed workflows |
| **Step 5** | Network | Distributed | Peer-to-peer | Adaptive collaboration |

## Troubleshooting

**Issue**: "Database not found"
- **Solution**: Run `python3 02_rag_lcel/ingest.py` first

**Issue**: Agents loop infinitely
- **Solution**:
  - Add iteration counter to state
  - Improve agent prompts to reduce unnecessary handoffs
  - Add max iterations check in routing function

**Issue**: Agent not receiving handoff correctly
- **Solution**:
  - Verify tool names match in handoff tools and routing map
  - Check that `handle_tool_calls()` extracts the reason correctly
  - Ensure agent checks for handoff in its logic

**Issue**: Final answer is incomplete
- **Solution**:
  - Check that agents set `next_agent: "FINISH"` when done
  - Verify routing function handles "FINISH" → "end"
  - Ensure agents don't handoff when they should provide final answer

**Issue**: Handoff reason not showing
- **Solution**:
  - Verify `handoff_reason` is being set in `handle_tool_calls()`
  - Check that agents print the reason when receiving handoff

## Next Steps

Congratulations! You've completed all 5 steps of the DevFest 2025 Local LLMs Workshop!

You now understand:
✅ How to run local LLMs with Ollama
✅ How to build RAG systems with vector stores
✅ How to use LCEL for sequential chains
✅ How to build ReAct agents with tool usage
✅ How to implement Supervisor multi-agent systems
✅ How to build Network/Swarm agent systems

### Further Exploration:

1. **Hybrid Patterns:**
   - Combine Supervisor and Network patterns
   - Supervisor of networks (groups of peer agents)
   - Network with occasional supervisor intervention

2. **Advanced Handoffs:**
   - Partial handoffs (share context, don't transfer control)
   - Broadcast handoffs (notify multiple agents)
   - Conditional handoffs (only if X condition met)

3. **State Persistence:**
   - Save network state between sessions
   - Resume conversations
   - Audit trail of all handoffs

4. **Performance Optimization:**
   - Parallel agent execution where possible
   - Cache handoff decisions
   - Smart agent selection based on task type

5. **Real-World Applications:**
   - Customer support with specialist escalation
   - Collaborative content creation
   - Multi-domain research assistants
   - Peer review systems

Happy building! 🚀
