# Step 3: From Chains to Graphs (ReAct Agent)

## Goal
Transition from linear chains to dynamic graphs with decision-making capabilities. This step demonstrates how to build a ReAct (Reasoning + Acting) agent that can decide when and how to use tools based on user queries.

## Prerequisites
Before starting this step, ensure you have:
- Completed Step 2 (RAG with LCEL)
- The knowledge base ingested into SQLite (run `02_rag_lcel/ingest.py`)
- Ollama running with models pulled
- Python virtual environment activated

## What You'll Learn
- The difference between **Chains** (linear) and **Graphs** (cyclic)
- What a **ReAct Agent** is and how it works (Reasoning + Acting)
- How to define and use **Tools** in LangGraph
- How to build a **state-based graph** with decision logic
- When to use graphs vs. chains

## Chains vs. Graphs

### Chains (LCEL)
```
┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐
│  A  │───▶│  B  │───▶│  C  │───▶│  D  │
└─────┘    └─────┘    └─────┘    └─────┘
```
- **Linear flow**: A → B → C → D
- **Deterministic**: Same path every time
- **No decisions**: Cannot branch based on conditions
- **Use cases**: Fixed pipelines, simple RAG, predictable workflows

### Graphs (LangGraph)
```
         ┌─────────────┐
         │    Agent    │◀──┐
         └──────┬──────┘   │
                │           │
         ┌──────▼──────┐   │
         │   Decision  │   │
         └──┬───────┬──┘   │
            │       │       │
     ┌──────▼──┐ ┌─▼────┐  │
     │ Tool A  │ │Tool B│  │
     └──────┬──┘ └─┬────┘  │
            │      │        │
            └──────┴────────┘
```
- **Cyclic flow**: Can loop back and iterate
- **Dynamic**: Different paths based on conditions
- **Decision-making**: Agent chooses tools or actions
- **Use cases**: Complex agents, multi-step reasoning, tool usage

## What is ReAct?

**ReAct** = **Rea**soning + **Act**ing

It's an agent pattern where the model:
1. **Reasons** about the task ("What information do I need?")
2. **Acts** by choosing a tool ("I'll use lookup_policy")
3. **Observes** the result ("The CEO is Alex Dupont")
4. **Reasons** again ("Now I can answer")
5. **Acts** with the final answer

### ReAct Loop
```
┌──────────────────────────────────────────────────────────┐
│                    ReAct Agent Loop                       │
└──────────────────────────────────────────────────────────┘

1. User Query: "Who is the CEO of DevFest Corp?"
   ↓
2. Agent Reasoning:
   "I need to look up information about DevFest Corp's CEO"
   ↓
3. Agent Action:
   Call tool: lookup_policy("CEO of DevFest Corp")
   ↓
4. Tool Execution:
   Returns: "Alex Dupont is the CEO of DevFest Corp..."
   ↓
5. Agent Observation:
   "I received information about the CEO"
   ↓
6. Agent Reasoning:
   "I now have enough information to answer"
   ↓
7. Agent Response:
   "The CEO of DevFest Corp is Alex Dupont"
```

## The Tools

Our ReAct agent has access to two tools:

### 1. `lookup_policy(query: str)` - Real Tool
- **Purpose**: Queries the SQLite vector store (knowledge base)
- **When to use**: Questions about DevFest Corp policies, people, or information
- **How it works**:
  - Embeds the query
  - Finds similar chunks in the vector store
  - Returns relevant context
- **Example queries**:
  - "Who is the CEO?"
  - "What is the vacation policy?"
  - "Tell me about the company culture"

### 2. `search_tech_events(query: str)` - Fictional Demo Tool
- **Purpose**: Demonstrates multi-tool capabilities with mock data
- **When to use**: Questions about tech conferences or events
- **How it works**: Returns fictional demo content about tech events
- **Example queries**:
  - "What tech conferences are happening this year?"
  - "Find events about AI and machine learning"
- **Note**: This tool returns clearly marked **DEMO CONTENT** to show how agents can work with multiple tools

## The Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 LANGGRAPH REACT ARCHITECTURE                 │
└─────────────────────────────────────────────────────────────┘

USER QUERY
   │
   ▼
┌──────────────┐
│  Agent Node  │  ← Receives query, decides what to do
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│  Should Continue?│  ← Graph decision point
└──────┬───────────┘
       │
       ├─── YES ───▶ ┌────────────────┐
       │             │   Tool Node    │  ← Execute chosen tool
       │             └────────┬───────┘
       │                      │
       │                      └─────┐ (loop back)
       │                            │
       ▼                            ▼
┌──────────────┐           ┌──────────────┐
│     END      │           │  Agent Node  │
└──────────────┘           └──────────────┘
```

## When to Use Chains vs. Graphs

### Use Chains (LCEL) When:
- ✅ Fixed, predictable flow (A → B → C)
- ✅ No branching or decision-making needed
- ✅ Simple RAG pipelines
- ✅ Data transformations (format → process → output)
- ✅ Performance is critical (chains are faster)

**Example**: Basic RAG query
```python
chain = retriever | format_docs | prompt | llm | parser
```

### Use Graphs (LangGraph) When:
- ✅ Need to make decisions ("Should I use a tool?")
- ✅ Multiple possible paths through logic
- ✅ Tool usage required
- ✅ Iterative reasoning (loop until done)
- ✅ Complex multi-agent systems

**Example**: ReAct agent that chooses tools
```python
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_conditional_edges("agent", should_continue)
```

## Implementation Overview

The agent script (`agent.py`) implements:

1. **Pydantic State Management**: Uses Pydantic `BaseModel` for type-safe state tracking
2. **Tools Definition**:
   - `lookup_policy` - Queries SQLite vector store for DevFest Corp information
   - `search_tech_events` - Returns mock tech event data (clearly marked as demo content)
3. **Agent Node**: Decides which tool to use (if any) based on the query
4. **Tool Node**: Built-in `ToolNode` that executes the chosen tool
5. **Conditional Logic**: `should_continue` function determines whether to use tools or end
6. **Graph Execution**: Runs the ReAct loop with streaming to show each step

## Running the Agent

### Prerequisites
Make sure you've run the ingestion script first to create the knowledge base:
```bash
python3 02_rag_lcel/ingest.py
```

### Basic Mode (Default Model)
```bash
python3 03_langgraph_react/agent.py
```

### Interactive Mode
```bash
python3 03_langgraph_react/agent.py --interactive
```

### Custom Question
```bash
python3 03_langgraph_react/agent.py --question "What tech events are happening?"
```

### Thinking Model Mode
```bash
python3 03_langgraph_react/agent.py --thinking
```

Or combine flags:
```bash
python3 03_langgraph_react/agent.py --interactive --thinking
```

## Expected Behavior

### Example 1: Using the Policy Tool
**Question**: "Who is the CEO of DevFest Corp?"

**Agent Flow**:
1. Agent receives query
2. Reasons: "I need information about DevFest Corp"
3. Acts: Calls `lookup_policy("CEO of DevFest Corp")`
4. Observes: Gets context about CEO from knowledge base
5. Reasons: "I have the information needed"
6. Acts: Responds with answer

**Output**: "The CEO of DevFest Corp is Alex Dupont..."

### Example 2: Using the Tech Events Tool
**Question**: "What AI conferences are happening this year?"

**Agent Flow**:
1. Agent receives query
2. Reasons: "This is about tech events"
3. Acts: Calls `search_tech_events("AI conferences")`
4. Observes: Gets demo content about tech events
5. Reasons: "I have information about events"
6. Acts: Responds with answer (including DEMO CONTENT note)

**Output**: "Here are some AI conferences... (DEMO CONTENT)"

### Example 3: No Tool Needed
**Question**: "What is 2 + 2?"

**Agent Flow**:
1. Agent receives query
2. Reasons: "This is a simple math question, no tools needed"
3. Acts: Responds directly with answer

**Output**: "2 + 2 equals 4."

## Understanding Agent State

LangGraph uses **typed state** to track the agent's progress. In this implementation, we use **Pydantic** for state management:

```python
class AgentState(BaseModel):
    messages: Annotated[list, add_messages] = Field(
        default_factory=list,
        description="List of messages in the conversation"
    )
```

**Benefits of using Pydantic for state:**
- **Type safety**: Automatic validation of state structure
- **Documentation**: Field descriptions make state clear
- **IDE support**: Better autocomplete and type hints
- **Serialization**: Easy conversion to/from JSON
- **Validation**: Ensures state stays consistent

- **messages**: List of all messages (user, agent, tool calls, tool responses)
- **add_messages**: Built-in reducer that appends new messages to the list
- **State updates**: Each node returns a dictionary that gets merged with existing state

## Tool Binding

Tools are bound to the LLM, which allows the model to:
1. Understand what tools are available
2. Know when to use each tool
3. Format tool calls correctly

```python
tools = [lookup_policy, search_tech_events]
llm_with_tools = llm.bind_tools(tools)
```

The model then decides:
- **Use a tool**: Returns a tool call message
- **Respond directly**: Returns a text message

## Code Structure

```
03_langgraph_react/
├── 03_langgraph_react.md    # This documentation
└── agent.py                  # ReAct agent implementation
```

## Key Concepts from LangGraph

### 1. StateGraph
Manages the state and flow of the agent:
```python
workflow = StateGraph(AgentState)
```

### 2. Nodes
Functions that process state and return updates:
```python
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
```

### 3. Edges
Define flow between nodes:
```python
workflow.set_entry_point("agent")  # Start here
workflow.add_edge("tools", "agent")  # Tools → Agent
```

### 4. Conditional Edges
Dynamic routing based on state:
```python
workflow.add_conditional_edges(
    "agent",
    should_continue,  # Function that returns next node
)
```

## Key Takeaways

1. **Chains are for pipelines**, graphs are for **agents**
2. **ReAct pattern** enables reasoning and tool use
3. **LangGraph provides state management** for complex flows
4. **Conditional edges** enable dynamic decision-making
5. **Tools extend agent capabilities** beyond the base model
6. **Graphs can loop**, chains cannot

## Next Steps

Once you've successfully:
- Run the ReAct agent with different queries
- Observed how it chooses between tools
- Understood the difference between chains and graphs
- Seen the agent reason about when to use tools

Proceed to **Step 4: Supervisor Pattern** where we'll build multi-agent systems with a supervisor coordinating multiple specialized agents.

## Troubleshooting

**Issue**: "Database not found"
- **Solution**: Run `python3 02_rag_lcel/ingest.py` first to create the knowledge base

**Issue**: "Tool not found" or tool errors
- **Solution**: Ensure tools are properly decorated with `@tool` and bound to the LLM

**Issue**: Agent loops infinitely
- **Solution**: Check the `should_continue` logic and ensure tools return valid responses

**Issue**: Agent doesn't use tools when expected
- **Solution**:
  - Verify tool descriptions are clear
  - Check that tools are bound to the LLM
  - Ensure the model understands when to use each tool

**Issue**: Thinking model not showing reasoning
- **Solution**: Use `--thinking` flag and ensure you're using `qwen3:8b`
