# Step 4: Supervisor Multi-Agent Pattern

## Goal
Learn how to build a multi-agent system with centralized control where a Supervisor agent orchestrates multiple specialized worker agents. This step demonstrates hierarchical agent coordination for complex tasks that require multiple domains of expertise.

## Prerequisites
Before starting this step, ensure you have:
- Completed Step 3 (LangGraph ReAct)
- The knowledge base ingested into SQLite (run `02_rag_lcel/ingest.py`)
- Ollama running with models pulled
- Python virtual environment activated

## What You'll Learn
- The difference between **ReAct agents** (autonomous) and **Supervisor pattern** (centralized control)
- How to build a **multi-agent system** with specialized workers
- How a **Supervisor delegates tasks** to appropriate specialists
- When to use the **Supervisor pattern** vs. single agents
- How to implement **conditional routing** based on agent decisions

## ReAct Agent vs. Supervisor Pattern

### ReAct Agent (Step 3)
```
┌──────────────────────────────────────────────────────────┐
│                    Single Agent + Tools                   │
└──────────────────────────────────────────────────────────┘

User Query: "Who is the CEO?"
   │
   ▼
┌──────────┐       ┌──────────┐
│  Agent   │──────▶│  Tool 1  │
│          │       │  Tool 2  │
│ (Decides │       │  Tool 3  │
│  which   │◀──────│          │
│  tool)   │       └──────────┘
└────┬─────┘
     │
     ▼
Response to User
```

**Characteristics:**
- **Single agent** makes all decisions
- **Autonomous tool selection** based on query
- **Direct tool execution** without coordination
- **Best for**: Simple tasks, clear tool purposes, independent operations

### Supervisor Pattern (Step 4)
```
┌──────────────────────────────────────────────────────────┐
│          Multi-Agent System with Supervisor              │
└──────────────────────────────────────────────────────────┘

User Query: "Who is the CEO?"
   │
   ▼
┌─────────────┐
│ SUPERVISOR  │──── Analyzes query
│             │     Delegates to workers
└──────┬──────┘     Synthesizes results
       │
       ├───────────────┬───────────────┬──────────────┐
       │               │               │              │
       ▼               ▼               ▼              ▼
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│Researcher│   │  Writer  │   │  Fact    │   │ [Other   │
│  Agent   │   │  Agent   │   │ Checker  │   │ Workers] │
└────┬─────┘   └────┬─────┘   └────┬─────┘   └──────────┘
     │              │              │
     └──────────────┴──────────────┘
                    │
                    ▼
            Back to Supervisor
                    │
                    ▼
          Final Synthesized Response
```

**Characteristics:**
- **Centralized control** through Supervisor
- **Specialized workers** with specific expertise
- **Explicit delegation** (Supervisor chooses workers)
- **Return-to-supervisor** architecture (workers always report back)
- **Best for**: Complex tasks, multiple specialists needed, coordinated workflows

## The Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              SUPERVISOR MULTI-AGENT SYSTEM                   │
└─────────────────────────────────────────────────────────────┘

USER QUERY: "Who is the CEO of DevFest Corp?"
   │
   ▼
┌────────────────┐
│  SUPERVISOR    │ ← Entry point, analyzes task
└────────┬───────┘
         │
         ▼
    [Decision: Need information?]
         │
         ├─ YES ──▶ ┌──────────────┐
         │          │  RESEARCHER  │ ← Queries knowledge base
         │          └──────┬───────┘
         │                 │
         │                 └─────┐ (return to supervisor)
         │                       │
         ▼                       ▼
    [Decision: Need content?]  SUPERVISOR
         │
         ├─ YES ──▶ ┌──────────────┐
         │          │    WRITER    │ ← Generates response
         │          └──────┬───────┘
         │                 │
         │                 └─────┐ (return to supervisor)
         │                       │
         ▼                       ▼
    [Decision: Need verification?] SUPERVISOR
         │
         ├─ YES ──▶ ┌──────────────┐
         │          │ FACT CHECKER │ ← Verifies accuracy
         │          └──────┬───────┘
         │                 │
         │                 └─────┐ (return to supervisor)
         │                       │
         ▼                       ▼
    [All tasks complete?]   SUPERVISOR
         │
         ├─ YES ──▶ Synthesize final answer
         │
         ▼
    FINAL RESPONSE
```

## The Agents

### 1. Supervisor Agent (Orchestrator)
**Role**: Central controller that coordinates all other agents

**Responsibilities:**
- Analyzes user queries
- Determines which workers are needed
- Delegates tasks to specialized agents
- Tracks workflow progress
- Synthesizes final answers

**Decision Logic:**
```python
if not has_researcher_output:
    # First step: gather information
    return {"next_agent": "researcher"}
elif has_researcher_output and not has_writer_output:
    # Second step: write the response
    return {"next_agent": "writer"}
elif has_writer_output and not has_fact_checker_output:
    # Third step: verify the response
    return {"next_agent": "fact_checker"}
else:
    # All workers completed - synthesize final answer
    return {"next_agent": "FINISH"}
```

**Key Function**: `supervisor_agent()` in supervisor.py:280

### 2. Researcher Agent (Information Gatherer)
**Role**: Retrieves information from the knowledge base

**Specialization:**
- Queries SQLite vector store
- Finds relevant company information
- Retrieves policy documents
- Extracts organizational details

**How it works:**
1. Receives task from Supervisor
2. Queries the DevFest Corp knowledge base
3. Performs similarity search (top 3 results)
4. Returns findings to Supervisor

**Key Function**: `researcher_agent()` in supervisor.py:80

### 3. Writer Agent (Content Generator)
**Role**: Creates well-formatted, professional responses

**Specialization:**
- Synthesizes information into readable content
- Maintains consistent tone and style
- Creates clear, concise answers
- Formats content professionally

**How it works:**
1. Receives research context from Supervisor
2. Uses LLM to craft response
3. Focuses on clarity and professionalism
4. Returns written content to Supervisor

**Key Function**: `writer_agent()` in supervisor.py:155

### 4. Fact Checker Agent (Quality Assurance)
**Role**: Validates information and ensures quality

**Specialization:**
- Verifies factual accuracy
- Checks policy compliance
- Identifies inconsistencies
- Ensures quality standards

**Note**: This is a **demonstration agent** with mocked output. In production, this would:
- Query external fact-checking APIs
- Verify against trusted data sources
- Run compliance rule engines
- Check citations and references

**Key Function**: `fact_checker_agent()` in supervisor.py:218

## Workflow Execution

### Example Flow: "Who is the CEO of DevFest Corp?"

**Step 1: Supervisor Receives Query**
```
👔 SUPERVISOR: Analyzing query...
Decision: Need information → Route to RESEARCHER
```

**Step 2: Researcher Gathers Information**
```
🔍 RESEARCHER: Querying knowledge base...
   ✓ Retrieved 3 relevant documents
   ✓ Found: CEO information
Returns: "[Researcher] I found the following information:
         Alex Dupont is the CEO of DevFest Corp..."
```

**Step 3: Supervisor Re-evaluates**
```
👔 SUPERVISOR: Researcher completed. Analyzing...
Decision: Have info, need content → Route to WRITER
```

**Step 4: Writer Crafts Response**
```
✍️  WRITER: Generating content...
   ✓ Synthesizing research findings
   ✓ Creating professional response
Returns: "[Writer] Alex Dupont serves as the CEO of
         DevFest Corp, leading the company's strategic..."
```

**Step 5: Supervisor Re-evaluates**
```
👔 SUPERVISOR: Writer completed. Analyzing...
Decision: Have content, need verification → Route to FACT CHECKER
```

**Step 6: Fact Checker Validates**
```
✅ FACT CHECKER: Verifying information...
   ✓ Information Sources: Verified
   ✓ Policy Compliance: Passed
   ✓ Factual Accuracy: Consistent
Returns: "[Fact Checker] Verification Report: APPROVED ✓"
```

**Step 7: Supervisor Synthesizes Final Answer**
```
👔 SUPERVISOR: All workers completed. Synthesizing...
Decision: All tasks complete → FINISH

Final Answer:
Based on the work of our specialized team:

Alex Dupont serves as the CEO of DevFest Corp,
leading the company's strategic...

---
(This answer was produced through coordination of
Researcher, Writer, and Fact Checker agents)
```

## Implementation Details

### State Management with Pydantic

```python
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
```

**Benefits:**
- **Type safety**: Automatic validation
- **Clear structure**: Documented fields
- **Message tracking**: Full conversation history
- **Routing control**: `next_agent` field determines flow

### Routing Logic

The `route_to_agent()` function implements conditional routing:

```python
def route_to_agent(state: SupervisorState) -> Literal["researcher", "writer", "fact_checker", "supervisor", "end"]:
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
```

**Key Function**: `route_to_agent()` in supervisor.py:354

### Graph Structure

```python
workflow = StateGraph(SupervisorState)

# Add all nodes
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("writer", writer_agent)
workflow.add_node("fact_checker", fact_checker_agent)

# Set entry point
workflow.set_entry_point("supervisor")

# Conditional edges from supervisor
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
```

**Key Function**: `create_supervisor_graph()` in supervisor.py:384

## Running the Supervisor System

### Prerequisites
Make sure you've run the ingestion script first:
```bash
python3 02_rag_lcel/ingest.py
```

### Basic Mode (Default Model)
```bash
python3 04_supervisor/supervisor.py
```

### Interactive Mode
```bash
python3 04_supervisor/supervisor.py --interactive
```

### Custom Question
```bash
python3 04_supervisor/supervisor.py --question "What is DevFest Corp's vacation policy?"
```

### Thinking Model Mode
Use `qwen3:8b` to see supervisor's reasoning process:
```bash
python3 04_supervisor/supervisor.py --thinking
```

Or combine flags:
```bash
python3 04_supervisor/supervisor.py --interactive --thinking
```

## When to Use Supervisor Pattern

### ✅ Use Supervisor Pattern When:

1. **Multiple Domains of Expertise**
   - Task requires specialized knowledge in different areas
   - Example: Legal review + Financial analysis + Content writing

2. **Complex Workflows with Coordination**
   - Tasks must be executed in a specific order
   - Outputs from one agent feed into another
   - Example: Research → Write → Review → Publish

3. **Quality Control Required**
   - Need verification and validation steps
   - Multiple checks before final output
   - Example: Content generation with fact-checking

4. **Explicit Control Over Workflow**
   - You want deterministic delegation logic
   - Need to enforce specific processes
   - Example: Compliance-heavy industries

5. **Resource Management**
   - Different workers have different costs/capabilities
   - Need to optimize which agent handles what
   - Example: Expensive specialized models vs. cheaper general models

### ❌ Don't Use Supervisor Pattern When:

1. **Simple, Single-Domain Tasks**
   - A single agent with tools is sufficient
   - Use ReAct pattern instead (Step 3)

2. **Independent Parallel Operations**
   - Tasks don't need coordination
   - Workers don't depend on each other's outputs
   - Use parallel tool execution instead

3. **Autonomous Decision-Making Needed**
   - Agent should freely choose approach
   - No fixed workflow required
   - Use ReAct or autonomous agents

4. **Performance is Critical**
   - Supervisor adds overhead (routing + synthesis)
   - Single agent is faster for simple tasks

## Comparing Agent Patterns

| Feature | ReAct Agent | Supervisor Pattern |
|---------|-------------|-------------------|
| **Control** | Autonomous | Centralized |
| **Decision-Making** | Agent chooses tools | Supervisor delegates |
| **Coordination** | None | Explicit |
| **Specialization** | Tools | Specialized agents |
| **Workflow** | Dynamic | Structured |
| **Best For** | Simple tasks | Complex multi-step tasks |
| **Overhead** | Low | Higher (routing + synthesis) |
| **Scalability** | Limited | High (add more workers) |

## Key Concepts

### 1. Hierarchical Structure
- **Supervisor** at top level makes strategic decisions
- **Workers** at lower level execute specialized tasks
- Clear chain of command

### 2. Return-to-Supervisor Architecture
- Workers **always** report back to supervisor
- Workers **never** communicate directly with each other
- Supervisor maintains complete control

### 3. Worker Specialization
- Each worker has a **specific domain** of expertise
- Workers use **specialized tools** or knowledge
- Better than one generalist agent for complex tasks

### 4. State-Based Routing
- Supervisor tracks **what's been done** via state
- Routes based on **completion status**
- Ensures all required steps are executed

### 5. Final Synthesis
- Supervisor **combines** all worker outputs
- Creates **coherent final response**
- Adds context about the multi-agent process

## Advanced Topics

### Adding New Workers

To add a new specialized worker:

1. **Define the worker function:**
```python
def analyst_agent(state: SupervisorState) -> dict:
    # Perform analysis
    result = analyze(state.messages)
    return {
        "messages": [AIMessage(content=f"[Analyst] {result}")],
        "next_agent": "supervisor"
    }
```

2. **Update supervisor decision logic:**
```python
def supervisor_agent(state: SupervisorState) -> dict:
    # ... existing logic ...

    elif needs_analysis and not has_analyst_output:
        return {"next_agent": "analyst"}
```

3. **Add to graph:**
```python
workflow.add_node("analyst", analyst_agent)
workflow.add_edge("analyst", "supervisor")
workflow.add_conditional_edges(
    "supervisor",
    route_to_agent,
    {
        # ... existing routes ...
        "analyst": "analyst",
    }
)
```

### Dynamic vs. Static Delegation

**Static Delegation** (Current Implementation):
- Fixed workflow: Researcher → Writer → Fact Checker
- Always executes in same order
- Predictable and reliable

**Dynamic Delegation** (Advanced):
- Supervisor uses LLM to decide next agent
- Adapts workflow based on query
- More flexible but less predictable

Example dynamic delegation:
```python
def supervisor_agent(state: SupervisorState) -> dict:
    # Ask LLM which agent to use next
    prompt = f"Given the task and current state, which agent should handle the next step? Options: {agent_list}"
    decision = llm.invoke(prompt)
    return {"next_agent": decision.content}
```

## Code Structure

```
04_supervisor/
├── 04_supervisor.md         # This documentation
└── supervisor.py             # Supervisor multi-agent implementation
```

## Key Takeaways

1. **Supervisor pattern provides centralized control** over multi-agent systems
2. **Specialized workers** handle specific domains better than generalists
3. **Return-to-supervisor architecture** ensures coordination and control
4. **State-based routing** enables complex, multi-step workflows
5. **Use for complex tasks** requiring multiple areas of expertise
6. **Trade-off**: More overhead but better coordination than single agents

## Comparison with Previous Steps

| Step | Pattern | Control | Tools/Agents | Best For |
|------|---------|---------|--------------|----------|
| **Step 1** | Basic LLM | None | None | Simple queries |
| **Step 2** | RAG Chain | Linear | RAG pipeline | Knowledge retrieval |
| **Step 3** | ReAct Agent | Autonomous | Multiple tools | Tool-based tasks |
| **Step 4** | Supervisor | Centralized | Multiple agents | Complex workflows |

## Troubleshooting

**Issue**: "Database not found"
- **Solution**: Run `python3 02_rag_lcel/ingest.py` first to create the knowledge base

**Issue**: Supervisor loops infinitely
- **Solution**: Check that workers always return `"next_agent": "supervisor"` in their output

**Issue**: Worker not being called
- **Solution**:
  - Verify the worker node is added to the graph
  - Check supervisor decision logic includes the worker
  - Ensure routing function handles the worker's name

**Issue**: Empty or incorrect final answer
- **Solution**:
  - Check that each worker returns properly formatted messages
  - Verify supervisor synthesis logic extracts the right content
  - Ensure workers tag their output (e.g., "[Researcher]", "[Writer]")

**Issue**: Thinking model not showing reasoning
- **Solution**: Use `--thinking` flag and ensure you're using `qwen3:8b`

## Next Steps

Congratulations! You've completed the DevFest 2025 Local LLMs Workshop. You now understand:

✅ How to run local LLMs with Ollama
✅ How to build RAG systems with vector stores
✅ How to use LCEL for sequential chains
✅ How to build ReAct agents with tool usage
✅ How to implement multi-agent systems with supervisors

### Further Exploration:

1. **Experiment with different workflows:**
   - Try different worker combinations
   - Implement dynamic delegation
   - Add new specialized workers

2. **Optimize performance:**
   - Cache embeddings
   - Batch similar queries
   - Use smaller models for simple workers

3. **Production considerations:**
   - Add error handling and retries
   - Implement logging and monitoring
   - Add authentication and rate limiting

4. **Advanced patterns:**
   - Peer-to-peer agent communication
   - Hierarchical supervisors (supervisor of supervisors)
   - Feedback loops and self-correction

5. **Real-world applications:**
   - Customer support systems
   - Content creation pipelines
   - Research and analysis tools
   - Compliance and audit systems

## Resources

- **LangGraph Documentation**: https://langchain-ai.github.io/langgraph/
- **LangChain Documentation**: https://python.langchain.com/
- **Ollama Models**: https://ollama.com/library
- **Workshop Repository**: Check the README for more examples

Happy building! 🚀
