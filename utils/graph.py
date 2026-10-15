import asyncio
import logging
from datetime import datetime
from uuid import uuid4

from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from utils.datamodels import IvyState, Memory
from utils.models import LLM
from utils.persona import IVY_PERSONA
from utils.tools import TOOLS

logger = logging.getLogger(__name__)

llm_with_tools = LLM.bind_tools(TOOLS)

# ---------- Helper Functions ---------- #
def filter_by_human_messages(messages, messages_to_keep: int): 
    human_message_index = [i for i, msg in enumerate(messages) if isinstance(msg, HumanMessage)]
    
    # If the oldest human message is within the last `messages_to_keep` messages, return the last `messages_to_keep` messages
    if human_message_index and human_message_index[0] >= len(human_message_index) - messages_to_keep:
        return messages[-messages_to_keep:]
    

async def _background_memory_task(messages, person, store, existing_summary):
    """Runs after the graph returns. Summarizes trimmed messages and
    extracts facts worth remembering into the store."""
    try:
        # --- 1. Summarize the trimmed conversation ---
        prior = ""
        if existing_summary:
            prior = (
                "\n\nA summary of even older conversation exists:\n"
                f"{existing_summary}\n"
                "Incorporate it into your new summary."
            )

        summary_response = await LLM.ainvoke([
            SystemMessage(content=(
                "Summarize the following conversation concisely. "
                "Capture key topics, decisions, unresolved questions, "
                "and emotional tone. This summary will be prepended to "
                "future conversations so the assistant retains context."
                f"{prior}"
            )),
            *messages,
        ])

        await store.aput(
            ("summaries", person),
            "latest",
            {
                "content": summary_response.content,
                "datetime": datetime.now().isoformat(),
            },
        )

        # --- 2. Extract long-term memories ---
        memory_response = await LLM.ainvoke([
            SystemMessage(content=(
                "Extract key facts, personal preferences, important events, "
                "and noteworthy information about the user from this conversation. "
                "Return each fact on its own line. Only include things genuinely "
                "worth remembering long-term. If there is nothing worth "
                "remembering, return NONE."
            )),
            *messages,
        ])

        if memory_response.content.strip().upper() != "NONE":
            facts = [
                line.strip()
                for line in memory_response.content.strip().split("\n")
                if line.strip()
            ]
            for fact in facts:
                id = str(uuid4())
                await store.aput(
                    ("memories", person),
                    id,
                    {
                        "id": id,
                        "content": fact,
                        "datetime": datetime.now().isoformat(),
                        "source": "conversation",
                        "category": "extracted",
                    },
                    index=["content"],
                )
            logger.info("Background memory task completed: %d memories extracted", len(facts))
        else:
            logger.info("Background memory task completed: no new memories")

    except Exception:
        logger.exception("Background memory task failed")


# ---------- Nodes ---------- #
async def reference_memories(state: IvyState, runtime: Runtime):
    recent_message = state['messages'][-1].content if isinstance(state['messages'][-1], HumanMessage) else None
    store = runtime.store
    if recent_message:
        memories = await store.asearch(
            ("memories", state["person"]),
            query=recent_message,
            limit=5,
        )
        if memories:
            relevant_memories: list[Memory] = [
                Memory(
                    id=item.key,
                    content=item.value["content"],
                    datetime=item.value["datetime"],
                    source=item.value.get("source", ""),
                    category=item.value.get("category", ""),
                )
                for item in memories
            ]
            return {"relevant_memories": relevant_memories}
        
        return {}
    

async def fetch_summary(state: IvyState, runtime: Runtime):
    store = runtime.store
    try:
        result = await store.aget(("summaries", state["person"]), "latest")
        if result:
            return {"summary": result.value.get("content")}
    except Exception:
        logger.exception("Failed to fetch summary")
    return {"summary": ""}
        
    
def ivy_node(state: IvyState, runtime: Runtime):
    system = SystemMessage(content=IVY_PERSONA.format(person=state["person"]))
    
    memories = state.get("relevant_memories", [])
    if memories:
        memory_text = "\n".join(
            f"- Memory: {memory['content']} (from Source: {memory['source']} on DateTime: {memory['datetime']})"
            for index, memory in enumerate(memories) if index < 10
        )
        system.content += f"\n\nHere are some relevant memories:\n{memory_text}"
        
    summary = state.get("summary")
    if summary:
        system.content += f"\n\nHere is a summary of the conversation before the messages cut off:\n{summary}"
    print("System Message Content:\n", system.content)
    response = llm_with_tools.invoke([system] + state["messages"])
    return {"messages": [response]}

    
def tool_condition(state: IvyState) -> str:
    last_message = state["messages"][-1]

    # 1. Does Ivy want to call a tool?
    if getattr(last_message, "tool_calls", []):
        return "tools"

    # 2. Is conversation history over the limit?
    if len(state["messages"]) > 50:
        return "memory_manager"

    # 3. Nothing else to do
    return END


async def memory_manager(state: IvyState, runtime: Runtime):
    """Trims old messages from state and fires a background task
    to summarize them and extract long-term memories."""
    messages = state["messages"]
    person = state["person"]
    store = runtime.store

    # Keep the last ~10 messages, trim the rest
    keep_count = 40
    if len(messages) <= keep_count:
        return {}

    trim_index = len(messages) - keep_count

    # Adjust forward to a HumanMessage boundary so we never split
    # a tool-call / tool-response pair in half
    while trim_index < len(messages) and not isinstance(messages[trim_index], HumanMessage):
        trim_index += 1

    if trim_index >= len(messages) - 1:
        return {}

    messages_to_trim = messages[:trim_index]

    # Remove trimmed messages from state immediately so the graph can finish
    remove_messages = [RemoveMessage(id=msg.id) for msg in messages_to_trim]

    # Fetch any existing summary to build on
    existing_summary = None
    try:
        result = await store.aget(("summaries", person), "latest")
        if result:
            existing_summary = result.value.get("content")
    except Exception:
        pass

    # Fire-and-forget background task
    asyncio.create_task(
        _background_memory_task(messages_to_trim, person, store, existing_summary)
    )

    return {"messages": remove_messages}

# ---------- Graph ---------- #
def build_graph(checkpointer, store):

    builder = StateGraph(IvyState)

    builder.add_node("reference_memories", reference_memories)
    builder.add_node("fetch_summary", fetch_summary)
    builder.add_node("ivy", ivy_node)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_node("memory_manager", memory_manager)

    builder.add_edge(START, "fetch_summary")
    builder.add_edge("fetch_summary", "ivy")
    builder.add_edge(START, "reference_memories")
    builder.add_edge("reference_memories", "ivy")
    builder.add_conditional_edges("ivy", tool_condition)
    builder.add_edge("tools", "ivy")
    builder.add_edge("memory_manager", END)

    ivy_graph = builder.compile(checkpointer=checkpointer, store=store)
    
    return ivy_graph
