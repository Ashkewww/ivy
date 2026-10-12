import os
import sqlite3

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from utils.datamodels import IvyState
from utils.models import LLM
from utils.persona import IVY_PERSONA
from utils.tools import TOOLS

llm_with_tools = LLM.bind_tools(TOOLS)


# ---------- Nodes ---------- #

def reference_memories(state: IvyState):
    recent_message = state['messages'][-1].content if isinstance(state['messages'][-1], HumanMessage) else None
    
    if recent_message:
        #TODO: Refer the memories here, Semantic Search for the relevant memories
        pass
    
def fetch_summary(state: IvyState):
    #TODO: Fetch summaries from PLSQL
    pass
    
def ivy_node(state: IvyState):
    system = SystemMessage(content=IVY_PERSONA.format(person=state["person"]))
    response = llm_with_tools.invoke([system] + state["messages"])
    return {"messages": [response]}
    
def tool_condition(state: IvyState):
    #TODO: write a condition to call a tool if the LLM calls for it, other wise go to check if messages left before summarization is less than 1 so summarize or go to  
    pass

def memory_manager(state: IvyState):
    #TODO: Summarizes the messages, stores the summary, cleans up the chat, remembers relevant memories (Asynchronosly)
    pass


# ---------- Graph ---------- #
def build_graph(checkpointer, store):

    builder = StateGraph(IvyState)

    builder.add_node("ivy", ivy_node)
    builder.add_node("tools", ToolNode(TOOLS))  # runs the requested tool(s)

    builder.add_edge(START, "ivy")
    
    builder.add_conditional_edges("ivy", tools_condition)
    
    builder.add_edge("tools", "ivy")

    ivy_graph = builder.compile(checkpointer=checkpointer, store=store)
    
    return ivy_graph


