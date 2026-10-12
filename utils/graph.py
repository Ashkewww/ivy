import os
import sqlite3

from langchain_core.messages import SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver  

from utils.state import IvyState
from utils.llm import LLM
from utils.persona import IVY_PERSONA
from utils.tools import TOOLS

llm_with_tools = LLM.bind_tools(TOOLS)


# ---------- Nodes ---------- #
def ivy_node(state: IvyState) -> dict:
    system = SystemMessage(content=IVY_PERSONA.format(person=state["person"]))
    response = llm_with_tools.invoke([system] + state["messages"])
    return {"messages": [response]}


# ---------- Graph ---------- #
def build_graph(checkpointer):

    builder = StateGraph(IvyState)

    builder.add_node("ivy", ivy_node)
    builder.add_node("tools", ToolNode(TOOLS))  # runs the requested tool(s)

    builder.add_edge(START, "ivy")
    # After Ivy speaks: did she request a tool? -> go to "tools". Otherwise -> END.
    builder.add_conditional_edges("ivy", tools_condition)
    # After a tool runs, go back to Ivy so she can turn the result into a reply.
    builder.add_edge("tools", "ivy")

    ivy_graph = builder.compile(checkpointer=checkpointer)
    
    return ivy_graph


