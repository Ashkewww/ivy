from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

class IvyState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    person: str
    summary: str
    relevant_memories: list[str]
    
class Memory(TypedDict):
    content: str
    datetime: str
