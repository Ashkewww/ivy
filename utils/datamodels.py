from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class Memory(TypedDict):
    id: str
    content: str
    datetime: str
    source: str
    category: str  
class IvyState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    person: str
    summary: str
    relevant_memories: list[Memory]  
class Note(TypedDict):
    id: str
    user_id: str
    content: str
    created_at: str
    updated_at: str
