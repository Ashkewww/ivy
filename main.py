import logging
import os
import dotenv
dotenv.load_dotenv(".env")

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver  
from langgraph.checkpoint.memory import InMemorySaver

from utils.graph import build_graph

logger = logging.getLogger(__name__)


# ---------- Chat loop ---------- #
def main():
    config = {"configurable": {"thread_id": "dev-thread-1"}}
    
    if os.environ['IVY_DATABASE_URL']:
        with PostgresSaver.from_conn_string(os.environ['IVY_DATABASE_URL']) as checkpointer:
            
            #checkpointer.setup() 
            
            ivy_graph = build_graph(checkpointer=checkpointer)
    
            while True:
                text = input("You: ").strip()
                if text.lower() in {"quit", "exit"}:
                    break

                print("Ivy: ", end="", flush=True)
                for chunk, meta in ivy_graph.stream(
                    {"messages": [HumanMessage(content=text)], "person": "Ayush"},
                    config,
                    stream_mode="messages",
                ):
                    if meta.get("langgraph_node") == "ivy" and isinstance(chunk.content, str):
                        print(chunk.content, end="", flush=True)
                print()
        

if __name__ == "__main__":
    main()
