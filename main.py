import logging
import os
import dotenv
dotenv.load_dotenv(".env")

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore

from utils.graph import build_graph
from utils.models import EMBEDDINGS

logger = logging.getLogger(__name__)

# ---------- Chat loop ---------- #
async def main():
    config = {"configurable": {"thread_id": "dev-thread-1"}}

    DB_URL = os.environ['IVY_DATABASE_URL']
    if DB_URL:
        async with (
            AsyncPostgresSaver.from_conn_string(DB_URL) as checkpointer,
            AsyncPostgresStore.from_conn_string(DB_URL, index={"embed": EMBEDDINGS, "dims": 1536}) as store 
            ):
            store.setup()
            #checkpointer.setup() 
            
            ivy_graph = build_graph(checkpointer=checkpointer, store=store)
    
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
