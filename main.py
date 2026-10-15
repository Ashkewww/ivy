import logging, asyncio, selectors

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from azure_postgresql_auth.psycopg3 import AsyncEntraConnection
from langgraph.store.postgres import AsyncPostgresStore
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

from utils.graph import build_graph
from utils.models import EMBEDDINGS
from utils.credentials import Credentials

logger = logging.getLogger(__name__)

# ---------- Chat loop ---------- #
async def main():
    
    config = {
        "configurable": {
                "thread_id": "dev-thread-1"
            }
        }

    async with AsyncConnectionPool(
        conninfo=(
            f"host={Credentials.POSTGRES_HOST} "
            f"port={Credentials.POSTGRES_PORT} "
            f"dbname={Credentials.POSTGRES_DATABASE} "
            f"user={Credentials.POSTGRES_USER} "
            "sslmode=require"
        ),
        connection_class=AsyncEntraConnection,
        kwargs={
            "credential": Credentials.ASYNC_AZURE_IDENTITY,
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
        min_size=1,
        max_size=5,
    ) as pool:
        
        checkpointer = AsyncPostgresSaver(pool)
        store = AsyncPostgresStore(
                pool,
                index={
                    "embed": EMBEDDINGS,
                    "dims": 1536,
                }
            )
        
        #await store.setup()
        #await checkpointer.setup()
        
        ivy_graph = build_graph(checkpointer=checkpointer, store=store)

        while True:
            text = (await asyncio.to_thread(input, "You: ")).strip()
            
            if text.lower() in {"quit", "exit"}:
                break

            print("Ivy: ", end="", flush=True)
            
            async for chunk, meta in ivy_graph.astream(
                {"messages": [HumanMessage(content=text)], "person": "Ayush-Dev-Thread"},
                config,
                stream_mode="messages"
            ):
                if meta.get("langgraph_node") == "ivy" and isinstance(chunk.content, str):
                    print(chunk.content, end="", flush=True)
            print()
        

if __name__ == "__main__":
    asyncio.run(
        main(),
        loop_factory=lambda: asyncio.SelectorEventLoop(
            selectors.SelectSelector()
        )
    )
