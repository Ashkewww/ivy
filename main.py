import logging
import dotenv
dotenv.load_dotenv(".env")

from langchain_core.messages import HumanMessage

from utils.graph import ivy_graph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- chat loop
def main():
    config = {"configurable": {"thread_id": "dev-thread-1"}}
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
