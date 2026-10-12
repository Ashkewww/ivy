import os

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_openai import ChatOpenAI
from langchain_azure_ai.embeddings import AzureAIOpenAIApiEmbeddingsModel


token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://ai.azure.com/.default"
)

EMBEDDINGS = AzureAIOpenAIApiEmbeddingsModel(
    project_endpoint=os.environ["FOUNDRY_PROJECT"],
    credential=DefaultAzureCredential(),
    model=os.environ["EMBEDDING_MODEL"],
)


LLM = ChatOpenAI(
    base_url=os.environ["FOUNDRY_ENDPOINT"],
    api_key=token_provider,
    model=os.environ["MODEL_DEPLOYMENT_NAME"],
)
