import os

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_openai import ChatOpenAI
from langchain_azure_ai.embeddings import AzureAIOpenAIApiEmbeddingsModel

from utils.credentials import Credentials


token_provider = get_bearer_token_provider(
    Credentials.AZURE_IDENTITY, "https://ai.azure.com/.default"
)

EMBEDDINGS = AzureAIOpenAIApiEmbeddingsModel(
    project_endpoint=Credentials.FOUNDRY_PROJECT,
    credential=DefaultAzureCredential(),
    model=Credentials.EMBEDDING_MODEL,
)


LLM = ChatOpenAI(
    base_url=Credentials.FOUNDRY_ENDPOINT,
    api_key=token_provider,
    model=Credentials.MODEL_DEPLOYMENT_NAME,
)
