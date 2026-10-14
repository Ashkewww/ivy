import os, dotenv, time
from typing import TypedDict
from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
from azure.identity import DefaultAzureCredential 
dotenv.load_dotenv('.env')

# ---------- Azure Credentials ---------- #
FOUNDRY_ENDPOINT = os.environ['FOUNDRY_ENDPOINT']
FOUNDRY_PROJECT = os.environ['FOUNDRY_PROJECT']
MODEL_DEPLOYMENT_NAME = os.environ['MODEL_DEPLOYMENT_NAME']
EMBEDDING_MODEL = os.environ['EMBEDDING_MODEL']
AZURE_IDENTITY = DefaultAzureCredential()
ASYNC_AZURE_IDENTITY = AsyncDefaultAzureCredential()


# ---------- Postgres Credentials ---------- #
POSTGRES_DATABASE = os.environ['POSTGRES_DATABASE']
POSTGRES_PORT = os.environ['POSTGRES_PORT']
POSTGRES_USER = os.environ['POSTGRES_USER']
POSTGRES_HOST = os.environ['POSTGRES_HOST']


# ---------- Ivy Timezone ---------- #
IVY_TIMEZONE = os.environ.get('IVY_TIMEZONE')  
IVY_HOME_CITY = os.environ.get('IVY_HOME_CITY')


# ---------- Check Credentials ---------- #
if not IVY_TIMEZONE:
    IVY_TIMEZONE = time.tzname[time.daylight]
    
if not FOUNDRY_ENDPOINT:
    raise ValueError(f"{FOUNDRY_ENDPOINT} is not set")

if not FOUNDRY_PROJECT:
    raise ValueError(f"{FOUNDRY_PROJECT} is not set")

if not MODEL_DEPLOYMENT_NAME:
    raise ValueError(f"{MODEL_DEPLOYMENT_NAME} is not set")

if not EMBEDDING_MODEL:
    raise ValueError(f"{EMBEDDING_MODEL} is not set")

if not POSTGRES_DATABASE:
    raise ValueError(f"{POSTGRES_DATABASE} is not set")

if not POSTGRES_PORT:
    raise ValueError(f"{POSTGRES_PORT} is not set")

if not POSTGRES_USER:
    raise ValueError(f"{POSTGRES_USER} is not set")

if not POSTGRES_HOST:
    raise ValueError(f"{POSTGRES_HOST} is not set")

class Credentials(TypedDict):
    FOUNDRY_ENDPOINT = FOUNDRY_ENDPOINT
    FOUNDRY_PROJECT =  FOUNDRY_PROJECT
    MODEL_DEPLOYMENT_NAME = MODEL_DEPLOYMENT_NAME
    EMBEDDING_MODEL = EMBEDDING_MODEL
    AZURE_IDENTITY = AZURE_IDENTITY
    ASYNC_AZURE_IDENTITY = ASYNC_AZURE_IDENTITY
    IVY_TIMEZONE = IVY_TIMEZONE
    POSTGRES_DATABASE = POSTGRES_DATABASE
    POSTGRES_PORT = POSTGRES_PORT
    POSTGRES_USER = POSTGRES_USER
    POSTGRES_HOST = POSTGRES_HOST
    IVY_HOME_CITY = IVY_HOME_CITY



