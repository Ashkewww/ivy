import os
import sqlite3
from dotenv import load_dotenv
load_dotenv()



sqlite_connection = sqlite3.connect(
    os.environ["SQLITE_PATH"], check_same_thread=False
)
