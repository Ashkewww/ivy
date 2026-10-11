

import os
import sqlite3


sqlite_connection = sqlite3.connect(
    os.environ["SQLITE_PATH"], check_same_thread=False
)
