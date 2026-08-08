import os
import tempfile

# Set writable SQLite path for serverless environments (Vercel /tmp)
if "VERCEL" in os.environ or os.getenv("SQLITE_PATH", "").startswith("data/"):
    tmp_db = os.path.join(tempfile.gettempdir(), "quantedge.db")
    os.environ["SQLITE_PATH"] = tmp_db

from quantedge.api.app import app
