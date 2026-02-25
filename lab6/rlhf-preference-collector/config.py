import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

APP_TITLE = "RLHF Preference Collector"

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

LOCAL_DATA_FILE = Path(os.getenv("LOCAL_DATA_FILE", "preference_data.jsonl"))
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "1024"))

TEMPERATURE_1 = 0.7
TEMPERATURE_2 = 0.9
