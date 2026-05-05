from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "demo_data"

MODEL_NAME = "Qwen3-8B"
OPENAI_API_BASE = "http://127.0.0.1:6006/v1"
OPENAI_API_KEY = "EMPTY"
TEMPERATURE = 0.3
MAX_MODEL_LEN = 8192

APP_TITLE = "Security Agent Demo"
APP_VERSION = "0.1.0"
