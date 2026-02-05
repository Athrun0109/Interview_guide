import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Fix OpenMP duplicate library conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add nvidia DLL paths so CTranslate2 can find cublas64_12.dll
_nvidia_dir = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
if os.path.isdir(_nvidia_dir):
    for pkg in os.listdir(_nvidia_dir):
        bin_dir = os.path.join(_nvidia_dir, pkg, "bin")
        if os.path.isdir(bin_dir):
            os.add_dll_directory(bin_dir)
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

# Whisper settings
WHISPER_MODEL_SIZE = "large-v3"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"

# Gemini settings
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_TEMPERATURE = 0.7
GEMINI_MAX_OUTPUT_TOKENS = 4096

# Serper settings
SERPER_ENDPOINT = "https://google.serper.dev/search"


def load_env_keys() -> dict:
    """Load API keys from environment variables."""
    return {
        "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
        "serper_api_key": os.getenv("SERPER_API_KEY", ""),
        "hf_token": os.getenv("HF_TOKEN", ""),
    }
