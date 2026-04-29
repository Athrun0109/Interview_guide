import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Fix OpenMP duplicate library conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# PyTorch 2.6+ flipped torch.load() default to weights_only=True, which rejects
# pyannote 3.4's checkpoints (they pickle omegaconf.ListConfig objects).
# Force weights_only=False globally — safe because pyannote weights are from HuggingFace.
import torch as _torch
_original_torch_load = _torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
_torch.load = _torch_load_compat

# Add nvidia DLL paths so CTranslate2 can find cublas64_12.dll
_nvidia_dir = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
if os.path.isdir(_nvidia_dir):
    for pkg in os.listdir(_nvidia_dir):
        bin_dir = os.path.join(_nvidia_dir, pkg, "bin")
        if os.path.isdir(bin_dir):
            os.add_dll_directory(bin_dir)
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

# Point pydub at the standalone ffmpeg/ffprobe shipped under modules/, bypassing
# the conda-installed copies (which suffer from DLL conflicts on Windows).
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_FFMPEG_DIR = os.path.join(_PROJECT_ROOT, "modules")
_FFMPEG_EXE = os.path.join(_FFMPEG_DIR, "ffmpeg.exe")
_FFPROBE_EXE = os.path.join(_FFMPEG_DIR, "ffprobe.exe")

if os.path.isfile(_FFMPEG_EXE) and os.path.isfile(_FFPROBE_EXE):
    # Prepend to PATH so any subprocess (whisperx, ctranslate2) also finds the right binaries first
    os.environ["PATH"] = _FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

    from pydub import AudioSegment
    AudioSegment.converter = _FFMPEG_EXE
    AudioSegment.ffmpeg = _FFMPEG_EXE
    AudioSegment.ffprobe = _FFPROBE_EXE

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
