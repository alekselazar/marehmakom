# src/marehmakom/config.py
import os
from dataclasses import dataclass

DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/marehmakom")

@dataclass
class Config:
    # Configure these via env vars or change defaults here
    tokenizer_file_id: str = os.getenv("MAREHMAKOM_TOKENIZER_ID", "18b1AB_xCz7WQuHtaW79EH3mDaA-HfgUx")
    checkpoint_file_id: str = os.getenv("MAREHMAKOM_CHECKPOINT_ID", "1Igva-lLvtPp-UbXW5_1kDN10N7FMd1H-")
    base_model_name: str = os.getenv("MAREHMAKOM_BASE", "sentence-transformers/LaBSE")
    cache_dir: str = os.getenv("MAREHMAKOM_CACHE", DEFAULT_CACHE_DIR)
    dropout: float = float(os.getenv("MAREHMAKOM_DROPOUT", "0.1"))
    max_query_len: int = int(os.getenv("MAREHMAKOM_MAX_Q", "128"))
    max_target_len: int = int(os.getenv("MAREHMAKOM_MAX_T", "480"))
    threshold: float = float(os.getenv("MAREHMAKOM_THRESHOLD", "0.5"))
    smooth_k: int = int(os.getenv("MAREHMAKOM_SMOOTH_K", "3"))
