# src/marehmakom/download.py
import os
import gdown
from .config import Config

TOKENIZER_NAME = "labse_tokenizer.json"
CHECKPOINT_NAME = "best_checkpoint.pt"

def _download(file_id: str, dest: str):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    if not os.path.exists(dest):
        if not file_id:
            raise RuntimeError("Google Drive file_id is empty. Set env vars MAREHMAKOM_TOKENIZER_ID / MAREHMAKOM_CHECKPOINT_ID.")
        print(f"[marehmakom] downloading {dest} ...")
        gdown.download(url, dest, quiet=False)
    return dest

def ensure_assets(cfg: Config | None = None):
    cfg = cfg or Config()
    tok_path = os.path.join(cfg.cache_dir, TOKENIZER_NAME)
    ckpt_path = os.path.join(cfg.cache_dir, CHECKPOINT_NAME)
    _download(cfg.tokenizer_file_id, tok_path)
    _download(cfg.checkpoint_file_id, ckpt_path)
    return tok_path, ckpt_path

def main():
    # CLI: `marehmakom-download`
    tok, ckpt = ensure_assets(silent=False)
    print("[marehmakom] assets ready:", tok, ckpt)
