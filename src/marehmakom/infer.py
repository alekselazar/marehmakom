# src/marehmakom/infer.py
from dataclasses import asdict
from typing import List, Tuple, Dict
import torch
from transformers import PreTrainedTokenizerFast
from .config import Config
from .download import ensure_assets, TOKENIZER_NAME, CHECKPOINT_NAME
from .model import MarehMakomModel

def _bin_to_spans(bits: List[int]) -> List[Tuple[int, int]]:
    spans = []; in_run = False; start = 0
    for i, v in enumerate(bits):
        if v and not in_run: in_run, start = True, i
        elif not v and in_run: in_run=False; spans.append((start, i))
    if in_run: spans.append((start, len(bits)))
    return spans

def _median_filter_1d(probs: torch.Tensor, k: int) -> torch.Tensor:
    pad = k // 2
    padded = torch.nn.functional.pad(probs.unsqueeze(0).unsqueeze(0), (pad, pad), mode='replicate')[0,0]
    windows = padded.unfold(0, k, 1)
    return windows.median(dim=1).values

class MarehMakomInference:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        tok_path, ckpt_path = ensure_assets(self.cfg, silent=True)

        # tokenizer
        tok_file = tok_path
        self.tok = PreTrainedTokenizerFast(tokenizer_file=tok_file)
        if self.tok.cls_token is None: self.tok.cls_token = "[CLS]"
        if self.tok.sep_token is None: self.tok.sep_token = "[SEP]"
        if self.tok.pad_token is None: self.tok.add_special_tokens({"pad_token": "[PAD]"})
        self.CLS, self.SEP, self.PAD = self.tok.cls_token_id, self.tok.sep_token_id, self.tok.pad_token_id

        # model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MarehMakomModel(self.cfg.base_model_name, dropout=self.cfg.dropout).to(self.device)
        sd = torch.load(ckpt_path, map_location=self.device)
        state = sd.get("model", sd)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def _encode_with_offsets(self, text: str, max_len: int):
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True,
                       truncation=True, max_length=max_len)
        return enc["input_ids"], enc["attention_mask"], enc["offset_mapping"]

    def _build_ids(self, ids, attn):
        return [self.CLS] + ids + [self.SEP], [1] + attn + [1]

    @torch.inference_mode()
    def predict(self, query: str, target: str) -> Dict:
        q_ids, q_attn, _ = self._encode_with_offsets(query, self.cfg.max_query_len)
        t_ids, t_attn, t_offsets = self._encode_with_offsets(target, self.cfg.max_target_len)
        q_ids2, q_attn2 = self._build_ids(q_ids, q_attn)
        t_ids2, t_attn2 = self._build_ids(t_ids, t_attn)

        q_ids_t  = torch.tensor([q_ids2], dtype=torch.long, device=self.device)
        q_attn_t = torch.tensor([q_attn2], dtype=torch.long, device=self.device)
        t_ids_t  = torch.tensor([t_ids2], dtype=torch.long, device=self.device)
        t_attn_t = torch.tensor([t_attn2], dtype=torch.long, device=self.device)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = self.model(q_ids_t, q_attn_t, t_ids_t, t_attn_t)[0]

        L = int(t_attn_t[0].sum().item())
        if L < 3:
            return {"probs": [], "token_mask": [], "token_spans": [], "char_spans": [], "snippets": []}

        probs = torch.sigmoid(logits[1:L-1])
        if self.cfg.smooth_k and self.cfg.smooth_k >= 3 and self.cfg.smooth_k % 2 == 1:
            probs = _median_filter_1d(probs, self.cfg.smooth_k)
        pred_bin = (probs >= self.cfg.threshold).to(torch.int).tolist()
        token_spans = _bin_to_spans(pred_bin)

        char_spans, snippets = [], []
        for s, e in token_spans:
            cs = t_offsets[s][0]
            ce = t_offsets[e-1][1] if e-1 < len(t_offsets) else t_offsets[-1][1]
            char_spans.append((cs, ce))
            snippets.append(target[cs:ce])

        return {
            "probs": probs.cpu().tolist(),
            "token_mask": pred_bin,
            "token_spans": token_spans,
            "char_spans": char_spans,
            "snippets": snippets,
            "config": asdict(self.cfg)
        }

def load() -> MarehMakomInference:
    return MarehMakomInference()

def main():
    import argparse, json, sys
    p = argparse.ArgumentParser(description="MarehMakom inference")
    p.add_argument("--query", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--json", action="store_true", help="print JSON")
    args = p.parse_args()
    infer = MarehMakomInference()
    out = infer.predict(args.query, args.target)
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print("Spans:", out["char_spans"])
        for s in out["snippets"]:
            print("→", s)
