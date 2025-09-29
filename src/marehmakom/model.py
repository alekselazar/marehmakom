# src/marehmakom/model.py
import torch
import torch.nn as nn
from transformers import AutoModel

class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=None):
        super().__init__()
        if p is None: p = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=p),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, k, padding=p),
            nn.ReLU()
        )
    def forward(self, x): return self.net(x)

class UNet1DHead(nn.Module):
    def __init__(self, hidden=768, dropout=0.1):
        super().__init__()
        self.enc1 = ConvBlock1D(hidden, 256)
        self.pool1 = nn.MaxPool1d(2, ceil_mode=True)
        self.enc2 = ConvBlock1D(256, 512)
        self.pool2 = nn.MaxPool1d(2, ceil_mode=True)
        self.bottleneck = ConvBlock1D(512, 1024)
        self.up2 = nn.ConvTranspose1d(1024, 512, 2, stride=2)
        self.dec2 = ConvBlock1D(1024, 512)
        self.up1 = nn.ConvTranspose1d(512, 256, 2, stride=2)
        self.dec1 = ConvBlock1D(512, 256)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Conv1d(256, 1, 1)

    @staticmethod
    def _crop_or_pad(x, T_target):
        B, C, T = x.shape
        if T == T_target: return x
        if T > T_target:
            start = (T - T_target) // 2
            return x[:, :, start:start+T_target]
        pad = T_target - T
        left = pad // 2; right = pad - left
        return nn.functional.pad(x, (left, right))

    def forward(self, hs_tgt):  # [B,T,H]
        T_orig = hs_tgt.size(1)
        x = hs_tgt.permute(0, 2, 1)
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        b  = self.bottleneck(p2)
        u2 = self.up2(b); e2c = self._crop_or_pad(e2, u2.size(-1))
        d2 = self.dec2(torch.cat([u2, e2c], dim=1))
        u1 = self.up1(d2); e1c = self._crop_or_pad(e1, u1.size(-1))
        d1 = self.dec1(torch.cat([u1, e1c], dim=1))
        d1 = self.drop(d1)
        logits = self.classifier(d1).squeeze(1)
        if logits.size(1) != T_orig:
            logits = self._crop_or_pad(logits.unsqueeze(1), T_orig).squeeze(1)
        return logits

class FiLMConditioner(nn.Module):
    def __init__(self, hidden=768):
        super().__init__()
        self.gamma = nn.Linear(hidden, hidden)
        self.beta  = nn.Linear(hidden, hidden)
    def forward(self, hs_tgt, q_vec):
        gamma = self.gamma(q_vec).unsqueeze(1)
        beta  = self.beta(q_vec).unsqueeze(1)
        return hs_tgt * (1.0 + gamma) + beta

class MarehMakomModel(nn.Module):
    """LaBSE encoder + FiLM conditioner + 1D-UNet head."""
    def __init__(self, base_model: str, dropout: float = 0.1, freeze_encoder: bool = True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        if freeze_encoder:
            for p in self.encoder.parameters(): p.requires_grad = False
        hidden = self.encoder.config.hidden_size
        self.conditioner = FiLMConditioner(hidden=hidden)
        self.head = UNet1DHead(hidden=hidden, dropout=dropout)

    @staticmethod
    def mean_pool(hs, attn):
        mask = attn.unsqueeze(-1).float()
        num = (hs * mask).sum(dim=1)
        den = mask.sum(dim=1).clamp_min(1.0)
        return num / den

    def forward(self, q_ids, q_attn, t_ids, t_attn):
        q_out = self.encoder(input_ids=q_ids, attention_mask=q_attn)
        q_vec = self.mean_pool(q_out.last_hidden_state, q_attn)
        t_out = self.encoder(input_ids=t_ids, attention_mask=t_attn)
        t_hs  = t_out.last_hidden_state
        t_cond = self.conditioner(t_hs, q_vec)
        logits = self.head(t_cond)
        return logits
