from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

from app.backend.model import SmokeFusionTCN

def clip_fft_features(luma_mean_seq: np.ndarray, k: int = 10) -> np.ndarray:
    x = np.asarray(luma_mean_seq, dtype=np.float32)
    x = x - x.mean()
    spec = np.fft.rfft(x)
    mag = np.abs(spec).astype(np.float32)
    mag = mag[1:]  
    out = np.zeros((k,), dtype=np.float32)
    out[: min(k, mag.shape[0])] = mag[:k]
    return out

@dataclass
class AppPaths:
    base: Path
    artifacts: Path
    idx: Path
    feat_eff: Path
    feat_phys: Path
    model_dir: Path

    @staticmethod
    def from_base(base: str) -> "AppPaths":
        b = Path(base)
        a = b / "artifacts"
        return AppPaths(
            base=b,
            artifacts=a,
            idx=a / "index",
            feat_eff=a / "features" / "effb0",
            feat_phys=a / "features" / "physics",
            model_dir=a / "models" / "stage5_fusion_pos_weight",
        )

class Predictor:
    def __init__(self, base_dir: str, device: str | None = None):
        self.paths = AppPaths.from_base(base_dir)

        self.clips_csv = self.paths.idx / "clips_index.csv.gz"
        self.video_inv_csv = self.paths.idx / "video_inventory.csv"
        self.splits_json = self.paths.idx / "splits.json"

        self.meta_eff_json = self.paths.feat_eff / "meta.json"
        self.meta_phys_json = self.paths.feat_phys / "meta.json"
        self.best_pt = self.paths.model_dir / "best.pt"

        for p in [self.clips_csv, self.video_inv_csv, self.splits_json, self.meta_eff_json, self.meta_phys_json, self.best_pt]:
            if not p.exists():
                raise FileNotFoundError(f"Missing required artifact: {p}")

        if self.clips_csv.suffix == ".gz":
            self.clips = pd.read_csv(self.clips_csv, compression="gzip")
        else:
            self.clips = pd.read_csv(self.clips_csv)
        self.videos = pd.read_csv(self.video_inv_csv).sort_values("video_id").reset_index(drop=True)
        self.splits = json.loads(self.splits_json.read_text())

        self.meta_eff = json.loads(self.meta_eff_json.read_text())
        self.meta_phys = json.loads(self.meta_phys_json.read_text())

        self.fps = 25.0
        self.T = int(round(self.fps * 2.0))      
        self.D_EMB = int(self.meta_eff.get("embedding_dim", 1280))
        self.D_PHYS = int(self.meta_phys.get("dims", 2)) if "dims" in self.meta_phys else 2
        self.D_SEQ = self.D_EMB + self.D_PHYS
        self.FFT_K = int(self.meta_phys.get("fft_k", 10)) if "fft_k" in self.meta_phys else 10

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = SmokeFusionTCN(seq_dim=self.D_SEQ, fft_k=self.FFT_K).to(self.device)
        state = torch.load(self.best_pt, map_location=self.device)
        if isinstance(state, dict) and "model_state" in state:
            self.model.load_state_dict(state["model_state"])
        else:
            self.model.load_state_dict(state)
        self.model.eval()

    def list_videos(self, split: str = "test"):
        vids = self.splits["videos"][split]
        return sorted(vids)

    def list_clips(self, video_id: str, split: str = "test", limit: int = 200):
        df = self.clips[(self.clips["split"] == split) & (self.clips["video_id"] == video_id)].copy()
        df = df.sort_values("t").head(limit)
        return df[["video_id", "t", "clip_start", "clip_end", "target"]].to_dict(orient="records")

    def _load_eff(self, video_id: str) -> np.ndarray:
        p = self.paths.feat_eff / f"{video_id}.npy"
        if not p.exists():
            raise FileNotFoundError(f"Missing eff features: {p}")
        return np.load(p, mmap_mode="r")

    def _load_phys(self, video_id: str) -> np.ndarray:
        p = self.paths.feat_phys / f"{video_id}.npy"
        if not p.exists():
            raise FileNotFoundError(f"Missing physics features: {p}")
        return np.load(p, mmap_mode="r")

    @torch.no_grad()
    def predict_clip(self, video_id: str, clip_start: int, clip_end: int, threshold: float = 0.5):
        eff = self._load_eff(video_id)
        phys = self._load_phys(video_id)

        x_eff = eff[clip_start:clip_end + 1]
        x_phys = phys[clip_start:clip_end + 1]

        if x_eff.shape[0] < self.T:
            pad_e = np.zeros((self.T - x_eff.shape[0], x_eff.shape[1]), dtype=np.float32)
            pad_p = np.zeros((self.T - x_phys.shape[0], x_phys.shape[1]), dtype=np.float32)
            x_eff = np.concatenate([x_eff, pad_e], axis=0)
            x_phys = np.concatenate([x_phys, pad_p], axis=0)
        elif x_eff.shape[0] > self.T:
            x_eff = x_eff[: self.T]
            x_phys = x_phys[: self.T]

        x_seq = np.concatenate([x_eff, x_phys], axis=1).astype(np.float32)
        luma_mean = x_phys[:, 0].astype(np.float32)
        x_fft = clip_fft_features(luma_mean, k=self.FFT_K).astype(np.float32)

        x_seq_t = torch.from_numpy(x_seq).unsqueeze(0).to(self.device)  
        x_fft_t = torch.from_numpy(x_fft).unsqueeze(0).to(self.device)  

        logit = self.model(x_seq_t, x_fft_t).item()
        prob = float(1.0 / (1.0 + np.exp(-logit)))
        decision = int(prob >= threshold)

        row = self.clips[
            (self.clips["video_id"] == video_id) &
            (self.clips["clip_start"] == clip_start) &
            (self.clips["clip_end"] == clip_end)
        ]
        gt = int(row.iloc[0]["target"]) if len(row) > 0 else None

        return {
            "video_id": video_id,
            "clip_start": int(clip_start),
            "clip_end": int(clip_end),
            "prob": prob,
            "threshold": float(threshold),
            "decision": decision,
            "ground_truth": gt,
        }