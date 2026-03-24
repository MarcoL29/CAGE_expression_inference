from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from affwild2_pipeline.data.sampling import Window, make_windows


_NUM_RE = re.compile(r"^(\d+)$")


def _parse_va_line(line: str) -> Tuple[float, float]:
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) < 2:
        return math.nan, math.nan
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return math.nan, math.nan


def _read_va_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    if not lines:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.bool_)

    data: List[Tuple[float, float]] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        v, a = _parse_va_line(line)
        data.append((v, a))

    arr = np.asarray(data, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.bool_)

    # AffWild2 VA labels are in [-1, 1]. The value -5 is a sentinel indicating missing labels.
    finite = np.isfinite(arr).all(axis=1)
    not_missing = (arr[:, 0] != -5.0) & (arr[:, 1] != -5.0)
    in_range = (arr[:, 0] >= -1.0) & (arr[:, 0] <= 1.0) & (arr[:, 1] >= -1.0) & (arr[:, 1] <= 1.0)
    valid = finite & not_missing & in_range
    arr[~valid] = 0.0
    return arr, valid


def _discover_frame_indices(frames_dir: Path) -> Tuple[Optional[int], bool, Optional[set[int]]]:
    """Return (max_index, is_contiguous_1_to_max, indices_set_if_sparse)."""
    if not frames_dir.exists():
        return None, False, None

    indices: List[int] = []
    for p in frames_dir.glob("*.jpg"):
        m = _NUM_RE.match(p.stem)
        if not m:
            continue
        indices.append(int(m.group(1)))

    if not indices:
        return None, False, None

    indices.sort()
    max_idx = indices[-1]
    is_contiguous = (indices[0] == 1) and (len(indices) == max_idx)
    if is_contiguous:
        return max_idx, True, None
    return max_idx, False, set(indices)


@dataclass(frozen=True)
class VideoRecord:
    video_id: str
    ann_path: Path
    frames_dir: Path
    va: torch.Tensor  # (N, 2)
    valid: torch.Tensor  # (N,)
    max_frame_idx: int  # 1-based max frame index available
    contiguous: bool
    indices_set: Optional[set[int]]


class AffWild2VADataset(Dataset):
    """AffWild2 valence/arousal dataset emitting temporal windows.

    Returns:
      frames: FloatTensor (T, C, H, W)
      targets: FloatTensor (T, 2)
      mask: BoolTensor (T,) where True indicates valid target
      meta: dict with video_id and start (0-based)
    """

    def __init__(
        self,
        annotation_root: str | Path,
        frames_root: str | Path,
        split: str,
        seq_len: int,
        stride: int,
        transform,
        image_size: int = 112,
        drop_last: bool = True,
        strict_alignment: bool = False,
    ) -> None:
        super().__init__()
        self.annotation_root = Path(annotation_root)
        self.frames_root = Path(frames_root)
        self.split = split
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.transform = transform
        self.image_size = int(image_size)
        self.drop_last = bool(drop_last)
        self.strict_alignment = bool(strict_alignment)

        ann_dir = self.annotation_root / split
        if not ann_dir.exists():
            raise FileNotFoundError(f"Annotation split folder not found: {ann_dir}")

        self.videos: Dict[str, VideoRecord] = {}
        self.windows: List[Window] = []

        for ann_path in sorted(ann_dir.glob("*.txt")):
            video_id = ann_path.stem
            frames_dir = self.frames_root / video_id
            va_np, valid_np = _read_va_file(ann_path)
            if va_np.shape[0] == 0:
                continue

            max_idx, contiguous, indices_set = _discover_frame_indices(frames_dir)
            if max_idx is None:
                if self.strict_alignment:
                    raise FileNotFoundError(f"No frames found for video {video_id}: {frames_dir}")
                continue

            n = min(int(va_np.shape[0]), int(max_idx))
            if n < self.seq_len:
                continue

            va = torch.from_numpy(va_np[:n])
            valid = torch.from_numpy(valid_np[:n])

            record = VideoRecord(
                video_id=video_id,
                ann_path=ann_path,
                frames_dir=frames_dir,
                va=va,
                valid=valid,
                max_frame_idx=max_idx,
                contiguous=contiguous,
                indices_set=indices_set,
            )
            self.videos[video_id] = record

        if not self.videos:
            raise RuntimeError(
                "No usable videos found. Check annotation_root/frames_root paths and split name."
            )

        for record in self.videos.values():
            starts = make_windows(
                num_frames=record.va.shape[0],
                seq_len=self.seq_len,
                stride=self.stride,
                drop_last=self.drop_last,
            )
            for s in starts:
                if (not record.contiguous) and record.indices_set is not None:
                    needed = range(s + 1, s + self.seq_len + 1)
                    if any((i not in record.indices_set) for i in needed):
                        continue
                self.windows.append(Window(video_id=record.video_id, start=s, length=self.seq_len))

        if not self.windows:
            raise RuntimeError("No windows generated (seq_len/stride may be too large).")

    def __len__(self) -> int:
        return len(self.windows)

    def _frame_path(self, frames_dir: Path, frame_idx_1based: int) -> Path:
        return frames_dir / f"{frame_idx_1based:05d}.jpg"

    def __getitem__(self, idx: int):
        w = self.windows[idx]
        record = self.videos[w.video_id]
        start = w.start
        end = start + w.length

        images = []
        for t in range(start, end):
            frame_idx = t + 1
            path = self._frame_path(record.frames_dir, frame_idx)
            if not path.exists():
                if self.strict_alignment:
                    raise FileNotFoundError(f"Missing frame: {path}")
                img = Image.new("RGB", (self.image_size, self.image_size))
            else:
                img = Image.open(path).convert("RGB")
            images.append(self.transform(img))

        frames = torch.stack(images, dim=0)  # (T, C, H, W)
        targets = record.va[start:end].to(torch.float32)
        mask = record.valid[start:end].to(torch.bool)

        meta = {"video_id": record.video_id, "start": int(start)}
        return {"frames": frames, "targets": targets, "mask": mask, "meta": meta}

    def get_video_record(self, video_id: str) -> VideoRecord:
        return self.videos[video_id]
