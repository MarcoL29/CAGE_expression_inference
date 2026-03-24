from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class Window:
    video_id: str
    start: int  # 0-based frame index into annotations
    length: int


def make_windows(num_frames: int, seq_len: int, stride: int, drop_last: bool = True) -> List[int]:
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")
    if num_frames <= 0:
        return []

    if num_frames < seq_len:
        return [] if drop_last else [0]

    starts = list(range(0, num_frames - seq_len + 1, stride))
    if not drop_last and starts and (starts[-1] + seq_len < num_frames):
        starts.append(num_frames - seq_len)
    return starts
