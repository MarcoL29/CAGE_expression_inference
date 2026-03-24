from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image

# Allow running this script directly without installing the package or setting PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from affwild2_pipeline.utils.config import load_train_config


_NUM_RE = re.compile(r"^(\d+)$")


@dataclass
class VideoAnnStats:
    video_id: str
    ann_path: Path
    header: str
    header_order: str  # 'va', 'av', or 'unknown'
    num_rows: int
    num_parsed: int
    num_non_finite: int
    num_missing_sentinel: int
    num_out_of_range: int
    num_valid: int
    v_mean: float
    v_std: float
    v_min: float
    v_max: float
    a_mean: float
    a_std: float
    a_min: float
    a_max: float


@dataclass
class FrameScanStats:
    video_id: str
    frames_dir: Path
    jpg_count: int
    min_index: Optional[int]
    max_index: Optional[int]
    contiguous_1_to_max: bool
    missing_first_frame: bool
    has_zero_frame: bool
    indices_set: Optional[set[int]]


def _iter_va_lines(path: Path) -> Iterable[str]:
    # Aff-Wild2 VA annotation text files often have a header row.
    # We keep this conservative: skip first line unconditionally.
    with path.open("r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    if len(lines) <= 1:
        return []
    return lines[1:]


def _read_header_line(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8") as f:
            first = f.readline()
        return first.strip()
    except Exception:  # noqa: BLE001
        return ""


def _infer_header_order(header: str) -> str:
    h = (header or "").lower().replace(" ", "")
    # Common forms: "valence,arousal" or "valence, arousal" etc.
    # If both words appear, infer which comes first.
    v_pos = h.find("valence")
    a_pos = h.find("arousal")
    if v_pos != -1 and a_pos != -1:
        return "va" if v_pos < a_pos else "av"
    return "unknown"


def _parse_va_line(line: str) -> tuple[float, float] | None:
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def analyze_annotation_file(path: Path) -> VideoAnnStats:
    video_id = path.stem
    header = _read_header_line(path)
    header_order = _infer_header_order(header)

    num_rows = 0
    num_parsed = 0
    num_non_finite = 0
    num_missing_sentinel = 0
    num_out_of_range = 0

    valid_v: list[float] = []
    valid_a: list[float] = []

    for line in _iter_va_lines(path):
        if not line.strip():
            continue
        num_rows += 1
        parsed = _parse_va_line(line)
        if parsed is None:
            continue

        v, a = parsed
        num_parsed += 1

        if not (math.isfinite(v) and math.isfinite(a)):
            num_non_finite += 1
            continue

        # AffWild2 sentinel for missing labels.
        if v == -5.0 or a == -5.0:
            num_missing_sentinel += 1
            continue

        # Expected VA range.
        if not (-1.0 <= v <= 1.0 and -1.0 <= a <= 1.0):
            num_out_of_range += 1
            continue

        valid_v.append(v)
        valid_a.append(a)

    if valid_v:
        v_arr = np.asarray(valid_v, dtype=np.float32)
        a_arr = np.asarray(valid_a, dtype=np.float32)
        v_mean = float(v_arr.mean())
        v_std = float(v_arr.std())
        v_min = float(v_arr.min())
        v_max = float(v_arr.max())
        a_mean = float(a_arr.mean())
        a_std = float(a_arr.std())
        a_min = float(a_arr.min())
        a_max = float(a_arr.max())
    else:
        v_mean = v_std = v_min = v_max = float("nan")
        a_mean = a_std = a_min = a_max = float("nan")

    return VideoAnnStats(
        video_id=video_id,
        ann_path=path,
        header=header,
        header_order=header_order,
        num_rows=num_rows,
        num_parsed=num_parsed,
        num_non_finite=num_non_finite,
        num_missing_sentinel=num_missing_sentinel,
        num_out_of_range=num_out_of_range,
        num_valid=len(valid_v),
        v_mean=v_mean,
        v_std=v_std,
        v_min=v_min,
        v_max=v_max,
        a_mean=a_mean,
        a_std=a_std,
        a_min=a_min,
        a_max=a_max,
    )


def scan_frames_dir(frames_dir: Path) -> FrameScanStats:
    video_id = frames_dir.name
    if not frames_dir.exists():
        return FrameScanStats(
            video_id=video_id,
            frames_dir=frames_dir,
            jpg_count=0,
            min_index=None,
            max_index=None,
            contiguous_1_to_max=False,
            missing_first_frame=True,
            has_zero_frame=False,
            indices_set=None,
        )

    has_zero_frame = (frames_dir / "00000.jpg").exists()

    indices: list[int] = []
    jpg_count = 0
    for p in frames_dir.glob("*.jpg"):
        jpg_count += 1
        m = _NUM_RE.match(p.stem)
        if not m:
            continue
        indices.append(int(m.group(1)))

    if not indices:
        return FrameScanStats(
            video_id=video_id,
            frames_dir=frames_dir,
            jpg_count=jpg_count,
            min_index=None,
            max_index=None,
            contiguous_1_to_max=False,
            missing_first_frame=True,
            has_zero_frame=has_zero_frame,
            indices_set=None,
        )

    indices.sort()
    min_index = indices[0]
    max_index = indices[-1]
    contiguous = (indices[0] == 1) and (len(indices) == max_index)
    missing_first = indices[0] != 1

    return FrameScanStats(
        video_id=video_id,
        frames_dir=frames_dir,
        jpg_count=jpg_count,
        min_index=min_index,
        max_index=max_index,
        contiguous_1_to_max=contiguous,
        missing_first_frame=missing_first,
        has_zero_frame=has_zero_frame,
        indices_set=set(indices) if not contiguous else None,
    )


def windows_count(num_frames: int, seq_len: int, stride: int, drop_last: bool) -> int:
    if num_frames < seq_len:
        return 0
    if drop_last:
        return 1 + (num_frames - seq_len) // stride

    # Non-drop-last includes a final partial window in some implementations, but this repo's
    # dataset only emits full windows anyway. Keep it aligned with that behavior.
    return 1 + (num_frames - seq_len) // stride


def windows_count_with_sparse_indices(
    num_frames: int,
    seq_len: int,
    stride: int,
    drop_last: bool,
    indices_set: Optional[set[int]],
) -> int:
    """Compute window count matching AffWild2VADataset behavior.

    If indices_set is provided, this checks that every required 1-based frame index exists.
    """

    naive = windows_count(num_frames, seq_len, stride, drop_last)
    if naive == 0:
        return 0
    if not indices_set:
        return naive

    # Generate the same start positions as make_windows() would.
    # This repo only emits full windows, so starts are 0..(num_frames-seq_len) step stride.
    last_start = num_frames - seq_len
    if last_start < 0:
        return 0

    count = 0
    for s in range(0, last_start + 1, stride):
        # Need 1-based indices [s+1, s+seq_len]
        ok = True
        for i in range(s + 1, s + seq_len + 1):
            if i not in indices_set:
                ok = False
                break
        if ok:
            count += 1
    return count


def resolve_config_path(config_arg: str) -> Path:
    """Resolve a config path argument to an existing file.

    Users often pass just "default.json" while the file lives under
    affwild2_pipeline/training/configs/default.json.
    """

    raw = Path(config_arg).expanduser()
    candidates: list[Path] = []

    # 1) As provided (relative to current working directory).
    candidates.append(raw)
    # 2) Relative to repo root.
    candidates.append(_REPO_ROOT / raw)

    # 3) Common config folders.
    candidates.append(_REPO_ROOT / "affwild2_pipeline" / "training" / "configs" / raw.name)

    # 4) If the user forgot the .json suffix.
    if raw.suffix == "":
        with_json = raw.with_suffix(".json")
        candidates.append(with_json)
        candidates.append(_REPO_ROOT / with_json)
        candidates.append(_REPO_ROOT / "affwild2_pipeline" / "training" / "configs" / with_json.name)

    for c in candidates:
        if c.exists() and c.is_file():
            return c

    # Produce a helpful error.
    searched = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(
        f"Config file not found: {config_arg}\nSearched:\n{searched}\n"
        f"Tip: try --config affwild2_pipeline/training/configs/default.json"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Sanity checks for Aff-Wild2 VA annotations + frames")
    ap.add_argument("--config", required=True, help="Path to training config JSON")
    ap.add_argument(
        "--splits",
        default="train,validate",
        help="Comma-separated splits to check (default: train,validate)",
    )
    ap.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Max videos per split to scan (0 = no limit)",
    )
    ap.add_argument(
        "--check-frames",
        action="store_true",
        help="Check that frames folders exist for each annotation file",
    )
    ap.add_argument(
        "--scan-frames",
        action="store_true",
        help="Scan frame indices (*.jpg) to compute contiguity/max index (slower)",
    )
    ap.add_argument(
        "--sample-open",
        type=int,
        default=0,
        help="Open N sample frames with PIL to verify readability (requires --scan-frames)",
    )
    ap.add_argument(
        "--print-va-samples",
        type=int,
        default=0,
        help="Print N raw (v,a) pairs from the first video per split to verify column order",
    )
    ap.add_argument(
        "--show-problems",
        type=int,
        default=0,
        help="When scanning frames, print up to N example video_ids for detected issues",
    )
    args = ap.parse_args()

    if args.sample_open and not args.scan_frames:
        print("ERROR: --sample-open requires --scan-frames (so we can pick valid frame indices).")
        return 2

    config_path = resolve_config_path(args.config)
    if str(config_path) != str(Path(args.config)):
        print(f"Resolved config: {config_path}")

    cfg = load_train_config(config_path)
    annotation_root = Path(cfg.annotation_root)
    frames_root = Path(cfg.frames_root)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        print("No splits provided.")
        return 2

    if not annotation_root.exists():
        print(f"ERROR: annotation_root not found: {annotation_root}")
        return 2
    if not frames_root.exists():
        print(f"ERROR: frames_root not found: {frames_root}")
        return 2

    print("=== Aff-Wild2 VA sanity check ===")
    print(f"config: {args.config}")
    print(f"annotation_root: {annotation_root}")
    print(f"frames_root: {frames_root}")
    print(f"seq_len={cfg.seq_len} train_stride={cfg.train_stride} val_stride={cfg.val_stride}")
    print(f"strict_alignment={cfg.strict_alignment}")

    any_errors = False

    for split in splits:
        ann_dir = annotation_root / split
        if not ann_dir.exists():
            print(f"\n--- split={split} ---")
            print(f"ERROR: missing annotation split dir: {ann_dir}")
            any_errors = True
            continue

        ann_files = sorted(ann_dir.glob("*.txt"))
        if args.max_videos and args.max_videos > 0:
            ann_files = ann_files[: args.max_videos]

        print(f"\n--- split={split} ---")
        print(f"annotation files: {len(ann_files)}")
        if not ann_files:
            print("ERROR: no .txt annotation files found.")
            any_errors = True
            continue

        split_stride = int(cfg.train_stride if split == "train" else cfg.val_stride)
        drop_last = bool(split == "train")

        totals = {
            "rows": 0,
            "parsed": 0,
            "non_finite": 0,
            "missing": 0,
            "out_of_range": 0,
            "valid": 0,
            "header_va": 0,
            "header_av": 0,
            "header_unknown": 0,
            "missing_frames_dir": 0,
            "empty_frames_dir": 0,
            "non_contiguous_frames": 0,
            "missing_first_frame": 0,
            "too_short_for_seq": 0,
            "low_label_variance": 0,
            "frames_for_windows": 0,
            "estimated_windows": 0,
            "estimated_windows_after_gaps": 0,
            "min_frame_index_gt1": 0,
        }

        examples_non_contig: list[str] = []
        examples_missing_first: list[str] = []
        examples_missing_first_with_zero: list[str] = []
        min_index_values_gt1: list[int] = []

        opened = 0
        printed_samples = False
        for ann_path in ann_files:
            s = analyze_annotation_file(ann_path)
            totals["rows"] += s.num_rows
            totals["parsed"] += s.num_parsed
            totals["non_finite"] += s.num_non_finite
            totals["missing"] += s.num_missing_sentinel
            totals["out_of_range"] += s.num_out_of_range
            totals["valid"] += s.num_valid

            if s.header_order == "va":
                totals["header_va"] += 1
            elif s.header_order == "av":
                totals["header_av"] += 1
            else:
                totals["header_unknown"] += 1

            if args.print_va_samples and not printed_samples:
                # Print a few raw pairs from this file.
                raw: list[tuple[float, float]] = []
                for line in _iter_va_lines(ann_path):
                    parsed = _parse_va_line(line)
                    if parsed is None:
                        continue
                    raw.append(parsed)
                    if len(raw) >= int(args.print_va_samples):
                        break
                print(f"sample header: {s.header}")
                if raw:
                    print("sample (col0, col1) pairs:")
                    for i, (v, a) in enumerate(raw, start=1):
                        print(f"  {i:02d}: ({v:.4f}, {a:.4f})")
                else:
                    print("sample: no parseable rows found")
                printed_samples = True

            # Match AffWild2VADataset: base windowability on annotation length, then optionally cap by frames.
            n_for_windows = int(s.num_parsed)
            indices_set_for_count: Optional[set[int]] = None

            # Heuristic: very small std often means constant/near-constant predictions are optimal,
            # or there is a parsing issue.
            if math.isfinite(s.v_std) and math.isfinite(s.a_std) and (s.v_std < 0.05 or s.a_std < 0.05):
                totals["low_label_variance"] += 1

            if args.check_frames or args.scan_frames:
                frames_dir = frames_root / s.video_id
                if not frames_dir.exists():
                    totals["missing_frames_dir"] += 1
                    continue

                if args.scan_frames:
                    fs = scan_frames_dir(frames_dir)
                    if fs.max_index is None:
                        totals["empty_frames_dir"] += 1
                    else:
                        if not fs.contiguous_1_to_max:
                            totals["non_contiguous_frames"] += 1
                            if args.show_problems and len(examples_non_contig) < int(args.show_problems):
                                examples_non_contig.append(s.video_id)
                        if fs.missing_first_frame:
                            totals["missing_first_frame"] += 1
                            if args.show_problems and len(examples_missing_first) < int(args.show_problems):
                                examples_missing_first.append(s.video_id)
                            if fs.has_zero_frame and args.show_problems and len(examples_missing_first_with_zero) < int(args.show_problems):
                                examples_missing_first_with_zero.append(s.video_id)
                        if fs.min_index is not None and fs.min_index > 1:
                            totals["min_frame_index_gt1"] += 1
                            min_index_values_gt1.append(int(fs.min_index))

                    # Match AffWild2VADataset: n = min(num_label_rows, max_frame_idx)
                    if fs.max_index is not None:
                        n_for_windows = min(n_for_windows, int(fs.max_index))
                    indices_set_for_count = fs.indices_set

                    if args.sample_open > 0 and opened < args.sample_open and fs.max_index is not None:
                        # Try opening the first frame that should exist.
                        frame_path = frames_dir / "00001.jpg"
                        if frame_path.exists():
                            try:
                                with Image.open(frame_path) as im:
                                    im.convert("RGB")
                                opened += 1
                            except Exception as e:  # noqa: BLE001
                                print(f"ERROR: failed to open {frame_path}: {e}")
                                any_errors = True

            if n_for_windows < cfg.seq_len:
                totals["too_short_for_seq"] += 1

            totals["frames_for_windows"] += max(n_for_windows, 0)
            totals["estimated_windows"] += windows_count(n_for_windows, int(cfg.seq_len), int(split_stride), bool(drop_last))
            if args.scan_frames:
                # When scanning frames, compute the exact number of windows remaining after removing windows
                # that would require missing frames (matching the dataset's filtering behavior).
                # Note: This assumes 1-based frame indexing like the dataset.
                totals["estimated_windows_after_gaps"] += windows_count_with_sparse_indices(
                    n_for_windows,
                    int(cfg.seq_len),
                    int(split_stride),
                    bool(drop_last),
                    indices_set_for_count,
                )

        valid = totals["valid"]
        parsed = totals["parsed"]
        rows = totals["rows"]
        denom = max(parsed, 1)
        print(
            "parsed_rows={rows} parsed_ok={parsed} valid={valid} "
            "missing(-5)={missing} non_finite={non_finite} out_of_range={oor}".format(
                rows=rows,
                parsed=parsed,
                valid=valid,
                missing=totals["missing"],
                non_finite=totals["non_finite"],
                oor=totals["out_of_range"],
            )
        )
        print(
            "fractions: valid={:.2%} missing={:.2%} non_finite={:.2%} out_of_range={:.2%}".format(
                valid / denom,
                totals["missing"] / denom,
                totals["non_finite"] / denom,
                totals["out_of_range"] / denom,
            )
        )

        print(
            "header order (if detectable): va={} av={} unknown={}".format(
                totals["header_va"], totals["header_av"], totals["header_unknown"]
            )
        )

        print(f"videos too short for seq_len ({cfg.seq_len}): {totals['too_short_for_seq']}/{len(ann_files)}")
        print(f"videos with low label variance: {totals['low_label_variance']}/{len(ann_files)}")

        if args.check_frames or args.scan_frames:
            print(f"missing frames_dir: {totals['missing_frames_dir']}/{len(ann_files)}")
        if args.scan_frames:
            print(f"empty frames_dir (no numeric jpgs): {totals['empty_frames_dir']}/{len(ann_files)}")
            print(f"non-contiguous frame indices: {totals['non_contiguous_frames']}/{len(ann_files)}")
            print(f"missing first frame (index 1): {totals['missing_first_frame']}/{len(ann_files)}")

            if args.show_problems and int(args.show_problems) > 0:
                if examples_non_contig:
                    print("examples non-contiguous:")
                    for vid in examples_non_contig:
                        print(f"  - {vid}")
                if examples_missing_first:
                    print("examples missing 00001.jpg:")
                    for vid in examples_missing_first:
                        print(f"  - {vid}")
                if examples_missing_first_with_zero:
                    print("examples missing 00001.jpg but has 00000.jpg (likely 0-based indexing):")
                    for vid in examples_missing_first_with_zero:
                        print(f"  - {vid}")

        print(
            "estimated windows (dataset-style): frames_considered={} -> windows={} (seq_len={}, stride={}, drop_last={})".format(
                totals["frames_for_windows"],
                totals["estimated_windows"],
                cfg.seq_len,
                split_stride,
                drop_last,
            )
        )

        if args.scan_frames:
            dropped = totals["estimated_windows"] - totals["estimated_windows_after_gaps"]
            denom_w = max(totals["estimated_windows"], 1)
            print(
                "estimated windows after gap-filter: {} (dropped {} / {:.2%})".format(
                    totals["estimated_windows_after_gaps"],
                    dropped,
                    dropped / denom_w,
                )
            )
            print(f"videos with min frame index > 1: {totals['min_frame_index_gt1']}/{len(ann_files)}")
            if min_index_values_gt1:
                arr = np.asarray(min_index_values_gt1, dtype=np.int64)
                print(
                    "min frame index (>1) stats: min={} p50={} p90={} max={}".format(
                        int(arr.min()),
                        int(np.percentile(arr, 50)),
                        int(np.percentile(arr, 90)),
                        int(arr.max()),
                    )
                )

        # Basic warnings
        if valid == 0:
            print("ERROR: no valid labels found in this split.")
            any_errors = True
        if totals["out_of_range"] > 0:
            print("WARNING: found out-of-range labels; expected VA in [-1, 1].")
        if totals["missing"] / denom > 0.3:
            print("WARNING: >30% labels are missing (-5 sentinel). Training signal may be weak.")
        if totals["missing_frames_dir"] > 0:
            print(
                "WARNING: some videos have annotations but no frames folder; they will be skipped (or crash if strict_alignment=true)."
            )

        if totals["header_av"] > 0:
            print(
                "WARNING: some files look like they have 'arousal' before 'valence' in the header. "
                "This training code assumes column 0=valence, column 1=arousal."
            )

    return 1 if any_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
