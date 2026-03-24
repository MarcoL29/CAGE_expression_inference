# AffWild2 Temporal ViT Pipeline (Valence/Arousal)

This folder contains a production-ready PyTorch pipeline to train a **temporal Vision Transformer** on **AffWild2** valence/arousal (VA) annotations using **cropped aligned face frames**.

## Dataset assumptions

- VA annotation files live under:
  - `...\5th_ABAW_Annotations\VA\train\<video>.txt`
  - `...\5th_ABAW_Annotations\VA\validate\<video>.txt`
- Each annotation file has **one header line**, then rows:
  - `valence_value, arousal_value`
- Cropped aligned frames live under:
  - `...\cropped_aligned\<video>\00001.jpg ...`
- Frame `00001.jpg` corresponds to the **first annotation row after the header**.

The pipeline aligns frames and labels by **1-based frame index**.

## Quick start

1) (Optional but recommended) create/activate a venv.

2) Ensure dependencies are installed (this repo already includes `torch`, `torchvision`, `numpy`, `tqdm`, `pillow`).

3) Edit the config:

- `affwild2_pipeline/training/configs/default.json`

Set:
- `annotation_root` to `...\5th_ABAW_Annotations\VA`
- `frames_root` to `...\cropped_aligned`

4) Train:

```bash
python -m affwild2_pipeline.training.train --config affwild2_pipeline/training/configs/default.json
```

5) Validate a checkpoint:

```bash
python -m affwild2_pipeline.validation.validate \
  --config affwild2_pipeline/training/configs/default.json \
  --checkpoint affwild2_pipeline/training/checkpoints/best.pt
```

## Outputs

- Checkpoints: `affwild2_pipeline/training/checkpoints/`
- Logs: `affwild2_pipeline/training/runs/<run_name>/train.log`
- Validation predictions and summary metrics:
  - `affwild2_pipeline/validation/results/<run_name>/metrics.json`
  - `affwild2_pipeline/validation/results/<run_name>/predictions.csv`

## Notes

- The model predicts **per-frame VA** for each temporal window (shape: `B x T x 2`).
- Validation uses overlapping windows and **averages predictions for frames** that appear in multiple windows.
- AffWild2 uses `-5` as a sentinel for missing valence/arousal; any frame where **valence or arousal is `-5`** is automatically **masked out** (ignored) in both loss computation and metrics.
