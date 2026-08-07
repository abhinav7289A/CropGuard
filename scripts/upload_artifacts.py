"""Publish evaluation artifacts to the Hub, from wherever the training run left them.

    export HF_TOKEN=hf_...
    python scripts/upload_artifacts.py --source artifacts
    python scripts/upload_artifacts.py --source /content/drive/MyDrive/cropguard/artifacts

Training happens on ephemeral compute, so the prediction files and the challenger weights end
up in a Drive folder that nothing else can reach. That makes them un-reviewable and un-testable
- `.github/workflows/promotion-gate.yml` pulls exactly these `.npz` files from the Hub, and
the demo panel pulls `challenger.onnx`. Putting them somewhere addressable is what turns a
finished experiment into a reproducible one.

Prediction files are small (~2 MB each) because they hold predictions, not images.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Files worth publishing, and why each one earns its place. Anything absent is skipped, so a
# run that only produced some of them still works.
ARTIFACTS = {
    "challenger.onnx": "challenger weights — the demo panel fetches these",
    "preds_baseline.npz": "baseline predictions — input to the promotion gate",
    "preds_challenger.npz": "challenger predictions — input to the promotion gate",
    "preds_challenger_val.npz": "challenger validation predictions — refits temperature",
    "ab_comparison.json": "the A/B report the gate produced",
    "challenger_calibration.json": "fitted temperature and ECE for the challenger",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="XiElonMAsk/cropguard-models")
    parser.add_argument("--source", default="artifacts", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.source.is_dir():
        sys.exit(f"No such directory: {args.source}")

    present = [(name, args.source / name) for name in ARTIFACTS if (args.source / name).exists()]
    for name, path in present:
        print(f"  {name:32s} {path.stat().st_size / 1e6:7.1f} MB   {ARTIFACTS[name]}")
    for name in ARTIFACTS:
        if not (args.source / name).exists():
            print(f"  {name:32s}   -- absent, skipping")

    if not present:
        sys.exit("Nothing to upload.")
    if args.dry_run:
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set. Create a write token at https://hf.co/settings/tokens")

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    # One commit, so a partial upload cannot leave predictions that disagree with the report
    # generated from them.
    api.create_commit(
        repo_id=args.repo,
        operations=[
            CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path))
            for name, path in present
        ],
        commit_message="Publish challenger weights and A/B evaluation artifacts",
    )
    print(f"\nhttps://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
