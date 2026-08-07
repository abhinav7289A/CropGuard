"""Publish the demo panel to a Hugging Face Space.

    export HF_TOKEN=hf_...            # a *write* token
    python scripts/deploy_space.py --repo XiElonMAsk/cropguard

Uploads only what the panel needs at runtime: the app, the two configs, the torch-free serving
package, and requirements.txt. Model weights are deliberately *not* uploaded — the Space pulls
them from the model repo on first use, so the demo and the API load the same artifact and a
new model never has to be copied to two places.

`spaces/README.md` becomes the Space card; its YAML front matter is what tells Spaces which
SDK to run and where the app file lives, so the upload would produce a broken Space without it.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# (source, destination-in-space). Paths are preserved so the app's REPO_ROOT-relative lookups
# resolve identically on the Space and on a laptop.
FILES = [
    ("spaces/README.md", "README.md"),
    ("spaces/Dockerfile", "Dockerfile"),
    ("requirements.txt", "requirements.txt"),
    ("app/streamlit_app.py", "app/streamlit_app.py"),
    ("configs/models.json", "configs/models.json"),
    ("configs/classes.json", "configs/classes.json"),
    ("src/cropguard/__init__.py", "src/cropguard/__init__.py"),
    ("src/cropguard/serving/__init__.py", "src/cropguard/serving/__init__.py"),
    ("src/cropguard/serving/model_loader.py", "src/cropguard/serving/model_loader.py"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Space id, e.g. username/cropguard")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="List the payload and exit")
    args = parser.parse_args()

    missing = [src for src, _ in FILES if not (REPO_ROOT / src).exists()]
    if missing:
        sys.exit("Missing required files:\n  " + "\n  ".join(missing))

    total = sum((REPO_ROOT / src).stat().st_size for src, _ in FILES)
    for src, dst in FILES:
        print(f"  {src:45s} -> {dst}")
    print(f"{len(FILES)} files, {total / 1024:.0f} KB")

    if args.dry_run:
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit(
            "HF_TOKEN is not set. Create a write token at https://huggingface.co/settings/tokens"
        )

    from huggingface_hub import HfApi

    # Docker, not streamlit: the Hub rejects `streamlit` as an SDK for new Spaces, accepting
    # only gradio, docker and static. spaces/Dockerfile runs the same app either way.
    api = HfApi(token=token)
    api.create_repo(
        args.repo, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True
    )

    # One commit rather than eight, so a half-finished upload cannot leave the Space running a
    # new app against old configs.
    from huggingface_hub import CommitOperationAdd

    api.create_commit(
        repo_id=args.repo,
        repo_type="space",
        operations=[
            CommitOperationAdd(path_in_repo=dst, path_or_fileobj=str(REPO_ROOT / src))
            for src, dst in FILES
        ],
        commit_message="Deploy CropGuard demo panel",
    )
    print(f"\nhttps://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
