"""Fetch model artifacts from HF Hub at container startup if not baked into the image.

Env: CROPGUARD_HF_REPO (e.g. "abhinavbhatia/cropguard-models"),
     CROPGUARD_MODEL_FILE (default "cropguard.int8.onnx").
No-op when the model file already exists or CROPGUARD_HF_REPO is unset.
"""

import os
from pathlib import Path


def main() -> None:
    repo = os.environ.get("CROPGUARD_HF_REPO")
    model_path = Path(os.environ.get("CROPGUARD_MODEL_PATH", "models/cropguard.int8.onnx"))
    if model_path.exists():
        print(f"Model already present: {model_path}")
        return
    if not repo:
        print("CROPGUARD_HF_REPO not set and no local model — API will start degraded.")
        return

    from huggingface_hub import hf_hub_download

    model_path.parent.mkdir(parents=True, exist_ok=True)
    filename = os.environ.get("CROPGUARD_MODEL_FILE", model_path.name)
    downloaded = hf_hub_download(repo_id=repo, filename=filename)
    model_path.write_bytes(Path(downloaded).read_bytes())
    print(f"Downloaded {repo}/{filename} -> {model_path}")


if __name__ == "__main__":
    main()
