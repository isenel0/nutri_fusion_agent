"""Upload the runtime model weights to the Hugging Face Hub in a single commit."""

from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO_ID = "kingkuntairfan/nutri-fusion"
ROOT = Path(__file__).resolve().parent

# local path (relative to this file) -> path in the HF repo
FILES = {
    "src/agents/vision/best.pt": "vision/best.pt",
    "src/agents/vision/Swin.pth": "vision/Swin.pth",
    "src/agents/text/model.safetensors": "text/model.safetensors",
    "src/agents/text/config.json": "text/config.json",
    "src/agents/text/label_map.json": "text/label_map.json",
    "src/agents/text/tokenizer.json": "text/tokenizer.json",
    "src/agents/text/tokenizer_config.json": "text/tokenizer_config.json",
}


def main() -> None:
    missing = [local for local in FILES if not (ROOT / local).exists()]
    if missing:
        raise FileNotFoundError(f"Missing model files: {missing}")

    api = HfApi()
    operations = [
        CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(ROOT / local))
        for local, remote in FILES.items()
    ]
    commit = api.create_commit(
        repo_id=REPO_ID,
        repo_type="model",
        operations=operations,
        commit_message="Upload Nutri Fusion model weights",
    )
    print(f"Uploaded {len(operations)} files: {commit.commit_url}")


if __name__ == "__main__":
    main()
