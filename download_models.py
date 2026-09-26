"""Download the model weights from the Hugging Face Hub into src/agents/."""

from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "kingkuntairfan/nutri-fusion"
AGENTS_DIR = Path(__file__).resolve().parent / "src" / "agents"

# Paths in the HF repo; they land at src/agents/<path>
FILES = [
    "vision/best.pt",
    "vision/Swin.pth",
    "text/model.safetensors",
]


def main() -> None:
    for filename in FILES:
        path = hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=AGENTS_DIR)
        print(f"OK  {Path(path).relative_to(AGENTS_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
