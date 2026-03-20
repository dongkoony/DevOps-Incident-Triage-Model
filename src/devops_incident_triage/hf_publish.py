from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload trained model artifacts to Hugging Face Hub."
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models/devops-incident-triage"))
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Format: <username>/<model-repo>",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="HF token or env HF_TOKEN",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--model-card-path", type=Path, default=Path("docs/model_card.md"))
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Add model artifacts from local training run",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.model_dir.exists():
        raise FileNotFoundError(f"{args.model_dir} not found. Train the model first.")
    if not args.token:
        raise ValueError("HF token is required. Set --token or export HF_TOKEN.")

    api = HfApi(token=args.token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        upload_dir = Path(tmp_dir) / "model_upload"
        shutil.copytree(args.model_dir, upload_dir)
        if args.model_card_path.exists() and not (upload_dir / "README.md").exists():
            shutil.copy(args.model_card_path, upload_dir / "README.md")

        api.upload_folder(
            repo_id=args.repo_id,
            folder_path=str(upload_dir),
            repo_type="model",
            commit_message=args.commit_message,
        )
    print(f"Uploaded model artifacts to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
