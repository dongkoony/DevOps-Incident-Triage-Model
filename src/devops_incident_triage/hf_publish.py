from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError


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


def get_repo_namespace(repo_id: str) -> str:
    namespace, _, _ = repo_id.partition("/")
    return namespace


def get_token_username(api: HfApi) -> str | None:
    try:
        whoami = api.whoami()
    except Exception:  # pragma: no cover - defensive fallback
        return None
    if not isinstance(whoami, dict):
        return None
    username = whoami.get("name")
    return username if isinstance(username, str) else None


def repo_exists(api: HfApi, repo_id: str) -> bool:
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
    except HfHubHTTPError as exc:
        if exc.response.status_code == 404:
            return False
        raise
    return True


def build_permission_error_message(api: HfApi, repo_id: str, action: str) -> str:
    namespace = get_repo_namespace(repo_id)
    username = get_token_username(api)
    token_hint = f"'{username}'" if username else "the current token"
    return (
        f"Permission denied while trying to {action} Hugging Face repo '{repo_id}'. "
        f"The current token appears to belong to {token_hint}. "
        f"Use a token that can write to namespace '{namespace}', or change --repo-id "
        "to a namespace the token owns."
    )


def create_repo_if_needed(api: HfApi, repo_id: str, private: bool) -> None:
    if repo_exists(api, repo_id):
        return
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
        )
    except HfHubHTTPError as exc:
        if exc.response.status_code == 403:
            raise PermissionError(build_permission_error_message(api, repo_id, "create")) from exc
        raise


def upload_model_folder(
    api: HfApi,
    repo_id: str,
    upload_dir: Path,
    commit_message: str,
) -> None:
    try:
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(upload_dir),
            repo_type="model",
            commit_message=commit_message,
        )
    except HfHubHTTPError as exc:
        if exc.response.status_code == 403:
            raise PermissionError(
                build_permission_error_message(api, repo_id, "upload to")
            ) from exc
        raise


def prepare_upload_dir(model_dir: Path, model_card_path: Path) -> Path:
    upload_root = Path(tempfile.mkdtemp(prefix="hf_upload_"))
    upload_dir = upload_root / "model_upload"
    shutil.copytree(model_dir, upload_dir)
    if model_card_path.exists() and not (upload_dir / "README.md").exists():
        shutil.copy(model_card_path, upload_dir / "README.md")
    return upload_dir


def cleanup_upload_dir(upload_dir: Path) -> None:
    if upload_dir.exists():
        shutil.rmtree(upload_dir.parent)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.model_dir.exists():
        raise FileNotFoundError(f"{args.model_dir} not found. Train the model first.")
    if not args.token:
        raise ValueError("HF token is required. Set --token or export HF_TOKEN.")

    api = HfApi(token=args.token)
    create_repo_if_needed(api, repo_id=args.repo_id, private=args.private)

    upload_dir = prepare_upload_dir(args.model_dir, args.model_card_path)
    try:
        upload_model_folder(
            api,
            repo_id=args.repo_id,
            upload_dir=upload_dir,
            commit_message=args.commit_message,
        )
    finally:
        cleanup_upload_dir(upload_dir)
    print(f"Uploaded model artifacts to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
