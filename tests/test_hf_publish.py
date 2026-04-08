from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError

from devops_incident_triage.hf_publish import (
    build_permission_error_message,
    cleanup_upload_dir,
    create_repo_if_needed,
    get_repo_namespace,
    prepare_upload_dir,
    repo_exists,
    upload_model_folder,
)


def make_hf_error(status_code: int) -> HfHubHTTPError:
    request = httpx.Request("GET", "https://huggingface.co/api/test")
    response = httpx.Response(status_code, request=request)
    return HfHubHTTPError(f"http {status_code}", response=response)


class FakeApi:
    def __init__(self) -> None:
        self.create_repo_called = False
        self.upload_folder_called = False
        self.repo_info_error: HfHubHTTPError | None = None
        self.create_repo_error: HfHubHTTPError | None = None
        self.upload_folder_error: HfHubHTTPError | None = None
        self.username = "sdhcokr"

    def whoami(self) -> dict[str, str]:
        return {"name": self.username}

    def repo_info(self, repo_id: str, repo_type: str) -> dict[str, str]:
        if self.repo_info_error is not None:
            raise self.repo_info_error
        return {"repo_id": repo_id, "repo_type": repo_type}

    def create_repo(self, **_: str) -> None:
        self.create_repo_called = True
        if self.create_repo_error is not None:
            raise self.create_repo_error

    def upload_folder(self, **_: str) -> None:
        self.upload_folder_called = True
        if self.upload_folder_error is not None:
            raise self.upload_folder_error


def test_get_repo_namespace_returns_owner() -> None:
    assert get_repo_namespace("dongkoony/devops-incident-triage") == "dongkoony"


def test_repo_exists_returns_false_on_404() -> None:
    api = FakeApi()
    api.repo_info_error = make_hf_error(404)

    assert repo_exists(api, "dongkoony/devops-incident-triage") is False


def test_create_repo_if_needed_skips_creation_when_repo_exists() -> None:
    api = FakeApi()

    create_repo_if_needed(api, repo_id="dongkoony/devops-incident-triage", private=False)

    assert api.create_repo_called is False


def test_create_repo_if_needed_raises_friendly_permission_error() -> None:
    api = FakeApi()
    api.repo_info_error = make_hf_error(404)
    api.create_repo_error = make_hf_error(403)

    with pytest.raises(PermissionError) as exc:
        create_repo_if_needed(api, repo_id="dongkoony/devops-incident-triage", private=False)

    message = str(exc.value)
    assert "sdhcokr" in message
    assert "dongkoony" in message


def test_upload_model_folder_raises_friendly_permission_error(tmp_path: Path) -> None:
    api = FakeApi()
    api.upload_folder_error = make_hf_error(403)
    upload_dir = tmp_path / "model"
    upload_dir.mkdir()

    with pytest.raises(PermissionError) as exc:
        upload_model_folder(
            api,
            repo_id="dongkoony/devops-incident-triage",
            upload_dir=upload_dir,
            commit_message="test upload",
        )

    message = str(exc.value)
    assert "upload to" in message
    assert "dongkoony" in message


def test_prepare_upload_dir_copies_model_card_and_cleanup(tmp_path: Path) -> None:
    model_dir = tmp_path / "model_dir"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    model_card_path = tmp_path / "model_card.md"
    model_card_path.write_text("# card", encoding="utf-8")

    upload_dir = prepare_upload_dir(model_dir, model_card_path)

    assert (upload_dir / "config.json").exists()
    assert (upload_dir / "README.md").read_text(encoding="utf-8") == "# card"

    cleanup_upload_dir(upload_dir)
    assert upload_dir.exists() is False


def test_build_permission_error_message_mentions_namespace_and_user() -> None:
    api = FakeApi()
    message = build_permission_error_message(
        api,
        repo_id="dongkoony/devops-incident-triage",
        action="upload to",
    )
    assert "sdhcokr" in message
    assert "dongkoony" in message
