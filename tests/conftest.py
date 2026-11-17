"""Shared fixtures for Shop's backend test suite.

Every test that touches persistence gets a fresh storage root under
``var/test-runs`` so the developer database at ``var/shop.db`` is never
modified. Storage roots must stay inside the project directory by contract.
"""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from shop_api.app import create_app
from shop_core.config import PROJECT_ROOT, Settings, load_settings

TEST_RUNS_ROOT = PROJECT_ROOT / "var" / "test-runs"


def _settings_for_root(root: Path) -> Settings:
    settings = load_settings().model_copy(deep=True)
    root.mkdir(parents=True, exist_ok=True)
    settings.storage.root = root
    return settings


@pytest.fixture(scope="session", autouse=True)
def _prepare_test_runs_root() -> Iterator[None]:
    TEST_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    # Remove leftovers from previous interrupted runs, keeping the newest ones.
    for path in sorted(TEST_RUNS_ROOT.iterdir(), key=lambda item: item.stat().st_mtime):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    yield
    for path in TEST_RUNS_ROOT.iterdir():
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


@pytest.fixture()
def storage_root() -> Iterator[Path]:
    root = TEST_RUNS_ROOT / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    (root / "uploads").mkdir()
    (root / "evidence").mkdir()
    (root / "thumbnails").mkdir()
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture()
def settings(storage_root: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.delenv("SHOP_API_TOKEN", raising=False)
    return _settings_for_root(storage_root)


@pytest.fixture()
def app_factory(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., tuple[object, TestClient]]:
    """Build an app against the isolated settings and open its lifespan."""

    def factory(*, token: str | None = None, **setting_overrides) -> tuple[object, TestClient]:
        if token is not None:
            monkeypatch.setenv("SHOP_API_TOKEN", token)
        else:
            monkeypatch.delenv("SHOP_API_TOKEN", raising=False)
        for dotted, value in setting_overrides.items():
            target = settings
            parts = dotted.split("__")
            for part in parts[:-1]:
                target = getattr(target, part)
            setattr(target, parts[-1], value)
        app = create_app(settings, auto_migrate=True)
        client = TestClient(app)
        client.__enter__()
        return app, client

    opened: list[TestClient] = []

    def tracked_factory(*args, **kwargs):  # noqa: ANN002, ANN003
        app, client = factory(*args, **kwargs)
        opened.append(client)
        return app, client

    try:
        yield tracked_factory
    finally:
        for client in reversed(opened):
            client.__exit__(None, None, None)


@pytest.fixture()
def client(app_factory):  # noqa: ANN001
    _app, test_client = app_factory()
    return test_client


@pytest.fixture()
def app(client):  # noqa: ANN001
    return client.app


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny real AVI encoded and decoded by the local OpenCV build."""
    root = tmp_path_factory.mktemp("shop-fixture-video")
    path = root / "sample.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (96, 64)
    )
    if not writer.isOpened():
        pytest.skip("local OpenCV build cannot encode the test fixture video")
    for index in range(20):
        frame = np.zeros((64, 96, 3), dtype=np.uint8)
        frame[:, :, 1] = index * 10
        writer.write(frame)
    writer.release()
    if not path.is_file() or path.stat().st_size == 0:
        pytest.skip("local OpenCV build produced no test fixture video")
    return path


@pytest.fixture()
def sample_video_bytes(sample_video: Path) -> bytes:
    return sample_video.read_bytes()
