"""Tests for application startup behavior."""

import pytest

from backend import main


@pytest.mark.asyncio
async def test_app_lifespan_fails_when_cnn_model_not_ready(monkeypatch):
    monkeypatch.setattr(main, "_get_cnn_reader", lambda: (None, "model not found"))

    with pytest.raises(RuntimeError, match="Failed to load CNN model at startup"):
        async with main._app_lifespan(main.app):
            pass


@pytest.mark.asyncio
async def test_app_lifespan_fails_when_reader_missing_without_error(monkeypatch):
    monkeypatch.setattr(main, "_get_cnn_reader", lambda: (None, None))

    with pytest.raises(RuntimeError, match="Failed to load CNN model at startup"):
        async with main._app_lifespan(main.app):
            pass


@pytest.mark.asyncio
async def test_app_lifespan_succeeds_when_cnn_model_ready(monkeypatch):
    monkeypatch.setattr(main, "_get_cnn_reader", lambda: (object(), None))

    async with main._app_lifespan(main.app):
        pass
