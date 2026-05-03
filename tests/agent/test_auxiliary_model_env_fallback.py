"""Regression tests for env-only main-model fallback in auxiliary routing."""

from unittest.mock import patch


def test_read_main_model_falls_back_to_hermes_inference_model(monkeypatch):
    monkeypatch.setenv("HERMES_INFERENCE_MODEL", "google/gemini-2.5-flash-preview")
    monkeypatch.delenv("HERMES_MODEL", raising=False)

    with patch("hermes_cli.config.load_config", return_value={"model": {}}):
        from agent.auxiliary_client import _read_main_model

        assert _read_main_model() == "google/gemini-2.5-flash-preview"


def test_read_main_model_falls_back_to_hermes_model(monkeypatch):
    monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)
    monkeypatch.setenv("HERMES_MODEL", "anthropic/claude-sonnet-4.6")

    with patch("hermes_cli.config.load_config", return_value={}):
        from agent.auxiliary_client import _read_main_model

        assert _read_main_model() == "anthropic/claude-sonnet-4.6"
