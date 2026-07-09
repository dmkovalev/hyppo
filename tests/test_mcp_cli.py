"""Smoke tests for hyppo.mcp.cli argparse."""

import pytest

from hyppo.mcp.cli import main


def test_help_does_not_crash(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "hyppo Model Context Protocol server" in out


def test_unknown_transport_rejected(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--transport", "carrierpigeon"])
    assert exc_info.value.code != 0
