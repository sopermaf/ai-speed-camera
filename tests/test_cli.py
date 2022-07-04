"""Test cases for the __main__ module."""
import pathlib

import pytest
from click.testing import CliRunner

from ai_speed_camera import __main__


SAMPLE_DIR = pathlib.Path(__file__).absolute().parent.parent / "sample_data"


@pytest.fixture
def runner() -> CliRunner:
    """Fixture for invoking command-line interfaces."""
    return CliRunner()


def test_main_succeeds(runner: CliRunner) -> None:
    """It exits with a status code of zero."""
    result = runner.invoke(
        __main__.main,
        [
            "--annotations",
            str(SAMPLE_DIR / "output.json"),
            "--video",
            str(SAMPLE_DIR / "dene_road.mp4"),
            "--distance",
            32,
            "--output",
            "dest.mp4",
        ],
    )
    assert result.exit_code == 0
