"""Load scanner.toml. Every threshold lives there; logic reads it from here."""

import tomllib
from pathlib import Path

PATH = Path(__file__).with_name("scanner.toml")


def load(path: str | Path = PATH) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)
