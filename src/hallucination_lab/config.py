"""Runtime settings, resolved from environment variables with safe defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_DIR = Path("lab_data")


@dataclass(frozen=True)
class Settings:
    data_dir: Path

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def collections_dir(self) -> Path:
        """User-created reference collections."""
        return self.data_dir / "collections"

    @property
    def indexes_dir(self) -> Path:
        """Processed (indexed) collections, regenerable with `hallab index build`."""
        return self.data_dir / "indexes"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(data_dir=Path(os.environ.get("HALLAB_DATA_DIR", DEFAULT_DATA_DIR)))
