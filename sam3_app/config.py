"""Application settings loaded from environment variables."""
from dataclasses import dataclass
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    device: str = os.getenv("SAM3_DEVICE", "cuda")
    checkpoint: Path = Path(os.getenv("SAM3_CHECKPOINT", str(PROJECT_ROOT / "sam3" / "sam3.pt")))
    source_dir: Path = Path(os.getenv("SAM3_SOURCE", str(PROJECT_ROOT / "sam3")))
    confidence_threshold: float = float(os.getenv("SAM3_CONFIDENCE", "0.35"))


settings = Settings()

