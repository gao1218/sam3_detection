"""Streamlit entry point for the SAM 3 image segmentation demo."""
from pathlib import Path
import sys

PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sam3_app.ui import main


if __name__ == "__main__":
    main()

