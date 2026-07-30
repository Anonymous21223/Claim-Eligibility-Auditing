from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crop_yield_xai.xai_pipeline import run_xai_pipeline  # noqa: E402


if __name__ == "__main__":
    run_xai_pipeline(PROJECT_ROOT)
