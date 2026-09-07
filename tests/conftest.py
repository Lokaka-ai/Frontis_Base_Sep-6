import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVO = ROOT / "upstream/OpenRSI/OpenMLE-Evo"
sys.path[:0] = [str(ROOT), str(EVO), str(EVO / "third_party/aira-evo/src")]
os.environ.setdefault("LOGGING_DIR", str(ROOT / "artifacts/test-logs"))
