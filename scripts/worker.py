"""Internal entry point: force imports from this snapshot, then install audit hooks."""

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVO = ROOT / "upstream/OpenRSI/OpenMLE-Evo"
AIRA = EVO / "third_party/aira-evo"
sys.path[:0] = [str(ROOT), str(AIRA / "src"), str(EVO)]
from frontis_mila.runtime import install

install()
runpy.run_path(
    str(AIRA / "examples/mle_bench/single_task_runner.py"), run_name="__main__"
)
