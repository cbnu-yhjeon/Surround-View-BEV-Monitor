"""스크립트 공통: 저장소 루트를 import 경로에 넣고, 결과 저장 도우미."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import RESULTS  # noqa: E402


def out_dir(name: str) -> Path:
    d = RESULTS / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(obj, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)


def machine_note() -> dict:
    """속도 수치 해석용: 측정 당시 부하."""
    try:
        load = os.getloadavg()
    except OSError:
        load = (float("nan"),) * 3
    return {"cpu_count": os.cpu_count(), "loadavg_1_5_15": [round(x, 2) for x in load]}


def md_table(rows: list[dict], cols: list[str], fmt: dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(format(v, fmt[c]) if c in fmt and isinstance(v, (int, float)) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)
