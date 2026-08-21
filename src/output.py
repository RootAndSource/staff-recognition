"""结构化输出：时间、人员 ID、bbox、工装标签、置信度，JSON 逐帧或降频写出。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .identity import PersonObject
from .uniform import UniformResult


class JsonOutputWriter:
    """追加式 JSONL 写入器（每行一条记录，便于 B/C/D 消费）。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
        self._file.write("[")
        self._first = True

    def write_frame(
        self,
        *,
        frame_id: int,
        timestamp_ms: float,
        records: list[dict[str, Any]],
    ) -> None:
        """写入一帧的结构化记录（records 由 pipeline 组装）。"""
        entry = {
            "frame_id": frame_id,
            "timestamp_ms": round(timestamp_ms, 3),
            "utc_time": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "people": records,
        }
        if not self._first:
            self._file.write(",")
        self._file.write("\n" + json.dumps(entry, ensure_ascii=False))
        self._first = False

    def close(self) -> None:
        if self._file is None:
            return
        self._file.write("\n]\n")
        self._file.close()
        self._file = None

    def __enter__(self) -> "JsonOutputWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def build_person_record(
    person: PersonObject,
) -> dict[str, Any]:
    """把稳定人员对象组装成 JSON 记录。"""
    return {
        "person_id": person.person_id,
        "bt_id": person.bt_id,
        "bbox": [float(v) for v in person.xyxy],
        "uniform": person.uniform,
        "uniform_state": person.uniform_state,
        "uniform_prob": round(person.uniform_prob, 4),
        "person_restored": person.restored,
        "restored_from": person.restored_from,
        "red_ratio_connected": round(person.red_ratio_connected, 4),
        "baseline_ratio": round(person.baseline_ratio, 4),
        "confidence": round(person.confidence, 4),
        "mask_polygon": (
            [[round(float(x), 6), round(float(y), 6)] for x, y in person.polygon_norm]
            if person.polygon_norm is not None
            else None
        ),
    }
